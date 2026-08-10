# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""FireTurret Studio main window: the demoable spine — edit a model parameter, Run a
simulation on a worker thread, and watch the live telemetry plots. Also hosts the
ballistics explorer, a status bar (progress + state + cancel), a global error
dialog, and an undo/redo-backed Edit menu.
"""
from __future__ import annotations

from datetime import datetime

from PySide6.QtCore import QSettings, Qt, QThread, Signal, qVersion
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDockWidget,
    QFileDialog,
    QHeaderView,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTableView,
    QToolBar,
    QVBoxLayout,
)

from ..ballistics import reach_bounds
from . import io
from .models.runs_table import RunRow, RunsTableModel
from .session import Session
from .sweep import SweepRunner, sweep_grid
from .views.advisories_view import AdvisoriesView
from .views.ballistics_view import BallisticsView
from .views.calibration_panel import CalibrationPanel
from .views.camera_view import CameraView
from .views.compare_view import RunCompareView
from .views.errors import install_excepthook
from .views.model_editor import ModelEditor
from .views.plots import TelemetryPlots
from .views.sweep_dialog import SweepDialog
from .views.sweep_view import SweepResultsView
from .views.topdown import TopDownView
from .worker import SimWorker

_LAYOUT_VERSION = 1  # bump when the dock set changes; restoreState mismatches -> default
_THREAD_EXIT_MS = 5000  # how long to join a finished run's thread before parking it


class MainWindow(QMainWindow):
    run_finished = Signal(object)     # SimReport (or None on failure)
    sweep_finished = Signal(object)   # list[SweepResult]

    def __init__(self, session: Session | None = None, max_frames: int = 6000,
                 settings: QSettings | None = None, parent=None) -> None:
        super().__init__(parent)
        self.session = session or Session()
        self._settings = settings or QSettings()
        self._max_frames = max_frames
        self._worker: SimWorker | None = None
        self._thread: QThread | None = None
        self._stuck_threads: list[QThread] = []  # never GC a QThread that is still running
        self._running = False
        self._reach = (3.0, 9.0)
        self._first_batch = True
        self._last_error: str | None = None
        self._last_report = None
        self._run_counter = 0
        self._session_path: str | None = None
        self._sweep_runner: SweepRunner | None = None
        self._sweep_window: QDialog | None = None
        self._sweep_spec = None
        self._sweep_metric = "water_l"
        self._sweeping = False  # a batch sweep is in flight (reentrancy guard)
        self._run_data: list[dict] = []  # per-run {samples, reach}, aligned with runs_model.rows
        self._compare_window: QDialog | None = None

        self.resize(1360, 880)

        self.plots = TelemetryPlots()
        self.setCentralWidget(self.plots)

        self.ballistics = BallisticsView()
        dock_b = QDockWidget("Ballistics", self)
        dock_b.setObjectName("dock_ballistics")
        dock_b.setWidget(self.ballistics)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock_b)

        self.topdown = TopDownView()
        dock_t = QDockWidget("Top-down", self)
        dock_t.setObjectName("dock_topdown")
        dock_t.setWidget(self.topdown)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock_t)
        self.tabifyDockWidget(dock_b, dock_t)

        self.camera = CameraView()
        dock_c = QDockWidget("Camera", self)
        dock_c.setObjectName("dock_camera")
        dock_c.setWidget(self.camera)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock_c)
        self.tabifyDockWidget(dock_b, dock_c)
        dock_b.raise_()
        self._sync_scene_views()

        self.editor = ModelEditor(self.session)
        self._model_dock = QDockWidget("Model", self)
        self._model_dock.setObjectName("dock_model")
        self._model_dock.setWidget(self.editor)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self._model_dock)

        self.calibration = CalibrationPanel(self.session)
        self._cal_dock = QDockWidget("Calibration", self)
        self._cal_dock.setObjectName("dock_calibration")
        self._cal_dock.setWidget(self.calibration)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self._cal_dock)
        self.tabifyDockWidget(self._model_dock, self._cal_dock)
        self._model_dock.raise_()

        self._build_runs_dock()
        self._build_status_bar()
        self._build_menu()

        self.session.changed.connect(self._on_session_changed)
        # capture the default arrangement, then restore any saved one
        self._default_geometry = self.saveGeometry()
        self._default_state = self.saveState(_LAYOUT_VERSION)
        self._restore_layout()
        install_excepthook(self._report_error)
        self._update_title()

    # -- construction helpers -----------------------------------------------

    def _tilt_bounds(self) -> tuple[float, float]:
        t = self.session.config.turret
        return (t.tilt_min_deg, t.tilt_max_deg)

    def _build_runs_dock(self) -> None:
        self.runs_model = RunsTableModel()
        self.runs_view = QTableView()
        self.runs_view.setModel(self.runs_model)
        self.runs_view.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.runs_view.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.runs_view.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.runs_view.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.runs_view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.runs_view.customContextMenuRequested.connect(self._runs_context_menu)
        self.runs_view.doubleClicked.connect(self._replay_run)  # double-click replays a run
        self._runs_dock = QDockWidget("Runs", self)
        self._runs_dock.setObjectName("dock_runs")
        self._runs_dock.setWidget(self.runs_view)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self._runs_dock)

        self.advisories_view = AdvisoriesView()
        adv_dock = QDockWidget("Advisories", self)
        adv_dock.setObjectName("dock_advisories")
        adv_dock.setWidget(self.advisories_view)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, adv_dock)
        self.tabifyDockWidget(self._runs_dock, adv_dock)
        self._runs_dock.raise_()

    def _runs_context_menu(self, pos) -> None:
        index = self.runs_view.indexAt(pos)
        if not index.isValid():
            return
        row = index.row()
        selected = self._selected_run_rows()
        menu = QMenu(self)
        replay_act = menu.addAction("Replay")
        compare_act = menu.addAction("Compare selected")
        compare_act.setEnabled(len(selected) >= 2)
        rename_act = menu.addAction("Rename…")
        delete_act = menu.addAction("Delete")
        chosen = menu.exec(self.runs_view.viewport().mapToGlobal(pos))
        if chosen is replay_act:
            self._replay_run(row)
        elif chosen is compare_act:
            self._compare_runs(selected)
        elif chosen is rename_act:
            self._rename_run(row)
        elif chosen is delete_act:
            self._delete_run(row)

    def _selected_run_rows(self) -> list[int]:
        return sorted({i.row() for i in self.runs_view.selectionModel().selectedRows()})

    def _rename_run(self, row: int) -> None:
        current = self.runs_model.rows[row].name
        name, ok = QInputDialog.getText(self, "Rename run", "Name:", text=current)
        if ok and name:
            self.runs_model.rename(row, name)

    def _delete_run(self, row: int) -> None:
        if 0 <= row < len(self._run_data):
            del self._run_data[row]
        self.runs_model.remove(row)

    def _replay_run(self, index) -> None:
        row = index.row() if hasattr(index, "row") else int(index)
        if 0 <= row < len(self._run_data) and self._run_data[row]["samples"]:
            data = self._run_data[row]
            self.plots.set_run(data["samples"], data["reach"])
            self._state_label.setText(f"replay: {self.runs_model.rows[row].name}")

    def _compare_runs(self, rows: list[int]) -> None:
        runs = [
            (self.runs_model.rows[r].name, self._run_data[r]["samples"])
            for r in rows
            if 0 <= r < len(self._run_data) and self._run_data[r]["samples"]
        ]
        if len(runs) < 2:
            return
        view = RunCompareView(runs)
        dialog = QDialog(self)
        dialog.setWindowTitle("Compare runs")
        dialog.resize(760, 520)
        layout = QVBoxLayout(dialog)
        layout.addWidget(view)
        self._compare_window = dialog
        dialog.show()

    def _build_status_bar(self) -> None:
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setMaximumWidth(220)
        self._state_label = QLabel("idle")
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.clicked.connect(self.stop)
        self._readout_label = QLabel("")  # crosshair readout from the plots
        self.plots.readout_changed.connect(self._readout_label.setText)
        sb = self.statusBar()
        sb.addWidget(self._readout_label, 1)
        sb.addPermanentWidget(self._state_label)
        sb.addPermanentWidget(self._progress)
        sb.addPermanentWidget(self._cancel_btn)

    def _build_menu(self) -> None:
        self._run_act = QAction("Run", self)
        self._run_act.setShortcut(QKeySequence("F5"))
        self._run_act.triggered.connect(self.run)
        self._stop_act = QAction("Stop", self)
        self._stop_act.setShortcut(QKeySequence("Shift+F5"))
        self._stop_act.triggered.connect(self.stop)

        tb = QToolBar("Main")
        tb.setObjectName("toolbar_main")
        tb.addAction(self._run_act)
        tb.addAction(self._stop_act)
        self.addToolBar(tb)

        mb = self.menuBar()
        self._build_file_menu(mb.addMenu("&File"))
        view_menu = mb.addMenu("&View")
        reset_act = QAction("Reset layout", self)
        reset_act.triggered.connect(self._reset_layout)
        view_menu.addAction(reset_act)
        run_menu = mb.addMenu("&Run")
        run_menu.addAction(self._run_act)
        run_menu.addAction(self._stop_act)
        run_menu.addSeparator()
        self._sweep_act = QAction("Batch sweep…", self)
        self._sweep_act.triggered.connect(self.open_sweep_dialog)
        run_menu.addAction(self._sweep_act)
        self._edit_menu = mb.addMenu("&Edit")
        self._install_edit_actions()
        help_menu = mb.addMenu("&Help")
        about_act = QAction("About FireTurret Studio", self)
        about_act.triggered.connect(self._show_about)
        help_menu.addAction(about_act)
        shortcuts_act = QAction("Keyboard shortcuts", self)
        shortcuts_act.triggered.connect(self._show_shortcuts)
        help_menu.addAction(shortcuts_act)

    def _build_file_menu(self, menu: QMenu) -> None:
        for text, key, slot in [
            ("New", QKeySequence.StandardKey.New, self.new_session),
            ("Open…", QKeySequence.StandardKey.Open, self.open_session),
            ("Save As…", QKeySequence.StandardKey.SaveAs, self.save_session_as),
        ]:
            act = QAction(text, self)
            act.setShortcut(key)
            act.triggered.connect(slot)
            menu.addAction(act)
        self._recent_menu = menu.addMenu("Recent")
        menu.addSeparator()
        save_run = QAction("Save last run…", self)
        save_run.triggered.connect(self.save_last_run)
        menu.addAction(save_run)
        menu.addSeparator()
        quit_act = QAction("Quit", self)
        quit_act.setShortcut(QKeySequence.StandardKey.Quit)
        quit_act.triggered.connect(self.close)
        menu.addAction(quit_act)
        self._rebuild_recent_menu()

    def _install_edit_actions(self) -> None:
        self._edit_menu.clear()
        undo = self.session.undo_stack.createUndoAction(self, "Undo")
        undo.setShortcut(QKeySequence.StandardKey.Undo)
        redo = self.session.undo_stack.createRedoAction(self, "Redo")
        redo.setShortcut(QKeySequence.StandardKey.Redo)
        self._edit_menu.addAction(undo)
        self._edit_menu.addAction(redo)

    # -- editing ------------------------------------------------------------

    def set_param(self, group: str, name: str, value) -> None:
        """Programmatic edit; the ModelEditor refreshes via `session.changed`. For tests."""
        self.session.set_field(group, name, value)

    # -- file ops -----------------------------------------------------------

    def new_session(self) -> None:
        self._rebind_session(Session())
        self._session_path = None

    def open_session(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open session", "", "FireTurret session (*.fireturret.json)")
        if path:
            self.open_session_path(path)

    def open_session_path(self, path) -> None:
        cfg, scenario = io.load_session(path)
        self._rebind_session(Session(cfg, scenario))
        self._session_path = str(path)
        self._add_recent(str(path))

    def save_session_as(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save session", "", "FireTurret session (*.fireturret.json)")
        if path:
            self.save_session_path(path)

    def save_session_path(self, path) -> None:
        io.save_session(path, self.session.config, self.session.scenario)
        self.session.mark_saved()
        self._session_path = str(path)
        self._add_recent(str(path))
        self._update_title()

    def save_last_run(self) -> None:
        if self._last_report is None:
            return
        dir_path = QFileDialog.getExistingDirectory(self, "Save run to folder")
        if dir_path:
            self.save_last_run_path(dir_path, f"run_{self._run_counter}")

    def save_last_run_path(self, dir_path, name: str = "run") -> str | None:
        if self._last_report is None:
            return None
        return io.save_run(dir_path, name, self.session.config, self.session.scenario,
                           7, self._last_report)

    def _rebind_session(self, new_session: Session) -> None:
        try:
            self.session.changed.disconnect(self._on_session_changed)
        except (TypeError, RuntimeError):
            pass
        self.session = new_session
        self.editor = ModelEditor(self.session)
        self._model_dock.setWidget(self.editor)
        self.calibration = CalibrationPanel(self.session)
        self._cal_dock.setWidget(self.calibration)
        # a whole-session swap supersedes every prior run; drop them so replay/
        # compare can't show telemetry from a different model as if it were current
        self.runs_model.clear()
        self._run_data.clear()
        self.plots.clear()
        self.advisories_view.set_report(None)
        self._sync_scene_views()
        self.session.changed.connect(self._on_session_changed)
        self._install_edit_actions()
        self._update_title()

    # -- recent files -------------------------------------------------------

    def _recent_paths(self) -> list:
        return self._settings.value("recent_files", [], type=list) or []

    def _add_recent(self, path: str) -> None:
        paths = [p for p in self._recent_paths() if p != path]
        paths.insert(0, path)
        self._settings.setValue("recent_files", paths[:10])
        self._rebuild_recent_menu()

    def _rebuild_recent_menu(self) -> None:
        self._recent_menu.clear()
        paths = self._recent_paths()
        if not paths:
            self._recent_menu.addAction("(none)").setEnabled(False)
            return
        for p in paths:
            act = self._recent_menu.addAction(p)
            act.triggered.connect(lambda checked=False, path=p: self.open_session_path(path))

    def _sync_scene_views(self) -> None:
        jet = self.session.config.jet
        scenario = self.session.scenario
        tilt = self._tilt_bounds()
        self.ballistics.set_jet(jet, tilt)
        self.ballistics.set_true_scales(scenario.velocity_coeff_scale, scenario.drag_scale)
        self.topdown.set_scene(scenario, jet, tilt)

    def _on_session_changed(self) -> None:
        self._sync_scene_views()
        if self.session.dirty:  # a real edit (not a save) invalidates past runs
            self.runs_model.mark_all_stale()
        self._update_title()

    def _update_title(self) -> None:
        self.setWindowTitle(("*" if self.session.dirty else "") + "FireTurret Studio")

    # -- run / stop ---------------------------------------------------------

    def is_running(self) -> bool:
        return self._running

    def _join_run_thread(self) -> None:
        """Stop the previous run's thread before anything can release it.

        Rebinding ``self._thread`` drops the last Python reference to the old
        QThread, so PySide destroys the C++ object. If that thread has not
        actually exited yet, Qt calls
        ``qFatal("QThread: Destroyed while thread is still running")`` and the
        process aborts -- "Fatal Python error: Aborted" pointing at the
        assignment, with no Python traceback to explain it.

        The window is easy to hit: ``worker.finished -> thread.quit()`` is a
        queued connection into the GUI thread, so a second Run started right
        after the first one's ``run_finished`` often arrives before ``quit()``
        has even been delivered, with the old thread still spinning its loop.

        Deliberately does NOT null ``_thread`` / ``_worker``: those references are
        what keep the old C++ objects alive until their replacements are bound.
        Clearing them here frees a QObject that still lives in the retiring
        thread, which trades the abort for an access violation - no improvement.
        """
        prev = self._thread
        if prev is None or not prev.isRunning():
            return
        prev.quit()
        if not prev.wait(_THREAD_EXIT_MS):
            # A thread that will not stop must still never be garbage-collected
            # while running: that is the abort path. Park the reference instead.
            # The leak is bounded by the number of stuck runs, and a stuck run is
            # a visible bug rather than a dead process.
            self._stuck_threads.append(prev)

    def run(self) -> None:
        if self._running:
            return
        self._join_run_thread()  # never rebind over a thread that is still running
        cfg = self.session.config
        self._reach = reach_bounds(cfg.jet, cfg.turret.tilt_min_deg, cfg.turret.tilt_max_deg)
        self.plots.clear()
        self.camera.clear()
        self._first_batch = True

        worker = SimWorker(cfg, self.session.scenario, seed=7,
                           max_frames=self._max_frames, record_telemetry=True)
        thread = QThread()
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.telemetry.connect(self._on_telemetry)
        worker.frame.connect(self.camera.set_frame)
        worker.progress.connect(self._on_progress)
        worker.finished.connect(self._on_finished)
        worker.failed.connect(self._on_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        # Qt's documented moveToThread idiom: the worker deletes itself off its
        # OWN finished/failed signal, while its thread's event loop is still
        # running to process the deferred-delete event. The earlier code instead
        # scheduled worker.deleteLater() off thread.finished, which fires only
        # after that event loop has already stopped -- posting a deferred delete
        # into a queue nothing may ever pump again. That is an undefined-timing
        # window at thread teardown, not a leak: which Qt/PySide point release is
        # resolved decides whether it is silently swallowed or corrupts memory,
        # which is consistent with the intermittent, offscreen/Linux-only SIGBUS
        # this test caught. Never null the Python refs inside a finished slot
        # (that crashes PySide via a different path); deleteLater is still the
        # only safe teardown call here.
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)

        self._worker = worker
        self._thread = thread
        self._running = True
        self._cancel_btn.setEnabled(True)
        self._state_label.setText("running")
        thread.start()

    def stop(self) -> None:
        if self._running and self._worker is not None:
            self._worker.cancel()

    # -- batch sweep --------------------------------------------------------

    def open_sweep_dialog(self) -> None:
        dialog = SweepDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            spec, metric = dialog.spec()
            self.start_sweep(spec, metric)

    def start_sweep(self, spec, metric: str) -> None:
        if self._sweeping:  # a sweep is already running: don't stack a second runner
            return
        self._sweep_spec = spec
        self._sweep_metric = metric
        cells = sweep_grid(self.session.config, self.session.scenario, spec)
        runner = SweepRunner(cells)
        self._sweep_runner = runner
        runner.progress.connect(self._on_sweep_progress)
        runner.finished.connect(self._on_sweep_finished)
        self._sweeping = True
        self._sweep_act.setEnabled(False)
        self._state_label.setText(f"sweep: 0/{len(cells)}")
        runner.start()

    def _on_sweep_progress(self, done: int, total: int) -> None:
        self._state_label.setText(f"sweep: {done}/{total}")
        self._progress.setValue(int(100 * done / total) if total else 0)

    def _on_sweep_finished(self, results) -> None:
        self._sweeping = False
        self._sweep_act.setEnabled(True)
        self._state_label.setText(f"sweep done ({len(results)} cells)")
        view = SweepResultsView()
        view.set_results(self._sweep_spec, results, self._sweep_metric)
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Sweep results — {self._sweep_metric}")
        dialog.resize(720, 520)
        layout = QVBoxLayout(dialog)
        layout.addWidget(view)
        self._sweep_window = dialog  # keep a ref so it isn't GC'd
        dialog.show()
        self.sweep_finished.emit(results)

    def _on_telemetry(self, batch) -> None:
        if not batch:
            return
        if self._first_batch:
            self.plots.set_run(batch, self._reach)
            self._first_batch = False
        else:
            self.plots.append(batch)
        self._state_label.setText(batch[-1].state)

    def _on_progress(self, frac: float) -> None:
        self._progress.setValue(int(frac * 100))

    def _on_finished(self, report) -> None:
        self._running = False
        self._cancel_btn.setEnabled(False)
        self._state_label.setText("extinguished" if report.extinguished else "stopped")
        self._progress.setValue(100)
        self._last_report = report
        self._run_counter += 1
        self.runs_model.add_run(
            RunRow.from_report(f"run {self._run_counter}", report, seed=7,
                               scenario="custom", timestamp=_timestamp())
        )
        # retain the telemetry (aligned with the rows) for replay + compare
        self._run_data.append({"samples": list(report.telemetry), "reach": self._reach})
        self.advisories_view.set_report(report)
        self.run_finished.emit(report)

    def _on_failed(self, msg: str) -> None:
        self._running = False
        self._cancel_btn.setEnabled(False)
        self._report_error(msg, msg)
        self.run_finished.emit(None)

    def _about_text(self) -> str:
        from importlib.metadata import PackageNotFoundError, version

        import pyqtgraph
        import PySide6

        try:
            fireturret_v = version("fireturret")
        except PackageNotFoundError:
            fireturret_v = "0.1.0"
        return (
            "<b>FireTurret Studio</b><br>Fire-suppression turret engineering workbench"
            f"<br><br>fireturret {fireturret_v}<br>PySide6 {PySide6.__version__}"
            f"<br>pyqtgraph {pyqtgraph.__version__}<br>default seed: 7"
        )

    def _show_about(self) -> None:
        QMessageBox.about(self, "About FireTurret Studio", self._about_text())

    def _show_shortcuts(self) -> None:
        QMessageBox.information(
            self, "Keyboard shortcuts",
            "Ctrl+N  New session\nCtrl+O  Open session\nCtrl+S  Save session\n"
            "Ctrl+Z / Ctrl+Y  Undo / Redo\nF5  Run\nShift+F5  Stop\n"
            "Right-click a run  Rename / Delete",
        )

    def _report_error(self, summary: str, details: str) -> None:
        self._last_error = summary
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Critical)
        box.setWindowTitle("FireTurret Studio — error")
        box.setText(summary)
        if details and details != summary:
            box.setDetailedText(details)
        box.setStandardButtons(QMessageBox.StandardButton.Ok)
        box.show()  # non-modal: never blocks

    # -- dock layout persistence --------------------------------------------

    def _save_layout(self) -> None:
        self._settings.setValue("layout/qt", qVersion())
        self._settings.setValue("layout/geometry", self.saveGeometry())
        self._settings.setValue("layout/state", self.saveState(_LAYOUT_VERSION))

    def _restore_layout(self) -> None:
        # saved dock-state is Qt-build-specific; a different Qt can decode a stale
        # blob to a garbled layout, so only restore when the Qt version matches
        if self._settings.value("layout/qt") != qVersion():
            return
        geometry = self._settings.value("layout/geometry")
        state = self._settings.value("layout/state")
        if geometry is not None:
            self.restoreGeometry(geometry)
        if state is not None and not self.restoreState(state, _LAYOUT_VERSION):
            # version/format mismatch -> keep the default arrangement
            self.restoreState(self._default_state, _LAYOUT_VERSION)

    def _reset_layout(self) -> None:
        self.restoreGeometry(self._default_geometry)
        self.restoreState(self._default_state, _LAYOUT_VERSION)
        self._settings.remove("layout")

    def closeEvent(self, event) -> None:
        if self._running and self._worker is not None:
            self._worker.cancel()
        # Unconditional: after a run reports finished its thread is usually still
        # winding down, and closing the window drops the last reference to it.
        self._join_run_thread()
        self._save_layout()
        super().closeEvent(event)


def _timestamp() -> str:
    return datetime.now().strftime("%H:%M:%S")
