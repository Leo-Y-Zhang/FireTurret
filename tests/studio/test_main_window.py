# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""P1a Task 10 + P1b: the spine (edit -> Run -> live plots), Runs dock, file ops."""
from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtCore import QSettings, QThread

from triton.config import TritonConfig
from triton.studio.main_window import MainWindow
from triton.studio.sweep import SweepAxis, SweepSpec


def _temp_settings(tmp_path) -> QSettings:
    # IniFormat at a temp path keeps recent-files out of the real registry
    return QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)


def test_initial_empty_state_and_clean_title(qapp):
    win = MainWindow(max_frames=120)
    assert win.plots._stack.currentWidget() is win.plots._empty
    assert win.windowTitle() == "Triton Studio"


def test_edit_marks_dirty_updates_title_and_ballistics(qapp):
    win = MainWindow(max_frames=120)
    before = win.ballistics.arc_curve.getData()[0][-1]
    win.set_param("jet", "drag_k", 0.4)  # more drag -> shorter arc
    assert win.session.config.jet.drag_k == 0.4
    assert win.session.dirty is True
    assert win.windowTitle().startswith("*")
    after = win.ballistics.arc_curve.getData()[0][-1]
    assert after < before  # the ballistics explorer re-drew from the edited jet


def test_run_drives_worker_and_populates_plots(qapp, qtbot):
    win = MainWindow(max_frames=150)
    with qtbot.waitSignal(win.run_finished, timeout=30000) as blocker:
        win.run()
    report = blocker.args[0]
    assert report is not None
    assert len(win.plots._samples) > 0
    assert win.plots._stack.currentWidget() is win.plots._glw
    assert win.is_running() is False
    win.close()
    qtbot.wait(50)  # let deleteLater process the worker/thread


def test_finished_run_appears_in_runs_table(qapp, qtbot):
    win = MainWindow(max_frames=150)
    assert win.runs_model.rowCount() == 0
    with qtbot.waitSignal(win.run_finished, timeout=30000):
        win.run()
    assert win.runs_model.rowCount() == 1
    assert win.runs_model.rows[0].seed == 7
    win.close()
    qtbot.wait(50)


def test_replay_and_compare_runs(qapp, qtbot, tmp_path):
    win = MainWindow(max_frames=150, settings=_temp_settings(tmp_path))
    for _ in range(2):  # two runs, each with retained telemetry
        with qtbot.waitSignal(win.run_finished, timeout=30000):
            win.run()
    assert win.runs_model.rowCount() == 2
    assert len(win._run_data) == 2 and win._run_data[0]["samples"]

    win._replay_run(0)  # replay row 0 into the plots
    assert len(win.plots._samples) == len(win._run_data[0]["samples"])

    win._compare_runs([0, 1])  # overlay both
    assert win._compare_window is not None

    # editing the model marks the existing runs stale
    win.set_param("jet", "drag_k", 0.42)
    assert win.runs_model.rows[0].stale is True

    win._delete_run(0)  # delete keeps run_data aligned with the rows
    assert win.runs_model.rowCount() == 1
    assert len(win._run_data) == 1
    win.close()
    qtbot.wait(50)


def test_joining_a_live_run_thread_stops_it_instead_of_abandoning_it(qapp):
    """A running QThread must be stopped before its last Python reference goes.

    PySide destroys the C++ QThread when the wrapper is collected; if the thread
    is still running Qt calls qFatal and the process aborts with
    "Fatal Python error: Aborted" and no traceback. Deterministic unit test of
    the join, with no dependence on the race that exposed it in CI.
    """
    win = MainWindow(max_frames=120)
    live = QThread()
    live.start()
    assert live.isRunning() is True

    win._thread = live
    win._join_run_thread()

    assert live.isRunning() is False, "join must stop the thread, not abandon it"
    # The reference is deliberately kept: dropping it here would free a QObject
    # that still belongs to the retiring thread.
    assert win._thread is live
    assert win._stuck_threads == []
    win._thread = None


def test_a_second_run_never_releases_the_first_runs_thread_while_it_lives(qapp, qtbot):
    """Regression for the intermittent CI abort at main_window.run.

    The crash frame was the `self._thread = thread` rebinding on the second run:
    worker.finished -> thread.quit() is queued into the GUI thread, so straight
    after run_finished the first thread is usually still spinning, and dropping
    its reference there aborted the process.
    """
    win = MainWindow(max_frames=150)
    with qtbot.waitSignal(win.run_finished, timeout=30000):
        win.run()
    first = win._thread
    assert first is not None

    with qtbot.waitSignal(win.run_finished, timeout=30000):
        win.run()
        # Checked the instant run() returns: the old thread was joined before the
        # new one replaced it, so letting `first` go can no longer abort.
        assert first.isRunning() is False
        assert win._thread is not first
        assert win._stuck_threads == []

    win.close()
    qtbot.wait(50)


def test_failed_routes_to_error_and_does_not_crash(qapp):
    win = MainWindow(max_frames=120)
    win._on_failed("RuntimeError: boom")
    assert win._last_error == "RuntimeError: boom"


def test_stop_with_no_worker_is_noop(qapp):
    win = MainWindow(max_frames=120)
    win.stop()  # must not raise


def test_layout_persistence_roundtrip(qapp, tmp_path):
    settings = _temp_settings(tmp_path)
    win = MainWindow(max_frames=120, settings=settings)
    win._save_layout()
    assert settings.value("layout/state") is not None
    # a fresh window restores the saved layout without error
    MainWindow(max_frames=120, settings=settings)
    win._reset_layout()  # reset clears the stored layout
    assert settings.value("layout/state") is None


def test_reset_layout_is_safe(qapp, tmp_path):
    win = MainWindow(max_frames=120, settings=_temp_settings(tmp_path))
    win._reset_layout()  # must not raise on a clean settings store


def test_about_text_has_versions(qapp):
    win = MainWindow(max_frames=120)
    txt = win._about_text()
    assert "Triton Studio" in txt
    assert "PySide6" in txt
    assert "pyqtgraph" in txt


def test_session_save_and_open_roundtrip(qapp, tmp_path):
    settings = _temp_settings(tmp_path)
    win = MainWindow(max_frames=120, settings=settings)
    win.set_param("jet", "drag_k", 0.33)
    assert win.session.dirty is True
    path = tmp_path / "s.triton.json"
    win.save_session_path(path)
    assert win.session.dirty is False  # mark_saved cleared it
    assert path.exists()

    win2 = MainWindow(max_frames=120, settings=settings)
    win2.open_session_path(path)
    assert win2.session.config.jet.drag_k == pytest.approx(0.33)
    # editor was rebound to the loaded session
    assert win2.editor.session is win2.session


def test_new_session_resets_to_defaults(qapp, tmp_path):
    win = MainWindow(max_frames=120, settings=_temp_settings(tmp_path))
    win.set_param("jet", "drag_k", 0.4)
    win.new_session()
    assert win.session.config.jet.drag_k == TritonConfig().jet.drag_k
    assert win.session.dirty is False


def test_new_session_clears_prior_runs(qapp, qtbot, tmp_path):
    win = MainWindow(max_frames=150, settings=_temp_settings(tmp_path))
    with qtbot.waitSignal(win.run_finished, timeout=30000):
        win.run()
    assert win.runs_model.rowCount() == 1
    win.new_session()  # a session swap supersedes prior runs
    assert win.runs_model.rowCount() == 0
    assert win._run_data == []
    win.close()
    qtbot.wait(50)


def test_recent_files_tracked_on_save(qapp, tmp_path):
    win = MainWindow(max_frames=120, settings=_temp_settings(tmp_path))
    path = tmp_path / "s.triton.json"
    win.save_session_path(path)
    assert str(path) in win._recent_paths()


def test_batch_sweep_runs_and_shows_results(qapp, qtbot, tmp_path):
    win = MainWindow(max_frames=120, settings=_temp_settings(tmp_path))
    spec = SweepSpec(axes=[SweepAxis("jet", "drag_k", [0.14, 0.18])],
                     seeds=[7], max_frames=120)
    with qtbot.waitSignal(win.sweep_finished, timeout=60000) as blocker:
        win.start_sweep(spec, "water_l")
    results = blocker.args[0]
    assert len(results) == 2
    assert win._sweep_window is not None


def test_start_sweep_is_not_reentrant(qapp, qtbot, tmp_path):
    """Launching a second sweep while one is in flight must be ignored, not stack
    a second runner (which would interleave progress + open duplicate result
    windows). The Batch-sweep action is disabled while a sweep runs."""
    win = MainWindow(max_frames=120, settings=_temp_settings(tmp_path))
    spec = SweepSpec(axes=[SweepAxis("jet", "drag_k", [0.14, 0.18])],
                     seeds=[7], max_frames=120)
    win.start_sweep(spec, "water_l")
    first_runner = win._sweep_runner
    assert win._sweeping is True
    assert win._sweep_act.isEnabled() is False
    win.start_sweep(spec, "water_l")  # second call while running
    assert win._sweep_runner is first_runner  # not replaced by a second runner
    # let the real sweep finish so the guard clears and the action re-enables
    with qtbot.waitSignal(win.sweep_finished, timeout=60000):
        pass
    assert win._sweeping is False
    assert win._sweep_act.isEnabled() is True


def test_save_last_run_writes_files(qapp, qtbot, tmp_path):
    win = MainWindow(max_frames=150, settings=_temp_settings(tmp_path))
    with qtbot.waitSignal(win.run_finished, timeout=30000):
        win.run()
    json_path = win.save_last_run_path(tmp_path, "myrun")
    assert json_path is not None
    assert Path(json_path).exists()
    assert (tmp_path / "myrun.csv").exists()
    win.close()
    qtbot.wait(50)
