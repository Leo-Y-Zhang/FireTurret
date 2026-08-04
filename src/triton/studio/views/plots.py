# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""Live telemetry plots: five stacked, X-linked pyqtgraph panels mirroring the
static ``analysis.write_report`` figure. Fed frozen TelemetrySample batches from
the SimWorker; the GUI never computes physics.

The five panels: mission-state timeline, aim errors, range vs the reachable
envelope, pump/valve, and fire intensity + cumulative water.
"""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QLabel, QStackedLayout, QWidget

_STATES = ["SEARCH", "ACQUIRE", "RANGE", "ALIGN", "SUPPRESS", "CONFIRM", "HOLD", "SAFE"]
_STATE_IDX = {s: i for i, s in enumerate(_STATES)}

INK = "#e8edf4"
ACCENT = "#4cc3e8"
HEAT = "#ff9f43"
OK = "#58c470"
WATER = "#4cc3e8"


class TelemetryPlots(QWidget):
    readout_changed = Signal(str)  # crosshair readout ("t=.. | az_err=.. | ..")

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.curves: dict[str, pg.PlotDataItem] = {}
        self._crosshairs: list = []
        self.data: dict[str, np.ndarray] = {}
        self.t: np.ndarray | None = None
        self._samples: list = []
        self._reach: tuple[float, float] | None = None

        self._stack = QStackedLayout(self)
        self._empty = QLabel("No run yet — press Run (F5)")
        self._empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._stack.addWidget(self._empty)

        self._glw = pg.GraphicsLayoutWidget()
        self._build_panels()
        self._stack.addWidget(self._glw)
        self._stack.setCurrentWidget(self._empty)

    def _build_panels(self) -> None:
        p_state = self._glw.addPlot(row=0, col=0)
        p_state.setLabel("left", "state")
        p_state.getAxis("left").setTicks([[(i, s) for i, s in enumerate(_STATES)]])
        self.curves["state"] = p_state.plot(pen=pg.mkPen(ACCENT, width=1.5))

        p_aim = self._glw.addPlot(row=1, col=0)
        p_aim.setLabel("left", "aim error")
        p_aim.addLegend(offset=(-5, 5))
        self.curves["az_err"] = p_aim.plot(pen=pg.mkPen(ACCENT), name="azimuth (deg)")
        self.curves["range_err"] = p_aim.plot(pen=pg.mkPen(HEAT), name="range (px)")

        p_range = self._glw.addPlot(row=2, col=0)
        p_range.setLabel("left", "range / miss (m)")
        # reachable envelope as a horizontal shaded band (auto-spans the X range)
        self._envelope = pg.LinearRegionItem(
            orientation="horizontal", movable=False,
            brush=pg.mkBrush(88, 196, 112, 40), pen=pg.mkPen(None),
        )
        self._envelope.setZValue(-10)
        p_range.addItem(self._envelope)
        self.curves["range_est"] = p_range.plot(pen=pg.mkPen(INK, width=1.2))
        self.curves["splash_miss"] = p_range.plot(pen=pg.mkPen(WATER))

        p_pump = self._glw.addPlot(row=3, col=0)
        p_pump.setLabel("left", "pump / valve")
        self.curves["pump"] = p_pump.plot(pen=pg.mkPen(WATER, width=1.2))
        self.curves["valve"] = p_pump.plot(pen=pg.mkPen(WATER, style=Qt.PenStyle.DashLine))

        p_energy = self._glw.addPlot(row=4, col=0)
        p_energy.setLabel("left", "intensity / water")
        p_energy.setLabel("bottom", "time (s)")
        self.curves["intensity"] = p_energy.plot(pen=pg.mkPen(HEAT, width=1.5))
        self.curves["water"] = p_energy.plot(pen=pg.mkPen(WATER))

        self.plots = [p_state, p_aim, p_range, p_pump, p_energy]
        for p in self.plots:
            p.showGrid(x=True, y=True, alpha=0.2)
            p.setClipToView(True)
            p.setDownsampling(auto=True, mode="peak")
        for p in self.plots[1:]:
            p.setXLink(self.plots[0])

        # a shared crosshair across the X-linked panels, driven by the mouse
        for p in self.plots:
            line = pg.InfiniteLine(angle=90, movable=False,
                                   pen=pg.mkPen("#8a94a6", style=Qt.PenStyle.DashLine))
            line.setVisible(False)
            p.addItem(line, ignoreBounds=True)
            self._crosshairs.append(line)
        self._proxy = pg.SignalProxy(self._glw.scene().sigMouseMoved,
                                     rateLimit=30, slot=self._on_mouse)

    # -- public API ----------------------------------------------------------

    def set_run(self, samples, reach: tuple[float, float]) -> None:
        self._samples = list(samples)
        self._reach = reach
        self._redraw()

    def append(self, samples) -> None:
        if not samples:
            return
        self._samples.extend(samples)
        self._redraw()

    def clear(self) -> None:
        self._samples = []
        self.data = {}
        self.t = None
        for c in self.curves.values():
            c.setData([], [])
        for line in self._crosshairs:
            line.setVisible(False)
        self._stack.setCurrentWidget(self._empty)

    # -- crosshair -----------------------------------------------------------

    def _on_mouse(self, evt) -> None:
        pos = evt[0]
        for p in self.plots:
            if p.sceneBoundingRect().contains(pos):
                self.set_crosshair(p.getViewBox().mapSceneToView(pos).x())
                return

    def set_crosshair(self, x: float) -> None:
        for line in self._crosshairs:
            line.setPos(x)
            line.setVisible(True)
        readout = self._readout_at(x)
        if readout:
            self.readout_changed.emit(readout)

    def _readout_at(self, x: float) -> str:
        if self.t is None or len(self.t) == 0:
            return ""
        i = int(np.argmin(np.abs(self.t - x)))
        parts = [f"t={float(self.t[i]):.1f}s"]
        for key in ("az_err", "range_est", "splash_miss", "pump", "intensity", "water"):
            parts.append(f"{key}={float(self.data[key][i]):.2f}")
        return "  |  ".join(parts)

    # -- internals -----------------------------------------------------------

    def _redraw(self) -> None:
        s = self._samples
        if not s:
            return
        n = len(s)
        t = np.fromiter((x.t for x in s), float, n)
        series = {
            "state": np.fromiter((_STATE_IDX.get(x.state, -1) for x in s), float, n),
            "az_err": np.fromiter((x.az_err_deg for x in s), float, n),
            "range_err": np.fromiter((x.range_err_px for x in s), float, n),
            "range_est": np.fromiter((x.range_est_m for x in s), float, n),
            "splash_miss": np.fromiter((x.splash_miss_m for x in s), float, n),
            "pump": np.fromiter((x.pump for x in s), float, n),
            "valve": np.fromiter((100.0 if x.valve else 0.0 for x in s), float, n),
            "intensity": np.fromiter((x.fire_intensity for x in s), float, n),
            "water": np.fromiter((x.water_l for x in s), float, n),
        }
        self.t = t
        self.data = series
        for key, arr in series.items():
            self.curves[key].setData(t, arr)
        if self._reach is not None:
            self._envelope.setRegion((float(self._reach[0]), float(self._reach[1])))
        self._stack.setCurrentWidget(self._glw)
