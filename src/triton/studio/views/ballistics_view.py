# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""Interactive ballistics / firing-solution explorer.

Pump % and elevation sliders drive the water arc (``ballistics.simulate_arc``),
redrawn live over the jet's reachable ground-range envelope (``reach_bounds``).
Read-only visualization of the already-tested ballistics module — no new physics,
no worker, no threading. The slot handlers are callable directly (tested).
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QGridLayout, QLabel, QSlider, QVBoxLayout, QWidget

from ...ballistics import exit_velocity, reach_bounds, simulate_arc
from ...config import JetConfig

ACCENT = "#4cc3e8"
WATER = "#4cc3e8"
HEAT = "#ff9f43"
GRID = "#253141"


class BallisticsView(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._jet: JetConfig | None = None
        self._tilt = (8.0, 50.0)
        self._pump = 60.0
        self._elevation = 22.0
        self._vc_scale = 1.0
        self._drag_scale = 1.0

        outer = QVBoxLayout(self)
        self._plot = pg.PlotWidget()
        self._plot.setLabel("bottom", "ground distance (m)")
        self._plot.setLabel("left", "height (m)")
        self._plot.showGrid(x=True, y=True, alpha=0.2)
        self._plot.addLegend(offset=(-5, 5))
        self.envelope = pg.LinearRegionItem(
            orientation="vertical", movable=False,
            brush=pg.mkBrush(88, 196, 112, 40), pen=pg.mkPen(None),
        )
        self.envelope.setZValue(-10)
        self._plot.addItem(self.envelope)
        self.arc_curve = self._plot.plot(pen=pg.mkPen(ACCENT, width=2), name="model")
        self.true_curve = self._plot.plot(
            pen=pg.mkPen(HEAT, width=2, style=Qt.PenStyle.DashLine), name="true (perturbed)")
        self.landing = pg.ScatterPlotItem(size=11, brush=pg.mkBrush(WATER), pen=pg.mkPen("w"))
        self._plot.addItem(self.landing)
        outer.addWidget(self._plot, 1)

        form = QGridLayout()
        self._pump_slider = _slider(0, 100, int(self._pump))
        self._pump_label = QLabel(f"{self._pump:.0f}")
        self._elev_slider = _slider(int(self._tilt[0]), int(self._tilt[1]), int(self._elevation))
        self._elev_label = QLabel(f"{self._elevation:.0f}")
        form.addWidget(QLabel("pump %"), 0, 0)
        form.addWidget(self._pump_slider, 0, 1)
        form.addWidget(self._pump_label, 0, 2)
        form.addWidget(QLabel("elevation °"), 1, 0)
        form.addWidget(self._elev_slider, 1, 1)
        form.addWidget(self._elev_label, 1, 2)
        outer.addLayout(form)

        self._pump_slider.valueChanged.connect(self.set_pump)
        self._elev_slider.valueChanged.connect(self.set_elevation)

    def set_jet(self, jet: JetConfig, tilt_bounds: tuple[float, float]) -> None:
        self._jet = jet
        self._tilt = tilt_bounds
        self._elev_slider.blockSignals(True)
        self._elev_slider.setRange(int(tilt_bounds[0]), int(tilt_bounds[1]))
        # setRange may have clamped the slider to the new bounds; because
        # valueChanged is blocked here, set_elevation won't run — so re-sync the
        # cached elevation + label from the clamped value, otherwise the drawn arc
        # would use a stale elevation outside the turret's tilt limits.
        self._elevation = float(self._elev_slider.value())
        self._elev_label.setText(f"{self._elevation:.0f}")
        self._elev_slider.blockSignals(False)
        lo, hi = reach_bounds(jet, tilt_bounds[0], tilt_bounds[1])
        self.envelope.setRegion((lo, hi))
        self._redraw()

    def set_true_scales(self, velocity_coeff_scale: float, drag_scale: float) -> None:
        """The sim's deliberate true-physics perturbation vs the model (from the
        scenario): draws a dashed 'true' arc so the model-vs-true gap is visible."""
        self._vc_scale = float(velocity_coeff_scale)
        self._drag_scale = float(drag_scale)
        self._redraw()

    def set_pump(self, pct) -> None:
        self._pump = float(pct)
        self._pump_label.setText(f"{self._pump:.0f}")
        self._redraw()

    def set_elevation(self, deg) -> None:
        self._elevation = float(deg)
        self._elev_label.setText(f"{self._elevation:.0f}")
        self._redraw()

    def _redraw(self) -> None:
        if self._jet is None:
            return
        speed = exit_velocity(self._pump, self._jet)
        pts = simulate_arc(speed, self._elevation, self._jet)
        xs = np.array([p[0] for p in pts])
        ys = np.array([p[1] for p in pts])
        self.arc_curve.setData(xs, ys)
        self.landing.setData([float(xs[-1])], [float(ys[-1])])

        # the true (perturbed) jet the sim actually uses
        true_jet = replace(
            self._jet,
            velocity_coeff=self._jet.velocity_coeff * self._vc_scale,
            drag_k=self._jet.drag_k * self._drag_scale,
        )
        tpts = simulate_arc(exit_velocity(self._pump, true_jet), self._elevation, true_jet)
        self.true_curve.setData(
            np.array([p[0] for p in tpts]), np.array([p[1] for p in tpts]))


def _slider(lo: int, hi: int, value: int) -> QSlider:
    s = QSlider(Qt.Orientation.Horizontal)
    s.setRange(lo, hi)
    s.setValue(value)
    return s
