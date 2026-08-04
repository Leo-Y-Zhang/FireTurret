# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""Read-only top-down scene view: the turret at the origin, the fire(s) and decoy
as markers, and the jet's reachable range as a shaded annulus. Bird's-eye layout
of the scenario, so you can see at a glance whether a fire is in reach.

Polar convention: azimuth 0 is straight ahead (+forward); +azimuth swings right.
"""
from __future__ import annotations

import math

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QVBoxLayout, QWidget

from ...ballistics import reach_bounds

INK = "#e8edf4"
HEAT = "#ff9f43"
OK = "#58c470"
DECOY = "#8a94a6"


def _polar(az_deg: float, rng: float) -> tuple[float, float]:
    a = math.radians(az_deg)
    return rng * math.sin(a), rng * math.cos(a)


class TopDownView(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._plot = pg.PlotWidget()
        self._plot.setAspectLocked(True)
        self._plot.setLabel("bottom", "x (m)")
        self._plot.setLabel("left", "forward (m)")
        self._plot.showGrid(x=True, y=True, alpha=0.2)
        self._plot.addLegend(offset=(-5, 5))
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._plot)

        self._ring_min = self._plot.plot(pen=pg.mkPen(OK, style=Qt.PenStyle.DashLine))
        self._ring_max = self._plot.plot(pen=pg.mkPen(OK, style=Qt.PenStyle.DashLine))
        self.turret = pg.ScatterPlotItem(
            [0.0], [0.0], size=15, symbol="s", brush=pg.mkBrush(INK), name="turret")
        self._plot.addItem(self.turret)
        self.fires = pg.ScatterPlotItem(
            size=17, symbol="o", brush=pg.mkBrush(HEAT), name="fire")
        self._plot.addItem(self.fires)
        self.decoy = pg.ScatterPlotItem(
            size=14, symbol="x", brush=pg.mkBrush(DECOY), pen=pg.mkPen(DECOY), name="decoy")
        self._plot.addItem(self.decoy)

    def set_scene(self, scenario, jet, tilt_bounds: tuple[float, float]) -> None:
        fires = [_polar(scenario.fire_azimuth_deg, scenario.fire_range_m)]
        fires += [_polar(az, rng) for az, rng in scenario.extra_fires]
        self.fires.setData([p[0] for p in fires], [p[1] for p in fires])

        if scenario.decoy is not None:
            dx, dy = _polar(scenario.decoy[0], scenario.decoy[1])
            self.decoy.setData([dx], [dy])
        else:
            self.decoy.setData([], [])

        lo, hi = reach_bounds(jet, tilt_bounds[0], tilt_bounds[1])
        theta = np.linspace(0.0, 2.0 * np.pi, 120)
        self._ring_min.setData(lo * np.cos(theta), lo * np.sin(theta))
        self._ring_max.setData(hi * np.cos(theta), hi * np.sin(theta))
