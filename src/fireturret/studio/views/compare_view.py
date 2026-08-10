# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""Overlay a chosen telemetry metric across several runs, for side-by-side
comparison. Fed the telemetry the Runs table retains for each finished run.
"""
from __future__ import annotations

import pyqtgraph as pg
from PySide6.QtWidgets import QComboBox, QVBoxLayout, QWidget

_METRICS = [
    "fire_intensity", "az_err_deg", "range_est_m", "pump", "water_l", "splash_miss_m",
]
_COLORS = ["#4cc3e8", "#ff9f43", "#58c470", "#e05c6e", "#b48ead", "#e8edf4", "#f2c14e"]


class RunCompareView(QWidget):
    def __init__(self, runs, parent=None) -> None:
        super().__init__(parent)
        self._runs = list(runs)  # list[(name, list[TelemetrySample])]
        self.data: dict[str, list[float]] = {}

        layout = QVBoxLayout(self)
        self._metric = QComboBox()
        self._metric.addItems(_METRICS)
        self._metric.currentTextChanged.connect(self.set_metric)
        layout.addWidget(self._metric)

        self._plot = pg.PlotWidget()
        self._plot.addLegend(offset=(-5, 5))
        self._plot.showGrid(x=True, y=True, alpha=0.2)
        self._plot.setLabel("bottom", "time (s)")
        layout.addWidget(self._plot, 1)

        self._curves: list = []
        self.set_metric(_METRICS[0])

    def set_metric(self, metric: str) -> None:
        for curve in self._curves:
            self._plot.removeItem(curve)
        self._curves = []
        self.data = {}
        self._plot.setLabel("left", metric)
        for i, (name, samples) in enumerate(self._runs):
            t = [s.t for s in samples]
            y = [getattr(s, metric) for s in samples]
            pen = pg.mkPen(_COLORS[i % len(_COLORS)], width=1.5)
            self._curves.append(self._plot.plot(t, y, pen=pen, name=name))
            self.data[name] = y
