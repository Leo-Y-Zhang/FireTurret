# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""Jet calibration panel: enter measured test shots, run the fit
(``ballistics.fit_jet``), see measured-vs-predicted range for the fitted model,
and apply the fitted velocity_coeff / drag_k to the session config.
"""
from __future__ import annotations

from dataclasses import replace

import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ...ballistics import exit_velocity, fit_jet, range_of
from ..session import Session

ACCENT = "#4cc3e8"
GRID = "#253141"


class CalibrationPanel(QWidget):
    def __init__(self, session: Session, parent=None) -> None:
        super().__init__(parent)
        self.session = session
        self._fit_result = None

        layout = QVBoxLayout(self)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["pump %", "elevation °", "measured range m"])
        layout.addWidget(self.table)

        buttons = QHBoxLayout()
        add = QPushButton("Add shot")
        add.clicked.connect(lambda: self._add_row())
        remove = QPushButton("Remove shot")
        remove.clicked.connect(self._remove_row)
        do_fit = QPushButton("Fit")
        do_fit.clicked.connect(self.fit)
        self._apply_btn = QPushButton("Apply to config")
        self._apply_btn.setEnabled(False)
        self._apply_btn.clicked.connect(self.apply_fit)
        for b in (add, remove, do_fit, self._apply_btn):
            buttons.addWidget(b)
        layout.addLayout(buttons)

        self.result_label = QLabel("Enter ≥3 shots and press Fit")
        layout.addWidget(self.result_label)

        self._plot = pg.PlotWidget()
        self._plot.setLabel("bottom", "measured range (m)")
        self._plot.setLabel("left", "predicted range (m)")
        self._plot.showGrid(x=True, y=True, alpha=0.2)
        self._refline = self._plot.plot(pen=pg.mkPen(GRID, style=Qt.PenStyle.DashLine))
        self._scatter = pg.ScatterPlotItem(size=11, brush=pg.mkBrush(ACCENT))
        self._plot.addItem(self._scatter)
        layout.addWidget(self._plot, 1)

    # -- shots ---------------------------------------------------------------

    def _add_row(self, pump: float = 50.0, elev: float = 20.0, rng: float = 5.0) -> None:
        r = self.table.rowCount()
        self.table.insertRow(r)
        for c, v in enumerate((pump, elev, rng)):
            self.table.setItem(r, c, QTableWidgetItem(str(v)))

    def _remove_row(self) -> None:
        r = self.table.currentRow()
        if r < 0:
            r = self.table.rowCount() - 1
        if r >= 0:
            self.table.removeRow(r)

    def set_shots(self, shots) -> None:
        self.table.setRowCount(0)
        for pump, elev, rng in shots:
            self._add_row(pump, elev, rng)

    def _shots(self) -> list[tuple[float, float, float]]:
        out = []
        for r in range(self.table.rowCount()):
            try:
                out.append(tuple(float(self.table.item(r, c).text()) for c in range(3)))
            except (AttributeError, ValueError):
                continue
        return out

    # -- fit -----------------------------------------------------------------

    def fit(self):
        shots = self._shots()
        if len(shots) < 3:
            self.result_label.setText("need ≥3 valid shots (pump, elevation, measured range)")
            return None
        result = fit_jet(shots, self.session.config.jet)
        self._fit_result = result
        self.result_label.setText(
            f"velocity_coeff = {result.velocity_coeff:.3f}   "
            f"drag_k = {result.drag_k:.3f}   RMSE = {result.rmse_m:.2f} m")
        self._apply_btn.setEnabled(True)

        fitted = replace(self.session.config.jet,
                         velocity_coeff=result.velocity_coeff, drag_k=result.drag_k)
        measured = [s[2] for s in shots]
        predicted = [range_of(exit_velocity(s[0], fitted), s[1], fitted) for s in shots]
        self._scatter.setData(measured, predicted)
        lim = max(measured + predicted + [1.0])
        self._refline.setData([0.0, lim], [0.0, lim])  # y = x reference
        return result

    def apply_fit(self) -> None:
        if self._fit_result is None:
            return
        self.session.set_field("jet", "velocity_coeff", self._fit_result.velocity_coeff)
        self.session.set_field("jet", "drag_k", self._fit_result.drag_k)
