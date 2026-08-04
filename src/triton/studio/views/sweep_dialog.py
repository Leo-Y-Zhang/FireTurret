# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""Dialog to configure a batch sweep: pick 1-2 parameter axes (value range +
steps), a seed set, an outcome metric, and per-run frame budget.
"""
from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLineEdit,
    QSpinBox,
)

from ..sweep import SweepAxis, SweepSpec
from .sweep_view import METRIC_NAMES

# curated sweepable numeric fields (label, group, name)
SWEEPABLE = [
    ("Jet max pressure (psi)", "jet", "max_pressure_psi"),
    ("Jet drag k (1/m)", "jet", "drag_k"),
    ("Jet velocity coeff", "jet", "velocity_coeff"),
    ("Suppress tilt (deg)", "servo", "suppress_tilt_deg"),
    ("Range gain (%/px)", "servo", "range_gain_pct_per_px"),
    ("Fire range (m)", "scenario", "fire_range_m"),
    ("Fire azimuth (deg)", "scenario", "fire_azimuth_deg"),
    ("Max spray (s)", "mission", "max_spray_s"),
]


def _range_spin(value: float) -> QDoubleSpinBox:
    s = QDoubleSpinBox()
    s.setRange(-1e6, 1e6)
    s.setDecimals(4)
    s.setValue(value)
    return s


class SweepDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Batch sweep")
        form = QFormLayout(self)

        self._a1_combo = self._param_combo(with_none=False)
        self._a1_start = _range_spin(0.1)
        self._a1_stop = _range_spin(0.3)
        self._a1_steps = self._steps_spin(5)
        form.addRow("Axis 1 parameter", self._a1_combo)
        form.addRow("  start", self._a1_start)
        form.addRow("  stop", self._a1_stop)
        form.addRow("  steps", self._a1_steps)

        self._a2_combo = self._param_combo(with_none=True)
        self._a2_start = _range_spin(6.0)
        self._a2_stop = _range_spin(8.0)
        self._a2_steps = self._steps_spin(3)
        form.addRow("Axis 2 parameter", self._a2_combo)
        form.addRow("  start", self._a2_start)
        form.addRow("  stop", self._a2_stop)
        form.addRow("  steps", self._a2_steps)

        self._seeds = QLineEdit("7")
        form.addRow("Seeds (comma-separated)", self._seeds)
        self._metric = QComboBox()
        self._metric.addItems(METRIC_NAMES)
        form.addRow("Outcome metric", self._metric)
        self._frames = QSpinBox()
        self._frames.setRange(100, 12000)
        self._frames.setValue(3000)
        form.addRow("Max frames / run", self._frames)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def _param_combo(self, with_none: bool) -> QComboBox:
        combo = QComboBox()
        if with_none:
            combo.addItem("(none)", None)
        for label, group, name in SWEEPABLE:
            combo.addItem(label, (group, name))
        return combo

    def _steps_spin(self, value: int) -> QSpinBox:
        s = QSpinBox()
        s.setRange(1, 40)
        s.setValue(value)
        return s

    def spec(self) -> tuple[SweepSpec, str]:
        axes = [self._axis(self._a1_combo, self._a1_start, self._a1_stop, self._a1_steps)]
        a2 = self._axis(self._a2_combo, self._a2_start, self._a2_stop, self._a2_steps)
        if a2 is not None:
            axes.append(a2)
        seeds = [int(s) for s in self._seeds.text().split(",") if s.strip()]
        spec = SweepSpec(axes=axes, seeds=seeds or [7], max_frames=self._frames.value())
        return spec, self._metric.currentText()

    @staticmethod
    def _axis(combo, start, stop, steps) -> SweepAxis | None:
        data = combo.currentData()
        if data is None:
            return None
        group, name = data
        values = [float(v) for v in np.linspace(start.value(), stop.value(), steps.value())]
        return SweepAxis(group, name, values)
