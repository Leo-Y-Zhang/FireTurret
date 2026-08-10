# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""P3: the batch-sweep configuration dialog builds a SweepSpec."""
from __future__ import annotations

from fireturret.studio.views.sweep_dialog import SweepDialog
from fireturret.studio.views.sweep_view import METRIC_NAMES


def test_dialog_builds_1axis_spec(qapp):
    d = SweepDialog()
    d._a1_combo.setCurrentIndex(1)  # jet.drag_k
    d._a1_start.setValue(0.1)
    d._a1_stop.setValue(0.2)
    d._a1_steps.setValue(3)
    d._seeds.setText("7,8")
    spec, metric = d.spec()
    assert len(spec.axes) == 1
    assert spec.axes[0].name == "drag_k"
    assert len(spec.axes[0].values) == 3
    assert spec.seeds == [7, 8]
    assert metric in METRIC_NAMES


def test_dialog_two_axes(qapp):
    d = SweepDialog()
    d._a2_combo.setCurrentIndex(1)  # first real param on axis 2
    spec, _ = d.spec()
    assert len(spec.axes) == 2


def test_dialog_default_seed(qapp):
    d = SweepDialog()
    d._seeds.setText("")
    spec, _ = d.spec()
    assert spec.seeds == [7]
