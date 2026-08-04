# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""P3: run comparison overlay (data layer)."""
from __future__ import annotations

from triton.analysis import TelemetrySample
from triton.studio.views.compare_view import RunCompareView


def _samples(base: float, n: int = 5):
    return [
        TelemetrySample(t=i * 0.1, state="SUPPRESS", pan_cmd=0.0, pan_act=0.0, tilt=15.0,
                        pump=40.0, valve=True, range_est_m=7.0, az_err_deg=0.0,
                        range_err_px=0.0, fire_intensity=base - 0.1 * i, water_l=0.05 * i,
                        splash_miss_m=0.0, warning="")
        for i in range(n)
    ]


def test_compare_overlays_selected_runs(qapp):
    runs = [("run 1", _samples(1.0)), ("run 2", _samples(0.8))]
    v = RunCompareView(runs)
    v.set_metric("fire_intensity")
    assert set(v.data) == {"run 1", "run 2"}
    assert v.data["run 1"] == [s.fire_intensity for s in runs[0][1]]
    assert len(v._curves) == 2


def test_compare_switches_metric(qapp):
    runs = [("a", _samples(1.0))]
    v = RunCompareView(runs)
    v.set_metric("water_l")
    assert v.data["a"] == [s.water_l for s in runs[0][1]]
