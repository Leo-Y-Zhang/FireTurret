# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""P1a Task 8: telemetry plots verified at the DATA layer (not offscreen pixels)."""
from __future__ import annotations

from triton.analysis import TelemetrySample
from triton.studio.views.plots import TelemetryPlots


def _sample(i: int) -> TelemetrySample:
    return TelemetrySample(
        t=i / 30.0, state="SUPPRESS" if i % 2 else "SEARCH",
        pan_cmd=float(i), pan_act=float(i), tilt=22.0, pump=float(i % 100),
        valve=bool(i % 2), range_est_m=7.0 + 0.01 * i, az_err_deg=float(i),
        range_err_px=float(2 * i), fire_intensity=1.0 - 0.001 * i,
        water_l=0.02 * i, splash_miss_m=0.1 * (i % 5), warning="",
    )


def test_set_run_populates_series(qapp):
    samples = [_sample(i) for i in range(20)]
    w = TelemetryPlots()
    w.set_run(samples, reach=(3.0, 9.0))
    assert list(w.data["az_err"]) == [s.az_err_deg for s in samples]
    assert list(w.data["intensity"]) == [s.fire_intensity for s in samples]
    assert list(w.data["valve"]) == [100.0 if s.valve else 0.0 for s in samples]
    # the curve received the data too
    assert w.curves["az_err"].yData is not None
    assert len(w.curves["az_err"].yData) == 20


def test_panels_are_x_linked(qapp):
    w = TelemetryPlots()
    vb0 = w.plots[0].getViewBox()
    for p in w.plots[1:]:
        vb = p.getViewBox()
        assert vb.linkedView(vb.XAxis) is vb0


def test_empty_state_then_populated_then_cleared(qapp):
    w = TelemetryPlots()
    assert w._stack.currentWidget() is w._empty
    w.set_run([_sample(i) for i in range(5)], (3.0, 9.0))
    assert w._stack.currentWidget() is w._glw
    w.clear()
    assert w._stack.currentWidget() is w._empty


def test_append_extends_series(qapp):
    w = TelemetryPlots()
    w.set_run([_sample(i) for i in range(5)], (3.0, 9.0))
    w.append([_sample(i) for i in range(5, 12)])
    assert len(w.data["water"]) == 12
    assert list(w.data["az_err"]) == [float(i) for i in range(12)]


def test_crosshair_readout(qapp, qtbot):
    w = TelemetryPlots()
    samples = [_sample(i) for i in range(20)]
    w.set_run(samples, (3.0, 9.0))
    with qtbot.waitSignal(w.readout_changed, timeout=1000) as blocker:
        w.set_crosshair(samples[5].t)
    text = blocker.args[0]
    assert "t=" in text and "intensity=" in text
    # crosshair lines became visible and readout picks the nearest sample
    assert all(line.isVisible() for line in w._crosshairs)
    assert f"{samples[5].fire_intensity:.2f}" in w._readout_at(samples[5].t)
