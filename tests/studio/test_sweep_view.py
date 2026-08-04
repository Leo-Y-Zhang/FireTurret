# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""P3: the sweep results view (1-axis scatter, 2-axis heatmap), data layer."""
from __future__ import annotations

import pytest

from triton.studio.sweep import SweepAxis, SweepResult, SweepSpec
from triton.studio.views.sweep_view import SweepResultsView


def test_one_axis_scatter(qapp):
    spec = SweepSpec(axes=[SweepAxis("jet", "drag_k", [0.1, 0.2])], seeds=[7, 8])
    results = [
        SweepResult(0, {"jet.drag_k": 0.1, "seed": 7}, True, 5.0, 0.7, 0.1, 0.2),
        SweepResult(1, {"jet.drag_k": 0.1, "seed": 8}, True, 7.0, 0.7, 0.1, 0.2),
        SweepResult(2, {"jet.drag_k": 0.2, "seed": 7}, True, 6.0, 0.8, 0.2, 0.3),
        SweepResult(3, {"jet.drag_k": 0.2, "seed": 8}, True, 8.0, 0.8, 0.2, 0.3),
    ]
    v = SweepResultsView()
    v.set_results(spec, results, metric="water_l")
    assert v.data["x"] == [0.1, 0.2]
    assert v.data["y"] == [6.0, 7.0]  # averaged over the two seeds


def test_two_axis_heatmap(qapp):
    spec = SweepSpec(
        axes=[SweepAxis("jet", "drag_k", [0.1, 0.2]),
              SweepAxis("scenario", "fire_range_m", [6.0, 7.0])],
        seeds=[7],
    )
    results = [
        SweepResult(0, {"jet.drag_k": 0.1, "scenario.fire_range_m": 6.0, "seed": 7}, True, 1.0, None, 0, 0),
        SweepResult(1, {"jet.drag_k": 0.1, "scenario.fire_range_m": 7.0, "seed": 7}, True, 2.0, None, 0, 0),
        SweepResult(2, {"jet.drag_k": 0.2, "scenario.fire_range_m": 6.0, "seed": 7}, True, 3.0, None, 0, 0),
        SweepResult(3, {"jet.drag_k": 0.2, "scenario.fire_range_m": 7.0, "seed": 7}, True, 4.0, None, 0, 0),
    ]
    v = SweepResultsView()
    v.set_results(spec, results, metric="water_l")
    grid = v.data["grid"]
    assert grid.shape == (2, 2)
    assert grid[0, 0] == 1.0
    assert grid[1, 1] == 4.0
    # the heatmap image must be positioned over the REAL swept values (0.1..0.2 on
    # x, 6.0..7.0 on y), NOT the default 0..N array-index coordinates
    r = v._image.mapRectToParent(v._image.boundingRect()).normalized()
    assert (r.left(), r.right()) == pytest.approx((0.1, 0.2))
    assert (min(r.top(), r.bottom()), max(r.top(), r.bottom())) == pytest.approx((6.0, 7.0))


def test_failed_cell_does_not_pollute_heatmap(qapp):
    spec = SweepSpec(
        axes=[SweepAxis("jet", "drag_k", [0.1, 0.2]),
              SweepAxis("scenario", "fire_range_m", [6.0, 7.0])],
        seeds=[7],
    )
    ok = SweepResult(0, {"jet.drag_k": 0.1, "scenario.fire_range_m": 6.0, "seed": 7}, True, 1.0, None, 0, 0)
    bad = SweepResult(3, {"jet.drag_k": 0.2, "scenario.fire_range_m": 7.0, "seed": 7}, False, 0.0, None, 0, 0,
                      failed=True, error="boom")
    v = SweepResultsView()
    v.set_results(spec, [ok, bad], metric="water_l")
    grid = v.data["grid"]
    assert grid[0, 0] == 1.0
    assert grid[1, 1] != grid[1, 1]  # NaN: the failed cell reads blank, not 0.0


def test_extinguished_metric(qapp):
    spec = SweepSpec(axes=[SweepAxis("jet", "drag_k", [0.1])], seeds=[7])
    results = [SweepResult(0, {"jet.drag_k": 0.1, "seed": 7}, True, 5.0, 0.7, 0.1, 0.2)]
    v = SweepResultsView()
    v.set_results(spec, results, metric="extinguished")
    assert v.data["y"] == [1.0]
