# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""P3: batch sweep engine (pure grid/cells) + the QThreadPool runner."""
from __future__ import annotations

import pytest

from fireturret.config import FireTurretConfig
from fireturret.rig.sim_rig import SimScenario
from fireturret.studio.sweep import (
    SweepAxis,
    SweepRunner,
    SweepSpec,
    run_cell,
    sweep_grid,
    with_field,
)


def test_with_field_config_and_scenario():
    cfg, scn = FireTurretConfig(), SimScenario()
    cfg2, scn2 = with_field(cfg, scn, "jet", "drag_k", 0.25)
    assert cfg2.jet.drag_k == 0.25
    assert scn2 is scn  # scenario untouched
    _cfg3, scn3 = with_field(cfg, scn, "scenario", "fire_range_m", 9.0)
    assert scn3.fire_range_m == 9.0


def test_with_field_validates():
    with pytest.raises(ValueError):
        with_field(FireTurretConfig(), SimScenario(), "jet", "max_pressure_psi", -1.0)
    with pytest.raises(ValueError):
        with_field(FireTurretConfig(), SimScenario(), "bogus", "x", 1.0)


def test_sweep_grid_cardinality_and_coords():
    spec = SweepSpec(
        axes=[SweepAxis("jet", "drag_k", [0.1, 0.16, 0.2])],
        seeds=[7, 8],
        max_frames=120,
    )
    cells = sweep_grid(FireTurretConfig(), SimScenario(), spec)
    assert len(cells) == 3 * 2  # 3 values x 2 seeds
    # each cell's config reflects its axis value
    for c in cells:
        assert c.config.jet.drag_k == c.coords["jet.drag_k"]
        assert c.seed == c.coords["seed"]
    assert [c.index for c in cells] == list(range(6))


def test_sweep_grid_two_axes():
    spec = SweepSpec(
        axes=[SweepAxis("jet", "drag_k", [0.1, 0.2]),
              SweepAxis("scenario", "fire_range_m", [6.0, 7.0, 8.0])],
        seeds=[7],
        max_frames=120,
    )
    cells = sweep_grid(FireTurretConfig(), SimScenario(), spec)
    assert len(cells) == 2 * 3 * 1


def test_run_cell_returns_metrics():
    spec = SweepSpec(axes=[SweepAxis("jet", "drag_k", [0.16])], seeds=[7], max_frames=120)
    cell = sweep_grid(FireTurretConfig(), SimScenario(), spec)[0]
    result = run_cell(cell)
    assert result.index == 0
    assert isinstance(result.water_l, float)
    assert result.cancelled is False


def test_run_cell_cancelled_shortcut():
    spec = SweepSpec(axes=[SweepAxis("jet", "drag_k", [0.16])], seeds=[7], max_frames=6000)
    cell = sweep_grid(FireTurretConfig(), SimScenario(), spec)[0]
    result = run_cell(cell, should_stop=lambda: True)  # cancel at entry
    assert result.cancelled is True


def test_sweep_runner_completes(qapp, qtbot):
    spec = SweepSpec(axes=[SweepAxis("jet", "drag_k", [0.14, 0.16, 0.18])],
                     seeds=[7], max_frames=120)
    cells = sweep_grid(FireTurretConfig(), SimScenario(), spec)
    runner = SweepRunner(cells)
    with qtbot.waitSignal(runner.finished, timeout=60000) as blocker:
        runner.start()
    results = blocker.args[0]
    assert len(results) == 3
    assert [r.index for r in results] == [0, 1, 2]  # sorted by index


def test_sweep_runner_isolates_a_failing_cell(qapp, qtbot):
    """A single cell that raises during drive must NOT hang the whole batch — it
    is caught and reported as a failed result so `finished` still fires. A
    negative seed reaches np.random.default_rng and raises on the pool thread."""
    spec = SweepSpec(axes=[SweepAxis("jet", "drag_k", [0.16])], seeds=[7, -1], max_frames=120)
    cells = sweep_grid(FireTurretConfig(), SimScenario(), spec)
    runner = SweepRunner(cells)
    with qtbot.waitSignal(runner.finished, timeout=60000) as blocker:
        runner.start()
    results = blocker.args[0]
    assert len(results) == 2  # both cells accounted for; the bad one did not hang it
    failed = [r for r in results if r.failed]
    assert len(failed) == 1
    assert failed[0].error  # carries a description of what went wrong


def test_sweep_runner_cancel_before_start(qapp, qtbot):
    spec = SweepSpec(axes=[SweepAxis("jet", "drag_k", [0.14, 0.16])],
                     seeds=[7], max_frames=6000)
    cells = sweep_grid(FireTurretConfig(), SimScenario(), spec)
    runner = SweepRunner(cells)
    runner.cancel()  # every cell shortcuts to cancelled -> finished still fires
    with qtbot.waitSignal(runner.finished, timeout=30000) as blocker:
        runner.start()
    results = blocker.args[0]
    assert len(results) == 2
    assert all(r.cancelled for r in results)
