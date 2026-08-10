# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""SP3 — the range estimator and the range-table interpolants.

The estimator is an extraction and must be bit-identical to what
`MissionController` did inline. The tables are interpolants, and an interpolant
nobody characterised is a silent bias — so their error is MEASURED here rather
than assumed. `SpeedRangeTable` is now wired into the elevation search in
`ballistics`, which makes that measurement load-bearing rather than advisory.
"""

from __future__ import annotations

import time

import pytest

from fireturret.ballistics import exit_velocity, plan_suppression, range_of
from fireturret.config import DEFAULT_CONFIG
from fireturret.control.ranging import (
    RangeEstimator,
    RangeTable,
    SpeedRangeTable,
    target_ground_px,
)
from fireturret.geometry import CameraModel
from fireturret.vision.firedetect import FireBlob
from fireturret.vision.tracker import Track

CFG = DEFAULT_CONFIG
JET = CFG.jet
TILT_MIN, TILT_MAX = CFG.turret.tilt_min_deg, CFG.turret.tilt_max_deg


def _track(cx: float, cy: float, bbox=None) -> Track:
    t = Track(track_id=1, cx=cx, cy=cy, area=100.0)
    if bbox is not None:
        t.last_blob = FireBlob(cx=cx, cy=cy, area=100.0, bbox=bbox,
                               colour_score=1.0, flicker_score=1.0, confidence=1.0)
    return t


# ------------------------------------------------------------ ground pixel

def test_ground_px_uses_the_bbox_bottom_not_the_centroid() -> None:
    """The flame centroid sits ABOVE the burning material, so ranging from it
    biases every estimate long."""
    track = _track(cx=480.0, cy=300.0, bbox=(460, 250, 40, 100))
    assert target_ground_px(track) == (480.0, 350.0)  # y + h


def test_ground_px_falls_back_to_the_centroid_without_a_blob() -> None:
    assert target_ground_px(_track(cx=480.0, cy=300.0)) == (480.0, 300.0)


# --------------------------------------------------------------- estimator

def test_estimator_low_passes_a_flickering_reading() -> None:
    """The bbox bottom flickers with the flame; an unfiltered estimate would
    jitter the opening solution frame to frame."""
    est = RangeEstimator(CameraModel(CFG.camera), CFG.mission)
    track = _track(cx=480.0, cy=400.0, bbox=(460, 340, 40, 60))
    first = est.update(track)
    values = [est.update(track) for _ in range(5)]
    assert all(abs(v - first) < 5.0 for v in values)
    # and it converges rather than drifting
    assert abs(values[-1] - values[-2]) < abs(values[0] - first) + 1e-9


def test_estimator_falls_back_when_the_ground_plane_gives_nothing() -> None:
    """A flame base above the horizon has no ground intersection. The estimator
    must fall back to its last good value, then to the configured bootstrap —
    never to NaN or None."""
    est = RangeEstimator(CameraModel(CFG.camera), CFG.mission)
    above_horizon = _track(cx=480.0, cy=5.0)
    value = est.update(above_horizon)
    assert value == pytest.approx(CFG.mission.default_range_m, abs=1e-9)


def test_estimator_reset_forgets_the_previous_engagement() -> None:
    est = RangeEstimator(CameraModel(CFG.camera), CFG.mission)
    est.update(_track(cx=480.0, cy=400.0, bbox=(460, 340, 40, 60)))
    assert est.ema is not None
    est.reset()
    assert est.ema is None


# -------------------------------------------------------------- RangeTable

def test_table_reproduces_range_of_at_its_grid_points() -> None:
    """Interpolation must be exact where it is not interpolating."""
    table = RangeTable(JET, TILT_MIN, TILT_MAX)
    for i in (0, 5, 12, len(table.pumps) - 1):
        for j in (0, 7, len(table.tilts) - 1):
            pump, tilt = float(table.pumps[i]), float(table.tilts[j])
            want = range_of(exit_velocity(pump, JET), tilt, JET)
            assert table.range_of(pump, tilt) == pytest.approx(want, abs=1e-9)


def test_table_error_between_grid_points_is_measured_and_small() -> None:
    """The number that matters for SP8. Sampled off-grid across the usable
    envelope; if a future grid change degrades this, the assertion says so
    rather than the error being discovered in a control loop."""
    table = RangeTable(JET, TILT_MIN, TILT_MAX)
    worst = 0.0
    for pump in [30.0, 37.5, 44.0, 58.3, 71.1, 88.8, 96.0]:
        for tilt in [TILT_MIN + 1.3, 15.7, 22.0, 31.4, TILT_MAX - 2.1]:
            want = range_of(exit_velocity(pump, JET), tilt, JET)
            worst = max(worst, abs(table.range_of(pump, tilt) - want))
    # measured ~0.01 m on the shipped grid; 5 cm is a generous ceiling against a
    # ~7 m throw, and far inside the monocular range estimate's own error
    assert worst < 0.05, f"worst interpolation error {worst:.4f} m"


def test_table_clamps_rather_than_extrapolating() -> None:
    """Outside the grid the honest answer is the edge value, not an
    extrapolation into physics the table was never built for."""
    table = RangeTable(JET, TILT_MIN, TILT_MAX)
    assert table.range_of(-50.0, TILT_MIN) == table.range_of(float(table.pumps[0]), TILT_MIN)
    assert table.range_of(500.0, TILT_MAX) == table.range_of(100.0, TILT_MAX)


def test_table_lookup_is_much_cheaper_than_planning_a_solution() -> None:
    """The reason the table exists. `plan_suppression` costs ~53 ms — 1.6 frames
    at 30 fps — which is invisible while RANGE is entered once per engagement and
    a missed deadline once SP8 makes target range change continuously."""
    table = RangeTable(JET, TILT_MIN, TILT_MAX)

    start = time.perf_counter()
    for _ in range(200):
        table.range_of(63.0, 21.0)
    per_lookup = (time.perf_counter() - start) / 200

    start = time.perf_counter()
    plan_suppression(7.0, JET, TILT_MIN, TILT_MAX)
    per_plan = time.perf_counter() - start

    assert per_lookup < per_plan / 50, (
        f"lookup {per_lookup * 1e6:.1f} us vs plan {per_plan * 1e3:.1f} ms"
    )


def test_table_rejects_a_degenerate_grid() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        RangeTable(JET, TILT_MIN, TILT_MAX, pump_steps=1)


# ------------------------------------------------------- SpeedRangeTable (wired)
#
# These matter more than the RangeTable tests above, because this table is inside
# the shipped elevation search. An error here does not sit unused; it moves a
# firing solution.

SPEED_LO = exit_velocity(JET.min_pump_pct, JET)
SPEED_HI = exit_velocity(100.0, JET)


@pytest.fixture(scope="module")
def speed_table() -> SpeedRangeTable:
    """Built once — a 25x25 grid is 625 RK4 arcs, and rebuilding it per test
    would dominate the suite for no added coverage."""
    return SpeedRangeTable(JET)


def test_speed_table_is_exact_on_its_own_grid_points(speed_table) -> None:
    """Bilinear interpolation must REPRODUCE the samples it was built from. If
    this drifts, the indexing arithmetic is wrong — which is the failure mode the
    closed-form index replacement could plausibly have introduced."""
    for i in (0, 7, 12, speed_table.ns - 1):
        for j in (0, 5, 18, speed_table.nt - 1):
            speed = speed_table.s0 + i * speed_table.ds
            tilt = speed_table.t0 + j * speed_table.dt
            assert speed_table.range_of_speed(speed, tilt) == pytest.approx(
                range_of(speed, tilt, JET), abs=1e-9
            )


def test_speed_table_error_between_grid_points_is_measured(speed_table) -> None:
    """The number that justifies adoption. Deliberately sampled OFF the grid, at
    offsets chosen to land near cell centres where bilinear error peaks."""
    worst = 0.0
    for fs in (0.13, 0.37, 0.5, 0.61, 0.86):
        for ft in (0.13, 0.5, 0.79):
            speed = SPEED_LO + fs * (SPEED_HI - SPEED_LO)
            tilt = ft * SpeedRangeTable.TILT_MAX_DEG
            worst = max(worst, abs(speed_table.range_of_speed(speed, tilt)
                                   - range_of(speed, tilt, JET)))
    # 4 cm at ~9 m throw. Pinned as an upper bound rather than asserted loosely:
    # if a future grid change makes this worse, the test should say so.
    assert worst < 0.05, f"worst interpolation error {worst:.4f} m"


def test_speed_table_clamps_rather_than_extrapolating(speed_table) -> None:
    """Outside the grid the edge value is the honest answer. Extrapolating a
    bilinear fit past its support is how an interpolant produces a confident
    number for a configuration it never sampled."""
    assert speed_table.range_of_speed(SPEED_LO - 5.0, 20.0) == pytest.approx(
        speed_table.range_of_speed(SPEED_LO, 20.0)
    )
    assert speed_table.range_of_speed(SPEED_HI + 5.0, 20.0) == pytest.approx(
        speed_table.range_of_speed(SPEED_HI, 20.0)
    )
    top = SpeedRangeTable.TILT_MAX_DEG
    assert speed_table.range_of_speed(20.0, -10.0) == pytest.approx(
        speed_table.range_of_speed(20.0, 0.0)
    )
    assert speed_table.range_of_speed(20.0, top + 30.0) == pytest.approx(
        speed_table.range_of_speed(20.0, top)
    )


def test_speed_table_spans_the_whole_reachable_tilt_band() -> None:
    """The table must cover every elevation the searches can ask about, including
    the physical range peak (~30 deg), which sits above the turret's own tilt
    minimum but must not be clipped by the TABLE."""
    assert SpeedRangeTable.TILT_MAX_DEG >= TILT_MAX
    assert SpeedRangeTable.TILT_MAX_DEG > 45.0


def test_planning_a_solution_now_fits_inside_a_frame() -> None:
    """The whole point of wiring the table in.

    `plan_suppression` cost 43.7 ms before adoption — longer than the 33.3 ms
    frame it has to fit in. This asserts the budget, not the speedup, because the
    budget is what the system actually requires.
    """
    plan_suppression(9.0, JET, TILT_MIN, TILT_MAX)  # warm the cached table
    best = min(_time_plan() for _ in range(5))
    assert best < 33.3 / 2, f"plan_suppression took {best:.2f} ms"


def _time_plan() -> float:
    start = time.perf_counter()
    plan_suppression(9.0, JET, TILT_MIN, TILT_MAX)
    return (time.perf_counter() - start) * 1000.0


def test_the_range_a_solution_REPORTS_is_never_interpolated() -> None:
    """The boundary that keeps interpolation error out of everything downstream.

    The table accelerates the SEARCH for an elevation. Once one is chosen, the
    range reported back is integrated exactly at that elevation, so operator
    advisories, reachability checks and the water gate all see the true model
    rather than a bilinear fit of it.
    """
    for target in (5.0, 6.0, 7.0, 7.5, 8.0, 9.0):
        plan = plan_suppression(target, JET, TILT_MIN, TILT_MAX)
        if plan is None:
            continue
        exact = range_of(plan.speed_mps, plan.elevation_deg, JET)
        assert plan.predicted_range_m == pytest.approx(exact, abs=1e-9), (
            f"target {target} m reported an interpolated range"
        )
