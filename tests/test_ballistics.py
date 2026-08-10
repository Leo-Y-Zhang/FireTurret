# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
from dataclasses import replace

import pytest

from fireturret import ballistics
from fireturret.config import JetConfig

JET = JetConfig()


def test_exit_velocity_monotonic_and_zero_at_zero() -> None:
    assert ballistics.exit_velocity(0, JET) == 0.0
    v50 = ballistics.exit_velocity(50, JET)
    v100 = ballistics.exit_velocity(100, JET)
    assert 0 < v50 < v100


def test_arc_lands_on_ground() -> None:
    v = ballistics.exit_velocity(80, JET)
    arc = ballistics.simulate_arc(v, 30, JET)
    assert arc[0][1] == pytest.approx(JET.nozzle_height_m)
    assert arc[-1][1] == pytest.approx(0.0, abs=1e-9)
    assert arc[-1][0] > 1.0


def test_range_monotonic_on_low_arc_branch() -> None:
    # With quadratic drag the range peaks well below 45 deg; the low-arc branch
    # (min elevation up to the peak) must be strictly increasing.
    v = ballistics.exit_velocity(80, JET)
    peak = ballistics.peak_elevation(v, JET, 8.0, 45.0)
    assert 20.0 < peak < 40.0
    elevations = [e for e in (8, 12, 16, 20, 24) if e < peak]
    ranges = [ballistics.range_of(v, e, JET) for e in elevations]
    assert all(b > a for a, b in zip(ranges, ranges[1:], strict=False))


def test_range_falls_off_past_the_peak() -> None:
    v = ballistics.exit_velocity(80, JET)
    peak = ballistics.peak_elevation(v, JET, 8.0, 45.0)
    assert ballistics.range_of(v, peak, JET) > ballistics.range_of(v, 45.0, JET)


def test_solve_elevation_picks_the_low_arc() -> None:
    v = ballistics.exit_velocity(80, JET)
    peak = ballistics.peak_elevation(v, JET, 8.0, 50.0)
    r_min = ballistics.range_of(v, 8.0, JET)
    r_peak = ballistics.range_of(v, peak, JET)
    target = (r_min + r_peak) / 2  # reachable on the low arc
    elev = ballistics.solve_elevation(target, v, JET, 8.0, 50.0)
    assert elev is not None
    assert elev <= peak  # low arc, not the high lob
    assert ballistics.range_of(v, elev, JET) == pytest.approx(target, abs=0.05)


def test_drag_reduces_range() -> None:
    v = ballistics.exit_velocity(80, JET)
    with_drag = ballistics.range_of(v, 30, JET)
    vacuum = ballistics.range_of(v, 30, replace(JET, drag_k=0.0))
    assert with_drag < vacuum


def test_solve_elevation_inverts_range() -> None:
    v = ballistics.exit_velocity(80, JET)
    peak = ballistics.peak_elevation(v, JET, 8.0, 50.0)
    r_min = ballistics.range_of(v, 8.0, JET)
    r_peak = ballistics.range_of(v, peak, JET)
    target = (r_min + r_peak) / 2  # squarely inside the low-arc band
    elev = ballistics.solve_elevation(target, v, JET, 8.0, 50.0)
    assert elev is not None
    assert ballistics.range_of(v, elev, JET) == pytest.approx(target, abs=0.05)


def test_solve_elevation_high_arc_for_close_target() -> None:
    # With a steep tilt ceiling, a target closer than the low-arc minimum is
    # reachable only by lobbing (elevation past the peak). Needs a high enough
    # max elevation that the lob range drops below the min-elevation range.
    v = ballistics.exit_velocity(60, JET)
    peak = ballistics.peak_elevation(v, JET, 8.0, 80.0)
    r_min = ballistics.range_of(v, 8.0, JET)
    r_lob = ballistics.range_of(v, 80.0, JET)
    assert r_lob < r_min  # a band only the high arc can reach exists
    close = (r_lob + r_min) / 2.0
    elev = ballistics.solve_elevation(close, v, JET, 8.0, 80.0)
    assert elev is not None
    assert elev > peak  # high arc
    assert ballistics.range_of(v, elev, JET) == pytest.approx(close, abs=0.15)


def test_solve_elevation_unreachable_returns_none() -> None:
    v = ballistics.exit_velocity(35, JET)
    max_r = ballistics.max_range(v, JET)
    assert ballistics.solve_elevation(max_r * 2, v, JET, 8.0, 50.0) is None


def test_choose_solution_hits_target_with_headroom() -> None:
    sol = ballistics.choose_solution(8.0, JET, 8.0, 50.0)
    assert sol is not None
    assert sol.predicted_range_m == pytest.approx(8.0, abs=0.15)
    # headroom: the chosen pressure can reach beyond the target
    speed = ballistics.exit_velocity(sol.pump_pct, JET)
    assert ballistics.max_range(speed, JET) > 8.0


def test_choose_solution_prefers_lower_pressure_for_short_range() -> None:
    near = ballistics.choose_solution(6.0, JET, 8.0, 50.0)
    far = ballistics.choose_solution(10.0, JET, 8.0, 50.0)
    assert near is not None and far is not None
    assert near.pump_pct <= far.pump_pct


def test_choose_solution_impossible_range_none() -> None:
    assert ballistics.choose_solution(500.0, JET, 8.0, 50.0) is None


def test_solve_pump_inverts_range_at_fixed_tilt() -> None:
    tilt = 15.0
    target = 8.0
    pump = ballistics.solve_pump(target, tilt, JET)
    assert pump is not None
    landed = ballistics.range_of(ballistics.exit_velocity(pump, JET), tilt, JET)
    assert landed == pytest.approx(target, abs=0.05)


def test_solve_pump_monotonic_more_pump_for_further_target() -> None:
    tilt = 15.0
    near = ballistics.solve_pump(7.0, tilt, JET)
    far = ballistics.solve_pump(10.0, tilt, JET)
    assert near is not None and far is not None
    assert far > near


def test_solve_pump_unreachable_returns_none() -> None:
    assert ballistics.solve_pump(50.0, 15.0, JET) is None  # too far at any pump


def test_plan_suppression_puts_target_mid_envelope() -> None:
    # the chosen pump should have headroom BOTH ways: its own range envelope at
    # the fixed tilt must straddle the target
    for target in (6.0, 7.5, 9.0):
        sol = ballistics.plan_suppression(target, JET, 8.0, 50.0)
        assert sol is not None, target
        assert sol.predicted_range_m == pytest.approx(target, abs=0.2)
        r_lo = ballistics.range_of(ballistics.exit_velocity(JET.min_pump_pct, JET), sol.elevation_deg, JET)
        r_hi = ballistics.range_of(ballistics.exit_velocity(100.0, JET), sol.elevation_deg, JET)
        assert r_lo < target < r_hi  # pump can trim shorter and longer


def test_plan_suppression_unreachable_returns_none() -> None:
    assert ballistics.plan_suppression(80.0, JET, 8.0, 50.0) is None
