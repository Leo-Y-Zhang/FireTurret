# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""SP1 — the keep-out sector must protect the SWEPT PATH, not just the aim.

`MissionController._apply_keepout` pushes a command that lies *inside* the
sector to the nearer edge, but a command *outside* the sector passes through
unchanged. So a slew from one side of a protected sector to the other traverses
it, and nothing closes the valve while that happens: the geofence protects the
aim, not the path the nozzle sweeps.

These tests pin the fix. `swept_intersects_keepout` is the pure predicate;
`MissionController._apply_swept_keepout` is the gate that consumes it.
"""

from __future__ import annotations

import pytest

from fireturret.config import DEFAULT_CONFIG, TurretConfig
from fireturret.control.mission import MissionController, swept_intersects_keepout
from fireturret.geometry import CameraModel
from fireturret.rig.interface import RigCommand, RigTelemetry

KEEPOUT = (-10.0, 10.0)


# --------------------------------------------------------------- pure predicate

@pytest.mark.parametrize(
    "a,b,expected",
    [
        # straddling the sector: the whole point of this fix
        (-30.0, 30.0, True),
        (30.0, -30.0, True),  # direction must not matter
        # one endpoint inside
        (-30.0, 0.0, True),
        (0.0, 30.0, True),
        # both inside
        (-5.0, 5.0, True),
        # touching an edge exactly counts as intersecting (conservative)
        (-30.0, -10.0, True),
        (10.0, 30.0, True),
        # cleanly outside, same side
        (-40.0, -20.0, False),
        (20.0, 40.0, False),
        # degenerate: no motion, outside
        (25.0, 25.0, False),
        # degenerate: no motion, inside
        (0.0, 0.0, True),
    ],
)
def test_swept_predicate(a: float, b: float, expected: bool) -> None:
    assert swept_intersects_keepout(a, b, KEEPOUT) is expected


def test_swept_predicate_none_keepout_never_intersects() -> None:
    assert swept_intersects_keepout(-180.0, 180.0, None) is False


# ------------------------------------------------------------------- the gate

def _mission(keepout=KEEPOUT) -> MissionController:
    cfg = DEFAULT_CONFIG
    cfg = type(cfg)(
        camera=cfg.camera,
        turret=TurretConfig(pan_keepout_deg=keepout),
        jet=cfg.jet,
        detector=cfg.detector,
        servo=cfg.servo,
        mission=cfg.mission,
    )
    return MissionController(cfg, CameraModel(cfg.camera), armed=True)


def _tel(pan: float) -> RigTelemetry:
    return RigTelemetry(
        pan_deg=pan, tilt_deg=20.0, pump_pct=0.0, valve=False,
        estop=False, ok=True, homed=True,
    )


def test_slew_across_sector_closes_the_valve() -> None:
    """The motivating case. Commanded pan is on the far side of the sector from
    where the turret actually is, so the nozzle sweeps through the protected
    sector. Water must be off for the whole transit."""
    m = _mission()
    wet = RigCommand(pan_deg=30.0, tilt_deg=22.0, pump_pct=55.0, valve=True)
    gated = m._apply_swept_keepout(wet, _tel(-30.0))
    assert gated.valve is False
    assert gated.pump_pct == 0.0


def test_aim_is_not_altered_by_the_water_gate() -> None:
    """The gate subtracts permission only. It must never re-aim the turret —
    that is the mission's job, and a safety layer that commands motion has its
    own failure surface."""
    m = _mission()
    wet = RigCommand(pan_deg=30.0, tilt_deg=22.0, pump_pct=55.0, valve=True)
    gated = m._apply_swept_keepout(wet, _tel(-30.0))
    assert gated.pan_deg == wet.pan_deg
    assert gated.tilt_deg == wet.tilt_deg


def test_spray_on_one_side_is_untouched() -> None:
    m = _mission()
    wet = RigCommand(pan_deg=30.0, tilt_deg=22.0, pump_pct=55.0, valve=True)
    gated = m._apply_swept_keepout(wet, _tel(28.0))
    assert gated == wet


def test_no_keepout_configured_is_a_no_op() -> None:
    m = _mission(keepout=None)
    wet = RigCommand(pan_deg=30.0, tilt_deg=22.0, pump_pct=55.0, valve=True)
    assert m._apply_swept_keepout(wet, _tel(-30.0)) == wet


def test_water_stays_closed_for_the_hold_after_the_transit_ends() -> None:
    """Water already in the air was launched during the transit, so clearing the
    sector does not immediately make it safe: the hold covers flight time."""
    m = _mission()
    wet = RigCommand(pan_deg=30.0, tilt_deg=22.0, pump_pct=55.0, valve=True)

    m._apply_swept_keepout(wet, _tel(-30.0))          # transit: blocks + arms hold
    m.tick_keepout_hold(0.05)                          # now clear of the sector...
    still_blocked = m._apply_swept_keepout(wet, _tel(28.0))
    assert still_blocked.valve is False, "hold must outlast the transit itself"

    m.tick_keepout_hold(m.keepout_hold_s + 0.01)       # ...and after the hold expires
    assert m._apply_swept_keepout(wet, _tel(28.0)) == wet


def test_hold_is_at_least_the_worst_case_flight_time() -> None:
    """The hold exists so water launched during a transit has landed. Anything
    shorter re-opens the valve while that water is still airborne."""
    from fireturret import ballistics

    m = _mission()
    worst = ballistics.flight_time_of(
        ballistics.exit_velocity(100.0, DEFAULT_CONFIG.jet),
        DEFAULT_CONFIG.servo.suppress_tilt_deg,
        DEFAULT_CONFIG.jet,
    )
    assert worst > 1.2, "sanity: measured ~1.284 s at full pump"
    assert m.keepout_hold_s >= worst


def test_dry_command_through_the_sector_is_allowed() -> None:
    """Slewing through a keep-out with the water OFF is legal and necessary —
    otherwise the turret could never traverse to reach a target beyond it."""
    m = _mission()
    dry = RigCommand(pan_deg=30.0, tilt_deg=22.0, pump_pct=0.0, valve=False)
    assert m._apply_swept_keepout(dry, _tel(-30.0)) == dry


# ------------------------------------------------------------- slew-rate hold

def test_fast_slew_holds_water_even_without_crossing_the_sector() -> None:
    """Water launched during a fast slew lands along an arc, not at the aim
    point, so the straddle test stops describing where the water goes. Near a
    protected sector that arc is exactly what must not stray."""
    m = _mission()
    wet = RigCommand(pan_deg=30.0, tilt_deg=22.0, pump_pct=55.0, valve=True)
    dt = 1 / 30

    # settle on one side so the slew is measured from a known previous command
    m._apply_swept_keepout(wet, _tel(30.0), dt)
    m.tick_keepout_hold(m.keepout_hold_s + 1.0)
    assert m._apply_swept_keepout(wet, _tel(30.0), dt) == wet, "steady aim must spray"

    # now jump 2 deg in one tick = 60 deg/s, well above the threshold, while
    # staying entirely on the far side of the sector
    fast = RigCommand(pan_deg=32.0, tilt_deg=22.0, pump_pct=55.0, valve=True)
    gated = m._apply_swept_keepout(fast, _tel(30.0), dt)
    assert gated.valve is False
    assert gated.pan_deg == fast.pan_deg, "the gate must not re-aim"


def test_tracking_rate_slew_still_sprays() -> None:
    """Closed-loop tracking of a moving fire runs at ~4 deg/s. If the hold fired
    at that rate the turret could never suppress a moving target near a sector."""
    m = _mission()
    dt = 1 / 30
    pan = 30.0
    m._apply_swept_keepout(
        RigCommand(pan_deg=pan, tilt_deg=22.0, pump_pct=55.0, valve=True), _tel(pan), dt)
    m.tick_keepout_hold(m.keepout_hold_s + 1.0)

    for _ in range(10):
        pan += 4.0 * dt  # 4 deg/s, the documented moving-fire tracking rate
        cmd = RigCommand(pan_deg=pan, tilt_deg=22.0, pump_pct=55.0, valve=True)
        gated = m._apply_swept_keepout(cmd, _tel(pan), dt)
        m.tick_keepout_hold(dt)
        assert gated.valve is True, f"tracking at 4 deg/s was gated at pan={pan:.2f}"


def test_slew_hold_is_inert_without_a_keepout() -> None:
    """The whole slew rule is gated on a configured sector, which no golden or
    e2e scenario sets. This is what makes SP1 behaviour-neutral for run_sim."""
    m = _mission(keepout=None)
    dt = 1 / 30
    slow = RigCommand(pan_deg=0.0, tilt_deg=22.0, pump_pct=55.0, valve=True)
    fast = RigCommand(pan_deg=60.0, tilt_deg=22.0, pump_pct=55.0, valve=True)
    assert m._apply_swept_keepout(slow, _tel(0.0), dt) == slow
    assert m._apply_swept_keepout(fast, _tel(0.0), dt) == fast


def test_first_tick_is_not_treated_as_an_infinite_slew() -> None:
    """With no previous command there is no rate to measure; inventing one from
    a zero default would gate the opening shot of every engagement."""
    m = _mission()
    wet = RigCommand(pan_deg=30.0, tilt_deg=22.0, pump_pct=55.0, valve=True)
    assert m._apply_swept_keepout(wet, _tel(30.0), 1 / 30) == wet
