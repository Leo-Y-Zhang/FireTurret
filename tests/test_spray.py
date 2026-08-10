# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""SP5 — spray dispersion and the deposition-aware keep-out.

The load-bearing idea: `pan_keepout_deg` constrains where the nozzle POINTS.
Water lands over an AREA. For the fire profile's solid stream those are the same
thing, so the gap is invisible. For a fan pattern it is not:

    a 10 degree fan at 7 m reaches 0.61 m either side of the aim
    an aim 3 degrees outside a sector edge is only 0.37 m outside it
    => 0.24 m of water lands INSIDE the sector, with the geofence satisfied

That arithmetic is checked below rather than quoted, and the acceptance test the
spec asks for is here verbatim: a wide-pattern aim 3 degrees outside an edge must
produce a `Violation` while `_apply_keepout` on the same pan returns it unchanged.
"""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from fireturret.config import DEFAULT_CONFIG, TurretConfig
from fireturret.control.mission import MissionController
from fireturret.geometry import CameraModel
from fireturret.spray.envelope import LEGACY_ENVELOPE, SprayEnvelope, fan_envelope
from fireturret.spray.keepout import check_deposition, wetted_interval_deg
from fireturret.vision.impact import ImpactObservation

CFG = DEFAULT_CONFIG
SECTOR = (-12.0, 12.0)
FAN = fan_envelope(10.0)


# ------------------------------------------------------------------ envelope

def test_the_legacy_envelope_is_a_point() -> None:
    """A solid stream has not broken up by the time it lands — breakup length
    ~19 m against a ~7 m throw — so a point footprint is the correct answer, not
    a simplification."""
    assert LEGACY_ENVELOPE.is_point
    assert LEGACY_ENVELOPE.wetted_half_width_m(7.0) == 0.0
    assert LEGACY_ENVELOPE.wetted_half_angle_deg(7.0) == 0.0


def test_a_ten_degree_fan_is_sixty_one_centimetres_wide_at_seven_metres() -> None:
    """The spec's arithmetic, checked rather than quoted."""
    assert FAN.wetted_half_width_m(7.0) == pytest.approx(0.61, abs=0.02)


def test_lateral_spread_grows_with_range_and_longitudinal_does_not() -> None:
    """Different units on purpose: lateral spread is set by the fan ANGLE and so
    scales with range; longitudinal spread is set by ballistic spread and does
    not. Storing both per-metre would quietly make one of them wrong."""
    assert FAN.lateral_sigma_m(10.0) == pytest.approx(FAN.lateral_sigma_m(5.0) * 2.0)
    envelope = SprayEnvelope(longitudinal_sigma_m=0.2, lateral_sigma_per_m=0.05)
    assert envelope.longitudinal_sigma_m == 0.2  # independent of range by construction


def test_the_footprint_is_inflated_to_two_sigma_by_default() -> None:
    """The question a keep-out asks is where water CAN land, not where it
    typically does."""
    assert FAN.wetted_half_width_m(7.0, sigmas=2.0) == pytest.approx(
        2.0 * FAN.lateral_sigma_m(7.0)
    )


@pytest.mark.parametrize("bad", [
    {"lateral_sigma_per_m": -0.1},
    {"longitudinal_sigma_m": -0.1},
    {"equivalent_drag_k": -1.0},
])
def test_negative_spreads_are_rejected(bad) -> None:
    with pytest.raises(ValueError):
        SprayEnvelope(**bad)


# ------------------------------------------------------- THE acceptance test

def test_an_aim_outside_the_sector_can_still_wet_it() -> None:
    """The spec's acceptance criterion, stated exactly: a wide-pattern aim three
    degrees outside a sector edge produces a Violation, while the AIM geofence
    on the same pan is perfectly satisfied."""
    aim = SECTOR[1] + 3.0  # 3 degrees clear of the edge
    range_m = 7.0

    violation = check_deposition(aim, range_m, SECTOR, FAN)
    assert violation is not None, "the wetted footprint did not register a violation"
    assert violation.overlap_deg > 0.0

    mission = MissionController(
        replace(CFG, turret=TurretConfig(pan_keepout_deg=SECTOR)),
        CameraModel(CFG.camera),
    )
    assert mission._apply_keepout(aim) == aim, "the aim geofence considers this legal"


def test_the_overlap_matches_the_hand_calculation() -> None:
    """0.61 m footprint, 0.37 m clearance => 0.24 m of water inside the sector."""
    range_m = 7.0
    clearance_deg = 3.0
    half_width_m = FAN.wetted_half_width_m(range_m)
    clearance_m = range_m * math.tan(math.radians(clearance_deg))
    assert half_width_m == pytest.approx(0.61, abs=0.02)
    assert clearance_m == pytest.approx(0.37, abs=0.02)
    assert half_width_m - clearance_m == pytest.approx(0.24, abs=0.03)


def test_a_solid_stream_never_violates_from_outside_the_sector() -> None:
    """Which is why this whole layer is behaviour-neutral for the fire profile:
    for a point footprint the deposition check IS the aim check."""
    assert check_deposition(SECTOR[1] + 0.1, 7.0, SECTOR, LEGACY_ENVELOPE) is None
    assert check_deposition(0.0, 7.0, SECTOR, LEGACY_ENVELOPE) is not None  # aim inside


def test_no_sector_configured_is_never_a_violation() -> None:
    assert check_deposition(0.0, 7.0, None, FAN) is None


def test_an_aim_far_outside_is_clear() -> None:
    assert check_deposition(60.0, 7.0, SECTOR, FAN) is None


def test_a_fans_angular_width_is_range_invariant_while_its_linear_width_is_not() -> None:
    """Worth pinning because it is counter-intuitive and it decides the design.

    A fan's LINEAR footprint grows with range — 0.61 m at 7 m, 1.05 m at 12 m —
    but its ANGULAR footprint does not, because the fan is defined by an angle in
    the first place. So a keep-out expressed in DEGREES is violated at the same
    aim regardless of range, and a keep-out expressed in metres would not be.
    Sectors here are in degrees, which is why range does not rescue a bad aim.
    """
    aim = SECTOR[1] + 3.0
    near = check_deposition(aim, 3.0, SECTOR, FAN)
    far = check_deposition(aim, 12.0, SECTOR, FAN)
    assert near is not None and far is not None
    assert near.half_angle_deg == pytest.approx(far.half_angle_deg)
    assert FAN.wetted_half_width_m(12.0) > FAN.wetted_half_width_m(3.0)


def test_a_longer_throw_wets_more_ground_for_the_same_angle() -> None:
    assert FAN.wetted_half_width_m(7.0) == pytest.approx(0.61, abs=0.02)
    assert FAN.wetted_half_width_m(12.0) == pytest.approx(1.05, abs=0.04)


def test_the_wetted_interval_is_centred_on_the_aim() -> None:
    left, right = wetted_interval_deg(20.0, 7.0, FAN)
    assert (left + right) / 2.0 == pytest.approx(20.0)
    assert right - left > 0.0


# ---------------------------------------------------------- mission wiring

def _mission(envelope, keepout=SECTOR) -> MissionController:
    m = MissionController(
        replace(CFG, turret=TurretConfig(pan_keepout_deg=keepout)),
        CameraModel(CFG.camera),
        armed=True,
    )
    m.spray_envelope = envelope
    m._ranger.ema = 7.0
    return m


def test_the_mission_withholds_water_on_a_predicted_violation() -> None:
    from fireturret.rig.interface import RigCommand

    mission = _mission(FAN)
    wet = RigCommand(pan_deg=SECTOR[1] + 3.0, tilt_deg=22.0, pump_pct=55.0, valve=True)
    gated = mission._apply_deposition_keepout(wet)
    assert gated.valve is False
    assert gated.pump_pct == 0.0
    assert gated.pan_deg == wet.pan_deg, "deny-only: it must not re-aim"


def test_the_default_envelope_leaves_the_command_untouched() -> None:
    """The property that keeps every fixture still."""
    from fireturret.rig.interface import RigCommand

    mission = _mission(LEGACY_ENVELOPE)
    wet = RigCommand(pan_deg=SECTOR[1] + 3.0, tilt_deg=22.0, pump_pct=55.0, valve=True)
    assert mission._apply_deposition_keepout(wet) == wet


def test_a_dry_command_is_not_affected() -> None:
    from fireturret.rig.interface import RigCommand

    mission = _mission(FAN)
    dry = RigCommand(pan_deg=SECTOR[1] + 3.0, tilt_deg=22.0, pump_pct=0.0, valve=False)
    assert mission._apply_deposition_keepout(dry) == dry


# ------------------------------------------------------------ base_y measurement

def test_impact_observation_defaults_preserve_the_old_semantics() -> None:
    """Every existing construction and comparison must behave identically."""
    old_style = ImpactObservation(cx=1.0, cy=2.0, area=3.0)
    assert old_style.base_y is None
    assert old_style.major_sigma_px == 0.0
    assert old_style == ImpactObservation(cx=1.0, cy=2.0, area=3.0)


def test_base_y_is_below_the_centroid_for_a_real_splash() -> None:
    """`base_y` exists because the plume is airborne and the centroid sits above
    the ground contact. It is a MEASUREMENT of that contact, so a wrong value
    still leaves a visible residual — unlike an asserted bias constant, which
    would move the loop's fixed point by exactly the amount asserted and leave
    nothing to detect the assertion being wrong."""
    import numpy as np

    from fireturret.vision.impact import SplashDetector

    det = SplashDetector(min_area_px=10)
    base = np.zeros((200, 300, 3), dtype=np.uint8)
    det.observe(base, target_px=(150, 100))

    frame = base.copy()
    frame[100:140, 130:170] = (205, 205, 200)  # a bright desaturated blob
    obs = det.observe(frame, target_px=(150, 100))

    assert obs is not None
    assert obs.base_y is not None
    assert obs.base_y > obs.cy, "ground contact must be below the plume centroid"
    assert obs.major_sigma_px > 0.0
