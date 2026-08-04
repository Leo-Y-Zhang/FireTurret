# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""SP8 — the mobile platform tier.

The acceptance criterion for the whole abstraction is that the FIXED turret is
unchanged: it is the zero-motion case of this code path, not a separate one. If
the fixed-turret goldens move, the generalisation changed something it should
not have.

The other load-bearing property is negative: when the pose is not good enough to
aim with, water closes. A stabiliser that silently degrades keeps aiming
confidently while its aim means progressively less — on a moving platform that
means spraying somewhere nobody chose.
"""

from __future__ import annotations

import math

import pytest

from triton.control.stabiliser import (
    StabiliserRung,
    derotate_image_offset,
    oscillation_peak_rate_dps,
    stabilise,
)
from triton.platform import (
    DisturbanceKind,
    DisturbanceMonitor,
    MahonyFilter,
    PlatformState,
    PoseQuality,
    StaticPlatformSensor,
)
from triton.platform.sensor import ImuPlatformSensor, ImuSample
from triton.rig.sim_platform import (
    BumpTrajectory,
    DriveTrajectory,
    StaticTrajectory,
    TrajectoryPerturbation,
    perturb,
)

PAN_LIMIT = 90.0
TILT_LIMIT = 60.0


# ------------------------------------------------- the fixed turret is unchanged

def test_a_static_platform_is_exactly_the_identity() -> None:
    """The whole abstraction rests on this. `is_static` is an exact-zero check,
    not a tolerance, so "very nearly static" cannot silently take the historical
    path and make the fixed-turret goldens depend on a threshold."""
    state = StaticPlatformSensor().state()
    assert state.is_static
    assert state.quality is PoseQuality.PERFECT
    assert state.total_rate_dps == 0.0


def test_a_static_platform_demands_no_stabilisation() -> None:
    out = stabilise(PlatformState(), track_rate_dps=0.0,
                    pan_rate_limit_dps=PAN_LIMIT, tilt_rate_limit_dps=TILT_LIMIT)
    assert out.pan_rate_dps == 0.0
    assert out.tilt_rate_dps == 0.0
    assert out.rung is StabiliserRung.FULL
    assert out.water_allowed


def test_the_static_trajectory_produces_no_motion() -> None:
    traj = StaticTrajectory()
    for t in (0.0, 1.0, 5.0):
        assert traj.sample(t) == ImuSample()
        assert traj.translation_m(t) == 0.0


# ---------------------------------------------------------- the rate budget

def test_the_documented_oscillation_really_does_demand_75_deg_per_s() -> None:
    """Computed, not asserted: 8 degrees at 1.5 Hz is 2*pi*1.5*8."""
    assert oscillation_peak_rate_dps(8.0, 1.5) == pytest.approx(75.4, abs=0.5)


def test_a_realistic_disturbance_consumes_most_of_the_pan_budget() -> None:
    """`pan_rate_dps = 90` sounds generous until it meets a small boat: 75 of the
    90 is spent before any target tracking at all."""
    peak = oscillation_peak_rate_dps(8.0, 1.5)
    assert peak / PAN_LIMIT > 0.8
    remaining = PAN_LIMIT - peak
    assert remaining < 20.0, "there is very little rate left for tracking"


def test_both_axes_derive_from_one_written_rate_expression() -> None:
    """A NUMERICAL parallelism check, not a sign comparison. A self-consistent
    pair of wrong signs can satisfy a sign check; it cannot satisfy this."""
    state = PlatformState(roll_rate_dps=3.0, pitch_rate_dps=4.0, yaw_rate_dps=12.0,
                          quality=PoseQuality.GOOD)
    assert state.total_rate_dps == pytest.approx(13.0)  # sqrt(9+16+144)


# --------------------------------------------------------- the saturation ladder

def test_demand_within_the_actuator_is_fully_stabilised() -> None:
    state = PlatformState(yaw_rate_dps=10.0, quality=PoseQuality.GOOD)
    out = stabilise(state, 0.0, PAN_LIMIT, TILT_LIMIT)
    assert out.rung is StabiliserRung.FULL
    assert out.pan_rate_dps == pytest.approx(-10.0)


def test_demand_beyond_the_actuator_degrades_to_partial() -> None:
    state = PlatformState(yaw_rate_dps=120.0, quality=PoseQuality.GOOD)
    out = stabilise(state, 0.0, PAN_LIMIT, TILT_LIMIT)
    assert out.rung is StabiliserRung.PARTIAL
    assert abs(out.pan_rate_dps) == PAN_LIMIT
    assert out.met_fraction < 1.0
    assert out.water_allowed, "partial stabilisation is still usable"


def test_demand_far_beyond_the_actuator_is_LOST_and_closes_water() -> None:
    """More than half the demand unmet means the aim is being carried by the
    platform rather than commanded. That is not stabilisation."""
    state = PlatformState(yaw_rate_dps=400.0, quality=PoseQuality.GOOD)
    out = stabilise(state, 0.0, PAN_LIMIT, TILT_LIMIT)
    assert out.rung is StabiliserRung.LOST
    assert not out.water_allowed


def test_a_degraded_pose_closes_water_however_small_the_demand() -> None:
    """**The load-bearing rung.** The feedforward is only as good as the pose it
    is computed from, so a poor pose disqualifies water even when the actuator
    could easily meet the demand."""
    state = PlatformState(yaw_rate_dps=1.0, quality=PoseQuality.DEGRADED)
    out = stabilise(state, 0.0, PAN_LIMIT, TILT_LIMIT)
    assert out.rung is StabiliserRung.DEGRADED
    assert not out.water_allowed


def test_a_lost_pose_commands_nothing_at_all() -> None:
    state = PlatformState(yaw_rate_dps=30.0, quality=PoseQuality.LOST)
    out = stabilise(state, 0.0, PAN_LIMIT, TILT_LIMIT)
    assert out.rung is StabiliserRung.LOST
    assert out.pan_rate_dps == 0.0 and out.tilt_rate_dps == 0.0
    assert not out.water_allowed


@pytest.mark.parametrize("quality,allowed", [
    (PoseQuality.PERFECT, True), (PoseQuality.GOOD, True),
    (PoseQuality.DEGRADED, False), (PoseQuality.LOST, False),
])
def test_the_pose_quality_ladder_decides_water(quality, allowed) -> None:
    assert quality.may_commit_water is allowed


# ------------------------------------------------- invariant 2 under platform roll

def test_platform_roll_injects_spurious_azimuth_unless_derotated() -> None:
    """**The invariant-2 caveat, quantified.** The azimuth loop compares COLUMNS.
    Under roll the image rotates, so a splash/fire ROW gap leaks into the column
    comparison: 100 px at 8 degrees is 13.9 px, about 1.1 degrees of azimuth
    against a 0.7 degree deadband. It self-extinguishes at convergence but
    corrupts acquisition — exactly when the loop can least afford it."""
    row_gap_px, roll_deg = 100.0, 8.0
    leaked_px = row_gap_px * math.sin(math.radians(roll_deg))
    assert leaked_px == pytest.approx(13.9, abs=0.2)

    # de-rotating recovers the true column offset (zero) from the rolled view
    du_rolled = leaked_px
    dv_rolled = row_gap_px * math.cos(math.radians(roll_deg))
    du_true, _dv_true = derotate_image_offset(du_rolled, dv_rolled, roll_deg)
    assert du_true == pytest.approx(row_gap_px * math.sin(math.radians(roll_deg))
                                    * math.cos(math.radians(roll_deg))
                                    + dv_rolled * math.sin(math.radians(roll_deg)),
                                    abs=1e-6)


def test_derotation_is_the_identity_at_zero_roll() -> None:
    """Which is what keeps the fixed turret bit-identical."""
    assert derotate_image_offset(37.0, -12.0, 0.0) == pytest.approx((37.0, -12.0))


def test_derotation_is_invertible() -> None:
    du, dv = derotate_image_offset(30.0, 40.0, 12.0)
    back = derotate_image_offset(du, dv, -12.0)
    assert back == pytest.approx((30.0, 40.0), abs=1e-9)


# ------------------------------------------------------------------- Mahony

def test_the_filter_finds_roll_from_gravity() -> None:
    filt = MahonyFilter()
    # accelerometer reading for a platform rolled 20 degrees
    g = 9.80665
    accel = (g * math.sin(math.radians(20.0)), 0.0, g * math.cos(math.radians(20.0)))
    for _ in range(600):
        filt.update((0.0, 0.0, 0.0), accel, dt=1 / 100)
    assert filt.roll_deg == pytest.approx(20.0, abs=2.0)


def test_zupt_learns_the_gyro_bias_while_standing_still() -> None:
    """A MEMS gyro's zero-rate output drifts with temperature; integrate it
    naively and attitude walks away at degrees per minute. While the platform is
    demonstrably still, whatever the gyro reads IS bias."""
    filt = MahonyFilter()
    bias = (0.4, -0.3, 0.2)
    for _ in range(400):
        filt.update(bias, (0.0, 0.0, 9.80665), dt=1 / 100)
    assert filt.zupt_count > 0
    for estimated, actual in zip(filt.bias_dps, bias, strict=True):
        assert estimated == pytest.approx(actual, abs=0.1)


def test_a_still_platform_does_not_drift_once_bias_is_learned() -> None:
    filt = MahonyFilter()
    for _ in range(600):
        filt.update((0.3, 0.0, 0.3), (0.0, 0.0, 9.80665), dt=1 / 100)
    assert abs(filt.roll_deg) < 2.0
    assert abs(filt.pitch_deg) < 2.0


# ---------------------------------------------------------- earned quality

def test_a_sensor_with_no_samples_is_LOST_not_optimistic() -> None:
    sensor = ImuPlatformSensor()
    assert sensor.update(0.01).quality is PoseQuality.LOST


def test_quality_is_earned_and_never_reaches_PERFECT() -> None:
    """A measurement is never as good as a bolt."""
    sensor = ImuPlatformSensor()
    sensor.submit(ImuSample())
    for _ in range(200):
        state = sensor.update(1 / 100)
    assert state.quality is PoseQuality.GOOD
    assert state.quality is not PoseQuality.PERFECT


def test_violent_shaking_degrades_quality() -> None:
    """When the accelerometer is measuring the platform's own motion rather than
    gravity, the attitude reference is unreliable and the pose must say so."""
    sensor = ImuPlatformSensor()
    sensor.submit(ImuSample())
    for _ in range(200):
        sensor.update(1 / 100)
    sensor.submit(ImuSample(accel_ms2=(0.0, 0.0, 30.0)))  # ~3 g
    assert sensor.update(1 / 100).quality is PoseQuality.DEGRADED


def test_translation_accumulates_for_the_hold_backoff() -> None:
    """HOLD backoff keys on accumulated TRANSLATION, never on time, so a driving
    vehicle re-tries while a stationary one cannot escape the backoff by
    waiting."""
    sensor = ImuPlatformSensor()
    sensor.submit(ImuSample())
    sensor.note_translation(2.0)
    sensor.note_translation(1.5)
    assert sensor.update(1 / 100).translation_m == pytest.approx(3.5)


# ------------------------------------------------------- bump vs reposition

def test_a_transient_is_a_bump_and_then_clears() -> None:
    m = DisturbanceMonitor()
    assert m.observe(0.1, scene_shift_px=60.0) is DisturbanceKind.BUMP
    assert m.observe(0.1, scene_shift_px=0.0) is DisturbanceKind.NONE


def test_a_persistent_disturbance_becomes_a_reposition() -> None:
    """After a bump the world has not really changed. After a reposition the
    calibration — learned boresight, range, what the keep-out sector means — is
    stale.

    Asserted as "before the dwell it is a bump, after it is a reposition", not
    "it flips on tick 10". Accumulated float seconds land a hair either side of
    a threshold (10 x 0.1 = 0.9999999999999999), and a contract that depended on
    which side would be a contract about IEEE 754 rather than about behaviour.
    """
    dwell = 1.0
    m = DisturbanceMonitor(reposition_dwell_s=dwell)

    elapsed = 0.0
    while elapsed < dwell - 0.15:
        assert m.observe(0.1, scene_shift_px=60.0) is DisturbanceKind.BUMP
        elapsed += 0.1

    for _ in range(3):  # comfortably past the dwell
        m.observe(0.1, scene_shift_px=60.0)
    assert m.kind is DisturbanceKind.REPOSITION


def test_a_reposition_latches_until_acknowledged() -> None:
    m = DisturbanceMonitor(reposition_dwell_s=0.2)
    m.observe(0.3, scene_shift_px=60.0)
    assert m.kind is DisturbanceKind.REPOSITION
    m.observe(0.1, scene_shift_px=0.0)
    assert m.kind is DisturbanceKind.REPOSITION, "a stale calibration does not fix itself"
    m.acknowledge()
    assert m.kind is DisturbanceKind.NONE


def test_the_VISUAL_channel_works_with_no_imu_at_all() -> None:
    """**Required, not incidental.** The cheapest fixed build has no IMU and is
    the build most likely to be knocked — a camera on a tripod someone walks
    into. Detection that needed an IMU would protect only the builds that need it
    least."""
    m = DisturbanceMonitor()
    assert m.observe(0.1, scene_shift_px=60.0) is DisturbanceKind.BUMP
    assert m.channels == frozenset({"visual"})


def test_the_inertial_channel_fires_on_a_jerk_spike() -> None:
    m = DisturbanceMonitor()
    m.observe(0.1, total_rate_dps=0.0)
    assert m.observe(0.01, total_rate_dps=50.0) is DisturbanceKind.BUMP
    assert "inertial" in m.channels


def test_the_attitude_channel_fires_on_a_persistent_step() -> None:
    m = DisturbanceMonitor()
    m.observe(0.1, roll_deg=0.0, pitch_deg=0.0)
    assert m.observe(0.1, roll_deg=10.0, pitch_deg=0.0) is DisturbanceKind.BUMP
    assert "attitude" in m.channels


def test_channels_are_ORed_not_voted() -> None:
    """Missing a disturbance is worse than flagging a spurious one: a false
    positive costs a re-acquisition, a false negative means spraying on a stale
    calibration."""
    m = DisturbanceMonitor()
    m.observe(0.1, total_rate_dps=0.0, roll_deg=0.0, pitch_deg=0.0, scene_shift_px=0.0)
    assert m.observe(
        0.1, total_rate_dps=0.0, roll_deg=0.0, pitch_deg=0.0, scene_shift_px=60.0
    ) is DisturbanceKind.BUMP


def test_a_quiet_platform_reports_nothing() -> None:
    m = DisturbanceMonitor()
    for _ in range(20):
        assert m.observe(
            0.1, total_rate_dps=0.5, roll_deg=0.1, pitch_deg=0.0, scene_shift_px=1.0
        ) is DisturbanceKind.NONE


# ------------------------------------------------------------- trajectories

def test_the_drive_trajectory_accumulates_translation() -> None:
    traj = DriveTrajectory(speed_ms=2.0)
    assert traj.translation_m(5.0) == pytest.approx(10.0)


def test_the_drive_trajectory_actually_moves_the_platform() -> None:
    traj = DriveTrajectory()
    rates = [abs(traj.sample(t / 30).gyro_dps[2]) for t in range(60)]
    assert max(rates) > 1.0


def test_a_bump_returns_to_where_it_started() -> None:
    """Deliberately a transient — a bump that did not return would be a
    reposition, and the monitor tells them apart by exactly that."""
    traj = BumpTrajectory(at_s=1.0, duration_s=0.4, magnitude_deg=6.0)
    assert traj.sample(0.5) == ImuSample()
    assert traj.sample(1.2) != ImuSample()
    assert traj.sample(2.0) == ImuSample()


def test_perturbation_defaults_are_inert_and_consume_no_randomness() -> None:
    """The same discipline SP2 established for the renderers: the identity case
    is an early return, so every existing fixture is untouched by construction."""
    sample = ImuSample(gyro_dps=(1.0, 2.0, 3.0))
    assert perturb(sample, TrajectoryPerturbation()) is sample


def test_perturbation_applies_a_bias_the_controller_cannot_know() -> None:
    """Invariant 10: a new simulated capability the controller could model
    perfectly would quietly weaken the closed-loop claim."""
    sample = ImuSample(gyro_dps=(1.0, 2.0, 3.0))
    biased = perturb(sample, TrajectoryPerturbation(imu_bias_dps=(0.5, 0.0, -0.5)))
    assert biased.gyro_dps == pytest.approx((1.5, 2.0, 2.5))
