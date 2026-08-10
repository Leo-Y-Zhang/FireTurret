# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""SP4 — range-loop truth.

Two defects, both measured rather than argued (spec §3.5, §3.6):

**§3.5 — the loop was measuring the wrong water.** The observation window sat
0.30-0.70 s after a correction while the water it observed was launched
0.94-1.28 s earlier. The median was therefore of the PREVIOUS cycle's pump
setting. Invariant 3 — observe, then correct — was not true of the shipped
system. The loop survived only because its gain was far below the stability
limit: too sluggish to oscillate on a 1-2 sample delay.

**§3.6 — control authority collapses across the envelope.** Ballistic authority
`dR/dpump` falls 2.91x from pump 25% to 85%; the OBSERVABLE loop gain falls
7.43x, because perspective foreshortening shrinks the pixel row spacing with
range. Range stays monotone in pump throughout, so a boolean monotonicity check
passes happily while authority dies by a factor of seven.

The numbers in this file are the spec's own measurements. If a future change
moves them, that is worth knowing about explicitly.
"""

from __future__ import annotations

import pytest

from fireturret.ballistics import arc_with_time, exit_velocity, range_of, simulate_arc
from fireturret.config import DEFAULT_CONFIG
from fireturret.control.plant import (
    MIN_PLANT_PX_PER_PCT,
    TARGET_LOOP_GAIN,
    plant_gain,
    schedule_cycle,
)
from fireturret.control.suppress import (
    FPS,
    OBSERVE_FRAMES,
    SETTLE_MARGIN_FRAMES,
    SuppressionCycle,
)
from fireturret.geometry import CameraModel

CFG = DEFAULT_CONFIG
JET = CFG.jet
CAM = CameraModel(CFG.camera)
SUPPRESS_TILT = CFG.servo.suppress_tilt_deg
BAND = [25.0, 40.0, 55.0, 70.0, 85.0]


# ------------------------------------------------------------- arc_with_time

def test_simulate_arc_is_an_unchanged_thin_wrapper() -> None:
    """Three source designs each wanted a different incompatible extension to
    `simulate_arc`. `arc_with_time` is the shape that serves all of them, and the
    old signature must keep returning exactly what it always did."""
    speed = exit_velocity(55.0, JET)
    assert simulate_arc(speed, SUPPRESS_TILT, JET) == arc_with_time(
        speed, SUPPRESS_TILT, JET
    ).points


@pytest.mark.parametrize("pump,expected_s", [(25.0, 0.944), (55.0, 1.136), (100.0, 1.284)])
def test_flight_time_matches_the_measured_values(pump, expected_s) -> None:
    """The measurements §3.5 is built on. Water is airborne for around a second —
    far longer than the 0.30-0.70 s window that was being used to observe it."""
    arc = arc_with_time(exit_velocity(pump, JET), SUPPRESS_TILT, JET)
    assert arc.flight_time_s == pytest.approx(expected_s, abs=0.02)


def test_arc_carries_impact_speed_and_range() -> None:
    arc = arc_with_time(exit_velocity(55.0, JET), SUPPRESS_TILT, JET)
    assert arc.impact_speed_mps > 0.0
    assert arc.range_m == pytest.approx(
        range_of(exit_velocity(55.0, JET), SUPPRESS_TILT, JET), abs=1e-9
    )


# --------------------------------------------------------------- geometry

@pytest.mark.parametrize("range_m", [3.0, 5.0, 7.0, 9.0, 11.0])
def test_row_for_range_inverts_range_from_row(range_m) -> None:
    """The forward projection the plant gain needs must be the exact inverse of
    the bootstrap that already existed, or the two disagree about where the
    ground is."""
    row = CAM.px_row_for_ground_range(range_m)
    assert CAM.ground_range_from_px(CAM.cx, row) == pytest.approx(range_m, rel=1e-9)


# ------------------------------------------------------- the measured collapse

@pytest.mark.parametrize("pump,d_range", [
    (25.0, 0.1148), (40.0, 0.0782), (55.0, 0.0591), (70.0, 0.0474), (85.0, 0.0394),
])
def test_ballistic_authority_matches_the_spec_measurements(pump, d_range) -> None:
    gain = plant_gain(pump, SUPPRESS_TILT, JET, CAM)
    assert gain.d_range_per_pct == pytest.approx(d_range, rel=0.05)


def test_ballistic_authority_falls_about_three_times_across_the_band() -> None:
    lo = plant_gain(25.0, SUPPRESS_TILT, JET, CAM).d_range_per_pct
    hi = plant_gain(85.0, SUPPRESS_TILT, JET, CAM).d_range_per_pct
    assert lo / hi == pytest.approx(2.91, rel=0.10)


def test_observable_authority_falls_much_faster_than_ballistic_authority() -> None:
    """The heart of §3.6, and the reason measuring in metres is misleading:
    perspective foreshortening means the loop SEES the collapse roughly 2.5x
    harder than the physics has it."""
    metres = (plant_gain(25.0, SUPPRESS_TILT, JET, CAM).d_range_per_pct
              / plant_gain(85.0, SUPPRESS_TILT, JET, CAM).d_range_per_pct)
    pixels = abs(plant_gain(25.0, SUPPRESS_TILT, JET, CAM).d_px_per_pct
                 / plant_gain(85.0, SUPPRESS_TILT, JET, CAM).d_px_per_pct)
    assert pixels == pytest.approx(7.43, rel=0.15)
    assert pixels > metres * 2.0


def test_range_stays_monotone_while_authority_dies() -> None:
    """Why a boolean monotonicity gate is not a substitute for a gain check: it
    passes at every point in the band where authority has collapsed 7x."""
    ranges = [range_of(exit_velocity(p, JET), SUPPRESS_TILT, JET) for p in BAND]
    assert ranges == sorted(ranges), "range should be monotone in pump"


# ------------------------------------------------------------ gain scheduling

@pytest.mark.parametrize("pump", BAND)
def test_scheduled_gain_holds_loop_gain_near_the_target(pump) -> None:
    """One fixed constant cannot serve a plant that varies 7.4x. The schedule
    holds `L = gain x plant` near TARGET_LOOP_GAIN wherever the clamps allow."""
    gain = plant_gain(pump, SUPPRESS_TILT, JET, CAM)
    scheduled = gain.scheduled_gain(lo=CFG.servo.range_gain_pct_per_px, hi=0.60)
    loop_gain = scheduled * abs(gain.d_px_per_pct)
    assert loop_gain <= TARGET_LOOP_GAIN + 1e-9, "scheduled gain exceeded the target"


def test_scheduled_gain_is_never_below_the_configured_floor() -> None:
    """The schedule may only ever make the loop MORE responsive than the shipped
    constant, never less — otherwise a modelling error could stall the loop."""
    for pump in BAND:
        gain = plant_gain(pump, SUPPRESS_TILT, JET, CAM)
        scheduled = gain.scheduled_gain(lo=CFG.servo.range_gain_pct_per_px, hi=0.60)
        assert scheduled >= CFG.servo.range_gain_pct_per_px


def test_scheduled_gain_is_clamped_at_the_top_of_the_envelope() -> None:
    """Near the top the computed gain grows without bound. A safety-relevant loop
    must not inherit a model's asymptote."""
    gain = plant_gain(99.5, SUPPRESS_TILT, JET, CAM)
    assert gain.scheduled_gain(lo=0.06, hi=0.60) <= 0.60


def test_the_fixed_constant_was_far_below_the_stability_limit() -> None:
    """Context for why the old loop survived a broken invariant 3: it was too
    sluggish to oscillate. Measured L was 0.068 at pump 25% against a target of
    0.3 — a factor of four unused."""
    gain = plant_gain(25.0, SUPPRESS_TILT, JET, CAM)
    old_loop_gain = CFG.servo.range_gain_pct_per_px * abs(gain.d_px_per_pct)
    assert old_loop_gain == pytest.approx(0.068, rel=0.15)
    assert old_loop_gain < TARGET_LOOP_GAIN


# ------------------------------------------------------- the envelope check

def test_a_pump_with_no_observable_authority_is_out_of_envelope() -> None:
    """"Out of envelope" must mean "a correction would produce no measurable
    response", not only "the pump hit its stop"."""
    gain = plant_gain(55.0, SUPPRESS_TILT, JET, CAM)
    assert gain.observable is True

    starved = type(gain)(
        pump_pct=99.0, tilt_deg=SUPPRESS_TILT, range_m=10.0,
        d_range_per_pct=1e-6, d_px_per_pct=MIN_PLANT_PX_PER_PCT / 2,
    )
    assert starved.observable is False
    assert starved.scheduled_gain(lo=0.06, hi=0.60) == 0.06


# ---------------------------------------------- THE contract SP4 establishes

@pytest.mark.parametrize("pump", [25.0, 40.0, 55.0, 70.0, 85.0, 100.0])
def test_every_observation_post_dates_the_correction_that_caused_it(pump) -> None:
    """**The contract this whole sub-project exists to establish.**

    A cycle applies a correction, then waits, then observes. For the observation
    to be evidence ABOUT that correction, the water it sees must have been
    launched after the correction took effect — i.e. the settle window must be at
    least the flight time. Before SP4 the settle window was 0.30 s against a
    flight time of ~1.1 s, so every observation was evidence about the PREVIOUS
    correction, and the loop was crediting each correction with the last one's
    result.
    """
    sched = schedule_cycle(
        pump, SUPPRESS_TILT, JET, CAM,
        base_gain=CFG.servo.range_gain_pct_per_px, max_gain=0.60, fps=FPS,
        settle_margin_frames=SETTLE_MARGIN_FRAMES, observe_frames=OBSERVE_FRAMES,
    )
    settle_s = sched.settle_frames / FPS
    assert settle_s >= sched.flight_time_s, (
        f"at pump {pump}% the loop starts observing {settle_s:.3f} s after a "
        f"correction, but its water is airborne for {sched.flight_time_s:.3f} s — "
        "so it is measuring the previous cycle's setting"
    )


def test_the_old_fixed_window_violated_that_contract() -> None:
    """Guards the claim, not just the fix. If this ever passes, the premise of
    SP4 was wrong and the numbers above need re-measuring."""
    from fireturret.control.suppress import SUPP_SETTLE_FRAMES

    old_settle_s = SUPP_SETTLE_FRAMES / FPS
    flight_s = arc_with_time(exit_velocity(55.0, JET), SUPPRESS_TILT, JET).flight_time_s
    assert old_settle_s < flight_s, "the old fixed settle window was not actually too short"


def test_reachability_is_judged_on_a_tighter_threshold_than_control() -> None:
    """Why HOLD was unreliable, stated as a property.

    The control deadband (12 px) exists to stop the loop dithering on noise. At
    long range it is far larger than the error a genuinely unreachable target
    produces: measured below, a shortfall of over a metre at 10 m is only a few
    pixels. Judging reach on the control deadband therefore reads "out of reach"
    as "on target", and HOLD only ever fired when noise happened to cross 12 px.
    """
    from fireturret.control.suppress import REACH_SHORTFALL_PX

    assert REACH_SHORTFALL_PX < CFG.servo.range_deadband_px

    short_by_m = 1.4
    target_m = 10.0
    rows = abs(CAM.px_row_for_ground_range(target_m)
               - CAM.px_row_for_ground_range(target_m - short_by_m))
    assert rows < CFG.servo.range_deadband_px, (
        f"a {short_by_m} m shortfall at {target_m} m is {rows:.1f} px — the premise "
        "of the reachability threshold is that this is INSIDE the control deadband"
    )
    assert rows > REACH_SHORTFALL_PX, "but it must be visible to the reach test"


def test_the_cycle_retimes_itself_from_the_current_solution() -> None:
    """Flight time varies with pump across the band, so one computed window is no
    better than one constant — it has to be recomputed per cycle."""
    from fireturret.control.servo import VisualServo

    cycle = SuppressionCycle(CFG, CAM, VisualServo(CFG.servo, CFG.turret, CAM), 30.0)
    cycle.retime(25.0, SUPPRESS_TILT)
    slow_pump = cycle.settle_frames
    cycle.retime(100.0, SUPPRESS_TILT)
    fast_pump = cycle.settle_frames
    assert fast_pump > slow_pump, "a longer flight time must produce a longer wait"
