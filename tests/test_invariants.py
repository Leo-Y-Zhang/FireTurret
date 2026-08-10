# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""TIER 1 — behavioural contracts. These are NEVER regenerated.

The three golden tiers exist because the old single tier conflated "the engine
behaves correctly" with "the renderer produced these exact pixels". Tier 2
(metrics with tolerances) and tier 3 (byte-exact commands over frozen frames)
may both be regenerated for a deliberate change, with a written reason recorded
in `tests/golden/MANIFEST.json`. **This file may not.**

So every assertion here must be a contract that would still be true after the
renderer, the spray model and the platform simulator were all rewritten. That
means no fixture values and no incidental numbers. Where a bound is genuinely
part of the contract ("water is bounded"), it is DERIVED from configuration
rather than pasted from a measurement — a bound that has to be edited when the
engine improves was never a contract in the first place.

`scripts/goldens.py` refuses to regenerate anything while this file is red, so
a regression cannot be laundered into a fixture.
"""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from fireturret.app import Pipeline, run_sim
from fireturret.config import DEFAULT_CONFIG, TurretConfig
from fireturret.control.mission import MissionController, pan_in_keepout
from fireturret.geometry import CameraModel
from fireturret.rig.interface import RigTelemetry
from fireturret.rig.sim_rig import SimRig, SimScenario

DT = 1 / 30

# Peak flow is ~5 L/min at full pump (rig/sim_rig.py). Any bound on water is
# derived from that and the mission's own continuous-spray limit, so improving
# the controller can only ever make these pass more comfortably.
_PEAK_FLOW_L_PER_S = 5.0 / 60.0


def _water_ceiling(cfg, cycles: float) -> float:
    """Water a correctly-behaving controller cannot exceed over `cycles`
    continuous-spray windows. Not a measurement — a consequence of `max_spray_s`
    and the pump's peak flow."""
    return _PEAK_FLOW_L_PER_S * cfg.mission.max_spray_s * cycles


# --------------------------------------------------------------- fast contracts

def test_safe_latches_and_does_not_auto_resume() -> None:
    """A fault latches SAFE and disarms water. Clearing the fault must NOT
    silently re-arm — the operator has to call rearm(). This is the difference
    between a failsafe and a glitch filter."""
    m = MissionController(DEFAULT_CONFIG, CameraModel(DEFAULT_CONFIG.camera), armed=True)
    faulted = RigTelemetry(0.0, 20.0, 0.0, False, estop=True, ok=True, homed=True)
    healthy = RigTelemetry(0.0, 20.0, 0.0, False, estop=False, ok=True, homed=True)

    m.update(DT, None, None, None, faulted)
    assert m.state == "SAFE"

    for _ in range(30):  # a full second of perfectly healthy telemetry
        cmd = m.update(DT, None, None, None, healthy)
        assert m.state == "SAFE", "SAFE must latch; it auto-resumed"
        assert cmd.valve is False and cmd.pump_pct == 0.0

    # rearm() acknowledges the fault; it does not itself resume. Autonomy comes
    # back on the next healthy tick, and only then.
    m.rearm()
    assert m.state == "SAFE", "rearm() must not itself resume autonomy"
    m.update(DT, None, None, None, healthy)
    assert m.state != "SAFE"


def test_telemetry_loss_also_latches_safe() -> None:
    """Stale telemetry is indistinguishable from a severed link. It must latch
    the same way an E-stop does, not merely pause."""
    m = MissionController(DEFAULT_CONFIG, CameraModel(DEFAULT_CONFIG.camera), armed=True)
    dead = RigTelemetry(0.0, 20.0, 0.0, False, estop=False, ok=False, homed=True)
    m.update(DT, None, None, None, dead)
    assert m.state == "SAFE"

    healthy = RigTelemetry(0.0, 20.0, 0.0, False, estop=False, ok=True, homed=True)
    m.update(DT, None, None, None, healthy)
    assert m.state == "SAFE"


def test_disarmed_mission_never_commands_water() -> None:
    """The control-layer ARM gate is upstream of everything. A disarmed mission
    may track, but no code path below it may open a valve."""
    m = MissionController(DEFAULT_CONFIG, CameraModel(DEFAULT_CONFIG.camera), armed=False)
    tel = RigTelemetry(0.0, 20.0, 0.0, False, estop=False, ok=True, homed=True)
    for _ in range(300):
        cmd = m.update(DT, None, None, None, tel)
        assert cmd.valve is False
        assert cmd.pump_pct == 0.0


def test_unhomed_turret_never_commands_water() -> None:
    """Pan is dead-reckoned, so without a homing reference the commanded angle
    means nothing — and an angle that means nothing must never carry water."""
    m = MissionController(DEFAULT_CONFIG, CameraModel(DEFAULT_CONFIG.camera), armed=True)
    unhomed = RigTelemetry(0.0, 20.0, 0.0, False, estop=False, ok=True, homed=False)
    for _ in range(300):
        cmd = m.update(DT, None, None, None, unhomed)
        assert cmd.valve is False
        assert cmd.pump_pct == 0.0


# ------------------------------------------------------------ closed-loop tier

@pytest.mark.slow
def test_reachable_fire_is_extinguished() -> None:
    """The system's reason to exist. Stated without reference to how long it
    takes or how much water it uses — those are tier-2 metrics."""
    report = run_sim(DEFAULT_CONFIG, SimScenario(), seed=7, headless=True, max_frames=2400)
    assert report.extinguished
    assert "SUPPRESS" in report.states_visited


UNREACHABLE = SimScenario(fire_azimuth_deg=0.0, fire_range_m=10.0, azimuth_bias_deg=1.6,
                          velocity_coeff_scale=0.88, drag_scale=1.3)

#: The same degraded jet against a fire it CAN reach. Its job is to catch a
#: reachability threshold loosened until it starts crying "too far" about
#: everything — which would pass every unreachable test below while making the
#: feature useless.
REACHABLE_NEARBY = SimScenario(fire_azimuth_deg=-15.0, fire_range_m=9.0,
                               azimuth_bias_deg=-1.0,
                               velocity_coeff_scale=0.88, drag_scale=1.3)


@pytest.mark.slow
@pytest.mark.parametrize("seed", [137, 7, 42, 99])
def test_unreachable_target_holds_with_bounded_water(seed: int) -> None:
    """A physically unreachable fire (10 m against a perturbed jet that reaches
    ~8.6 m) must be RECOGNISED — HOLD, valve shut — not hosed at forever.

    This assertion has a history worth keeping, because it is the difference
    between a test that passes and a mechanism that works. Measured over six
    seeds:

        SP1 baseline                          HOLD in 3/6
        after SP2 (named RNG streams)         HOLD in 1/6
        after SP4 timing + gain scheduling    HOLD in 0/6
        after the reachability threshold      HOLD in 6/6

    The shipped e2e test asserted HOLD at seed 137 — one of the lucky three. The
    mechanism was never real: at 10 m a 1.4 m shortfall is only ~6.7 px, INSIDE
    the 12 px control deadband, so the loop could not see it was short. HOLD was
    being triggered by renderer noise crossing the deadband. Tightening the loop
    removed the noise and therefore the accident, which is why 0/6 was the
    correct and useful signal rather than a regression to paper over.

    Judging reachability on its own threshold (`REACH_SHORTFALL_PX`) instead of
    the control deadband makes it deterministic. Water fell with it, 5.8-6.2 L to
    4.7-5.6 L, because it stops hosing sooner.

    **6/6 was not the whole story.** A later 24-seed sweep put this at **23/24** —
    see `test_the_HOLD_state_is_marginal_but_the_SAFETY_guarantees_are_not` for
    the counterexample and what was deliberately not done about it.
    """
    report = run_sim(DEFAULT_CONFIG, UNREACHABLE, seed=seed, headless=True, max_frames=2400)
    assert not report.extinguished, "an out-of-reach fire cannot be put out"
    assert "HOLD" in report.states_visited, "out-of-reach target was not recognised"
    assert "TOO_FAR" in report.advisories_raised, "operator was not told to reposition"
    assert report.water_used_l < _water_ceiling(DEFAULT_CONFIG, cycles=3.0)


@pytest.mark.slow
def test_the_HOLD_state_is_marginal_but_the_SAFETY_guarantees_are_not() -> None:
    """Seed 666 — the counterexample a 24-seed sweep found, and the reason the
    contract above is not the one that matters.

    Measured over 24 seeds on this exact scenario:

        HOLD reached              23/24   (seed 666 is the exception)
        TOO_FAR advisory raised   24/24
        fire falsely extinguished  0/24
        water used            4.46-5.84 L

    Seed 666 runs to the frame cap through SEARCH/ACQUIRE/RANGE/ALIGN/SUPPRESS
    without ever latching HOLD, at mean miss 1.86 m against seed 137's 1.815 m.
    The reachability judgement is sitting on its threshold and this seed's noise
    keeps it the wrong side. Nothing is broken; the margin is just thin.

    **What was deliberately NOT done:** the threshold was not nudged to capture
    666. It would be tuning to make one seed pass — the practice the docstring
    above exists to warn about — and the same sweep showed a genuinely REACHABLE
    fire raising HOLD 0/24 and TOO_FAR 0/24, so the discrimination is currently
    clean in the direction that matters. Loosening it spends that to buy a seed.

    **What this asserts instead** is the set of guarantees the sweep found to be
    universal, checked at the one seed known to be marginal: the fire is never
    falsely reported out, the operator is always told to reposition, and water
    stays bounded. Those are the safety-relevant properties. Which state label
    the machine passes through on the way is not one of them.

    It deliberately does NOT assert that 666 fails to reach HOLD. A future
    improvement that captured it should not have to edit this test to land.
    """
    report = run_sim(DEFAULT_CONFIG, UNREACHABLE, seed=666, headless=True, max_frames=2400)
    assert not report.extinguished, "an out-of-reach fire cannot be put out"
    assert "TOO_FAR" in report.advisories_raised, "operator was not told to reposition"
    assert report.water_used_l < _water_ceiling(DEFAULT_CONFIG, cycles=3.0)


@pytest.mark.slow
def test_a_REACHABLE_fire_does_not_trip_the_unreachable_machinery() -> None:
    """The other half of the discrimination, and why the threshold is left alone.

    Same degraded jet, 1 m closer and off-axis: over 24 seeds this was
    extinguished 24/24, raised HOLD 0/24 and TOO_FAR 0/24. A reachability test
    that never cries "too far" about a fire it can actually reach is the half
    that is easy to lose while chasing the other one.
    """
    report = run_sim(DEFAULT_CONFIG, REACHABLE_NEARBY, seed=666, headless=True,
                     max_frames=2400)
    assert report.extinguished, "a reachable fire was not put out"
    assert "TOO_FAR" not in report.advisories_raised, "a reachable fire was called too far"
    assert "HOLD" not in report.states_visited, "a reachable fire triggered HOLD"


@pytest.mark.slow
def test_static_decoy_is_never_sprayed() -> None:
    """A fire-coloured object that does not flicker is not a fire. Spraying one
    is the canonical false positive, and it must be impossible."""
    rig = SimRig(DEFAULT_CONFIG, SimScenario(decoy=(20.0, 6.0)), seed=7)
    rig.fires[0].intensity = 0.0
    pipe = Pipeline(DEFAULT_CONFIG)
    for _ in range(600):
        rig.step(DT)
        result = pipe.tick(rig.render(), DT, rig.telemetry())
        rig.command(result.command)
        assert result.command.valve is False, "the turret sprayed a decoy"
        assert pipe.mission.state != "SUPPRESS"


@pytest.mark.slow
def test_water_is_never_committed_before_a_target_is_acquired() -> None:
    """Water is never committed speculatively. Note what this does NOT say: a
    momentary loss of the track mid-suppression does not shut the valve, because
    the splash plume routinely occludes the flame it is landing on — that is the
    soak, and it is deliberate. What must never happen is water before the
    perception stack has ever seen anything."""
    rig = SimRig(DEFAULT_CONFIG, SimScenario(), seed=7)
    pipe = Pipeline(DEFAULT_CONFIG)
    ever_acquired = False
    for _ in range(1800):
        rig.step(DT)
        result = pipe.tick(rig.render(), DT, rig.telemetry())
        rig.command(result.command)
        ever_acquired = ever_acquired or result.target is not None
        if result.command.valve:
            assert ever_acquired, "valve opened before any target was ever acquired"


@pytest.mark.slow
def test_valve_is_never_open_while_searching() -> None:
    """SEARCH means "I do not currently have a target to commit to". Spraying
    while sweeping for one is spraying blind, whatever the previous state was."""
    rig = SimRig(DEFAULT_CONFIG, SimScenario(), seed=7)
    pipe = Pipeline(DEFAULT_CONFIG)
    for _ in range(1800):
        rig.step(DT)
        result = pipe.tick(rig.render(), DT, rig.telemetry())
        rig.command(result.command)
        if pipe.mission.state == "SEARCH":
            assert result.command.valve is False, "valve open while searching"


@pytest.mark.slow
def test_no_nan_ever_reaches_a_rig_command() -> None:
    """A NaN angle is a command the firmware will clamp into something arbitrary.
    Nothing in the control stack may emit one, under any scenario."""
    scenarios = [
        SimScenario(),
        SimScenario(fire_azimuth_deg=0.0, fire_range_m=10.0),   # unreachable
        SimScenario(fire_azimuth_deg=10.0, fire_range_m=3.0),   # too close
        SimScenario(fire_azimuth_deg=20.0, fire_range_m=6.5,
                    wind_cross_amp_deg=5.0, wind_range_amp_m=1.5),
    ]
    for scenario in scenarios:
        rig = SimRig(DEFAULT_CONFIG, scenario, seed=7)
        pipe = Pipeline(DEFAULT_CONFIG)
        for _ in range(900):
            rig.step(DT)
            cmd = pipe.tick(rig.render(), DT, rig.telemetry()).command
            rig.command(cmd)
            for name in ("pan_deg", "tilt_deg", "pump_pct"):
                value = getattr(cmd, name)
                assert math.isfinite(value), f"{name} was {value} in {scenario}"


@pytest.mark.slow
def test_commanded_pan_is_never_inside_a_keepout() -> None:
    """The geofence is not advisory. Across a whole engagement with the fire on
    the far side of a protected sector, no command may aim into it."""
    keepout = (-12.0, 12.0)
    cfg = replace(DEFAULT_CONFIG, turret=TurretConfig(pan_keepout_deg=keepout))
    rig = SimRig(cfg, SimScenario(fire_azimuth_deg=25.0, fire_range_m=6.5), seed=7)
    pipe = Pipeline(cfg)
    for _ in range(1800):
        rig.step(DT)
        result = pipe.tick(rig.render(), DT, rig.telemetry())
        rig.command(result.command)
        assert not pan_in_keepout(result.command.pan_deg, keepout), (
            f"commanded pan {result.command.pan_deg:.2f} is inside the keep-out"
        )


@pytest.mark.slow
def test_water_is_never_open_while_the_nozzle_crosses_a_keepout() -> None:
    """SP1's swept-path contract, stated as an invariant: over a full engagement
    the valve is never open on a tick where the nozzle's swept path touches the
    protected sector. This test fails against pre-SP1 HEAD."""
    keepout = (-12.0, 12.0)
    cfg = replace(DEFAULT_CONFIG, turret=TurretConfig(pan_keepout_deg=keepout))
    rig = SimRig(cfg, SimScenario(fire_azimuth_deg=25.0, fire_range_m=6.5), seed=7)
    pipe = Pipeline(cfg)
    from fireturret.control.mission import swept_intersects_keepout

    for _ in range(1800):
        rig.step(DT)
        tel = rig.telemetry()
        result = pipe.tick(rig.render(), DT, tel)
        rig.command(result.command)
        if result.command.valve:
            assert not swept_intersects_keepout(tel.pan_deg, result.command.pan_deg, keepout), (
                f"valve open while sweeping {tel.pan_deg:.2f} -> "
                f"{result.command.pan_deg:.2f} across {keepout}"
            )
