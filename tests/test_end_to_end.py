# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""End-to-end and robustness tests: the complete pipeline — real detector on
rendered frames, tracker, mission, servo, splash feedback — driving simulated
fires whose TRUE physics (discharge coefficient, drag, boresight bias) differ
from the controller's model. Passing proves the closed loop corrects model error.
"""

import pytest

from fireturret.app import Pipeline, run_sim
from fireturret.config import DEFAULT_CONFIG
from fireturret.rig.sim_rig import SimRig, SimScenario

# Full-length closed-loop runs (~145 s total). Deselected by default
# (addopts = -m 'not slow'); run explicitly with `pytest -m slow`.
pytestmark = pytest.mark.slow

# Reachable scenarios (target well inside the perturbed jet's envelope): the
# system MUST extinguish each. Varied range, azimuth, boresight bias, and
# physics perturbation (weaker/stronger jet, more/less drag).
REACHABLE = [
    SimScenario(),  # default: 7 m, +24°, +1.6° bias, weak jet
    SimScenario(fire_azimuth_deg=-35.0, fire_range_m=6.0, azimuth_bias_deg=-1.0,
                velocity_coeff_scale=1.05, drag_scale=0.85),
    SimScenario(fire_azimuth_deg=40.0, fire_range_m=5.5, azimuth_bias_deg=1.6,
                velocity_coeff_scale=0.88, drag_scale=1.3),
    SimScenario(fire_azimuth_deg=-28.0, fire_range_m=7.5, azimuth_bias_deg=-1.5,
                velocity_coeff_scale=0.92, drag_scale=1.2),
]


@pytest.mark.parametrize("scenario", REACHABLE, ids=lambda s: f"r{s.fire_range_m}_az{s.fire_azimuth_deg}")
def test_reachable_fire_is_extinguished(scenario: SimScenario) -> None:
    seed = 1 + int(abs(scenario.fire_azimuth_deg)) + int(scenario.fire_range_m * 10)
    report = run_sim(DEFAULT_CONFIG, scenario, seed=seed, headless=True, max_frames=2400)
    assert report.extinguished, (
        f"not extinguished in {report.sim_seconds:.0f}s; "
        f"states={sorted(report.states_visited)} final={report.final_intensity:.2f}"
    )
    assert "SUPPRESS" in report.states_visited
    assert report.water_used_l > 0


def test_unreachable_target_holds_and_bounds_water() -> None:
    # 10 m target vs a weak jet that reaches only ~8.6 m: physically impossible.
    # The controller must recognise it (HOLD), not hose water indefinitely.
    #
    # This assertion used to pass at seed 137 by luck — SP2 measured HOLD at 3/6
    # seeds, and SP4 traced that to renderer noise crossing the control deadband
    # rather than to any detection. It is now deterministic (6/6); see
    # tests/test_invariants.py::test_unreachable_target_holds_with_bounded_water
    # for the full measurement history.
    scenario = SimScenario(fire_azimuth_deg=0.0, fire_range_m=10.0, azimuth_bias_deg=1.6,
                           velocity_coeff_scale=0.88, drag_scale=1.3)
    report = run_sim(DEFAULT_CONFIG, scenario, seed=137, headless=True, max_frames=2400)
    assert not report.extinguished  # cannot be reached
    assert "HOLD" in report.states_visited  # recognised as out of reach
    assert report.water_used_l < 9.0  # backoff keeps water bounded, no endless hosing
    assert "TOO_FAR" in report.advisories_raised  # operator told to move the turret closer


def test_no_nans_and_search_happens() -> None:
    report = run_sim(DEFAULT_CONFIG, SimScenario(), seed=7, headless=True, max_frames=2400)
    assert report.extinguished
    assert "SEARCH" in report.states_visited


def test_static_decoy_is_never_sprayed() -> None:
    # A fire-coloured but non-flickering object (a red cloth, a warning light)
    # and NO real fire: the flicker gate must reject it, so the turret searches
    # but never opens the valve.
    rig = SimRig(DEFAULT_CONFIG, SimScenario(decoy=(20.0, 6.0)), seed=7)
    rig.fires[0].intensity = 0.0  # remove the real fire; only the decoy remains
    pipe = Pipeline(DEFAULT_CONFIG)
    dt = 1 / 30
    sprayed = False
    for _ in range(600):  # 20 s
        rig.step(dt)
        result = pipe.tick(rig.render(), dt, rig.telemetry())
        rig.command(result.command)
        sprayed = sprayed or result.command.valve
        assert pipe.mission.state != "SUPPRESS"
    assert not sprayed


def test_static_decoy_does_not_stop_the_sweep() -> None:
    """The liveness half of the decoy guarantee, which was missing.

    "Never sprayed" is a negative assertion, and a paralysed turret satisfies it
    perfectly. It used to: a persistent candidate that never confirms made
    _do_search stop the sweep, ACQUIRE time out, and the next frame stop it
    again - zero net pan movement, forever, while the operator sees a machine
    sitting in ACQUIRE looking busy. Measured on the state machine directly, 900
    frames gave {'ACQUIRE': 900} and a final pan of 0.000 deg, so a real fire
    anywhere outside the current 66 degree field of view was never found.

    A suppression system that silently stops suppressing is the worst failure
    this project has, so the decoy scenario must assert the machine is still
    doing its job, not merely that it is not doing the wrong one.
    """
    rig = SimRig(DEFAULT_CONFIG, SimScenario(decoy=(20.0, 6.0)), seed=7)
    rig.fires[0].intensity = 0.0  # only the decoy remains
    pipe = Pipeline(DEFAULT_CONFIG)
    dt = 1 / 30

    commanded: list[float] = []
    for _ in range(600):  # 20 s
        rig.step(dt)
        result = pipe.tick(rig.render(), dt, rig.telemetry())
        rig.command(result.command)
        commanded.append(result.command.pan_deg)

    traversed = max(commanded) - min(commanded)
    assert traversed > 20.0, (
        "the sweep must keep moving past a decoy it cannot confirm; commanded "
        f"pan only traversed {traversed:.1f} deg in 20 s"
    )


def test_real_fire_engaged_despite_decoy() -> None:
    # a decoy on one side and a real fire on the other: engage the fire
    scenario = SimScenario(fire_azimuth_deg=25.0, fire_range_m=6.5, decoy=(-30.0, 6.0))
    report = run_sim(DEFAULT_CONFIG, scenario, seed=7, headless=True, max_frames=3000)
    assert report.extinguished


def test_spreading_fire_is_tracked() -> None:
    # a fire that drifts laterally (spreads) while it burns: the suppress servo
    # fully tracks the fire's azimuth each cycle and follows it.
    scenario = SimScenario(fire_azimuth_deg=5.0, fire_range_m=5.0,
                           fire_drift_az_dps=1.5, fire_drift_range_mps=0.05)
    report = run_sim(DEFAULT_CONFIG, scenario, seed=7, headless=True, max_frames=3000)
    assert report.extinguished, f"moving fire not tracked; final={report.final_intensity:.2f}"


def test_receding_fire_extends_reach_via_tilt() -> None:
    # a fire receding in range within the jet's envelope: when the pump
    # saturates, the servo raises tilt toward the range peak to keep reaching it.
    scenario = SimScenario(fire_azimuth_deg=10.0, fire_range_m=5.0,
                           fire_drift_range_mps=0.1, fire_drift_az_dps=0.5)
    report = run_sim(DEFAULT_CONFIG, scenario, seed=7, headless=True, max_frames=3000)
    assert report.extinguished, f"receding fire not reached; final={report.final_intensity:.2f}"


def test_too_close_fire_holds_not_hoses() -> None:
    # A fire nearer than the low-arc minimum reach: the jet overshoots and the
    # splash lands occluded behind the flame, so the servo gets no feedback.
    # It must recognise this and HOLD, not spray blind indefinitely.
    rig = SimRig(DEFAULT_CONFIG, SimScenario(fire_azimuth_deg=10.0, fire_range_m=3.0), seed=7)
    pipe = Pipeline(DEFAULT_CONFIG)
    dt = 1 / 30
    states = set()
    for _ in range(1400):
        rig.step(dt)
        result = pipe.tick(rig.render(), dt, rig.telemetry())
        rig.command(result.command)
        states.add(pipe.mission.state)
    assert "HOLD" in states  # recognised, not hosed forever
    assert rig.water_used_l < 3.0  # bounded, no endless blind spraying


def test_gusting_wind_is_rejected() -> None:
    # A slowly gusting crosswind + head/tailwind that the ballistic model knows
    # nothing about. The closed loop tracks the splash and compensates — the
    # whole point of visual servoing over open-loop aiming.
    scenario = SimScenario(fire_azimuth_deg=20.0, fire_range_m=6.5,
                           wind_cross_amp_deg=5.0, wind_range_amp_m=1.5, wind_period_s=8.0)
    report = run_sim(DEFAULT_CONFIG, scenario, seed=7, headless=True, max_frames=3000)
    assert report.extinguished, f"wind not rejected; final={report.final_intensity:.2f}"


def test_reachable_out_unreachable_warns() -> None:
    # a realistic mix: one fire the turret can reach and one it can't. It must
    # extinguish the reachable one and raise a warning for the unreachable one
    # (so the operator repositions the turret) rather than getting stuck.
    scenario = SimScenario(fire_azimuth_deg=20.0, fire_range_m=6.0,
                           extra_fires=((-25.0, 10.0),),
                           velocity_coeff_scale=0.88, drag_scale=1.3)
    rig = SimRig(DEFAULT_CONFIG, scenario, seed=7)
    pipe = Pipeline(DEFAULT_CONFIG)
    dt = 1 / 30
    advisories: set[str] = set()
    for _ in range(3600):
        rig.step(dt)
        result = pipe.tick(rig.render(), dt, rig.telemetry())
        rig.command(result.command)
        advisories.update(a.kind for a in pipe.mission.advisories)
    assert rig.fires[0].intensity == 0.0  # reachable fire extinguished
    assert rig.fires[1].intensity > 0.0  # unreachable fire remains
    assert "TOO_FAR" in advisories  # operator warned to reposition


def test_two_fires_extinguished_sequentially() -> None:
    # a second fire at a different azimuth: after CONFIRMing the first out, the
    # mission must re-SEARCH, find the second, and extinguish it too
    scenario = SimScenario(fire_azimuth_deg=25.0, fire_range_m=7.0,
                           extra_fires=((-30.0, 6.0),))
    report = run_sim(DEFAULT_CONFIG, scenario, seed=7, headless=True, max_frames=3600)
    assert report.extinguished  # ALL fires out
    assert "CONFIRM" in report.states_visited  # first fire confirmed before the next
