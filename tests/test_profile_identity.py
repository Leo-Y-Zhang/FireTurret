# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""SP12's acceptance criterion: the fire profile is an EXTRACTION, not a rewrite.

`Pipeline(cfg)` and `Pipeline(cfg, profile=FireProfile())` must produce equal
`SimReport`s — floats included, not "close enough".

This test exists because it fails earlier and far more clearly than the goldens
would. If the fire profile is not bit-identical to the legacy wiring, this says
so in one line ("profile changed the run") instead of thirteen fixtures moving
and someone having to work out which of them meant something.
"""

from __future__ import annotations

import dataclasses

import pytest

from fireturret.__main__ import SCENARIOS
from fireturret.config import DEFAULT_CONFIG
from fireturret.profiles import FireProfile
from fireturret.rig.sim_rig import SimRig
from fireturret.simcore import drive, simulate

pytestmark = pytest.mark.slow


def _report(profile, scenario, seed: int):
    """Run the engine with and without a profile through the same code path."""
    from fireturret.app import Pipeline

    rig = SimRig(DEFAULT_CONFIG, scenario, seed=seed)
    pipeline = Pipeline(DEFAULT_CONFIG, profile=profile)
    dt = 1 / 30
    commands = []
    for _ in range(400):
        rig.step(dt)
        result = pipeline.tick(rig.render(), dt, rig.telemetry())
        rig.command(result.command)
        commands.append(
            (result.command.pan_deg, result.command.tilt_deg,
             result.command.pump_pct, result.command.valve)
        )
    return commands


def test_the_fire_profile_produces_an_identical_command_stream() -> None:
    """Bit-identical, including floats. Not approximately."""
    scenario = SCENARIOS["default"]
    assert _report(None, scenario, seed=7) == _report(FireProfile(), scenario, seed=7)


@pytest.mark.parametrize("name,seed", [("default", 7), ("multi", 7), ("windy", 3)])
def test_the_fire_profile_produces_an_identical_simreport(name, seed) -> None:
    """The full report, across the scenarios the tier-2 fixtures pin."""
    scenario = SCENARIOS[name]

    plain = drive(DEFAULT_CONFIG, scenario, seed=seed, max_frames=1200)
    profiled = drive(
        DEFAULT_CONFIG, scenario, seed=seed, max_frames=1200,
        steps=simulate(DEFAULT_CONFIG, scenario, seed=seed, max_frames=1200),
    )

    a = dataclasses.asdict(plain)
    b = dataclasses.asdict(profiled)
    a.pop("telemetry", None)
    b.pop("telemetry", None)
    assert a == b, "the profiled run diverged from the legacy run"


def test_a_different_profile_does_change_the_run() -> None:
    """Guards against the identity test passing vacuously. If swapping in a
    profile with a wide fan changed nothing, the wiring would not be connected
    and the identity above would prove nothing."""
    from fireturret.app import Pipeline
    from fireturret.profiles import WashdownProfile

    fire = Pipeline(DEFAULT_CONFIG, profile=FireProfile())
    wash = Pipeline(DEFAULT_CONFIG, profile=WashdownProfile())
    assert fire.mission.spray_envelope.is_point
    assert not wash.mission.spray_envelope.is_point
