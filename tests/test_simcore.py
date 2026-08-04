# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""Unit tests for the Qt-free simulation core (simulate + drive)."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from triton.__main__ import SCENARIOS
from triton.analysis import TelemetrySample
from triton.app import run_sim
from triton.config import DEFAULT_CONFIG
from triton.simcore import SimStep, drive, simulate

_DT = 1.0 / 30.0


def test_simulate_yields_simsteps_with_wellformed_samples():
    steps = list(simulate(DEFAULT_CONFIG, seed=7, max_frames=40))
    assert len(steps) == 40
    s = steps[10]
    assert isinstance(s, SimStep)
    assert isinstance(s.sample, TelemetrySample)
    assert isinstance(s.frame, np.ndarray)
    assert s.frame.shape == (DEFAULT_CONFIG.camera.height, DEFAULT_CONFIG.camera.width, 3)
    assert s.frame_index == 10
    assert s.sim_seconds == sum([_DT] * 11)  # the running += dt fold (order matters)
    assert isinstance(s.advisories, tuple)
    assert isinstance(s.miss_recorded, bool)
    assert s.sample.state in (
        "SEARCH", "ACQUIRE", "RANGE", "ALIGN", "SUPPRESS", "CONFIRM", "HOLD", "SAFE"
    )


def _norm(rep):
    d = dataclasses.asdict(rep)
    d["states_visited"] = sorted(d["states_visited"])
    d["advisories_raised"] = sorted(d["advisories_raised"])
    d["telemetry_len"] = len(d.pop("telemetry"))
    return d


@pytest.mark.parametrize("name,seed", [("default", 7), ("multi", 7)])
def test_drive_matches_run_sim_short(name, seed):
    a = run_sim(
        DEFAULT_CONFIG, scenario=SCENARIOS[name], seed=seed,
        headless=True, max_frames=300, record_telemetry=True,
    )
    b = drive(
        DEFAULT_CONFIG, scenario=SCENARIOS[name], seed=seed,
        max_frames=300, record_telemetry=True,
    )
    assert _norm(b) == _norm(a)


def test_drive_multi_advisory_union_survives():
    b = drive(
        DEFAULT_CONFIG, scenario=SCENARIOS["unreachable"], seed=7,
        max_frames=400, record_telemetry=True,
    )
    assert b.advisories_raised  # non-empty: unreachable raises advisories


def test_per_fire_outcomes_for_multi_fire():
    b = drive(DEFAULT_CONFIG, scenario=SCENARIOS["multi"], seed=7, max_frames=3000)
    assert len(b.per_fire) == 2  # multi has a primary + one extra fire
    # both are extinguished in the reachable multi scenario, each with a time
    for outcome in b.per_fire:
        assert outcome.extinguished is True
        assert outcome.extinguish_time_s is not None
        assert outcome.residual_intensity <= 0.03


def test_per_fire_single_fire_default():
    b = drive(DEFAULT_CONFIG, seed=7, max_frames=2000)
    assert len(b.per_fire) == 1


def test_drive_should_stop_cancels_early():
    n = {"i": 0}

    def stop():
        n["i"] += 1
        return n["i"] > 20

    b = drive(DEFAULT_CONFIG, seed=7, max_frames=6000, should_stop=stop)
    assert b.frames <= 21
