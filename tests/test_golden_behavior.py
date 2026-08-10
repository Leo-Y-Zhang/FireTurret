# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""TIER 2 and TIER 3 golden guards. (Tier 1 lives in `test_invariants.py`.)

The old version of this file did `dataclasses.asdict(rep)` and compared the whole
thing exactly, which had two bad consequences:

* **Any new `SimReport` field broke all four report fixtures**, whatever its
  default — so "additive and therefore inert" was unachievable for any change
  that reported a new metric.
* It conflated "the engine behaves correctly" with "the renderer produced these
  exact pixels", so a cosmetic renderer change looked identical to a control
  regression.

So the guard is split by what each thing is actually for:

**TIER 2 — metrics, with declared tolerances.** How much water, how many frames,
how far off the splash landed: real numbers whose *magnitude* is the contract,
not their last decimal. Categorical fields — did it extinguish, which states were
visited, which advisories fired — are compared EXACTLY, because those are
behaviour rather than measurement.

**TIER 3 — byte-exact commands over frozen frames.** Byte-exactness moves off the
stochastic renderer and onto fixed input: 120 recorded frames plus the telemetry
that accompanied them, replayed through a fresh `Pipeline`. This pins the
perception and control stack precisely while surviving a rewrite of the
renderer, the spray model or the platform simulator — none of which it touches.

Regenerate ONLY via `scripts/goldens.py`, which refuses to run while tier 1 is
red and records a mandatory written reason in `tests/golden/MANIFEST.json`.
"""

from __future__ import annotations

import dataclasses
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from fireturret.__main__ import SCENARIOS
from fireturret.app import Pipeline, run_sim
from fireturret.config import DEFAULT_CONFIG

GOLDEN = Path(__file__).parent / "golden"
CASES = [("default", 7), ("unreachable", 7), ("multi", 7), ("windy", 3)]
DT = 1 / 30

# Compared EXACTLY: these are behaviour, not measurement. If the set of states a
# mission visits changes, something real changed.
EXACT_FIELDS = ("extinguished", "states_visited", "advisories_raised")

# Compared with tolerance. ("rel", x) = within x fraction; ("abs", x) = within x.
# Rationale for the loose ones: acquisition time depends on where the search
# sweep happens to be when the fire enters frame, so it is inherently coarse;
# frame count inherits that same offset.
#
# `peak_miss_m` is deliberately looser than `mean_miss_m`, and the asymmetry is
# measured rather than guessed. Running the `unreachable` scenario across eight
# seeds with the code held completely fixed:
#
#     peak_miss_m   2.538 - 3.228   spread 0.690 m
#     mean_miss_m   1.676 - 1.911   spread 0.235 m
#
# A mean over a 200 s engagement is a stable statistic; a maximum is an extremum,
# and an extremum of a noisy closed-loop chase moves by ~0.7 m between equivalent
# realisations. Both fields previously shared an 0.30 m bound, which put peak's
# bound BELOW its own noise floor: it could only pass while the engine was
# byte-identical, and any correct change would trip it. That is a tolerance that
# reports "different" while claiming to report "wrong".
#
# 0.75 m is the measured spread rounded up. `mean_miss_m` keeps 0.30 m because
# measurement says it deserves it — the point is to specify each bound from the
# statistic's real variability, not to loosen until things pass.
#
# The categorical safety fields (extinguished, states_visited, advisories_raised)
# are compared EXACTLY in EXACT_FIELDS and are untouched by any of this.
TOLERANCES: dict[str, tuple[str, float]] = {
    "water_used_l": ("rel", 0.15),
    "frames": ("rel", 0.20),
    "telemetry_len": ("rel", 0.20),
    "sim_seconds": ("rel", 0.20),
    "mean_miss_m": ("abs", 0.30),
    "peak_miss_m": ("abs", 0.75),
    "time_to_first_suppress_s": ("rel", 0.50),
    "final_intensity": ("abs", 0.05),
}


def _close(name: str, got, want) -> None:
    if want is None or got is None:
        assert got == want, f"{name}: {got!r} vs golden {want!r}"
        return
    kind, tol = TOLERANCES[name]
    if kind == "rel":
        limit = abs(want) * tol
        assert abs(got - want) <= limit + 1e-9, (
            f"{name}: {got:.4f} vs golden {want:.4f} (tolerance +/-{tol:.0%} = {limit:.4f})"
        )
    else:
        assert abs(got - want) <= tol + 1e-9, (
            f"{name}: {got:.4f} vs golden {want:.4f} (tolerance +/-{tol})"
        )


def _report_json(rep) -> dict:
    d = dataclasses.asdict(rep)
    d["states_visited"] = sorted(d["states_visited"])
    d["advisories_raised"] = sorted(d["advisories_raised"])
    d["telemetry_len"] = len(d.pop("telemetry"))
    return d


# --------------------------------------------------------------------- tier 2

@pytest.mark.slow
@pytest.mark.parametrize("name,seed", CASES)
def test_simreport_matches_golden_within_tolerance(name, seed):
    rep = run_sim(DEFAULT_CONFIG, scenario=SCENARIOS[name], seed=seed,
                  headless=True, record_telemetry=True)
    got = _report_json(rep)
    want = json.loads((GOLDEN / f"report_{name}_seed{seed}.json").read_text())

    for field in EXACT_FIELDS:
        assert got[field] == want[field], f"{field}: {got[field]!r} vs golden {want[field]!r}"
    for field in TOLERANCES:
        _close(field, got[field], want[field])

    # per-fire outcomes: whether each fire went out is categorical; when it went
    # out is a measurement.
    assert len(got["per_fire"]) == len(want["per_fire"])
    for i, (g, w) in enumerate(zip(got["per_fire"], want["per_fire"], strict=True)):
        assert g["extinguished"] == w["extinguished"], f"per_fire[{i}].extinguished"
        _close("final_intensity", g["residual_intensity"], w["residual_intensity"])
        if w["extinguish_time_s"] is not None:
            _close("sim_seconds", g["extinguish_time_s"], w["extinguish_time_s"])


def test_new_simreport_fields_do_not_break_tier2():
    """The defect this tier exists to fix: under a wholesale `asdict` compare, a
    new field with a default broke every fixture. Adding one must be inert."""
    want = json.loads((GOLDEN / "report_default_seed7.json").read_text())
    got = dict(want, brand_new_metric_from_a_future_sub_project=0.0)
    for field in EXACT_FIELDS:
        assert got[field] == want[field]
    for field in TOLERANCES:
        _close(field, got[field], want[field])


_TOKEN = re.compile(r"(\w+)=(\[[^\]]*\]|\S+)")


def _parse_stdout(text: str) -> dict[str, str]:
    return dict(_TOKEN.findall(text))


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_cli_stdout_matches_golden(name):
    """The CLI's printed surface. Keys and categorical values must match exactly
    (a missing field or a flipped `extinguished` is a real change); the numbers
    it prints are the same measurements tier 2 already tolerances."""
    out = subprocess.run(
        [sys.executable, "-m", "fireturret", "sim", "--scenario", name, "--headless"],
        capture_output=True, text=True,
    ).stdout
    got = _parse_stdout(out)
    want = _parse_stdout((GOLDEN / f"stdout_{name}.txt").read_text())

    assert set(got) == set(want), f"stdout fields changed: {set(got) ^ set(want)}"
    for key, want_value in want.items():
        got_value = got[key]
        number = re.fullmatch(r"(-?\d+(?:\.\d+)?)([a-z%]*)", want_value)
        if number and re.fullmatch(r"-?\d+(?:\.\d+)?[a-z%]*", got_value):
            want_num = float(number.group(1))
            got_num = float(re.match(r"-?\d+(?:\.\d+)?", got_value).group(0))
            # printed to 1-2 dp, so allow the wider of 20% or one printed unit
            limit = max(abs(want_num) * 0.20, 0.05)
            assert abs(got_num - want_num) <= limit, (
                f"stdout {key}: {got_value} vs golden {want_value}"
            )
        else:
            assert got_value == want_value, f"stdout {key}: {got_value} vs {want_value}"


# --------------------------------------------------------------------- tier 3

def _load_frozen():
    import numpy as np

    from fireturret.rig.interface import RigTelemetry

    with np.load(GOLDEN / "frames_default.npz") as data:
        frames = data["frames"]
        telemetry = [
            RigTelemetry(
                pan_deg=float(data["pan"][i]), tilt_deg=float(data["tilt"][i]),
                pump_pct=float(data["pump"][i]), valve=bool(data["valve"][i]),
                estop=bool(data["estop"][i]), ok=bool(data["ok"][i]),
                homed=bool(data["homed"][i]),
            )
            for i in range(len(frames))
        ]
    return frames, telemetry


@pytest.mark.slow
def test_commands_over_frozen_frames_are_byte_exact():
    """The strongest guard in the suite, and the cheapest to keep honest: given
    exactly these frames and exactly this telemetry, the perception and control
    stack must emit exactly these commands. No simulator, no renderer, no RNG."""
    frames, telemetry = _load_frozen()
    want = json.loads((GOLDEN / "commands_default.json").read_text())
    assert len(want) == len(frames)

    pipeline = Pipeline(DEFAULT_CONFIG)
    for i, (frame, tel, expected) in enumerate(zip(frames, telemetry, want, strict=True)):
        cmd = pipeline.tick(frame, DT, tel).command
        got = {"pan_deg": cmd.pan_deg, "tilt_deg": cmd.tilt_deg,
               "pump_pct": cmd.pump_pct, "valve": cmd.valve,
               "laser": cmd.laser, "warn": cmd.warn}
        assert got == expected, f"command diverged at frozen frame {i}"


def test_frozen_frames_fixture_is_self_consistent():
    """Cheap structural check so a truncated or half-written fixture fails loudly
    here rather than as a confusing divergence in the replay test."""
    frames, telemetry = _load_frozen()
    want = json.loads((GOLDEN / "commands_default.json").read_text())
    assert len(frames) == len(telemetry) == len(want)
    assert frames.dtype.name == "uint8"
    assert frames.shape[1:] == (DEFAULT_CONFIG.camera.height, DEFAULT_CONFIG.camera.width, 3)


# -------------------------------------------------------------------- manifest

def test_there_is_exactly_one_golden_generator_and_it_has_the_gate():
    """`scripts/make_goldens.py` used to regenerate fixtures with no tier-1
    check. A second generator without the gate makes the gate optional, which
    makes it worthless — so the gate is pinned here, not just implemented."""
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    generators = [p.name for p in scripts.glob("*golden*.py")]
    assert generators == ["goldens.py"], (
        f"more than one golden generator: {generators}. Every path that writes a "
        "fixture must go through the tier-1 gate."
    )
    source = (scripts / "goldens.py").read_text(encoding="utf-8")
    assert "require_tier1_green()" in source
    assert "--reason" in source, "a fixture with no written reason is unreviewed"


def test_every_fixture_is_recorded_in_the_manifest():
    """A fixture nobody wrote a reason for is a fixture nobody reviewed."""
    manifest = json.loads((GOLDEN / "MANIFEST.json").read_text())
    recorded = set(manifest["fixtures"])
    on_disk = {p.name for p in GOLDEN.iterdir() if p.name != "MANIFEST.json"}
    assert on_disk == recorded, f"unrecorded/stale fixtures: {on_disk ^ recorded}"
    for name, entry in manifest["fixtures"].items():
        assert entry["reason"].strip(), f"{name} has no written reason"
        assert entry["tier"] in (2, 3), f"{name}: tier 1 is never a fixture"
        assert len(entry["sha256"]) == 64
