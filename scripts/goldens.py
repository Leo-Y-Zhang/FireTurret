# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""Regenerate golden fixtures — the ONLY sanctioned way to move them.

    python scripts/goldens.py all --reason "SP2: named RNG streams"
    python scripts/goldens.py frames --reason "..."      # tier 3 input
    python scripts/goldens.py commands --reason "..."    # tier 3 expectation
    python scripts/goldens.py reports --reason "..."     # tier 2
    python scripts/goldens.py stdout --reason "..."      # tier 2 (CLI surface)

Three tiers, because the old single tier conflated "the engine behaves
correctly" with "the renderer produced these exact pixels":

  TIER 1  tests/test_invariants.py — behavioural contracts, NEVER regenerated.
  TIER 2  report_*.json / stdout_*.txt — metrics with declared tolerances and
          categorical fields compared exactly.
  TIER 3  frames_default.npz -> commands_default.json — a byte-exact RigCommand
          sequence over FROZEN input frames.

Tier 3 is the important structural change. Byte-exactness moves off the
stochastic renderer and onto fixed input, which is both stronger (it pins the
perception and control stack exactly) and cheaper (it survives the renderer, the
spray model and the platform simulator all being rewritten).

**This script refuses to run while tier 1 is red.** That is the whole point: a
regression must not be launderable into a fixture by regenerating it. There is
no override flag, deliberately. The gate runs the full tier-1 file including its
slow closed-loop contracts, so expect it to take a few minutes.

Every run records what it did — and WHY, from a mandatory --reason — into
tests/golden/MANIFEST.json, alongside a sha256 per fixture and the engine commit
the fixtures were produced from.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "tests" / "golden"
MANIFEST = GOLDEN / "MANIFEST.json"

sys.path.insert(0, str(ROOT / "src"))

from fireturret.__main__ import SCENARIOS  # noqa: E402
from fireturret.app import Pipeline, run_sim  # noqa: E402
from fireturret.config import DEFAULT_CONFIG  # noqa: E402
from fireturret.rig.interface import RigTelemetry  # noqa: E402
from fireturret.rig.sim_rig import SimRig  # noqa: E402

CASES = [("default", 7), ("unreachable", 7), ("multi", 7), ("windy", 3)]

# Tier 3: how many frames of the `default` scenario are frozen as input. Long
# enough to cover acquisition, ranging, alignment and the start of suppression —
# i.e. every state transition that the perception and control stack can take
# before the fire starts visibly going out.
FRAME_COUNT = 120
FRAME_SCENARIO = "default"
FRAME_SEED = 7
DT = 1 / 30

TIERS = {
    "frames_default.npz": 3,
    "commands_default.json": 3,
    **{f"report_{n}_seed{s}.json": 2 for n, s in CASES},
    **{f"stdout_{n}.txt": 2 for n in SCENARIOS},
}


# ------------------------------------------------------------------ tier 1 gate

def require_tier1_green() -> None:
    """Refuse to regenerate anything while the behavioural contracts are red."""
    print("gate: running tier-1 invariants (includes slow contracts)...", flush=True)
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", str(ROOT / "tests" / "test_invariants.py"),
         "-q", "-m", "slow or not slow", "-p", "no:cacheprovider"],
        cwd=ROOT,
    )
    if proc.returncode != 0:
        raise SystemExit(
            "\nREFUSING to regenerate goldens: tier-1 invariants are RED.\n"
            "Tier 1 is the set of contracts that may never be regenerated, so a\n"
            "failure there is a regression in behaviour — not a stale fixture.\n"
            "Fix the behaviour. There is deliberately no override for this."
        )
    print("gate: tier-1 green.\n", flush=True)


# ------------------------------------------------------------------- generators

def report_to_json(rep) -> dict:
    d = dataclasses.asdict(rep)
    d["states_visited"] = sorted(d["states_visited"])
    d["advisories_raised"] = sorted(d["advisories_raised"])
    d["telemetry_len"] = len(d.pop("telemetry"))  # store length, not the rows
    return d


def gen_reports() -> list[str]:
    written = []
    for name, seed in CASES:
        rep = run_sim(DEFAULT_CONFIG, scenario=SCENARIOS[name], seed=seed,
                      headless=True, record_telemetry=True)
        path = GOLDEN / f"report_{name}_seed{seed}.json"
        path.write_text(json.dumps(report_to_json(rep), indent=2, sort_keys=True))
        written.append(path.name)
        print(f"  wrote {path.name}")
    return written


def gen_stdout() -> list[str]:
    written = []
    for name in sorted(SCENARIOS):
        # `fireturret sim` exits 1 when the fire is not extinguished (e.g. the
        # `unreachable` scenario) — correct behaviour, so do NOT check=True.
        out = subprocess.run(
            [sys.executable, "-m", "fireturret", "sim", "--scenario", name, "--headless"],
            capture_output=True, text=True, cwd=ROOT,
        ).stdout
        path = GOLDEN / f"stdout_{name}.txt"
        path.write_text(out)
        written.append(path.name)
        print(f"  wrote {path.name}")
    return written


def gen_frames() -> list[str]:
    """Freeze the perception stack's INPUT: rendered frames plus the telemetry
    that accompanied each one. Everything downstream of this is then testable
    without running the simulator at all."""
    rig = SimRig(DEFAULT_CONFIG, SCENARIOS[FRAME_SCENARIO], seed=FRAME_SEED)
    pipeline = Pipeline(DEFAULT_CONFIG)
    frames, pan, tilt, pump, valve, estop, ok, homed = [], [], [], [], [], [], [], []

    for _ in range(FRAME_COUNT):
        rig.step(DT)
        frame = rig.render()
        tel = rig.telemetry()
        frames.append(frame)
        pan.append(tel.pan_deg)
        tilt.append(tel.tilt_deg)
        pump.append(tel.pump_pct)
        valve.append(tel.valve)
        estop.append(tel.estop)
        ok.append(tel.ok)
        homed.append(tel.homed)
        # keep the loop closed so the frozen frames follow a REAL engagement
        rig.command(pipeline.tick(frame, DT, tel).command)

    path = GOLDEN / "frames_default.npz"
    np.savez_compressed(
        path,
        frames=np.asarray(frames, dtype=np.uint8),
        pan=np.asarray(pan), tilt=np.asarray(tilt), pump=np.asarray(pump),
        valve=np.asarray(valve), estop=np.asarray(estop),
        ok=np.asarray(ok), homed=np.asarray(homed),
    )
    mb = path.stat().st_size / 1e6
    raw = np.asarray(frames, dtype=np.uint8).nbytes / 1e6
    print(f"  wrote {path.name} — {mb:.2f} MB ({raw:.1f} MB raw, {raw / mb:.0f}x)")
    return [path.name]


def load_frames() -> tuple[np.ndarray, list[RigTelemetry]]:
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


def commands_from_frozen_frames() -> list[dict]:
    """Replay the frozen frames through a fresh pipeline. Pure function of the
    fixture: no simulator, no renderer, no RNG."""
    frames, telemetry = load_frames()
    pipeline = Pipeline(DEFAULT_CONFIG)
    out = []
    for frame, tel in zip(frames, telemetry, strict=True):
        cmd = pipeline.tick(frame, DT, tel).command
        out.append({
            "pan_deg": cmd.pan_deg, "tilt_deg": cmd.tilt_deg,
            "pump_pct": cmd.pump_pct, "valve": cmd.valve,
            "laser": cmd.laser, "warn": cmd.warn,
        })
    return out


def gen_commands() -> list[str]:
    path = GOLDEN / "commands_default.json"
    path.write_text(json.dumps(commands_from_frozen_frames(), indent=2))
    print(f"  wrote {path.name}")
    return [path.name]


# --------------------------------------------------------------------- manifest

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def engine_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def update_manifest(written: list[str], reason: str) -> None:
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {"fixtures": {}}
    stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    sha = engine_sha()
    for name in written:
        path = GOLDEN / name
        manifest["fixtures"][name] = {
            "tier": TIERS.get(name, 2),
            "sha256": sha256(path),
            "engine_sha": sha,
            "regenerated": stamp,
            "reason": reason,
        }
    manifest["note"] = (
        "Tier 1 (tests/test_invariants.py) is NEVER regenerated and has no entry "
        "here. Tier 2 fixtures are compared with the tolerances declared in "
        "tests/test_golden_behavior.py; their categorical fields are compared "
        "exactly. Tier 3 pins a byte-exact command sequence over frozen frames. "
        "Every entry below was written by scripts/goldens.py, which refuses to "
        "run while tier 1 is red."
    )
    manifest["canonical_platform"] = (
        "Fixtures are produced on the maintainer's development machine. Tier-3 "
        "floats are the product of OpenCV kernels and long float accumulation and "
        "will NOT reproduce bit-for-bit on a differently-compiled OpenCV or on "
        "arm64. That is why tier 2 carries tolerances and only tier 3 is exact."
    )
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"\nmanifest: recorded {len(written)} fixture(s) — {reason}")


# ------------------------------------------------------------------------- main

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("what", choices=["all", "frames", "commands", "reports", "stdout"])
    parser.add_argument("--reason", required=True,
                        help="why these fixtures are moving; recorded in MANIFEST.json")
    args = parser.parse_args()

    GOLDEN.mkdir(parents=True, exist_ok=True)
    require_tier1_green()

    written: list[str] = []
    if args.what in ("all", "frames"):
        written += gen_frames()
    if args.what in ("all", "reports"):
        written += gen_reports()
    if args.what in ("all", "stdout"):
        written += gen_stdout()
    # commands LAST: they are derived from the frames fixture, so regenerating
    # both in one run must read the frames this run just wrote.
    if args.what in ("all", "commands"):
        written += gen_commands()

    update_manifest(written, args.reason)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
