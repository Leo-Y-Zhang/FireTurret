# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""Session and run persistence.

Uses explicit ``config_from_dict``/``scenario_from_dict`` rather than bare
``asdict`` + constructor: the top-level constructor would leave nested sub-configs
as plain dicts and skip their ``__post_init__`` validation. Tuple fields
(``extra_fires``, ``decoy``) are coerced back to tuples so ``==`` round-trips
through JSON (which turns tuples into lists).

- A **session** is ``*.fireturret.json``: config + scenario.
- A **run** is the existing ``analysis.write_csv`` telemetry file plus a small
  JSON sidecar (config snapshot, scenario, seed, metrics).
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from ..config import FireTurretConfig, config_from_dict, config_to_dict
from ..rig.sim_rig import SimScenario

__all__ = [
    "config_to_dict", "config_from_dict", "scenario_to_dict", "scenario_from_dict",
    "save_session", "load_session", "save_run", "load_run_meta", "report_metrics",
]


def scenario_to_dict(sc: SimScenario) -> dict:
    return asdict(sc)


def scenario_from_dict(d: dict) -> SimScenario:
    d = dict(d)
    if d.get("extra_fires") is not None:
        d["extra_fires"] = tuple(tuple(x) for x in d["extra_fires"])
    if d.get("decoy") is not None:
        d["decoy"] = tuple(d["decoy"])
    return SimScenario(**d)


def save_session(path, config: FireTurretConfig, scenario: SimScenario) -> None:
    data = {
        "version": 1,
        "config": config_to_dict(config),
        "scenario": scenario_to_dict(scenario),
    }
    Path(path).write_text(json.dumps(data, indent=2))


def load_session(path) -> tuple[FireTurretConfig, SimScenario]:
    data = json.loads(Path(path).read_text())
    return config_from_dict(data["config"]), scenario_from_dict(data["scenario"])


def report_metrics(report) -> dict:
    return {
        "extinguished": report.extinguished,
        "frames": report.frames,
        "sim_seconds": report.sim_seconds,
        "water_used_l": report.water_used_l,
        "final_intensity": report.final_intensity,
        "time_to_first_suppress_s": report.time_to_first_suppress_s,
        "mean_miss_m": report.mean_miss_m,
        "peak_miss_m": report.peak_miss_m,
        "states_visited": sorted(report.states_visited),
        "advisories_raised": sorted(report.advisories_raised),
    }


def save_run(dir_path, name: str, config: FireTurretConfig, scenario: SimScenario,
             seed: int, report) -> str:
    """Write ``<name>.csv`` (telemetry) + ``<name>.json`` (metadata). Returns the
    JSON sidecar path."""
    from ..analysis import write_csv

    base = Path(dir_path) / name
    if report.telemetry:
        write_csv(report.telemetry, str(base) + ".csv")
    meta = {
        "name": name,
        "seed": seed,
        "config": config_to_dict(config),
        "scenario": scenario_to_dict(scenario),
        "metrics": report_metrics(report),
    }
    json_path = str(base) + ".json"
    Path(json_path).write_text(json.dumps(meta, indent=2))
    return json_path


def load_run_meta(json_path) -> dict:
    return json.loads(Path(json_path).read_text())
