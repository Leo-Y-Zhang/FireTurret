# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""P1b: session/run persistence with explicit from_dict + tuple coercion."""
from __future__ import annotations

import json
from dataclasses import replace

import pytest

from triton.config import TritonConfig
from triton.rig.sim_rig import SimScenario
from triton.studio.io import (
    config_from_dict,
    config_to_dict,
    load_run_meta,
    load_session,
    save_run,
    save_session,
    scenario_from_dict,
    scenario_to_dict,
)


def _json_roundtrip(d):
    return json.loads(json.dumps(d))


def test_config_roundtrip_equal():
    cfg = TritonConfig()
    cfg2 = config_from_dict(_json_roundtrip(config_to_dict(cfg)))
    assert cfg2 == cfg


def test_config_from_dict_runs_validation():
    d = config_to_dict(TritonConfig())
    d["jet"]["max_pressure_psi"] = -5.0
    with pytest.raises(ValueError):
        config_from_dict(d)


def test_scenario_roundtrip_preserves_tuples():
    sc = SimScenario(extra_fires=((-30.0, 6.0), (20.0, 5.0)), decoy=(10.0, 4.0))
    sc2 = scenario_from_dict(_json_roundtrip(scenario_to_dict(sc)))
    assert sc2 == sc
    assert isinstance(sc2.extra_fires, tuple)
    assert isinstance(sc2.extra_fires[0], tuple)
    assert isinstance(sc2.decoy, tuple)


def test_session_save_load(tmp_path):
    cfg = replace(TritonConfig(), jet=replace(TritonConfig().jet, drag_k=0.2))
    sc = SimScenario(fire_range_m=8.0, decoy=(5.0, 4.0))
    path = tmp_path / "s.triton.json"
    save_session(path, cfg, sc)
    cfg2, sc2 = load_session(path)
    assert cfg2 == cfg
    assert sc2 == sc


def test_save_run_and_load_meta(tmp_path):
    from triton.config import DEFAULT_CONFIG
    from triton.simcore import drive

    report = drive(DEFAULT_CONFIG, seed=7, max_frames=120, record_telemetry=True)
    json_path = save_run(tmp_path, "run1", DEFAULT_CONFIG, SimScenario(), 7, report)
    meta = load_run_meta(json_path)
    assert meta["seed"] == 7
    assert meta["metrics"]["frames"] == report.frames
    assert (tmp_path / "run1.csv").exists()
    assert (tmp_path / "run1.json").exists()
