# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""Load/save a calibrated FireTurretConfig from a file (calibration as data, not
hardcoded source). Interchangeable with Studio session files."""
from __future__ import annotations

import json
from dataclasses import replace

import pytest

from fireturret.config import (
    DEFAULT_CONFIG,
    FireTurretConfig,
    TurretConfig,
    config_from_dict,
    config_to_dict,
    load_config,
    save_config,
)


def test_roundtrip_equal():
    cfg = replace(DEFAULT_CONFIG, jet=replace(DEFAULT_CONFIG.jet, drag_k=0.22))
    cfg2 = config_from_dict(json.loads(json.dumps(config_to_dict(cfg))))
    assert cfg2 == cfg


def test_save_load_file(tmp_path):
    cfg = replace(DEFAULT_CONFIG, turret=replace(DEFAULT_CONFIG.turret, pan_max_deg=120.0))
    path = tmp_path / "turret.json"
    save_config(path, cfg)
    assert load_config(path) == cfg


def test_load_accepts_session_file(tmp_path):
    # a Studio session file has {"config": {...}, "scenario": {...}} — load the config out
    path = tmp_path / "session.fireturret.json"
    path.write_text(json.dumps({"config": config_to_dict(DEFAULT_CONFIG), "scenario": {}}))
    assert load_config(path) == DEFAULT_CONFIG


def test_validation_on_load():
    d = config_to_dict(FireTurretConfig())
    d["jet"]["max_pressure_psi"] = -5.0
    with pytest.raises(ValueError):
        config_from_dict(d)


def test_keepout_roundtrips_as_tuple():
    cfg = replace(DEFAULT_CONFIG,
                  turret=replace(DEFAULT_CONFIG.turret, pan_keepout_deg=(-30.0, 30.0)))
    cfg2 = config_from_dict(json.loads(json.dumps(config_to_dict(cfg))))
    assert cfg2 == cfg
    assert isinstance(cfg2.turret.pan_keepout_deg, tuple)


def test_keepout_validation():
    with pytest.raises(ValueError):
        TurretConfig(pan_keepout_deg=(30.0, 30.0))  # not ascending
    with pytest.raises(ValueError):
        TurretConfig(pan_keepout_deg=(-200.0, 0.0))  # outside the pan limits
