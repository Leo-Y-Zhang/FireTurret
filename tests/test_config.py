# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
import pytest

from fireturret.config import (
    DEFAULT_CONFIG,
    CameraConfig,
    FireTurretConfig,
    JetConfig,
    TurretConfig,
)


def test_default_config_is_valid() -> None:
    assert isinstance(DEFAULT_CONFIG, FireTurretConfig)  # constructs without error


def test_turret_rejects_bad_limits() -> None:
    with pytest.raises(ValueError):
        TurretConfig(tilt_min_deg=50.0, tilt_max_deg=10.0)
    with pytest.raises(ValueError):
        TurretConfig(pan_min_deg=10.0, pan_max_deg=-10.0)
    with pytest.raises(ValueError):
        TurretConfig(pan_rate_dps=0.0)


def test_jet_rejects_bad_values() -> None:
    with pytest.raises(ValueError):
        JetConfig(max_pressure_psi=0.0)
    with pytest.raises(ValueError):
        JetConfig(min_pump_pct=0.0)
    with pytest.raises(ValueError):
        JetConfig(min_pump_pct=100.0)


def test_camera_rejects_bad_values() -> None:
    with pytest.raises(ValueError):
        CameraConfig(width=0)
    with pytest.raises(ValueError):
        CameraConfig(hfov_deg=200.0)
    with pytest.raises(ValueError):
        CameraConfig(mount_height_m=-1.0)
