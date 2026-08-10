# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
import math

import pytest

from fireturret.config import CameraConfig
from fireturret.geometry import CameraModel, clamp, wrap_deg


@pytest.fixture
def cam() -> CameraModel:
    return CameraModel(CameraConfig())


def test_centre_pixel_is_zero_angles(cam: CameraModel) -> None:
    ang = cam.px_to_angles(cam.cx, cam.cy)
    assert abs(ang.azimuth_deg) < 1e-9
    assert abs(ang.elevation_deg) < 1e-9


def test_right_of_centre_is_positive_azimuth(cam: CameraModel) -> None:
    assert cam.px_to_angles(cam.cx + 100, cam.cy).azimuth_deg > 0
    assert cam.px_to_angles(cam.cx - 100, cam.cy).azimuth_deg < 0
    # image y grows down; above centre = positive elevation
    assert cam.px_to_angles(cam.cx, cam.cy - 100).elevation_deg > 0


def test_angle_pixel_roundtrip(cam: CameraModel) -> None:
    u, v = cam.angles_to_px(12.0, -5.0)
    ang = cam.px_to_angles(u, v)
    assert ang.azimuth_deg == pytest.approx(12.0, abs=1e-6)
    assert ang.elevation_deg == pytest.approx(-5.0, abs=1e-6)


def test_edge_azimuth_is_half_fov(cam: CameraModel) -> None:
    ang = cam.px_to_angles(cam.cfg.width, cam.cy)
    assert ang.azimuth_deg == pytest.approx(cam.cfg.hfov_deg / 2, abs=0.2)


def test_ground_range_centre_pixel(cam: CameraModel) -> None:
    # centre pixel looks along the mount pitch: depression 8 deg
    r = cam.ground_range_from_px(cam.cx, cam.cy)
    assert r is not None
    expected = cam.cfg.mount_height_m / math.tan(math.radians(-cam.cfg.mount_pitch_deg))
    assert r == pytest.approx(expected, rel=1e-6)


def test_ground_range_monotonic_with_pixel_height(cam: CameraModel) -> None:
    low = cam.ground_range_from_px(cam.cx, cam.cfg.height - 10)
    mid = cam.ground_range_from_px(cam.cx, cam.cy)
    assert low is not None and mid is not None
    assert low < mid  # lower in the image = nearer


def test_ground_range_none_above_horizon(cam: CameraModel) -> None:
    assert cam.ground_range_from_px(cam.cx, 0) is None


def test_wrap_and_clamp() -> None:
    assert wrap_deg(190) == pytest.approx(-170)
    assert wrap_deg(-190) == pytest.approx(170)
    assert wrap_deg(0) == 0
    assert clamp(5, 0, 3) == 3
    assert clamp(-1, 0, 3) == 0
