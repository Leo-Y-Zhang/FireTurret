# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""SP16 — the ROS 2 bridge's translation layer.

Everything here runs with no ROS installed, because everything here is a pure
function over plain data. That is the point: field orderings, unit conversions
and colour orders are where bridges actually break, and none of it needs a
running ROS graph to check. Only the node wiring needs `rclpy`, and that is the
part with the least logic in it.
"""

from __future__ import annotations

import math

import pytest

from fireturret.rig.interface import RigCommand, RigTelemetry
from fireturret.ros.translate import (
    FRAME_CAMERA,
    RosImage,
    RosImu,
    command_to_twist,
    has_rclpy,
    image_to_frame,
    imu_to_platform_sample,
    telemetry_to_joint_state,
)


def _image(encoding: str, pixel: tuple[int, int, int]) -> RosImage:
    h, w = 2, 3
    data = bytes(list(pixel) * (h * w))
    return RosImage(height=h, width=w, encoding=encoding, step=w * 3, data=data)


# ------------------------------------------------------------------- images

def test_bgr8_passes_through_unchanged() -> None:
    frame = image_to_frame(_image("bgr8", (10, 20, 30)))
    assert frame.shape == (2, 3, 3)
    assert tuple(frame[0, 0]) == (10, 20, 30)


def test_rgb8_is_reordered_to_bgr() -> None:
    """Getting this backwards would invert the saturation test that separates
    water from fire (invariant 4) — and it would look like a detector tuning
    problem rather than a bridge bug."""
    frame = image_to_frame(_image("rgb8", (10, 20, 30)))
    assert tuple(frame[0, 0]) == (30, 20, 10)


def test_an_unknown_encoding_raises_rather_than_guessing() -> None:
    with pytest.raises(ValueError, match="unsupported image encoding"):
        image_to_frame(_image("mono8", (10, 20, 30)))


def test_a_truncated_image_is_rejected() -> None:
    msg = _image("bgr8", (1, 2, 3))
    msg.data = msg.data[:-3]
    with pytest.raises(ValueError, match="expected"):
        image_to_frame(msg)


def test_the_decoded_frame_is_writable() -> None:
    """`np.frombuffer` returns a read-only view, and OpenCV will refuse to draw
    on it. A bridge that returned one would fail only in the overlay path."""
    frame = image_to_frame(_image("bgr8", (5, 5, 5)))
    frame[0, 0] = (1, 2, 3)  # must not raise


def test_the_camera_frame_id_matches_the_kinematic_chain() -> None:
    assert _image("bgr8", (0, 0, 0)).frame_id == FRAME_CAMERA


# -------------------------------------------------------------------- units

def test_imu_angular_velocity_converts_rad_per_s_to_deg_per_s() -> None:
    """ROS uses rad/s; this codebase uses deg/s everywhere. Getting it backwards
    produces a stabiliser that over-corrects by 57x — obvious in a test,
    catastrophic on hardware."""
    msg = RosImu(angular_velocity=(math.pi, 0.0, -math.pi / 2))
    sample = imu_to_platform_sample(msg)
    assert sample["gyro_dps"][0] == pytest.approx(180.0)
    assert sample["gyro_dps"][2] == pytest.approx(-90.0)


def test_linear_acceleration_is_already_in_si_and_is_not_converted() -> None:
    msg = RosImu(linear_acceleration=(0.0, 9.81, 0.0))
    assert imu_to_platform_sample(msg)["accel_ms2"] == (0.0, 9.81, 0.0)


def test_joint_state_positions_are_radians() -> None:
    telemetry = RigTelemetry(90.0, 45.0, 55.0, True, estop=False, ok=True, homed=True)
    js = telemetry_to_joint_state(telemetry)
    assert js.name == ["fireturret_pan", "fireturret_tilt"]
    assert js.position[0] == pytest.approx(math.pi / 2)
    assert js.position[1] == pytest.approx(math.pi / 4)


def test_pump_is_reported_as_effort_not_position() -> None:
    telemetry = RigTelemetry(0.0, 20.0, 62.0, True, estop=False, ok=True, homed=True)
    assert telemetry_to_joint_state(telemetry).effort[1] == 62.0


def test_a_twist_is_produced_in_radians() -> None:
    twist = command_to_twist(
        RigCommand(pan_deg=10.0, tilt_deg=20.0, pump_pct=0.0, valve=False),
        pan_rate_dps=90.0, tilt_rate_dps=60.0,
    )
    assert twist.angular[1] == pytest.approx(math.radians(90.0))
    assert twist.angular[0] == pytest.approx(math.radians(60.0))


# --------------------------------------------------------- the optional extra

def test_the_core_never_imports_rclpy() -> None:
    """The `[ros]` extra follows the `[ml]` pattern: the core install stays small
    and a missing ROS is a skip, never a failure."""
    import subprocess
    import sys

    code = (
        "import sys, fireturret.simcore; "
        "print('rclpy' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         check=True).stdout.strip()
    assert out == "False"


def test_translation_needs_no_ros_installed() -> None:
    """This whole module is testable without a ROS runtime — which is the reason
    the translation layer is separated from the node wiring in the first place."""
    assert image_to_frame(_image("bgr8", (1, 1, 1))) is not None
    assert isinstance(has_rclpy(), bool)


@pytest.mark.skipif(not has_rclpy(), reason="rclpy not installed (optional [ros] extra)")
def test_the_node_layer_imports_when_ros_is_present() -> None:  # pragma: no cover
    from fireturret.ros import node  # noqa: F401
