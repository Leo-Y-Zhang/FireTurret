# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""Message translation — the part where the real bugs live, and the part that
needs no ROS installed to test.

Standard messages only (`sensor_msgs/Image`, `sensor_msgs/Imu`,
`geometry_msgs/TwistStamped`, `sensor_msgs/JointState`), mapped onto the
interfaces that already exist — `TurretRig`, `PlatformSensor` — rather than
introducing a parallel abstraction. A ROS bridge that invented its own rig
concept would double the number of places a sign convention can be wrong.

Everything here is a pure function over plain dataclasses that mirror the ROS
message layouts. That is deliberate: field orderings, unit conversions and frame
names are exactly where bridges break, and none of that needs a running ROS
graph to check.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..rig.interface import RigCommand, RigTelemetry

# The tf2 frame names, matching fireturret.frames' chain so the two cannot drift.
FRAME_WORLD = "fireturret/world"
FRAME_PLATFORM = "fireturret/platform"
FRAME_PAN = "fireturret/pan"
FRAME_TILT = "fireturret/tilt"
FRAME_CAMERA = "fireturret/camera"


@dataclass
class RosImage:
    """Mirrors `sensor_msgs/Image` for the fields that matter."""

    height: int
    width: int
    encoding: str
    step: int
    data: bytes
    frame_id: str = FRAME_CAMERA


@dataclass
class RosImu:
    """Mirrors `sensor_msgs/Imu`. ROS angular velocity is **rad/s**."""

    angular_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    linear_acceleration: tuple[float, float, float] = (0.0, 0.0, 0.0)
    orientation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)  # x,y,z,w
    frame_id: str = FRAME_PLATFORM


@dataclass
class RosTwist:
    """Mirrors `geometry_msgs/TwistStamped`. Angular units are **rad/s**."""

    linear: tuple[float, float, float] = (0.0, 0.0, 0.0)
    angular: tuple[float, float, float] = (0.0, 0.0, 0.0)
    frame_id: str = FRAME_PLATFORM


@dataclass
class RosJointState:
    """Mirrors `sensor_msgs/JointState`. Joint positions are **radians**."""

    name: list[str] = field(default_factory=list)
    position: list[float] = field(default_factory=list)
    effort: list[float] = field(default_factory=list)


def image_to_frame(msg: RosImage) -> np.ndarray:
    """`sensor_msgs/Image` -> the BGR ndarray the pipeline expects.

    Only `bgr8` and `rgb8` are accepted, and anything else raises rather than
    being guessed at. A silently mis-decoded colour order would break the
    saturation test that separates splash from fire — invariant 4 — and it would
    do so in a way that looks like a detector tuning problem.
    """
    if msg.encoding not in ("bgr8", "rgb8"):
        raise ValueError(
            f"unsupported image encoding {msg.encoding!r}; expected bgr8 or rgb8. "
            "Guessing would silently invert the colour test that separates water "
            "from fire."
        )
    expected = msg.height * msg.step
    if len(msg.data) != expected:
        raise ValueError(f"image data is {len(msg.data)} bytes, expected {expected}")

    arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
    return arr[:, :, ::-1].copy() if msg.encoding == "rgb8" else arr.copy()


def command_to_twist(cmd: RigCommand, pan_rate_dps: float, tilt_rate_dps: float) -> RosTwist:
    """A rig command expressed as a twist, in ROS units (rad/s).

    Degrees are the repo's unit at every API surface; ROS's is radians. The
    conversion lives here, once.
    """
    return RosTwist(
        angular=(
            math.radians(tilt_rate_dps) if cmd.tilt_deg else 0.0,
            math.radians(pan_rate_dps) if cmd.pan_deg else 0.0,
            0.0,
        ),
        frame_id=FRAME_PLATFORM,
    )


def telemetry_to_joint_state(telemetry: RigTelemetry) -> RosJointState:
    """Rig telemetry as `sensor_msgs/JointState`, positions in radians."""
    return RosJointState(
        name=["fireturret_pan", "fireturret_tilt"],
        position=[math.radians(telemetry.pan_deg), math.radians(telemetry.tilt_deg)],
        effort=[0.0, telemetry.pump_pct],
    )


def imu_to_platform_sample(msg: RosImu) -> dict:
    """`sensor_msgs/Imu` -> the platform sensor's units (**deg/s**).

    ROS reports angular velocity in rad/s and this codebase uses deg/s
    everywhere. Getting that backwards produces a stabiliser that over-corrects
    by 57x, which is the kind of bug that is obvious in a test and catastrophic
    on hardware.
    """
    wx, wy, wz = msg.angular_velocity
    return {
        "gyro_dps": (math.degrees(wx), math.degrees(wy), math.degrees(wz)),
        "accel_ms2": msg.linear_acceleration,
        "frame_id": msg.frame_id,
    }


def has_rclpy() -> bool:
    """Is a ROS 2 runtime importable? Used to skip, never to change behaviour."""
    try:  # pragma: no cover - depends on the environment
        import rclpy  # noqa: F401
    except ImportError:
        return False
    return True
