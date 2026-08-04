# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""ROS 2 bridge — an OPTIONAL extra, following the repo's established `[ml]` pattern.

The core never imports this package, `rclpy` is not a dependency of anything
else, and the tests skip when it is absent. That convention already exists here
for `onnxruntime`, so this adds a pattern rather than inventing one.

## Safety note, carried here and in docs/SAFETY.md

**A pure-ROS rig has no independent hardware watchdog.** The firmware's E-stop and
500 ms heartbeat live on the microcontroller; a rig that speaks only ROS topics
has neither. So a ROS-driven turret must either be paired with the firmware
E-stop or run **dry-aim only**. This is stated rather than designed around,
because designing around it would mean claiming a safety property the
architecture does not have.

## What is testable without rclpy

The message translation layer — which is where the real bugs live. Field
orderings, unit conversions, frame names, sign conventions: all pure functions
over plain data, all testable with no ROS installed. `translate.py` holds them,
and `tests/test_ros.py` exercises them unconditionally. Only the node wiring
needs `rclpy`, and that is the part with the least logic in it.
"""

from .translate import (
    RosImage,
    RosImu,
    RosTwist,
    command_to_twist,
    image_to_frame,
    imu_to_platform_sample,
    telemetry_to_joint_state,
)

__all__ = [
    "RosImage",
    "RosImu",
    "RosTwist",
    "command_to_twist",
    "image_to_frame",
    "imu_to_platform_sample",
    "telemetry_to_joint_state",
]
