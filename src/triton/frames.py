# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""The kinematic chain, with every sign convention in exactly one place.

    WORLD -> PLATFORM -> PAN -> TILT -> NOZZLE
                          |
                          +-> CAMERA

**Invariant 2 — the camera rides the PAN stage — is what this diagram encodes,
and for the first time it is machine-checkable rather than a comment.**
`MountConfig.camera_parent` names the joint the camera hangs off, and
`tests/test_egomotion.py` asserts it is PAN. That matters because the invariant is
load-bearing: the camera turning with the pan axis is precisely what lets the
azimuth loop correct the boresight offset from the observed splash. Hang the
camera off the base instead and the whole servo argument collapses — silently.

Rotations are quaternions rather than Euler triples. Not for elegance: Euler
angles need a stated order, and a stated order is a thing two subsystems can
disagree about while both look correct. There is no order to disagree about here.

Today every pose in the chain below PLATFORM is driven by an encoder, and
PLATFORM itself is the identity. SP8 makes PLATFORM move; nothing else about the
chain changes, which is the point of building it now.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# The named joints. Strings rather than an enum so a plugin can name one without
# importing this module, and so a config file can spell it.
WORLD = "WORLD"
PLATFORM = "PLATFORM"
PAN = "PAN"
TILT = "TILT"
NOZZLE = "NOZZLE"
CAMERA = "CAMERA"

#: parent -> child. The camera branches off PAN, and that is invariant 2.
CHAIN: dict[str, str] = {
    PLATFORM: WORLD,
    PAN: PLATFORM,
    TILT: PAN,
    NOZZLE: TILT,
    CAMERA: PAN,
}


@dataclass(frozen=True)
class Quaternion:
    """A unit quaternion (w, x, y, z). Rotations compose by multiplication."""

    w: float = 1.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    @staticmethod
    def identity() -> Quaternion:
        return Quaternion()

    @staticmethod
    def from_axis_angle(axis: tuple[float, float, float], angle_deg: float) -> Quaternion:
        ax, ay, az = axis
        norm = math.sqrt(ax * ax + ay * ay + az * az)
        if norm == 0.0:
            raise ValueError("rotation axis cannot be zero-length")
        half = math.radians(angle_deg) / 2.0
        s = math.sin(half) / norm
        return Quaternion(math.cos(half), ax * s, ay * s, az * s)

    @staticmethod
    def from_yaw(deg: float) -> Quaternion:
        """Pan: rotation about the world's vertical (Y) axis."""
        return Quaternion.from_axis_angle((0.0, 1.0, 0.0), deg)

    @staticmethod
    def from_pitch(deg: float) -> Quaternion:
        """Tilt: rotation about the lateral (X) axis. Positive = nose up."""
        return Quaternion.from_axis_angle((1.0, 0.0, 0.0), deg)

    @staticmethod
    def from_roll(deg: float) -> Quaternion:
        """Roll about the forward (Z) axis. Zero for a fixed turret; SP8's
        platform supplies it."""
        return Quaternion.from_axis_angle((0.0, 0.0, 1.0), deg)

    def __mul__(self, other: Quaternion) -> Quaternion:
        a, b = self, other
        return Quaternion(
            a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
            a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
            a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
            a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        )

    def conjugate(self) -> Quaternion:
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def norm(self) -> float:
        return math.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)

    def normalised(self) -> Quaternion:
        n = self.norm()
        if n == 0.0:
            raise ValueError("cannot normalise a zero quaternion")
        return Quaternion(self.w / n, self.x / n, self.y / n, self.z / n)

    def matrix(self) -> np.ndarray:
        w, x, y, z = self.w, self.x, self.y, self.z
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ], dtype=float)

    def rotate(self, v: tuple[float, float, float]) -> tuple[float, float, float]:
        out = self.matrix() @ np.asarray(v, dtype=float)
        return float(out[0]), float(out[1]), float(out[2])


@dataclass(frozen=True)
class Transform:
    """A rigid-body transform: rotate, then translate."""

    rotation: Quaternion = Quaternion()
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @staticmethod
    def identity() -> Transform:
        return Transform()

    def apply(self, point: tuple[float, float, float]) -> tuple[float, float, float]:
        rx, ry, rz = self.rotation.rotate(point)
        tx, ty, tz = self.translation
        return (rx + tx, ry + ty, rz + tz)

    def compose(self, child: Transform) -> Transform:
        """`self` then `child`, as a single transform (parent-to-grandchild)."""
        rotation = self.rotation * child.rotation
        moved = self.rotation.rotate(child.translation)
        tx, ty, tz = self.translation
        return Transform(rotation, (moved[0] + tx, moved[1] + ty, moved[2] + tz))

    def inverse(self) -> Transform:
        inv = self.rotation.conjugate()
        back = inv.rotate(self.translation)
        return Transform(inv, (-back[0], -back[1], -back[2]))


def chain_to(joint: str) -> list[str]:
    """The named path from WORLD down to `joint`, parent first.

    Raises for an unknown joint rather than returning something plausible — a
    typo'd frame name that silently resolved would be a sign error waiting to
    happen, and sign errors in this file are the expensive kind.
    """
    if joint == WORLD:
        return [WORLD]
    if joint not in CHAIN:
        raise KeyError(f"unknown frame {joint!r}; known: {sorted(CHAIN) + [WORLD]}")
    path = [joint]
    while path[-1] != WORLD:
        path.append(CHAIN[path[-1]])
    return list(reversed(path))


def camera_parent() -> str:
    """The joint the camera hangs off. Invariant 2 says PAN, and this is the
    single place that fact is written down in code."""
    return CHAIN[CAMERA]
