# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""Simulated platform trajectories: Static, Drive, Bump. Only those three.

The spec is explicit that this is the list, and the restraint is the point.
Boat motion, hand-carry and composite trajectories are *data*, not architecture —
each is a few numbers, each is a new adversarial test case, and each can be added
later without touching anything. Building an elaborate trajectory framework
before there is a single real IMU log would be building for imagined
requirements.

## Every trajectory carries its own perturbation knobs

Invariant 10 says the simulator's truth is deliberately different from the
controller's model, and the end-to-end test passing is what proves the closed
loop corrects model error. A new simulated capability that the controller could
model perfectly would quietly weaken that.

So `imu_bias_dps`, `imu_noise_dps` and `imu_latency_s` exist, all defaulting to
zero and written as early returns so they consume no RNG draws — which keeps
every existing fixture untouched by construction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..platform.sensor import ImuSample


@dataclass(frozen=True)
class TrajectoryPerturbation:
    """Sensor error the CONTROLLER does not know about (invariant 10)."""

    imu_bias_dps: tuple[float, float, float] = (0.0, 0.0, 0.0)
    imu_noise_dps: float = 0.0
    imu_latency_s: float = 0.0

    @property
    def is_identity(self) -> bool:
        return (
            self.imu_bias_dps == (0.0, 0.0, 0.0)
            and self.imu_noise_dps == 0.0
            and self.imu_latency_s == 0.0
        )


class StaticTrajectory:
    """A bolted-down turret. The zero-motion case of the same code path."""

    name = "static"

    def sample(self, t_s: float) -> ImuSample:
        return ImuSample()

    def translation_m(self, t_s: float) -> float:
        return 0.0


@dataclass
class DriveTrajectory:
    """A vehicle driving in a straight line at constant speed, with the gentle
    pitch and roll of suspension over an imperfect surface."""

    speed_ms: float = 2.0
    roughness_deg: float = 1.5
    frequency_hz: float = 0.8
    name: str = "drive"

    def sample(self, t_s: float) -> ImuSample:
        omega = 2.0 * math.pi * self.frequency_hz
        roll_rate = self.roughness_deg * omega * math.cos(omega * t_s)
        pitch_rate = 0.6 * self.roughness_deg * omega * math.cos(omega * t_s + 1.0)
        roll = self.roughness_deg * math.sin(omega * t_s)
        return ImuSample(
            gyro_dps=(pitch_rate, 0.0, roll_rate),
            accel_ms2=_gravity_in_body(roll_deg=roll, pitch_deg=0.0),
        )

    def translation_m(self, t_s: float) -> float:
        return self.speed_ms * t_s


@dataclass
class BumpTrajectory:
    """A single knock: a sharp transient the platform returns from.

    Deliberately a transient. A bump that did not return would be a reposition,
    and `platform/disturbance.py` tells them apart by exactly that.
    """

    at_s: float = 2.0
    magnitude_deg: float = 6.0
    duration_s: float = 0.4
    name: str = "bump"

    def sample(self, t_s: float) -> ImuSample:
        if not (self.at_s <= t_s < self.at_s + self.duration_s):
            return ImuSample()
        # one full sine cycle: out and back, so attitude returns to where it was
        phase = (t_s - self.at_s) / self.duration_s
        omega = 2.0 * math.pi / self.duration_s
        rate = self.magnitude_deg * omega * math.cos(2.0 * math.pi * phase)
        roll = self.magnitude_deg * math.sin(2.0 * math.pi * phase)
        return ImuSample(
            gyro_dps=(0.0, 0.0, rate),
            accel_ms2=_gravity_in_body(roll_deg=roll, pitch_deg=0.0),
        )

    def translation_m(self, t_s: float) -> float:
        return 0.0


def _gravity_in_body(roll_deg: float, pitch_deg: float) -> tuple[float, float, float]:
    """Gravity as an accelerometer on a tilted platform would read it."""
    r, p = math.radians(roll_deg), math.radians(pitch_deg)
    g = 9.80665
    return (g * math.sin(r), -g * math.sin(p), g * math.cos(r) * math.cos(p))


def perturb(
    sample: ImuSample, perturbation: TrajectoryPerturbation, rng=None
) -> ImuSample:
    """Apply sensor error the controller does not know about.

    Early return on the identity case, so the default path consumes **no RNG
    draws** and every existing fixture is untouched by construction — the same
    discipline SP2 established for the renderers.
    """
    if perturbation.is_identity:
        return sample

    bias = perturbation.imu_bias_dps
    gyro = [g + b for g, b in zip(sample.gyro_dps, bias, strict=True)]
    if perturbation.imu_noise_dps and rng is not None:
        gyro = [g + float(rng.normal(0.0, perturbation.imu_noise_dps)) for g in gyro]
    return ImuSample(gyro_dps=(gyro[0], gyro[1], gyro[2]), accel_ms2=sample.accel_ms2)
