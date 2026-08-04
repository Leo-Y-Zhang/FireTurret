# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""Mahony passive complementary filter on SO(3), with explicit gyro-bias action.

Roughly forty lines of arithmetic, no scipy, no new core dependency. The choice
of filter is deliberate and the spec argues it: an ESKF or a VIO front-end would
be more capable on paper, but **there is no test that distinguishes them from
this without real IMU data**. Shipping both would mean shipping one that is
unvalidated and one that is redundant, with no way to tell which is which.

So: Mahony now, and a `PlatformSensor` plugin protocol so anyone with real data
can drop in a better estimator without forking.

## The bias integrator is the point

A MEMS gyro's zero-rate output drifts with temperature. Integrate it naively and
attitude walks away at degrees per minute, which is fatal for a machine that
holds an aim for tens of seconds. The integral term estimates that bias from the
accelerometer's long-run agreement with gravity and subtracts it.

## ZUPT

While the platform is demonstrably still — gyro near zero and accelerometer
magnitude near 1 g — any measured rotation rate IS bias, by definition. That is
the cheapest and most reliable bias observation available, so stillness is used
to re-estimate it directly rather than waiting for the integrator to converge.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

G_MS2 = 9.80665

# Mahony gains. Kp sets how hard the accelerometer pulls attitude toward gravity;
# Ki sets how fast the bias estimate moves. Ki is small on purpose: a fast bias
# integrator will happily absorb a sustained real acceleration as "bias" and
# then take just as long to give it back.
DEFAULT_KP = 1.0
DEFAULT_KI = 0.05

# ZUPT thresholds. Generous, because a false ZUPT is much worse than a missed
# one: it teaches the filter that real motion is bias.
ZUPT_GYRO_DPS = 1.0
ZUPT_ACCEL_TOLERANCE = 0.05  # fraction of g


@dataclass
class MahonyFilter:
    """Attitude from gyro + accelerometer. Yaw is NOT observable from these two.

    Gravity gives roll and pitch. It says nothing about heading, so `yaw_deg`
    here integrates the gyro and will drift — that is a property of the sensor
    set, not a defect, and it is why the stabiliser uses yaw RATE (which is
    measured) rather than yaw angle (which is dead-reckoned).
    """

    kp: float = DEFAULT_KP
    ki: float = DEFAULT_KI
    roll_deg: float = 0.0
    pitch_deg: float = 0.0
    yaw_deg: float = 0.0
    bias_dps: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    zupt_count: int = 0

    def update(
        self,
        gyro_dps: tuple[float, float, float],
        accel_ms2: tuple[float, float, float],
        dt: float,
    ) -> None:
        if dt <= 0:
            return

        gx = gyro_dps[0] - self.bias_dps[0]
        gy = gyro_dps[1] - self.bias_dps[1]
        gz = gyro_dps[2] - self.bias_dps[2]

        if self._is_still(gyro_dps, accel_ms2):
            # ZUPT: standing still, so whatever the gyro reads IS bias.
            self.zupt_count += 1
            for i in range(3):
                self.bias_dps[i] += 0.1 * (gyro_dps[i] - self.bias_dps[i])
            gx = gy = gz = 0.0

        ax, ay, az = accel_ms2
        norm = math.sqrt(ax * ax + ay * ay + az * az)
        if norm > 1e-6:
            ax, ay, az = ax / norm, ay / norm, az / norm

            # Attitude implied by gravity, versus the attitude we currently hold.
            measured_roll = math.degrees(math.atan2(ax, az))
            measured_pitch = math.degrees(math.atan2(-ay, math.hypot(ax, az)))

            roll_err = _wrap(measured_roll - self.roll_deg)
            pitch_err = _wrap(measured_pitch - self.pitch_deg)

            # Proportional: pull attitude toward gravity.
            self.roll_deg += self.kp * roll_err * dt
            self.pitch_deg += self.kp * pitch_err * dt
            # Integral: the persistent part of that pull is gyro bias.
            self.bias_dps[0] -= self.ki * pitch_err * dt
            self.bias_dps[2] -= self.ki * roll_err * dt

        # Gyro integration (all three axes; yaw has only this).
        self.roll_deg = _wrap(self.roll_deg + gz * dt)
        self.pitch_deg = _wrap(self.pitch_deg + gx * dt)
        self.yaw_deg = _wrap(self.yaw_deg + gy * dt)

    def _is_still(
        self, gyro_dps: tuple[float, float, float], accel_ms2: tuple[float, float, float]
    ) -> bool:
        if max(abs(v) for v in gyro_dps) > ZUPT_GYRO_DPS:
            return False
        magnitude = math.sqrt(sum(v * v for v in accel_ms2))
        return abs(magnitude - G_MS2) / G_MS2 <= ZUPT_ACCEL_TOLERANCE


def _wrap(deg: float) -> float:
    d = math.fmod(deg, 360.0)
    if d > 180.0:
        d -= 360.0
    if d <= -180.0:
        d += 360.0
    return d
