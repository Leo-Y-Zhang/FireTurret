# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""The platform sensor plugin point, and its zero-motion default.

`StaticPlatformSensor` is what a fixed turret uses, and it is not a stub: it is
the honest description of a bolted-down machine. Its pose is PERFECT because the
pose really is known exactly, not because nothing is measuring it.

That distinction matters downstream. `PoseQuality.PERFECT` licenses committing
water; a sensor that reported GOOD-because-unknown would be claiming a
measurement it never made.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .mahony import MahonyFilter
from .state import PlatformState, PoseQuality


@dataclass(frozen=True)
class ImuSample:
    """One inertial measurement. Gyro in **deg/s**, accelerometer in **m/s²**."""

    gyro_dps: tuple[float, float, float] = (0.0, 0.0, 0.0)
    accel_ms2: tuple[float, float, float] = (0.0, 0.0, 9.80665)


@runtime_checkable
class PlatformSensor(Protocol):
    def update(self, dt: float) -> PlatformState: ...
    def state(self) -> PlatformState: ...


class StaticPlatformSensor:
    """A bolted-down turret. PERFECT quality, because it genuinely is known."""

    def update(self, dt: float) -> PlatformState:
        return PlatformState()

    def state(self) -> PlatformState:
        return PlatformState()


class ImuPlatformSensor:
    """Attitude from an IMU through the Mahony filter.

    Quality is earned, not assumed:

      LOST      no sample has arrived, or samples have stopped
      DEGRADED  converging, or shaking hard enough that the accelerometer is
                measuring the platform's own motion rather than gravity
      GOOD      converged and consistent

    It never reports PERFECT. A measurement is never as good as a bolt.
    """

    # Consecutive good samples before the filter is trusted for water. Not a
    # ritual number: the accelerometer only constrains attitude on AVERAGE, so a
    # handful of consistent samples is the minimum that means anything.
    CONVERGENCE_SAMPLES = 30

    # Above this the accelerometer is dominated by the platform's own
    # acceleration rather than gravity, so the attitude reference is unreliable.
    SHAKE_G_TOLERANCE = 0.25

    def __init__(self, filt: MahonyFilter | None = None) -> None:
        self.filter = filt or MahonyFilter()
        self._samples = 0
        self._good_streak = 0
        self._last: ImuSample | None = None
        self._translation_m = 0.0
        self._state = PlatformState(quality=PoseQuality.LOST)

    def submit(self, sample: ImuSample) -> None:
        self._last = sample

    def note_translation(self, metres: float) -> None:
        """Accumulated ground distance. HOLD backoff keys on this rather than on
        time, so a driving vehicle re-tries while a stationary one cannot escape
        the backoff by simply waiting."""
        self._translation_m += abs(metres)

    def update(self, dt: float) -> PlatformState:
        if self._last is None:
            self._state = PlatformState(quality=PoseQuality.LOST)
            return self._state

        sample = self._last
        self.filter.update(sample.gyro_dps, sample.accel_ms2, dt)
        self._samples += 1

        magnitude = sum(v * v for v in sample.accel_ms2) ** 0.5
        shaking = abs(magnitude - 9.80665) / 9.80665 > self.SHAKE_G_TOLERANCE
        self._good_streak = 0 if shaking else self._good_streak + 1

        if self._good_streak >= self.CONVERGENCE_SAMPLES:
            quality = PoseQuality.GOOD
        elif self._samples > 0:
            quality = PoseQuality.DEGRADED
        else:  # pragma: no cover - unreachable, kept for the ladder's shape
            quality = PoseQuality.LOST

        self._state = PlatformState(
            roll_deg=self.filter.roll_deg,
            pitch_deg=self.filter.pitch_deg,
            yaw_deg=self.filter.yaw_deg,
            roll_rate_dps=sample.gyro_dps[2],
            pitch_rate_dps=sample.gyro_dps[0],
            yaw_rate_dps=sample.gyro_dps[1],
            translation_m=self._translation_m,
            quality=quality,
        )
        return self._state

    def state(self) -> PlatformState:
        return self._state
