# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""Line-of-sight stabilisation, and the ladder that gives up honestly.

## The rate budget, from one written expression

Total demand on the pan axis is the platform's own rotation plus whatever
tracking the target needs:

    omega_total = omega_platform + omega_track

`TurretConfig.pan_rate_dps = 90` sounds generous until it is compared with a real
disturbance. An 8 degree, 1.5 Hz oscillation — a small boat, a vehicle on rough
ground — has a peak rate of

    omega_peak = 2 * pi * f * A = 2 * pi * 1.5 * 8 = 75 deg/s

so **75 of the available 90 deg/s is spent before any target tracking at all**.
Both the pan and tilt requirements come from that single expression
(`PlatformState.total_rate_dps`), so they cannot drift apart, and the test is a
NUMERICAL parallelism check rather than a sign convention comparison — a
self-consistent pair of wrong signs can satisfy a sign check and cannot satisfy
this one.

## The four-rung ladder

    FULL       demand fits; stabilisation is complete
    PARTIAL    demand exceeds the actuator; null what can be nulled
    DEGRADED   pose quality is too poor to trust the feedforward
    LOST       no usable pose, or demand far beyond the actuator

**Water closes at DEGRADED and below.** That is the load-bearing rung. A
stabiliser that silently degrades keeps aiming confidently while its aim means
progressively less — which on a moving platform means spraying somewhere nobody
chose. Giving up is a feature.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum

from ..platform.state import PlatformState, PoseQuality


class StabiliserRung(IntEnum):
    LOST = 0
    DEGRADED = 1
    PARTIAL = 2
    FULL = 3

    @property
    def may_commit_water(self) -> bool:
        return self >= StabiliserRung.PARTIAL


@dataclass(frozen=True)
class StabiliserOutput:
    """Feedforward rates, and how much of the demand was actually met."""

    pan_rate_dps: float
    tilt_rate_dps: float
    rung: StabiliserRung
    demanded_dps: float
    met_fraction: float

    @property
    def water_allowed(self) -> bool:
        return self.rung.may_commit_water


def oscillation_peak_rate_dps(amplitude_deg: float, frequency_hz: float) -> float:
    """Peak angular rate of a sinusoidal disturbance: `2 pi f A`.

    Written as a function so the documented 8 deg / 1.5 Hz / 75 deg/s example is
    computed rather than asserted, and so a test can check the rate budget
    against the same expression the docstring quotes.
    """
    return 2.0 * math.pi * frequency_hz * amplitude_deg


def stabilise(
    platform: PlatformState,
    track_rate_dps: float,
    pan_rate_limit_dps: float,
    tilt_rate_limit_dps: float,
) -> StabiliserOutput:
    """Null the platform's line-of-sight motion, and say how well it went.

    Feedforward only — this predicts the platform's contribution from measured
    rates and cancels it. It does not close a loop of its own, because a second
    loop around the same actuator is a second set of stability arguments.
    """
    if platform.quality is PoseQuality.LOST:
        return StabiliserOutput(0.0, 0.0, StabiliserRung.LOST, 0.0, 0.0)

    # Yaw drives pan; roll and pitch drive tilt's line of sight.
    pan_demand = -platform.yaw_rate_dps + track_rate_dps
    tilt_demand = -math.hypot(platform.pitch_rate_dps, platform.roll_rate_dps)
    demanded = math.hypot(pan_demand, tilt_demand)

    pan_out = max(-pan_rate_limit_dps, min(pan_rate_limit_dps, pan_demand))
    tilt_out = max(-tilt_rate_limit_dps, min(tilt_rate_limit_dps, tilt_demand))
    met = math.hypot(pan_out, tilt_out)
    fraction = 1.0 if demanded <= 1e-9 else min(1.0, met / demanded)

    if platform.quality is PoseQuality.DEGRADED:
        # The feedforward is only as good as the pose it is computed from.
        rung = StabiliserRung.DEGRADED
    elif fraction >= 0.999:
        rung = StabiliserRung.FULL
    elif fraction >= 0.5:
        rung = StabiliserRung.PARTIAL
    else:
        # More than half the demand unmet: the aim is being carried by the
        # platform rather than commanded. That is not stabilisation.
        rung = StabiliserRung.LOST

    return StabiliserOutput(pan_out, tilt_out, rung, demanded, fraction)


def derotate_image_offset(
    du_px: float, dv_px: float, roll_deg: float
) -> tuple[float, float]:
    """Undo platform roll before comparing image COLUMNS.

    **The invariant-2 caveat, handled.** The azimuth loop compares the splash's
    column with the fire's. Under platform roll phi that comparison is corrupted,
    because the image rotates:

        u' = cx + du*cos(phi) + dv*sin(phi)

    A 100 px splash-to-fire ROW gap at 8 degrees of roll injects
    100*sin(8 deg) = 13.9 px of spurious COLUMN difference — about 1.1 degrees of
    azimuth, against a 0.7 degree pan deadband. It self-extinguishes at
    convergence (the row gap goes to zero) but it corrupts acquisition, which is
    exactly when the loop can least afford it.

    So de-rotate first, then compare.
    """
    phi = math.radians(roll_deg)
    cos_phi, sin_phi = math.cos(phi), math.sin(phi)
    return (du_px * cos_phi + dv_px * sin_phi, -du_px * sin_phi + dv_px * cos_phi)
