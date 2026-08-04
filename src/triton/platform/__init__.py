# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""The mobile platform tier — where the turret's base is allowed to move.

The fixed turret is the **zero-motion case of this code path**, not a separate
one. `StaticPlatformSensor` reports a perfect, motionless pose, the pose quality
is PERFECT, and every calculation reduces to what it was before. That is why the
fixed-turret goldens are the acceptance criterion for the whole abstraction: if
they move, the generalisation changed something it should not have.

## The honest part

Platform motion breaks invariant 3 ("observe then correct") outright — it assumes
a still camera. This tier does not repair that by pretending; it reconstructs the
invariant as "still in the ego-motion-warped frame" and then **gates suppression
on how well that reconstruction is working**. When the warp residual is large,
the median splash position is corrupted, and the correct response is to HOLD
rather than to servo on it. The difference between "works on a boat" and "sprays
confidently in the wrong direction on a boat" is exactly that gate.
"""

from .disturbance import DisturbanceKind, DisturbanceMonitor
from .mahony import MahonyFilter
from .sensor import ImuSample, PlatformSensor, StaticPlatformSensor
from .state import PlatformState, PoseQuality

__all__ = [
    "DisturbanceKind",
    "DisturbanceMonitor",
    "ImuSample",
    "MahonyFilter",
    "PlatformSensor",
    "PlatformState",
    "PoseQuality",
    "StaticPlatformSensor",
]
