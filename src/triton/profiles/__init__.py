# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""Mission profiles — what the machine is FOR, separated from how it aims.

The core loop is domain-neutral once stated properly:

    **drive the observed impact onto a commanded image setpoint**

Fire suppression is the case where the setpoint offset is zero and the success
criterion is "the fire went out". Wash-down is the case where the policy is a
raster sweep and success is "the area was covered". Neither needs a parallel
control law; they need different *policies* around the same one.

`FireProfile` reconstructs today's wiring object-for-object, and
`tests/test_profile_identity.py` asserts that a run with it is bit-identical to a
run without — which is what makes this an extraction rather than a rewrite.
"""

from .base import (
    EngagementPolicy,
    MissionProfile,
    SafetyConstraints,
)
from .fire import FireProfile
from .washdown import WashdownProfile

__all__ = [
    "EngagementPolicy",
    "FireProfile",
    "MissionProfile",
    "SafetyConstraints",
    "WashdownProfile",
]
