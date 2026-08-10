# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""The flagship profile — and, crucially, a reconstruction rather than a rewrite.

`FireProfile` describes what the system already does: aim the water AT the fire
(setpoint offset zero), stop when it is out, solid-stream nozzle, the existing
advisory vocabulary. `tests/test_profile_identity.py` asserts that a run driven
by this profile is bit-identical to a run without one — floats included.

That test is the whole point of the exercise. An abstraction whose flagship case
is not provably identical to the code it replaced has not been extracted; it has
been re-implemented, and the difference will surface later as a behaviour change
nobody meant to make.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..spray.envelope import LEGACY_ENVELOPE
from .base import MissionProfile, SafetyConstraints


@dataclass(frozen=True)
class FireEngagement:
    """Put the water on the fire, and stop when it is out."""

    name: str = "fire"

    def setpoint_offset_px(self, target) -> tuple[float, float]:
        # Zero: fire suppression is the case where you aim AT the thing. Every
        # other offset in the codebase (the open-short pump bias, invariant 8)
        # lives in the ballistics, not here.
        return (0.0, 0.0)

    def is_complete(self, target, elapsed_s: float) -> bool:
        # The mission state machine owns extinguish confirmation (CONFIRM's soak
        # and clear timers). A profile that duplicated that logic would be a
        # second source of truth for when to stop spraying.
        return False


def FireProfile() -> MissionProfile:  # noqa: N802 - reads as a constructor
    """The shipped fire-suppression profile."""
    return MissionProfile(
        name="fire",
        policy=FireEngagement(),
        spray_envelope=LEGACY_ENVELOPE,
        constraints=SafetyConstraints(require_armed=True),
        advisory_vocabulary=("TOO_FAR", "TOO_CLOSE", "OBSTRUCTED", "TRAVERSE", "SAFETY"),
        nozzle="solid_stream",
    )
