# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""The `MissionProfile` contract.

A profile supplies the domain-specific parts of a mission — what counts as a
target, where to put the water relative to it, when the job is done, what the
nozzle should be, what the operator is told, and what safety constraints apply —
while the control loop, the water gate and the failsafes stay common.

## Safety constraints may only ever NARROW

`SafetyConstraints.intersect` takes the element-wise **strictest** value. That is
the whole design of the safety story here: a profile is a way to describe a job,
not a way to acquire permissions. There is no combination of profiles, and no
plugin, that can widen a keep-out sector, raise a spray limit, or arm water that
the base configuration did not allow.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol, runtime_checkable

from ..spray.envelope import LEGACY_ENVELOPE, SprayEnvelope


@dataclass(frozen=True)
class SafetyConstraints:
    """Limits a profile operates under. Narrowing only — see the module docstring.

    `None` means "this profile adds no opinion", NOT "no limit": intersecting
    with None yields the other side unchanged.
    """

    max_spray_s: float | None = None
    max_pump_pct: float | None = None
    pan_keepout_deg: tuple[float, float] | None = None
    require_armed: bool = True

    def intersect(self, other: SafetyConstraints) -> SafetyConstraints:
        """Element-wise strictest. Never widens, in any field, ever."""
        return SafetyConstraints(
            max_spray_s=_stricter_min(self.max_spray_s, other.max_spray_s),
            max_pump_pct=_stricter_min(self.max_pump_pct, other.max_pump_pct),
            pan_keepout_deg=_widest_sector(self.pan_keepout_deg, other.pan_keepout_deg),
            # require_armed is a safety requirement, so True wins
            require_armed=self.require_armed or other.require_armed,
        )


def _stricter_min(a: float | None, b: float | None) -> float | None:
    """The smaller of two ceilings; None means "no opinion"."""
    if a is None:
        return b
    if b is None:
        return a
    return min(a, b)


def _widest_sector(
    a: tuple[float, float] | None, b: tuple[float, float] | None
) -> tuple[float, float] | None:
    """The union of two protected sectors.

    Note the asymmetry with the numeric fields, and that it is not a mistake: a
    keep-out is a region water must NOT enter, so "stricter" means a LARGER
    sector. Taking the intersection here would let a second profile shrink a
    protected zone, which is exactly the widening this class exists to prevent.
    """
    if a is None:
        return b
    if b is None:
        return a
    return (min(a[0], b[0]), max(a[1], b[1]))


@runtime_checkable
class EngagementPolicy(Protocol):
    """How the profile wants targets engaged.

    Deliberately thin. The policy decides WHERE to put water relative to a
    target and WHEN the job is done; it does not get to decide how the servo
    reaches that point, because that is the part with the safety argument
    attached.
    """

    name: str

    def setpoint_offset_px(self, target) -> tuple[float, float]:
        """Where to land the water, relative to the target's ground pixel.
        `(0, 0)` means "on it", which is fire suppression."""
        ...

    def is_complete(self, target, elapsed_s: float) -> bool:
        """Has this target been dealt with?"""
        ...


@dataclass(frozen=True)
class MissionProfile:
    """Everything domain-specific about a mission, in one object."""

    name: str
    policy: EngagementPolicy
    spray_envelope: SprayEnvelope = LEGACY_ENVELOPE
    constraints: SafetyConstraints = SafetyConstraints()
    advisory_vocabulary: tuple[str, ...] = ()
    nozzle: str = "solid_stream"

    def with_constraints(self, extra: SafetyConstraints) -> MissionProfile:
        """A copy narrowed by `extra`. Cannot widen — `intersect` will not."""
        return replace(self, constraints=self.constraints.intersect(extra))
