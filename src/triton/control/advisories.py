# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""Operator advisories.

The turret is FIXED: when it cannot engage a fire on its own — the fire is
out of reach, too close, near the traverse limit, or the water impact can't be
seen — the right response is for a human to reposition the turret. This module
turns those conditions into structured, actionable advisories: *what is wrong*
and *what to do about it*. They are surfaced in the operator overlay and, on real
hardware, drive a warning indicator (see `RigCommand.warn`).
"""

from __future__ import annotations

from dataclasses import dataclass

# severities, ordered by urgency
INFO = "INFO"
WARN = "WARN"
CRITICAL = "CRITICAL"
_RANK = {INFO: 0, WARN: 1, CRITICAL: 2}


@dataclass(frozen=True)
class Advisory:
    kind: str  # TOO_FAR | TOO_CLOSE | OUT_OF_TRAVERSE | OBSTRUCTED | SAFETY
    severity: str
    message: str  # what is wrong
    action: str  # what the operator should do

    @property
    def rank(self) -> int:
        return _RANK.get(self.severity, 0)


def reachability_advisory(
    range_m: float, min_reach_m: float, max_reach_m: float, severity: str = WARN
) -> Advisory | None:
    """Warn (with a distance to move) if the fire is outside the jet's envelope."""
    if range_m > max_reach_m:
        return Advisory(
            "TOO_FAR", severity,
            f"Fire out of reach: ~{range_m:.1f} m (max ~{max_reach_m:.1f} m)",
            f"Move the turret ~{max(1, round(range_m - max_reach_m))} m closer",
        )
    if range_m < min_reach_m:
        return Advisory(
            "TOO_CLOSE", severity,
            f"Fire too close: ~{range_m:.1f} m (min ~{min_reach_m:.1f} m)",
            f"Move the turret ~{max(1, round(min_reach_m - range_m))} m back",
        )
    return None


def traverse_advisory(
    fire_world_az_deg: float, pan_min_deg: float, pan_max_deg: float, margin_deg: float = 8.0
) -> Advisory | None:
    """Warn if aiming at the fire needs a pan beyond (or near) the traverse limit."""
    if fire_world_az_deg > pan_max_deg - margin_deg:
        return Advisory(
            "OUT_OF_TRAVERSE", WARN, "Fire at the right traverse limit",
            "Rotate the turret base clockwise (to the right)",
        )
    if fire_world_az_deg < pan_min_deg + margin_deg:
        return Advisory(
            "OUT_OF_TRAVERSE", WARN, "Fire at the left traverse limit",
            "Rotate the turret base counter-clockwise (to the left)",
        )
    return None


def most_urgent(advisories: list[Advisory]) -> Advisory | None:
    return max(advisories, key=lambda a: a.rank, default=None)
