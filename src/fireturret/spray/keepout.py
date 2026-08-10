# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""Deposition-aware keep-out: where the water LANDS, not where the nozzle points.

## The gap

`TurretConfig.pan_keepout_deg` constrains the **aim**. `MissionController` pushes
any commanded angle inside the sector to its nearer edge, and SP1 added a hold
covering the swept path. Both reason entirely about a single direction.

Water does not land in a single direction. For a solid stream that distinction is
academic — the jet has not broken up by the time it lands, so the footprint is a
point, and every check here collapses to the aim. For a fan or fog pattern it is
the whole problem:

    a 10 degree fan at 7 m has a half-width of 0.61 m
    an aim 3 degrees outside a keep-out edge is 0.37 m outside it
    => water still lands 0.24 m INSIDE the protected sector

The turret is obeying the geofence perfectly and wetting the thing the geofence
exists to protect.

## The response

Project the controller's **inflated 2-sigma** footprint onto the ground, convert
it to an angular interval, and test THAT against the sector rather than the aim.
Two sigma because the question is where water *can* land, not where it typically
does.

A predicted violation is routed through the **existing HOLD path** rather than
getting its own handling, so it inherits invariant 9's exponential backoff for
free — and so there is one place in the codebase that means "stop, this target
cannot be engaged safely right now".

Deny-only, like every other safety layer here: this reports a violation, and the
mission responds by withholding water. It never re-aims.
"""

from __future__ import annotations

from dataclasses import dataclass

from .envelope import SprayEnvelope


@dataclass(frozen=True)
class Violation:
    """A predicted wetting of a protected sector."""

    sector: tuple[float, float]
    aim_pan_deg: float
    range_m: float
    half_angle_deg: float
    overlap_deg: float

    @property
    def reason(self) -> str:
        return (
            f"spray would wet the keep-out sector {self.sector} "
            f"(aim {self.aim_pan_deg:.1f} deg, footprint +/-{self.half_angle_deg:.1f} deg, "
            f"overlap {self.overlap_deg:.1f} deg)"
        )


def wetted_interval_deg(
    aim_pan_deg: float,
    range_m: float,
    envelope: SprayEnvelope,
    sigmas: float = 2.0,
) -> tuple[float, float]:
    """The angular interval the spray is expected to wet at this range."""
    half = envelope.wetted_half_angle_deg(range_m, sigmas)
    return (aim_pan_deg - half, aim_pan_deg + half)


def check_deposition(
    aim_pan_deg: float,
    range_m: float,
    keepout: tuple[float, float] | None,
    envelope: SprayEnvelope,
    sigmas: float = 2.0,
) -> Violation | None:
    """Would spraying at this aim wet the protected sector?

    Returns None when it would not — including whenever no sector is configured,
    and (correctly) for a solid stream, whose footprint is a point and so is
    already fully covered by the aim geofence.
    """
    if keepout is None:
        return None
    lo, hi = keepout
    left, right = wetted_interval_deg(aim_pan_deg, range_m, envelope, sigmas)
    if right < lo or left > hi:
        return None

    overlap = min(right, hi) - max(left, lo)
    return Violation(
        sector=keepout,
        aim_pan_deg=aim_pan_deg,
        range_m=range_m,
        half_angle_deg=(right - left) / 2.0,
        overlap_deg=max(0.0, overlap),
    )
