# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""`SprayEnvelope` — four numbers, and the only spray concept `control/` may see.

The whole spray model reduces, for control purposes, to four quantities:

    equivalent_drag_k     one drag constant standing in for the whole ensemble
    longitudinal_sigma_m  1-sigma spread ALONG the throw, at impact
    lateral_sigma_per_m   1-sigma spread ACROSS the throw, per metre of range
    impact_angle_deg      how steeply the water arrives

Everything else about atomisation — droplet size distributions, breakup length,
evaporation, per-diameter drag — exists to *produce* these four numbers, and runs
at solution time rather than per tick. Control imports the envelope; it never
imports the thing that computed it. That is the boundary, and it is enforced by
test rather than by convention.

## Why lateral spread is per-metre and longitudinal is not

Lateral spread is set by the nozzle's fan ANGLE, so it grows linearly with range:
a 10 degree fan is 0.61 m wide at 7 m and 0.87 m at 10 m. Longitudinal spread is
set by the spread in droplet *ballistics* — different sizes land at different
distances — which does not scale the same way. Storing them in different units is
deliberate; storing both "per metre" would quietly make one of them wrong.

## LEGACY_ENVELOPE

Reproduces today's behaviour exactly: a solid stream modelled as a point. Its
sigmas are zero, so every existing fixture is untouched and every deposition
check on it reduces to the aim itself.

It is also the only answer this module is willing to give for a solid stream —
see `envelope_from_ensemble`. The governing spec justified the point model by
asserting a ~19 m breakup length against a ~7 m throw. That number is not
reproducible: across four standard correlations at a 6 mm orifice the intact
length comes out 0.73-8.42 m against throws of 6.34-10.80 m, so on the published
physics the jet has probably ALREADY broken up when it lands. The point model
survives anyway, for a different and better reason — the ensemble that would
replace it is invalid here in both regimes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SprayEnvelope:
    """The spray's shape, as far as control is concerned."""

    equivalent_drag_k: float = 0.0
    longitudinal_sigma_m: float = 0.0
    lateral_sigma_per_m: float = 0.0
    impact_angle_deg: float = 0.0

    def __post_init__(self) -> None:
        if self.longitudinal_sigma_m < 0 or self.lateral_sigma_per_m < 0:
            raise ValueError("spray spreads cannot be negative")
        if self.equivalent_drag_k < 0:
            raise ValueError("drag cannot be negative")

    # ---------------------------------------------------------------- geometry

    def lateral_sigma_m(self, range_m: float) -> float:
        """1-sigma lateral spread at this range."""
        if range_m < 0:
            raise ValueError("range cannot be negative")
        return self.lateral_sigma_per_m * range_m

    def wetted_half_width_m(self, range_m: float, sigmas: float = 2.0) -> float:
        """Half-width of the wetted footprint, inflated to `sigmas`.

        Two sigma by default — deliberately not one. This number feeds a
        keep-out decision, and the point of the exercise is to bound where water
        *can* land, not where it typically does.
        """
        if sigmas < 0:
            raise ValueError("sigmas cannot be negative")
        return self.lateral_sigma_m(range_m) * sigmas

    def wetted_half_angle_deg(self, range_m: float, sigmas: float = 2.0) -> float:
        """The same half-width as an ANGLE, which is the unit the keep-out sector
        is expressed in. Zero at zero range: a footprint at the nozzle has no
        angular extent to speak of."""
        if range_m <= 0:
            return 0.0
        return math.degrees(math.atan(self.wetted_half_width_m(range_m, sigmas) / range_m))

    @property
    def is_point(self) -> bool:
        """True for a solid stream, where the footprint is a point and every
        deposition check reduces to the aim itself."""
        return self.lateral_sigma_per_m == 0.0 and self.longitudinal_sigma_m == 0.0


# Today's behaviour, exactly: a solid stream treated as a point impact.
LEGACY_ENVELOPE = SprayEnvelope()


def envelope_from_ensemble(
    nozzle,
    speed_ms: float,
    elevation_deg: float,
    target_range_m: float,
    wind_ms: tuple[float, float] = (0.0, 0.0),
    n_classes: int = 7,
) -> SprayEnvelope:
    """Compute the four numbers by integrating a droplet ensemble.

    **This is the direction of the dependency**: the ensemble PRODUCES the
    envelope that `control/` consumes. Control never imports the ensemble, which
    keeps a per-droplet integration out of a 33 ms loop — this runs once per
    firing solution.

    ## A solid stream short-circuits unconditionally, and MUST

    A solid stream returns `LEGACY_ENVELOPE` without integrating anything. That
    is not an optimisation — it is a refusal, and the body says why in detail.
    The short version: a droplet ensemble models spheres with sphere drag, and a
    coherent jet is not a sphere, while a broken-up jet does not start at the
    orifice. Neither regime is one this function can integrate honestly.

    An earlier version made the refusal conditional on a breakup-length
    correlation, so a stream judged "already atomised" would fall through to the
    integration. SP5b showed that correlation carries an order of magnitude of
    spread between published forms, which made a safety-relevant branch turn on
    the least certain number in the module. The branch is now unconditional and
    the correlation informs nothing but advisory text.

    `target_range_m` is the range the CONTROLLER planned for, from the calibrated
    ballistics model — the ensemble's own predicted range comes from a drag model
    the caller is not using.
    """
    from .arcs import integrate_ensemble
    from .dropletdist import equal_volume_classes
    from .nozzle import SprayPattern

    if nozzle.pattern is SprayPattern.SOLID_STREAM:
        # A droplet ensemble cannot model a solid stream in EITHER regime, so it
        # returns the legacy point envelope rather than a number it cannot stand
        # behind. Measured, both halves:
        #
        #  - Before breakup the jet is a coherent column, not a sphere. Sphere
        #    drag gives k = 0.0675 against the jet's fitted 0.160 — 2.4x too
        #    little — and an arc integrated with it lands at 13.7 m where the
        #    calibrated model says 8.8 m.
        #  - After breakup the droplets do not start at the nozzle; they inherit
        #    the jet's state wherever it disintegrated. Integrating them from the
        #    orifice, as this ensemble does, is the wrong initial condition.
        #
        # An earlier version returned a "small but non-zero" envelope for the
        # marginal case. That number was resting on the first error above, so it
        # was precision without accuracy — worse than declining to answer.
        return LEGACY_ENVELOPE

    classes = equal_volume_classes(
        nozzle.d63_m(speed_ms), nozzle.spread_exponent(), n_classes
    )
    fan_half = nozzle.pattern.nominal_fan_angle_deg / 2.0
    result = integrate_ensemble(
        classes, speed_ms, elevation_deg,
        fan_half_angle_deg=fan_half, wind_ms=wind_ms,
    )
    # Lateral spread is normalised against the range the CONTROLLER expects, not
    # the ensemble's own — for the same reason as the coherence test above.
    return SprayEnvelope(
        equivalent_drag_k=0.0,
        longitudinal_sigma_m=result.longitudinal_sigma_m,
        lateral_sigma_per_m=result.lateral_sigma_m / max(target_range_m, 1e-6),
        impact_angle_deg=abs(elevation_deg),
    )


def fan_envelope(fan_angle_deg: float, longitudinal_sigma_m: float = 0.15,
                 impact_angle_deg: float = 30.0) -> SprayEnvelope:
    """An envelope for a flat-fan or cone nozzle of a given full fan angle.

    The fan's half-angle is taken as roughly 2 sigma of the lateral distribution:
    the edges of a fan are its tails, not a hard boundary, so equating the
    nominal fan angle with the 2-sigma width is the honest reading rather than
    treating it as a wall the water never crosses.
    """
    if fan_angle_deg <= 0:
        raise ValueError("fan angle must be positive")
    half_angle_rad = math.radians(fan_angle_deg / 2.0)
    lateral_per_m_at_2sigma = math.tan(half_angle_rad)
    return SprayEnvelope(
        equivalent_drag_k=0.0,
        longitudinal_sigma_m=longitudinal_sigma_m,
        lateral_sigma_per_m=lateral_per_m_at_2sigma / 2.0,
        impact_angle_deg=impact_angle_deg,
    )
