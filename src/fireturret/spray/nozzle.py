# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""Nozzle patterns, and whether the jet has broken up at all by the time it lands.

## The result that makes SP5b honest about the flagship

For the fire profile's **solid stream** at a 6 mm orifice, the computed breakup
length is around 19 m against a throw of about 7 m. The jet has not broken up
when it lands. Every droplet class is the same, the ensemble is degenerate, and
`LEGACY_ENVELOPE` — a point footprint — is the CORRECT answer rather than a
simplification.

That is worth having computed rather than assumed, because it is the load-bearing
justification for the fire profile being unaffected by all of this. And it is
asserted by test: the ensemble must *reproduce* the legacy envelope, not be
excused from matching it.

For a **flat fan** or **fog** the same machinery gives a genuinely different
answer, which is what the wash-down profile needs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from .fluid import (
    WATER_DENSITY,
    ohnesorge_number,
    surface_tension,
    water_viscosity,
    weber_number,
)


class SprayPattern(Enum):
    SOLID_STREAM = "solid_stream"
    FLAT_FAN = "flat_fan"
    FULL_CONE = "full_cone"
    HOLLOW_CONE = "hollow_cone"
    FOG = "fog"

    @property
    def nominal_fan_angle_deg(self) -> float:
        return {
            SprayPattern.SOLID_STREAM: 0.0,
            SprayPattern.FLAT_FAN: 25.0,
            SprayPattern.FULL_CONE: 45.0,
            SprayPattern.HOLLOW_CONE: 60.0,
            SprayPattern.FOG: 90.0,
        }[self]

    @property
    def atomises_immediately(self) -> bool:
        """Fan, cone and fog nozzles atomise at the orifice by design. A solid
        stream does not — that is what makes it a stream."""
        return self is not SprayPattern.SOLID_STREAM


@dataclass(frozen=True)
class Nozzle:
    orifice_m: float = 0.006
    pattern: SprayPattern = SprayPattern.SOLID_STREAM
    temperature_c: float = 20.0

    def __post_init__(self) -> None:
        if self.orifice_m <= 0:
            raise ValueError("orifice diameter must be positive")

    # ------------------------------------------------------------- numbers

    def weber(self, velocity_ms: float) -> float:
        return weber_number(
            WATER_DENSITY, velocity_ms, self.orifice_m, surface_tension(self.temperature_c)
        )

    def ohnesorge(self) -> float:
        return ohnesorge_number(
            water_viscosity(self.temperature_c), WATER_DENSITY, self.orifice_m,
            surface_tension(self.temperature_c),
        )

    def breakup_length_m(self, velocity_ms: float) -> float:
        """Distance before a coherent jet disintegrates.

        Classical correlation for the second wind-induced regime:
        `L/d ~ 6 (We)^0.5 / (1 + 3 Oh)`. A fan or fog nozzle atomises at the
        orifice by design, so its breakup length is effectively zero.
        """
        if self.pattern.atomises_immediately:
            return 0.0
        we = self.weber(velocity_ms)
        oh = self.ohnesorge()
        return self.orifice_m * 6.0 * math.sqrt(max(we, 0.0)) / (1.0 + 3.0 * oh)

    def is_coherent_at(self, velocity_ms: float, range_m: float) -> bool:
        """Is the jet still a stream when it gets there?

        For the flagship at 6 mm this is True across the whole usable envelope,
        which is why the fire profile's point footprint is right.
        """
        return self.breakup_length_m(velocity_ms) >= range_m

    def sauter_mean_diameter_m(self, velocity_ms: float) -> float:
        """Representative droplet size after atomisation.

        For a coherent stream this is the orifice itself — the "droplet" is the
        jet. For an atomising pattern, a Weber-scaled correlation: higher
        velocity means finer droplets, which is why a fog nozzle at pressure
        produces mist and the same nozzle at a dribble produces drips.
        """
        if not self.pattern.atomises_immediately:
            return self.orifice_m
        we = max(self.weber(velocity_ms), 1.0)
        return self.orifice_m * 3.0 * we**-0.4

    def d63_m(self, velocity_ms: float) -> float:
        """Rosin-Rammler characteristic diameter. ~1.2 x SMD for typical sprays."""
        return self.sauter_mean_diameter_m(velocity_ms) * 1.2

    def spread_exponent(self) -> float:
        """Rosin-Rammler `n`. Higher means a narrower size distribution — a
        coherent stream is (by definition) perfectly uniform."""
        if not self.pattern.atomises_immediately:
            return 20.0  # effectively monodisperse
        return {
            SprayPattern.FLAT_FAN: 2.6,
            SprayPattern.FULL_CONE: 2.2,
            SprayPattern.HOLLOW_CONE: 2.4,
            SprayPattern.FOG: 1.8,
        }[self.pattern]
