# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""Multi-class droplet trajectories — the ensemble that PRODUCES the envelope.

## Where this runs, and why that answers the objection

At **solution time**, not per tick. That single decision is what makes droplet
physics affordable at all: the ensemble is integrated once when a firing solution
is computed, and its output is the four-number `SprayEnvelope` that `control/`
already consumes from SP5. Control never imports this module — the import
boundary test enforces it — because control does not need to. The ensemble
*produces* what control imports.

## Why per-diameter drag is the whole point

A single drag constant cannot represent a spray. Drag coefficient depends on
Reynolds number, Reynolds number depends on diameter, and a fan's droplets span
more than an order of magnitude in size. So small droplets decelerate fast and
fall short while large ones carry — which is exactly the longitudinal spread the
envelope is trying to describe, and exactly what one constant averages away.

It is also why wind SKEWS a footprint rather than translating it: small droplets
have more drag per unit mass, so they are pushed further downwind than large
ones. A model with one drag constant moves the whole footprint sideways and
cannot produce that asymmetry.

## Evaporation

d²-law with the Ranz-Marshall convective correction. Small droplets have enormous
surface-to-volume ratios and genuinely do shrink over a second of flight, which
shifts the size distribution during the trajectory rather than only at the
nozzle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .dropletdist import DropletClass, diameters, volume_fractions
from .fluid import (
    AIR_DENSITY_REF,
    WATER_DENSITY,
    air_viscosity,
    drag_coefficient,
    reynolds_number,
)

G = 9.81
_DT = 0.002
_MAX_STEPS = 4000

# d^2-law evaporation constant, m^2/s, for water in still air at ~20 C. Small,
# and it matters only for the finest classes — which is the correct behaviour,
# not a reason to drop the term.
EVAPORATION_K = 2.0e-9

# Lateral launch angles sampled across a fan, per size class. The ensemble is the
# outer product of sizes and angles — see `integrate_ensemble` for why coupling
# them one-to-one is wrong.
_FAN_ANGLE_SAMPLES = 5


@dataclass
class EnsembleResult:
    """Where each droplet class landed, and the envelope that summarises them."""

    ranges_m: np.ndarray
    lateral_m: np.ndarray
    volume_fractions: np.ndarray
    flight_times_s: np.ndarray
    diameters_m: np.ndarray

    @property
    def mean_range_m(self) -> float:
        return float(np.average(self.ranges_m, weights=self.volume_fractions))

    @property
    def longitudinal_sigma_m(self) -> float:
        mean = self.mean_range_m
        var = float(np.average((self.ranges_m - mean) ** 2, weights=self.volume_fractions))
        return math.sqrt(max(var, 0.0))

    @property
    def lateral_sigma_m(self) -> float:
        mean = float(np.average(self.lateral_m, weights=self.volume_fractions))
        var = float(np.average((self.lateral_m - mean) ** 2, weights=self.volume_fractions))
        return math.sqrt(max(var, 0.0))

    @property
    def lateral_skew(self) -> float:
        """Third standardised moment of the lateral landing distribution.

        Non-zero means the footprint is SKEWED rather than merely translated —
        the signature of small droplets drifting further downwind than large
        ones, which a single-drag model cannot produce.
        """
        sigma = self.lateral_sigma_m
        if sigma < 1e-9:
            return 0.0
        mean = float(np.average(self.lateral_m, weights=self.volume_fractions))
        third = float(np.average((self.lateral_m - mean) ** 3, weights=self.volume_fractions))
        return third / sigma**3


def integrate_ensemble(
    classes: list[DropletClass],
    speed_ms: float,
    elevation_deg: float,
    nozzle_height_m: float = 0.55,
    fan_half_angle_deg: float = 0.0,
    wind_ms: tuple[float, float] = (0.0, 0.0),
    air_density: float = AIR_DENSITY_REF,
    temperature_c: float = 20.0,
    evaporate: bool = True,
) -> EnsembleResult:
    """Integrate every droplet class to the ground, vectorised across classes.

    `wind_ms` is (downrange, lateral). Drag acts on AIRSPEED, matching the
    single-drag model SP7 corrected — the same physics, applied per class.
    """
    base_d = diameters(classes)
    base_fractions = volume_fractions(classes)
    if base_d.size == 0:
        raise ValueError("no droplet classes to integrate")

    mu_air = air_viscosity(temperature_c)
    theta = math.radians(elevation_deg)

    # Size and launch ANGLE are independent, and getting that wrong is an easy
    # and invisible mistake. Mapping one class to one angle (index order = size
    # order) makes the smallest droplets always launch to the same side, which
    # manufactures a size/direction correlation the nozzle does not have — and
    # then wind appears to REDUCE skew, because it fights the artifact.
    #
    # So the ensemble is the outer product: every size class is launched across
    # every angle, each carrying an equal share of that class's volume.
    if fan_half_angle_deg > 0.0:
        angle_samples = np.linspace(-1.0, 1.0, _FAN_ANGLE_SAMPLES) * math.radians(
            fan_half_angle_deg
        )
    else:
        angle_samples = np.zeros(1)

    d = np.repeat(base_d, angle_samples.size)
    fractions = np.repeat(base_fractions / angle_samples.size, angle_samples.size)
    offsets = np.tile(angle_samples, base_d.size)
    n = d.size

    x = np.zeros(n)          # downrange
    y = np.full(n, nozzle_height_m)
    z = np.zeros(n)          # lateral
    vx = speed_ms * math.cos(theta) * np.cos(offsets)
    vy = np.full(n, speed_ms * math.sin(theta))
    vz = speed_ms * math.cos(theta) * np.sin(offsets)
    diameter = d.copy()

    landed = np.zeros(n, dtype=bool)
    land_x = np.zeros(n)
    land_z = np.zeros(n)
    land_t = np.zeros(n)
    wx, wz = wind_ms

    t = 0.0
    for _ in range(_MAX_STEPS):
        if landed.all():
            break
        rel_x, rel_y, rel_z = vx - wx, vy, vz - wz
        speed_rel = np.sqrt(rel_x**2 + rel_y**2 + rel_z**2) + 1e-12

        # Per-diameter, Reynolds-dependent drag. This is the term a single
        # constant cannot represent.
        re = np.array([
            reynolds_number(air_density, s, dd, mu_air)
            for s, dd in zip(speed_rel, diameter, strict=True)
        ])
        cd = np.array([drag_coefficient(r) for r in re])
        mass = WATER_DENSITY * math.pi / 6.0 * diameter**3
        area = math.pi / 4.0 * diameter**2
        drag = 0.5 * air_density * cd * area * speed_rel / np.maximum(mass, 1e-15)

        ax = -drag * rel_x
        ay = -G - drag * rel_y
        az = -drag * rel_z

        active = ~landed
        vx[active] += ax[active] * _DT
        vy[active] += ay[active] * _DT
        vz[active] += az[active] * _DT
        x[active] += vx[active] * _DT
        y[active] += vy[active] * _DT
        z[active] += vz[active] * _DT

        if evaporate:
            # d^2-law with the Ranz-Marshall convective correction.
            sh = 2.0 + 0.6 * np.sqrt(np.maximum(re, 0.0))
            shrink = EVAPORATION_K * sh / np.maximum(diameter, 1e-9)
            diameter[active] = np.maximum(diameter[active] - shrink[active] * _DT, 1e-7)

        t += _DT
        newly = active & (y <= 0.0)
        if newly.any():
            land_x[newly] = x[newly]
            land_z[newly] = z[newly]
            land_t[newly] = t
            landed |= newly

    # Anything still airborne at the step cap is recorded where it is, rather
    # than dropped — losing a class would silently reweight the distribution.
    still = ~landed
    if still.any():
        land_x[still] = x[still]
        land_z[still] = z[still]
        land_t[still] = t

    return EnsembleResult(
        ranges_m=land_x, lateral_m=land_z, volume_fractions=fractions,
        flight_times_s=land_t, diameters_m=d,
    )
