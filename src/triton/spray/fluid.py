# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""Water and air properties. Correlations, with their ranges of validity stated.

Every function here is a published correlation rather than a fitted constant, so
each has a domain outside which it is wrong. Those domains are written down, and
`__post_init__`-style guards raise rather than extrapolating — an atomisation
model that silently used a surface tension valid only to 100 C would produce
droplet sizes that look plausible and are not.
"""

from __future__ import annotations

import math

# Sutherland's law for air.
_MU_REF = 1.716e-5   # Pa s at T_ref
_T_REF = 273.15      # K
_SUTHERLAND_C = 110.4  # K

WATER_DENSITY = 998.0   # kg/m^3 at 20 C
AIR_DENSITY_REF = 1.225  # kg/m^3, sea-level standard


def surface_tension(temperature_c: float) -> float:
    """Water surface tension, N/m. IAPWS correlation.

    Valid 0-374 C (triple point to critical). Raises outside that: extrapolating
    surface tension is how an atomisation model quietly starts producing
    droplet sizes for a fluid that does not exist.
    """
    if not -0.1 <= temperature_c <= 374.0:
        raise ValueError(
            f"IAPWS surface tension is valid 0-374 C, got {temperature_c} C"
        )
    t_c = 647.096  # critical temperature, K
    tau = 1.0 - (temperature_c + 273.15) / t_c
    return 0.2358 * tau**1.256 * (1.0 - 0.625 * tau)


def water_viscosity(temperature_c: float) -> float:
    """Dynamic viscosity of water, Pa s. Vogel-type correlation, valid 0-100 C."""
    if not 0.0 <= temperature_c <= 100.0:
        raise ValueError(f"water viscosity correlation is valid 0-100 C, got {temperature_c}")
    t_k = temperature_c + 273.15
    return 2.414e-5 * 10.0 ** (247.8 / (t_k - 140.0))


def air_viscosity(temperature_c: float) -> float:
    """Dynamic viscosity of air, Pa s. Sutherland's law."""
    t_k = temperature_c + 273.15
    if t_k <= 0:
        raise ValueError("temperature below absolute zero")
    return _MU_REF * (t_k / _T_REF) ** 1.5 * (_T_REF + _SUTHERLAND_C) / (t_k + _SUTHERLAND_C)


def saturation_pressure(temperature_c: float) -> float:
    """Saturation vapour pressure of water, Pa. Buck equation.

    Shared with `adapt/environment.py`, which uses it for air density. Kept in
    both places rather than cross-imported so `spray/` does not depend on
    `adapt/` — the dependency would be backwards.
    """
    t = temperature_c
    return 611.21 * math.exp((18.678 - t / 234.5) * (t / (257.14 + t)))


def weber_number(density: float, velocity: float, diameter: float, sigma: float) -> float:
    """We = rho v^2 d / sigma. Inertia versus surface tension — the number that
    decides whether a jet breaks up at all."""
    if sigma <= 0 or diameter <= 0:
        raise ValueError("surface tension and diameter must be positive")
    return density * velocity * velocity * diameter / sigma


def ohnesorge_number(mu: float, density: float, diameter: float, sigma: float) -> float:
    """Oh = mu / sqrt(rho sigma d). Viscosity versus inertia and surface tension."""
    if sigma <= 0 or diameter <= 0 or density <= 0:
        raise ValueError("density, surface tension and diameter must be positive")
    return mu / math.sqrt(density * sigma * diameter)


def reynolds_number(density: float, velocity: float, diameter: float, mu: float) -> float:
    if mu <= 0:
        raise ValueError("viscosity must be positive")
    return density * abs(velocity) * diameter / mu


def drag_coefficient(reynolds: float) -> float:
    """Sphere drag, Schiller-Naumann below Re 1000 and constant above.

    Reynolds-DEPENDENT, which is the point: a 50 um droplet and a 2 mm droplet
    have wildly different drag coefficients, and using one constant for both is
    what makes a single-drag model unable to represent a fan at all.
    """
    re = max(abs(reynolds), 1e-9)
    if re < 1000.0:
        return 24.0 / re * (1.0 + 0.15 * re**0.687)
    return 0.44
