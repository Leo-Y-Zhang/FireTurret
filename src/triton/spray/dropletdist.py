# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""Droplet size distribution, discretised by equal VOLUME.

## Rosin-Rammler

The standard sprays distribution: the volume fraction of droplets smaller than
`d` is `1 - exp(-(d/d63)^n)`. Two parameters — a characteristic diameter and a
spread exponent — and it fits real nozzle data well enough to be the industry
default.

## Equal-volume quadrature, not equal-diameter

The discretisation matters more than it looks. Splitting the distribution into
equal-DIAMETER bins puts almost all the water in the last one or two bins, so
most of the computational classes carry almost no water and the classes that
matter are under-resolved.

Splitting by equal VOLUME gives every class the same mass of water, so every
class is equally worth integrating. Seven classes is enough to represent a fan's
spread; more costs time and buys very little, and this runs at solution time so
the budget is real but not tight.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

DEFAULT_CLASSES = 7


@dataclass(frozen=True)
class DropletClass:
    """One representative droplet size, carrying an equal share of the volume."""

    diameter_m: float
    volume_fraction: float


def rosin_rammler_cdf(diameter_m: float, d63_m: float, spread: float) -> float:
    """Volume fraction below `diameter_m`."""
    if d63_m <= 0 or spread <= 0:
        raise ValueError("d63 and spread must be positive")
    if diameter_m <= 0:
        return 0.0
    return 1.0 - math.exp(-((diameter_m / d63_m) ** spread))


def rosin_rammler_inverse(fraction: float, d63_m: float, spread: float) -> float:
    """The diameter below which `fraction` of the VOLUME lies."""
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must be within (0, 1)")
    return d63_m * (-math.log(1.0 - fraction)) ** (1.0 / spread)


def equal_volume_classes(
    d63_m: float, spread: float, n_classes: int = DEFAULT_CLASSES
) -> list[DropletClass]:
    """Discretise the distribution into `n_classes` of EQUAL volume.

    Each class's representative diameter is taken at the midpoint of its volume
    interval, so the class stands for the water it actually carries rather than
    for a range of sizes weighted by count.
    """
    if n_classes < 1:
        raise ValueError("need at least one droplet class")
    share = 1.0 / n_classes
    classes: list[DropletClass] = []
    for i in range(n_classes):
        mid_fraction = (i + 0.5) * share
        classes.append(
            DropletClass(
                diameter_m=rosin_rammler_inverse(mid_fraction, d63_m, spread),
                volume_fraction=share,
            )
        )
    return classes


def sauter_mean_diameter(classes: list[DropletClass]) -> float:
    """D32 — the diameter with the same volume-to-surface ratio as the spray.

    The number that governs evaporation and heat transfer, and therefore the one
    worth reporting when a distribution is summarised to a single figure.
    """
    if not classes:
        raise ValueError("no droplet classes")
    volume = sum(c.volume_fraction for c in classes)
    # volume fraction / diameter is proportional to surface area
    surface = sum(c.volume_fraction / c.diameter_m for c in classes)
    return volume / surface if surface > 0 else 0.0


def diameters(classes: list[DropletClass]) -> np.ndarray:
    return np.array([c.diameter_m for c in classes], dtype=float)


def volume_fractions(classes: list[DropletClass]) -> np.ndarray:
    return np.array([c.volume_fraction for c in classes], dtype=float)
