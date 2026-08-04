# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""Wash-down / dust suppression — the profile that actually stress-tests the core.

A second fire-like profile would prove nothing: it would exercise the same
closed-loop point-targeting path and agree with it by construction. Wash-down is
chosen precisely because it does NOT work that way.

    fire        closed-loop servo onto a detected point target, stop when out
    wash-down   open-loop raster coverage of an AREA, stop when covered

If the core can express both, the abstraction is real. If it can only express
the first with the second bolted on beside it, the abstraction was a rename.

It also has no regulatory exposure, which agricultural spraying would (see the
spec's §7.2 on why the dose-ledger idea is refused rather than deferred).

Coverage is tracked as swept azimuth rather than as elapsed time, because time is
a proxy that stops being true the moment the pan rate changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..spray.envelope import fan_envelope
from .base import MissionProfile, SafetyConstraints


@dataclass
class SweepEngagement:
    """Raster coverage: sweep the sector, count what has been wetted."""

    sector_deg: tuple[float, float] = (-30.0, 30.0)
    swath_deg: float = 6.0
    name: str = "washdown_sweep"
    _covered: set[int] = field(default_factory=set)

    def setpoint_offset_px(self, target) -> tuple[float, float]:
        # Area coverage has no point target to be offset from; the sweep decides
        # the aim directly. Zero keeps the servo law well-defined if a target
        # does happen to be supplied.
        return (0.0, 0.0)

    # ------------------------------------------------------------- coverage

    def swath_index(self, pan_deg: float) -> int:
        return int((pan_deg - self.sector_deg[0]) // self.swath_deg)

    def total_swaths(self) -> int:
        span = self.sector_deg[1] - self.sector_deg[0]
        return max(1, int(round(span / self.swath_deg)))

    def note_wetted(self, pan_deg: float) -> None:
        lo, hi = self.sector_deg
        if lo <= pan_deg <= hi:
            self._covered.add(self.swath_index(pan_deg))

    def coverage_fraction(self) -> float:
        return len(self._covered) / self.total_swaths()

    def is_complete(self, target, elapsed_s: float) -> bool:
        return self.coverage_fraction() >= 1.0

    def next_aim_deg(self) -> float | None:
        """The centre of the first swath not yet wetted, or None when done."""
        for i in range(self.total_swaths()):
            if i not in self._covered:
                return self.sector_deg[0] + (i + 0.5) * self.swath_deg
        return None

    def reset(self) -> None:
        self._covered.clear()


def WashdownProfile(  # noqa: N802 - reads as a constructor
    sector_deg: tuple[float, float] = (-30.0, 30.0),
    fan_angle_deg: float = 25.0,
) -> MissionProfile:
    """Dust suppression / wash-down over a sector.

    A wide fan, which is also what makes SP5's deposition keep-out load-bearing
    rather than academic: with a 25 degree fan the wetted footprint is far wider
    than the aim, so "the aim is outside the sector" stops implying "the water
    is".
    """
    return MissionProfile(
        name="washdown",
        policy=SweepEngagement(sector_deg=sector_deg),
        spray_envelope=fan_envelope(fan_angle_deg),
        # Narrower than the fire profile: area coverage runs at modest pressure,
        # and a profile may only ever narrow.
        constraints=SafetyConstraints(max_pump_pct=60.0, require_armed=True),
        advisory_vocabulary=("TRAVERSE", "SAFETY"),
        nozzle="flat_fan",
    )
