# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""Explainable target triage.

## What this is, and what it explicitly is not

`tracker.primary()` ranks candidates by `confidence x area`. That is crude — it
ignores growth rate, reachability, and how close a fire is to something the
keep-out sector exists to protect, all of which the system already measures. So
there is real information being thrown away.

What it is **not** is an optimal scheduler. An earlier design justified this
layer with a scheduling theorem; the theorem does not transfer — its assumptions
(known processing times, no preemption cost, independent jobs) are all false
here. Dropping the bad justification is not the same as dropping the feature, so
this ships as **an explainable heuristic with operator-visible reasons** and makes
no optimality claim anywhere.

The reasons are the deliverable as much as the ranking is. Every term is
normalised to 0..1 and individually reportable, so an operator asking "why is it
spraying THAT one" gets an answer per term rather than a number.

## Default off

`FireProfile` leaves triage disabled. The existing CONFIRM -> SEARCH -> next-fire
sequencing is what `report_multi_seed7.json` pins, and switching the order by
default would move that fixture for a change that has not yet been measured
against the incumbent on the `multi` scenario. Turning it on is a deliberate act
until that measurement exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..vision.tracker import Track


@dataclass(frozen=True)
class TriageWeights:
    """Relative importance of each term. Not tuned — chosen to be legible, and
    documented as such. A weight set nobody can explain is worse than a crude
    ranking everybody understands."""

    growth: float = 0.30       # a fire getting worse fast is worth reaching first
    size: float = 0.20         # bigger fires are worth more water
    reachability: float = 0.25 # a fire we cannot reach scores nothing, however bad
    proximity: float = 0.15    # closer fires are engaged sooner, all else equal
    spread_risk: float = 0.10  # spreading TOWARD a keep-out is worse than away

    def total(self) -> float:
        return self.growth + self.size + self.reachability + self.proximity + self.spread_risk


@dataclass
class TriageTerm:
    """One normalised contribution, kept separately so it can be shown."""

    name: str
    value: float      # 0..1
    weight: float

    @property
    def contribution(self) -> float:
        return self.value * self.weight

    def describe(self) -> str:
        return f"{self.name}={self.value:.2f} (x{self.weight:.2f} = {self.contribution:.3f})"


@dataclass
class TriageScore:
    track_id: int
    terms: list[TriageTerm] = field(default_factory=list)

    @property
    def total(self) -> float:
        return sum(t.contribution for t in self.terms)

    def explain(self) -> str:
        """The operator-facing answer to "why that one?" — term by term."""
        parts = ", ".join(t.describe() for t in self.terms)
        return f"track {self.track_id}: score {self.total:.3f} [{parts}]"


def _normalise(value: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return max(0.0, min(1.0, (value - lo) / (hi - lo)))


def score_target(
    track: Track,
    *,
    range_m: float | None,
    growth_rate_per_s: float = 0.0,
    min_reach_m: float = 0.0,
    max_reach_m: float = 10.0,
    keepout: tuple[float, float] | None = None,
    bearing_deg: float | None = None,
    weights: TriageWeights | None = None,
    max_area_px: float = 20_000.0,
) -> TriageScore:
    """Score one candidate. Every term is normalised and individually reported.

    `reachability` is a hard zero outside the jet's envelope rather than a small
    number: a fire that cannot be reached is not a lower priority, it is not a
    candidate at all, and blending it into a weighted sum would let a large
    enough unreachable fire outrank a reachable one.
    """
    w = weights or TriageWeights()
    terms: list[TriageTerm] = []

    terms.append(TriageTerm("growth", _normalise(growth_rate_per_s, 0.0, 0.5), w.growth))
    terms.append(TriageTerm("size", _normalise(track.area, 0.0, max_area_px), w.size))

    if range_m is None:
        reach = 0.5  # unknown range: neither credit nor penalty
    elif min_reach_m <= range_m <= max_reach_m:
        reach = 1.0
    else:
        reach = 0.0
    terms.append(TriageTerm("reachability", reach, w.reachability))

    if range_m is None:
        proximity = 0.5
    else:
        proximity = 1.0 - _normalise(range_m, min_reach_m, max(max_reach_m, min_reach_m + 1e-6))
    terms.append(TriageTerm("proximity", proximity, w.proximity))

    # Spread risk: a fire near a protected sector is worse than one away from it,
    # because the consequence of losing control of it is worse.
    risk = 0.0
    if keepout is not None and bearing_deg is not None:
        lo, hi = keepout
        distance_deg = 0.0 if lo <= bearing_deg <= hi else min(
            abs(bearing_deg - lo), abs(bearing_deg - hi)
        )
        risk = 1.0 - _normalise(distance_deg, 0.0, 45.0)
    terms.append(TriageTerm("spread_risk", risk, w.spread_risk))

    return TriageScore(track_id=track.track_id, terms=terms)


def rank(scores: list[TriageScore]) -> list[TriageScore]:
    """Highest score first. Ties broken by track id so the order is
    deterministic — an unstable ranking would make the golden fixtures a
    function of dict iteration order."""
    return sorted(scores, key=lambda s: (-s.total, s.track_id))


def choose(scores: list[TriageScore]) -> TriageScore | None:
    ordered = rank(scores)
    return ordered[0] if ordered else None
