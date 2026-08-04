# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""SP14 — explainable target triage.

Two properties matter here and neither is "it picks the best target", because
nothing in this module claims to know what best means:

1. **It is explainable.** Every term is reportable, so an operator asking why the
   turret chose that fire gets an answer per term rather than a number.
2. **It is off by default**, so the existing `multi` sequencing — which
   `report_multi_seed7.json` pins — is undisturbed until triage has actually been
   measured against the incumbent `confidence x area` ranking.
"""

from __future__ import annotations

import pytest

from triton.triage.score import (
    TriageWeights,
    choose,
    rank,
    score_target,
)
from triton.vision.tracker import Track


def _track(track_id: int, area: float) -> Track:
    return Track(track_id=track_id, cx=480.0, cy=300.0, area=area, hits=10, confidence=0.8)


# ------------------------------------------------------------------ the terms

def test_every_term_is_normalised_and_reported() -> None:
    score = score_target(_track(1, 5000.0), range_m=7.0, growth_rate_per_s=0.2,
                         min_reach_m=3.0, max_reach_m=9.0)
    names = [t.name for t in score.terms]
    assert names == ["growth", "size", "reachability", "proximity", "spread_risk"]
    assert all(0.0 <= t.value <= 1.0 for t in score.terms)


def test_the_explanation_names_every_term() -> None:
    """The operator surface is the deliverable as much as the ranking is."""
    score = score_target(_track(7, 5000.0), range_m=7.0, growth_rate_per_s=0.2)
    text = score.explain()
    assert "track 7" in text
    for term in ("growth", "size", "reachability", "proximity", "spread_risk"):
        assert term in text


def test_weights_sum_to_one_so_scores_are_comparable() -> None:
    assert TriageWeights().total() == pytest.approx(1.0)


def test_an_unreachable_fire_scores_zero_reachability_not_a_small_number() -> None:
    """A hard zero, deliberately. If reachability merely lowered the score, a
    large enough unreachable fire would outrank a reachable one — and water
    aimed at something out of reach is water wasted, whatever its size."""
    far = score_target(_track(1, 20_000.0), range_m=30.0, min_reach_m=3.0, max_reach_m=9.0)
    reach_term = next(t for t in far.terms if t.name == "reachability")
    assert reach_term.value == 0.0

    near = score_target(_track(2, 100.0), range_m=6.0, min_reach_m=3.0, max_reach_m=9.0)
    assert near.total > far.total, "a tiny reachable fire must outrank a huge unreachable one"


def test_growth_rate_can_invert_the_size_ordering() -> None:
    """The spec's acceptance case: a small fast-growing fire versus a large
    static one. `confidence x area` cannot express this at all."""
    small_growing = score_target(_track(1, 2_000.0), range_m=6.0, growth_rate_per_s=0.5,
                                 min_reach_m=3.0, max_reach_m=9.0)
    large_static = score_target(_track(2, 12_000.0), range_m=6.0, growth_rate_per_s=0.0,
                                min_reach_m=3.0, max_reach_m=9.0)
    assert small_growing.total > large_static.total
    assert choose([large_static, small_growing]).track_id == 1


def test_proximity_prefers_the_nearer_of_two_equals() -> None:
    near = score_target(_track(1, 5_000.0), range_m=4.0, min_reach_m=3.0, max_reach_m=9.0)
    far = score_target(_track(2, 5_000.0), range_m=8.5, min_reach_m=3.0, max_reach_m=9.0)
    assert near.total > far.total


def test_a_fire_near_a_keepout_scores_higher_spread_risk() -> None:
    """Spreading toward something the geofence exists to protect is worse than
    spreading away from it, because losing control of it costs more."""
    near_sector = score_target(_track(1, 5_000.0), range_m=6.0,
                               keepout=(-10.0, 10.0), bearing_deg=12.0)
    away = score_target(_track(2, 5_000.0), range_m=6.0,
                        keepout=(-10.0, 10.0), bearing_deg=70.0)
    risk_near = next(t for t in near_sector.terms if t.name == "spread_risk").value
    risk_away = next(t for t in away.terms if t.name == "spread_risk").value
    assert risk_near > risk_away
    assert risk_away == 0.0


def test_an_unknown_range_is_neither_credited_nor_penalised() -> None:
    unknown = score_target(_track(1, 5_000.0), range_m=None)
    for name in ("reachability", "proximity"):
        assert next(t for t in unknown.terms if t.name == name).value == 0.5


# ------------------------------------------------------------------- ranking

def test_ranking_is_deterministic_on_ties() -> None:
    """An unstable ranking would make fixtures a function of iteration order."""
    a = score_target(_track(2, 5_000.0), range_m=6.0)
    b = score_target(_track(1, 5_000.0), range_m=6.0)
    assert [s.track_id for s in rank([a, b])] == [1, 2]
    assert [s.track_id for s in rank([b, a])] == [1, 2]


def test_choosing_from_nothing_is_none_not_an_error() -> None:
    assert choose([]) is None


# ------------------------------------------------------------- default is off

def test_triage_is_not_wired_into_the_mission() -> None:
    """Default-off, and enforced rather than asserted: `report_multi_seed7.json`
    pins the existing CONFIRM -> SEARCH -> next-fire sequencing, and changing the
    order by default would move that fixture for a change nobody has measured
    against the incumbent yet."""
    import inspect

    from triton.control import mission
    from triton.vision import tracker

    for module in (mission, tracker):
        assert "triage" not in inspect.getsource(module), (
            f"{module.__name__} wires in triage; it must stay opt-in until measured"
        )


def test_no_optimality_is_claimed_anywhere() -> None:
    """The scheduling theorem an earlier design leaned on does not transfer.
    Dropping the bad justification is not the same as dropping the feature — but
    the docs must not carry the claim."""
    from triton.triage import score

    text = (score.__doc__ or "").lower()
    assert "optimal" in text, "the module should address optimality explicitly"
    assert "not" in text
    assert "heuristic" in text
