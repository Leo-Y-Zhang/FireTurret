# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""SP11 — the self-calibrating ballistics learner.

The tests that matter here are the ones asserting the learner CANNOT do damage.
An adaptive layer on a safety-relevant machine earns its place by being provably
bounded, not by converging nicely — so the poisoned-learner test, the
authority-box tests and the "a deleted calibration file changes nothing" test are
the load-bearing ones. Convergence is the easy part.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from triton.adapt.ballistics_learner import (
    HOLDOUT_EVERY,
    MAX_LENGTHEN_PCT,
    MAX_SHORTEN_PCT,
    BallisticsLearner,
    apply_to_opening,
    basis,
)
from triton.adapt.rls import DEFAULT_TRACE_BOUND, RecursiveLeastSquares

# ------------------------------------------------------------------- the RLS

def test_rls_converges_on_a_clean_linear_relationship() -> None:
    rls = RecursiveLeastSquares(n_terms=2)
    truth = np.array([3.0, -1.5])
    rng = np.random.default_rng(0)
    for _ in range(300):
        phi = np.array([1.0, rng.uniform(-1.0, 1.0)])
        rls.update(phi, float(phi @ truth))
    assert rls.theta == pytest.approx(truth, abs=0.1)


def test_the_covariance_trace_is_hard_bounded() -> None:
    """**The important guard.** With forgetting < 1 and no persistent excitation,
    P grows without bound — and P is the gain on new evidence, so an inflated P
    means the next noisy datum swings theta arbitrarily. A well-converged turret
    sitting on target is exactly in that state."""
    rls = RecursiveLeastSquares(n_terms=2, forgetting=0.9, trace_bound=DEFAULT_TRACE_BOUND)
    for _ in range(5000):
        rls.update(np.array([1.0, 0.0]), 1.0)  # no excitation in the second term
    assert float(np.trace(rls.P)) <= DEFAULT_TRACE_BOUND + 1e-6


def test_an_unexcited_filter_does_not_become_infinitely_credulous() -> None:
    """The failure the trace bound prevents, exercised end to end: after a long
    quiet period, one wild observation must not move theta arbitrarily far."""
    rls = RecursiveLeastSquares(n_terms=2, forgetting=0.9)
    for _ in range(3000):
        rls.update(np.array([1.0, 0.0]), 1.0)
    before = rls.theta.copy()
    rls.update(np.array([1.0, 0.0]), 10_000.0)
    assert np.linalg.norm(rls.theta - before) < 50.0, "one bad datum swung the model"


def test_the_huber_gate_caps_a_single_outlier() -> None:
    """Squared error weights an outlier quadratically, so one reflection or
    detector glitch moves theta more than a hundred good observations."""
    clean = RecursiveLeastSquares(n_terms=1)
    for _ in range(50):
        clean.update(np.array([1.0]), 5.0)
    settled = clean.theta.copy()

    accepted = clean.update(np.array([1.0]), 500.0)
    assert accepted is False, "the outlier was not gated"
    assert abs(clean.theta[0] - settled[0]) < 5.0


def test_non_finite_input_is_rejected_not_absorbed() -> None:
    rls = RecursiveLeastSquares(n_terms=2)
    assert rls.update(np.array([1.0, float("nan")]), 1.0) is False
    assert rls.update(np.array([1.0, 0.0]), float("inf")) is False
    assert np.all(np.isfinite(rls.theta))


@pytest.mark.parametrize("kwargs", [
    {"n_terms": 0}, {"n_terms": 2, "forgetting": 0.0},
    {"n_terms": 2, "forgetting": 1.5}, {"n_terms": 2, "trace_bound": 0.0},
])
def test_degenerate_configuration_is_rejected(kwargs) -> None:
    with pytest.raises(ValueError):
        RecursiveLeastSquares(**kwargs)


# ------------------------------------------------------------- the two terms

def test_the_basis_has_exactly_two_terms() -> None:
    """Four of the six an obvious design would use are NOT identifiable from data
    this system can generate, and fitting unidentifiable parameters produces
    confident nonsense rather than weak estimates."""
    assert basis(7.0).shape == (2,)
    assert basis(7.0)[1] == pytest.approx(0.0)  # ln(r/r_ref) = 0 at r_ref
    assert basis(14.0)[1] == pytest.approx(math.log(2.0))


def test_the_basis_survives_a_degenerate_range() -> None:
    assert np.all(np.isfinite(basis(0.0)))
    assert np.all(np.isfinite(basis(-5.0)))


# ------------------------------------------------- THE authority box

def test_the_learner_may_shorten_freely() -> None:
    learner = BallisticsLearner()
    learner.rls.theta = np.array([-50.0, 0.0])  # wants a huge reduction
    learner.shots = 1  # not a holdout
    assert learner.correction_pct(7.0) == pytest.approx(-MAX_SHORTEN_PCT)


def test_the_learner_may_lengthen_only_marginally_until_trusted() -> None:
    """**Invariant 8 protected.** Short is recoverable; long lands behind the
    flame, occluded, starving the servo of feedback exactly when it has
    overshot."""
    learner = BallisticsLearner()
    learner.rls.theta = np.array([+50.0, 0.0])
    learner.shots = 1
    assert learner.rls.confident is False
    assert learner.correction_pct(7.0) == pytest.approx(MAX_LENGTHEN_PCT)
    assert MAX_LENGTHEN_PCT < MAX_SHORTEN_PCT, "authority must be asymmetric"


def test_trust_requires_excitation_across_the_basis() -> None:
    """Observing the same range sixty times teaches nothing about the RANGE term,
    so it must not buy trust. Confidence in a direction nothing has measured is
    exactly the failure the covariance bound exists to prevent."""
    learner = BallisticsLearner()
    for _ in range(60):
        learner.observe(7.0, 2.0)  # r == r_ref, so the second term is never excited
    assert learner.rls.confident is False


def test_a_trusted_learner_may_lengthen_further_but_still_less_than_it_may_shorten() -> None:
    learner = BallisticsLearner()
    learner.shots = 1
    for r in (4.0, 5.5, 7.0, 8.5, 10.0) * 20:  # excite BOTH terms
        learner.observe(r, 2.0)
    assert learner.rls.confident is True
    learner.rls.theta = np.array([+50.0, 0.0])
    correction = learner.correction_pct(7.0)
    assert correction > MAX_LENGTHEN_PCT
    assert correction < MAX_SHORTEN_PCT


def test_the_open_short_subtraction_is_applied_AFTER_the_learner() -> None:
    """Order is the point: the other way round, a learned correction could cancel
    the open-short bias, and invariant 8 exists because a long shot goes blind."""
    learner = BallisticsLearner()
    learner.shots = 1
    learner.rls.theta = np.array([+50.0, 0.0])  # wants maximum lengthening

    with_learner = apply_to_opening(60.0, learner, 7.0, open_short_pct=12.0, min_pump_pct=20.0)
    nominal = apply_to_opening(60.0, None, 7.0, open_short_pct=12.0, min_pump_pct=20.0)
    assert with_learner > nominal, "the learner had no effect at all"
    assert with_learner <= 60.0 + MAX_LENGTHEN_PCT - 12.0 + 1e-9, (
        "the learner bypassed the open-short subtraction"
    )


def test_the_opening_is_clamped_to_the_pump_envelope() -> None:
    learner = BallisticsLearner()
    learner.shots = 1
    learner.rls.theta = np.array([-500.0, 0.0])
    assert apply_to_opening(30.0, learner, 7.0, 12.0, min_pump_pct=25.0) == 25.0


# ----------------------------------------------------- THE poisoned learner

def test_a_poisoned_learner_cannot_produce_an_illegal_command() -> None:
    """The spec's acceptance test. A protocol-conforming learner returning absurd
    values must not be able to command anything outside the envelope — because
    the authority box is enforced at the point of use, not trusted to the
    learner."""
    class Poisoned(BallisticsLearner):
        def correction_pct(self, range_m: float) -> float:
            return 10_000.0

    poisoned = Poisoned()
    for nominal in (25.0, 55.0, 100.0):
        for r in (3.0, 7.0, 12.0):
            pump = apply_to_opening(nominal, poisoned, r, 12.0, min_pump_pct=20.0)
            assert 20.0 <= pump <= 100.0, f"poisoned learner produced pump {pump}"


def test_a_learner_returning_nonsense_degrades_to_nominal() -> None:
    learner = BallisticsLearner()
    learner.shots = 1
    learner.rls.theta = np.array([float("nan"), float("inf")])
    assert learner.correction_pct(7.0) == 0.0


def test_a_deleted_calibration_file_gives_exactly_nominal_behaviour() -> None:
    """**The property that makes this safe to ship.** A missing, stale, corrupt or
    wrong learner must degrade to today's behaviour — not to something new."""
    nominal = apply_to_opening(55.0, None, 7.0, 12.0, 20.0)
    fresh = apply_to_opening(55.0, BallisticsLearner(), 7.0, 12.0, 20.0)
    assert fresh == nominal


@pytest.mark.parametrize("corrupt", [
    {}, {"theta": [1.0]}, {"theta": "nonsense"},
    {"theta": [float("nan"), 0.0]}, {"theta": [1.0, 2.0], "r_ref_m": 99.0},
])
def test_a_corrupt_calibration_file_degrades_to_fresh(corrupt) -> None:
    """A corrupt calibration file costs today's behaviour, not the turret's
    ability to start."""
    learner = BallisticsLearner.from_dict(corrupt)
    assert learner.rls.updates == 0
    assert np.all(learner.rls.theta == 0.0)


def test_a_calibration_round_trips() -> None:
    learner = BallisticsLearner()
    for _ in range(30):
        learner.observe(7.0, 4.0)
    restored = BallisticsLearner.from_dict(learner.to_dict())
    assert restored.rls.theta == pytest.approx(learner.rls.theta)


# ------------------------------------------------------------- the holdout

def test_every_fifth_opening_shot_ignores_the_learner() -> None:
    """One mechanism, three purposes: persistent excitation (without which the
    covariance guard is doing all the work), an unbiased baseline for detecting
    divergence, and a number with a control group behind it."""
    learner = BallisticsLearner()
    learner.rls.theta = np.array([-8.0, 0.0])

    holdouts = []
    for _ in range(20):
        holdouts.append(learner.is_holdout())
        learner.note_shot()

    assert sum(holdouts) == 20 // HOLDOUT_EVERY
    assert holdouts[0] is True and holdouts[1] is False


def test_a_holdout_shot_uses_the_nominal_solution_exactly() -> None:
    learner = BallisticsLearner()
    learner.rls.theta = np.array([-8.0, 0.0])
    learner.shots = 0  # a holdout
    assert learner.correction_pct(7.0) == 0.0
    assert apply_to_opening(55.0, learner, 7.0, 12.0, 20.0) == apply_to_opening(
        55.0, None, 7.0, 12.0, 20.0
    )


# --------------------------------------------------------------- convergence

def test_the_learner_reduces_a_consistent_opening_error() -> None:
    """The easy part, and last for a reason. Convergence is worth nothing if the
    bounds above do not hold.

    A SHORTENING error, deliberately: that is the direction the learner has full
    authority in, so convergence is observable without the authority box
    truncating it. A +6% lengthening error would legitimately be capped at +3%
    until trust is earned, which is the design working rather than failing.
    """
    learner = BallisticsLearner()
    learner.shots = 1  # off the holdout cycle
    true_error_pct = -6.0
    for _ in range(60):
        learner.observe(7.0, true_error_pct)
    assert learner.correction_pct(7.0) == pytest.approx(true_error_pct, abs=1.5)


def test_a_lengthening_error_is_truncated_by_the_authority_box() -> None:
    """The same convergence, in the direction the box restricts — and it must be
    truncated, because invariant 8 says a long shot goes blind."""
    learner = BallisticsLearner()
    learner.shots = 1
    for _ in range(60):
        learner.observe(7.0, +6.0)
    assert learner.correction_pct(7.0) == pytest.approx(MAX_LENGTHEN_PCT)


def test_the_learner_captures_range_dependence() -> None:
    learner = BallisticsLearner()
    learner.shots = 1
    for r in (4.0, 5.5, 7.0, 8.5, 10.0) * 25:
        learner.observe(r, 2.0 + 3.0 * math.log(r / 7.0))
    assert learner.correction_pct(10.0) > learner.correction_pct(4.0)


# ------------------------------------------------ the boundary it must not cross

def test_the_learner_is_not_reachable_from_the_suppression_law() -> None:
    """It writes ONLY to the opening solution. A learner with write access to the
    closed loop could move the loop's fixed point, and then a wrong model
    converges confidently to the wrong place."""
    import inspect

    from triton.control import suppress

    source = inspect.getsource(suppress)
    for forbidden in ("learner", "BallisticsLearner", "adapt"):
        assert forbidden not in source, (
            f"control/suppress.py references {forbidden!r}; the learner must never "
            "reach the closed-loop correction law"
        )
