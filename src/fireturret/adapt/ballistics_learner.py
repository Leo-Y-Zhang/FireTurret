# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""Learning the opening shot — with two terms, and asymmetric authority.

## Two terms, not six

The obvious basis has a term for everything that might matter: constant, range,
tilt, pump, wind, temperature. Four of those **cannot be identified from data
this system can generate**, and fitting unidentifiable parameters does not
produce weak estimates — it produces confident nonsense, because the fit will
happily attribute the residual to whichever unexcited direction is cheapest.

    tilt      invariant 5 holds tilt FIXED during suppression, so a tilt term is
              never excited. Not weakly excited — never.
    pump      pump and ln(range) are near-collinear, because pump IS the range
              control. Two collinear regressors share a residual arbitrarily.
    wind      without an anemometer, wind is not separable from the constant
              (see adapt/wind.py — the same identifiability problem, stated
              there in full).

So the basis is `[1, ln(r / r_ref)]`. A constant, and a range-dependence. That is
what the data supports, and `fit_jet`'s identifiability check is what will say
when a third term becomes real.

## What it is allowed to touch

**Only the opening solution and a pan pre-offset.** It never touches
`control/suppress.py` — the closed-loop correction law — and the import-boundary
test enforces that. The reason is the failure mode: a learner with write access
to the closed loop can move the loop's fixed point, and then a wrong model
converges confidently to the wrong place. A learner that only sets the OPENING
shot can be arbitrarily wrong and the closed loop still walks the water onto
target; it just takes longer.

A deleted, stale, wrong or actively poisoned learner therefore degrades to
exactly today's behaviour. That is asserted by test.

## Asymmetric authority — invariant 8, protected

The learner may **shorten** the opening freely within its box, and may
**lengthen** it only marginally until trust is high. That asymmetry is not
caution for its own sake: invariant 8 says a long shot lands behind the flame,
occluded, starving the servo of the feedback it needs exactly when it has
overshot. Short is recoverable; long is blind.

And the existing `-12 %` open-short subtraction is applied **after** the learner,
so no learned correction can bypass it.

## The holdout probe

Every fifth opening shot uses the **nominal** solver instead of the learned one.
One mechanism, three purposes: it supplies persistent excitation (without which
the RLS covariance guard is doing all the work), it gives an unbiased baseline
for detecting divergence, and it is a number an operator can be shown —
"learned openings are landing 40 % closer than nominal" is a claim with a control
group behind it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .rls import RecursiveLeastSquares

# Reference range for the log basis. Any positive constant works; this one keeps
# ln(r/r_ref) near zero across the usable envelope, so the constant term carries
# the bulk and the range term is a correction rather than a competitor.
R_REF_M = 7.0

# How far the learner may move the opening pump command, in percent.
MAX_SHORTEN_PCT = 12.0   # freely, within the box
MAX_LENGTHEN_PCT = 3.0   # only marginally, until trust is high
MAX_LENGTHEN_TRUSTED_PCT = 8.0

# Every Nth opening shot ignores the learner entirely.
HOLDOUT_EVERY = 5


def basis(range_m: float) -> np.ndarray:
    """`[1, ln(r / r_ref)]`. Two terms, for the reasons in the module docstring."""
    r = max(float(range_m), 0.1)
    return np.array([1.0, math.log(r / R_REF_M)], dtype=float)


@dataclass
class BallisticsLearner:
    """Learns a correction to the OPENING pump command. Nothing else."""

    rls: RecursiveLeastSquares = field(
        default_factory=lambda: RecursiveLeastSquares(n_terms=2)
    )
    shots: int = 0
    holdout_every: int = HOLDOUT_EVERY
    _last_was_holdout: bool = False

    # ------------------------------------------------------------ observing

    def observe(self, range_m: float, pump_error_pct: float) -> bool:
        """Fold in one measured opening error.

        `pump_error_pct` is how much the opening pump SHOULD have differed — a
        positive value means the shot landed short and more pump was needed.
        """
        if not math.isfinite(pump_error_pct) or not math.isfinite(range_m):
            return False
        return self.rls.update(basis(range_m), pump_error_pct)

    # ------------------------------------------------------------ predicting

    def is_holdout(self) -> bool:
        """Is the NEXT opening shot a nominal-solver holdout probe?"""
        return self.holdout_every > 0 and (self.shots % self.holdout_every) == 0

    def correction_pct(self, range_m: float) -> float:
        """The learned adjustment to the opening pump command, in percent.

        Returns 0.0 on a holdout shot, on a non-finite prediction, or whenever
        the correction would fall outside the authority box.
        """
        if self.is_holdout():
            return 0.0
        raw = self.rls.predict(basis(range_m))
        if not math.isfinite(raw):
            return 0.0
        return self._clamp(raw)

    def _clamp(self, correction: float) -> float:
        """Asymmetric authority. Shorten freely; lengthen only marginally."""
        lengthen_cap = (
            MAX_LENGTHEN_TRUSTED_PCT if self.rls.confident else MAX_LENGTHEN_PCT
        )
        # A POSITIVE correction adds pump, which makes the shot land LONGER.
        return max(-MAX_SHORTEN_PCT, min(lengthen_cap, correction))

    def note_shot(self) -> None:
        """Record that an opening shot was taken (advances the holdout cycle)."""
        self._last_was_holdout = self.is_holdout()
        self.shots += 1

    @property
    def last_was_holdout(self) -> bool:
        return self._last_was_holdout

    # ------------------------------------------------------------ persistence

    def to_dict(self) -> dict:
        return {
            "theta": self.rls.theta.tolist(),
            "updates": self.rls.updates,
            "shots": self.shots,
            "r_ref_m": R_REF_M,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> BallisticsLearner:
        """Rebuild from a calibration file.

        Every failure — missing keys, wrong length, non-finite values, a file
        from a different `r_ref` — degrades to a FRESH learner rather than
        raising. A corrupt calibration file must cost today's behaviour, not the
        turret's ability to start.
        """
        learner = cls()
        try:
            theta = np.asarray(raw["theta"], dtype=float)
            if theta.shape != (2,) or not np.all(np.isfinite(theta)):
                return cls()
            if float(raw.get("r_ref_m", R_REF_M)) != R_REF_M:
                return cls()  # a different basis; the numbers do not transfer
            learner.rls.theta = theta
            learner.rls.updates = int(raw.get("updates", 0))
            learner.shots = int(raw.get("shots", 0))
        except (KeyError, TypeError, ValueError):
            return cls()
        return learner


def apply_to_opening(
    nominal_pump_pct: float,
    learner: BallisticsLearner | None,
    range_m: float,
    open_short_pct: float,
    min_pump_pct: float,
) -> float:
    """The ONE place a learned correction reaches a command.

    Order matters and is the point: the learner adjusts the nominal solution,
    and **then** the open-short subtraction is applied. Doing it the other way
    would let a learned correction cancel the open-short bias, and invariant 8
    exists because a long shot lands occluded and starves the loop.
    """
    pump = float(nominal_pump_pct)
    if learner is not None:
        pump += learner.correction_pct(range_m)
    pump -= open_short_pct  # invariant 8, applied LAST so nothing can bypass it
    return max(min_pump_pct, min(100.0, pump))
