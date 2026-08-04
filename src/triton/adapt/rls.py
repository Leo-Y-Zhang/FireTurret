# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""Recursive least squares, with the three guards that make it safe to leave running.

Plain RLS with a forgetting factor is a well-known trap in exactly this
situation, and all three guards below exist to close a specific failure:

## 1. The covariance trace bound — the important one

With forgetting factor λ < 1 and **no persistent excitation**, the covariance P
grows without bound. That is not a slow degradation; it is a time bomb. P is the
gain on new evidence, so an inflated P means the next datum — however noisy —
swings θ arbitrarily far. A turret that has been sitting on target, receiving
consistent data, is in precisely that state: well-converged, no excitation, P
quietly inflating. Then one bad splash observation arrives and the model jumps.

So the trace of P is hard-bounded. When it would exceed the bound, P is rescaled.
This costs adaptation speed after a genuinely long quiet period, and that is the
correct trade.

## 2. Directional forgetting

Forgetting should apply in directions the data actually informs. Forgetting
uniformly discards knowledge in directions nothing has been measured in, which is
how a parameter nobody has excited drifts away from a perfectly good prior.

## 3. A Huber outlier gate

A splash observation can be badly wrong — a reflection, a passing object, a
detector glitch. Squared error weights those quadratically, so one bad datum
moves θ more than a hundred good ones. The Huber gate caps a single observation's
influence.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Maximum trace of P. See guard 1 — this is what stops a well-converged,
# unexcited filter from becoming infinitely credulous.
DEFAULT_TRACE_BOUND = 100.0

# Residuals beyond this many sigma are down-weighted rather than trusted.
DEFAULT_HUBER_DELTA = 2.0


@dataclass
class RecursiveLeastSquares:
    """RLS with directional forgetting, a trace bound and a Huber gate."""

    n_terms: int
    forgetting: float = 0.995
    trace_bound: float = DEFAULT_TRACE_BOUND
    huber_delta: float = DEFAULT_HUBER_DELTA
    initial_variance: float = 10.0

    theta: np.ndarray = field(init=False)
    P: np.ndarray = field(init=False)
    updates: int = field(default=0, init=False)
    rejected: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.n_terms < 1:
            raise ValueError("need at least one term")
        if not 0.0 < self.forgetting <= 1.0:
            raise ValueError("forgetting factor must be within (0, 1]")
        if self.trace_bound <= 0:
            raise ValueError("trace bound must be positive")
        self.theta = np.zeros(self.n_terms, dtype=float)
        self.P = np.eye(self.n_terms, dtype=float) * self.initial_variance

    # ---------------------------------------------------------------- update

    def predict(self, phi: np.ndarray) -> float:
        """The model's value at `phi`, or NaN if the model is not finite.

        Checked rather than left to numpy, which would emit a RuntimeWarning and
        return NaN anyway. Callers treat NaN as "no opinion" and fall back to
        nominal, so a corrupted model degrades quietly instead of noisily.
        """
        if not np.all(np.isfinite(self.theta)):
            return float("nan")
        return float(np.dot(np.asarray(phi, dtype=float), self.theta))

    def update(self, phi: np.ndarray, y: float) -> bool:
        """Fold in one observation. Returns False if it was gated out.

        `phi` is the regressor (basis) vector, `y` the measured value.
        """
        phi = np.asarray(phi, dtype=float).reshape(-1)
        if phi.shape[0] != self.n_terms:
            raise ValueError(f"expected {self.n_terms} terms, got {phi.shape[0]}")
        if not np.all(np.isfinite(phi)) or not np.isfinite(y):
            self.rejected += 1
            return False

        residual = y - self.predict(phi)

        # --- Huber gate: cap a single observation's influence.
        denom = float(phi @ self.P @ phi) + 1.0
        sigma = np.sqrt(max(denom, 1e-12))
        weight = 1.0
        if abs(residual) > self.huber_delta * sigma:
            weight = self.huber_delta * sigma / abs(residual)
            self.rejected += 1

        # --- directional forgetting: only where this datum carries information.
        information = float(phi @ phi)
        lam = self.forgetting if information > 1e-9 else 1.0

        gain = (self.P @ phi) / (lam + denom - 1.0 + 1e-12)
        self.theta = self.theta + weight * gain * residual
        self.P = (self.P - np.outer(gain, phi @ self.P)) / lam
        self.P = 0.5 * (self.P + self.P.T)  # keep it symmetric against drift

        self._bound_trace()
        self.updates += 1
        # `bool(...)` because numpy comparisons yield np.bool_, which is not
        # `False` under `is` — and a caller writing `if result is False` would
        # silently never fire.
        return bool(weight == 1.0)

    def _bound_trace(self) -> None:
        """Guard 1. Rescale P if its trace would exceed the bound."""
        trace = float(np.trace(self.P))
        if not np.isfinite(trace) or trace <= 0:
            self.P = np.eye(self.n_terms) * self.initial_variance
            return
        if trace > self.trace_bound:
            self.P *= self.trace_bound / trace

    # ----------------------------------------------------------------- state

    @property
    def confident(self) -> bool:
        """Has this seen enough consistent data to be trusted with authority?

        Deliberately conservative: the learner's authority is asymmetric
        (`ballistics_learner`), and this is what unlocks the half that can make
        a shot land LONG.
        """
        return self.updates >= 20 and float(np.trace(self.P)) < self.initial_variance

    def reset(self) -> None:
        self.theta = np.zeros(self.n_terms, dtype=float)
        self.P = np.eye(self.n_terms, dtype=float) * self.initial_variance
        self.updates = 0
        self.rejected = 0
