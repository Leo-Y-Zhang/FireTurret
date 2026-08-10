# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""Asynchronous safety vetoes — a slow predicate, safely.

A person detector on a Raspberry Pi 5 CPU costs 120-300 ms per inference. The
control loop runs at 33 ms. So the predicate CANNOT run on the loop, and the loop
must act on an answer that is always somewhat out of date. This module is the
plumbing for that, built and proven **before** any detector exists — because the
plumbing is the part that can be tested exhaustively, and the detector's
judgement is the part that cannot.

## The fail-safe table

Every row is a test in `tests/test_veto.py`:

| situation | result |
|---|---|
| predicate said allow, answer is fresh | allow |
| predicate said deny | **deny, and latch** |
| answer older than `max_age_s` | **deny** |
| predicate raised | **deny** |
| no answer has ever arrived | **deny** |
| any veto in the chain denies | **deny** (no combination can rescue it) |

"I don't know" is never "yes". A stale answer is indistinguishable from a dead
worker, and a dead safety worker is exactly when you least want the layer to
abstain.

## Deny-only, and why there is no `forced_pan_deg`

A veto may withhold water and freeze the aim. It may NOT command motion. A layer
that re-aims acquires its own failure surface, and it would have to project a
detector's bounding box through the same monocular range model the whole
architecture distrusts. Re-aiming stays with the mission, whose next command
passes through this gate anyway.

## Why `max_age_s` is derived rather than borrowed

The obvious number to reach for is the firmware's 500 ms heartbeat. It is the
wrong number, and using it produces a layer that gets switched off:

At a 5 Hz submit cadence with 120-300 ms inference, a result is already
200 ms + inference old when it ARRIVES. With 300 ms inference that is 500 ms on
arrival — so a 500 ms threshold marks results stale the instant they land, and
the valve chatters at the detector's beat frequency. **An unusable safety layer
gets disabled, which is worse than a slower one.** `derive_max_age_s` does the
arithmetic explicitly from measured latency instead.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from ..control.gate import GateContext, Verdict

# Multiplier on the worst-case answer age. Two is not a ritual: it covers one
# missed submission cycle without tripping, which is the difference between "the
# worker is dead" and "the worker had a bad frame".
STALENESS_MARGIN = 2.0


def derive_max_age_s(
    p99_inference_s: float, submit_period_s: float, margin: float = STALENESS_MARGIN
) -> float:
    """Staleness threshold from MEASURED latency, with the arithmetic in the open.

    A result's age on arrival is at worst one full submit period (it was
    computed from a frame grabbed just after the last submission) plus the
    inference time itself. Anything below that marks fresh results stale.
    """
    if p99_inference_s < 0 or submit_period_s <= 0:
        raise ValueError("latency must be non-negative and the period positive")
    worst_age_on_arrival = submit_period_s + p99_inference_s
    return worst_age_on_arrival * margin


@dataclass(frozen=True)
class VetoAnswer:
    """One answer from the off-loop predicate, stamped when it was COMPUTED."""

    allowed: bool
    computed_at: float
    reason: str = ""


class AsyncVeto:
    """A `WaterVeto` fed asynchronously by a slow predicate.

    Clock-injected so the staleness behaviour is testable without sleeping —
    a safety threshold verified by `time.sleep` is a flaky test, and a flaky
    safety test is one that eventually gets deleted.
    """

    def __init__(
        self,
        name: str = "async_safety",
        max_age_s: float = 1.0,
        clock=time.monotonic,
        *,
        latching: bool = True,
    ) -> None:
        if max_age_s <= 0:
            raise ValueError("max_age_s must be positive")
        self.name = name
        self.max_age_s = max_age_s
        self.latching = latching
        self._clock = clock
        self._answer: VetoAnswer | None = None
        self._latched_reason = ""

    # ------------------------------------------------------- worker-side API

    def submit(self, answer: VetoAnswer) -> None:
        """Called from the worker thread when a new answer is ready."""
        self._answer = answer
        if not answer.allowed and self.latching:
            self._latched_reason = answer.reason or "denied by safety predicate"

    def submit_failure(self, exc: BaseException) -> None:
        """The predicate raised. Recorded as a DENIAL, not as an absence — a
        crashed detector must not read as 'nobody is there'."""
        reason = f"safety predicate failed ({exc.__class__.__name__})"
        self._answer = VetoAnswer(allowed=False, computed_at=self._clock(), reason=reason)
        if self.latching:
            self._latched_reason = reason

    def clear(self) -> None:
        """Operator acknowledgement. Clears a latched denial; it does NOT make a
        stale or currently-denying answer fresh."""
        self._latched_reason = ""

    # ------------------------------------------------------- loop-side API

    @property
    def latched(self) -> bool:
        return bool(self._latched_reason)

    def age_s(self) -> float | None:
        if self._answer is None:
            return None
        return self._clock() - self._answer.computed_at

    def evaluate(self, ctx: GateContext) -> Verdict:
        if self._latched_reason:
            return Verdict.deny(f"latched: {self._latched_reason}")
        if self._answer is None:
            return Verdict.deny("no safety answer yet")
        age = self._clock() - self._answer.computed_at
        if age > self.max_age_s:
            return Verdict.deny(f"safety answer stale ({age:.2f}s > {self.max_age_s:.2f}s)")
        if not self._answer.allowed:
            return Verdict.deny(self._answer.reason or "denied by safety predicate")
        return Verdict.allow()


class AlwaysAllowVeto:
    """The ship-first implementation: correct plumbing, no opinion.

    Its purpose is to prove the wiring end to end before a detector exists. It
    still goes through `AsyncVeto`, so it is stale-checked and latching like any
    other — a permanently-allowing veto that bypassed those would test nothing.
    """

    def __init__(self, name: str = "always_allow") -> None:
        self.name = name

    def evaluate(self, ctx: GateContext) -> Verdict:
        return Verdict.allow()


class NeverStaleClock:
    """Test helper: a clock the caller advances explicitly."""

    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds
