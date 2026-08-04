# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""Wind estimation — and an honest statement of what cannot be estimated.

## The thing that has to be said first

**Constant wind and constant model bias are NOT separately identifiable from
splash observations alone.** A jet landing consistently 0.4 m left of where the
model predicts is equally well explained by a steady crosswind, by a boresight
error, or by a discharge coefficient that is slightly wrong. No amount of splash
data distinguishes them, because they produce the same observation.

This is not a limitation to be engineered around; it is a property of the
problem. Pretending otherwise produces an estimator that confidently attributes a
mounting error to the weather.

So the filter here uses a **mean-reverting (Ornstein-Uhlenbeck) prior**. Genuine
gusts — which vary — are tracked. A persistent offset decays back toward zero
rather than being absorbed as permanent wind, which leaves it visible to SP11's
ballistics learner, where a persistent, repeatable offset actually belongs.
`test_a_constant_wind_with_no_anemometer_decays_toward_zero` asserts exactly
that, and it is a feature.

An anemometer breaks the ambiguity, because it measures wind directly. With
`anemometer=True` the mean reversion is relaxed and the measurement is trusted.

## The crosswind drift rule

From exterior ballistics, the lateral drift of a projectile in a crosswind is

    drift = w * (t_f - r / v0)

the difference between the actual flight time and the time a drag-free round
would have taken. It needs `t_f`, which is exactly what SP4's `arc_with_time`
exposes — the two sub-projects meet here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# Mean-reversion rate for the OU prior, per second. Chosen so a persistent
# offset decays with a time constant of ~20 s: long enough to track a real gust
# front, short enough that a mounting error does not masquerade as weather for a
# whole engagement.
OU_THETA = 0.05

# Process and measurement noise. Deliberately modest: the splash residual is a
# noisy, indirect measurement of wind, and a filter that trusted it heavily would
# chase detector noise.
PROCESS_VAR = 0.05
MEASUREMENT_VAR = 1.0


def crosswind_drift_m(
    crosswind_ms: float, flight_time_s: float, range_m: float, muzzle_speed_ms: float
) -> float:
    """Lateral drift from the exterior-ballistics lag-time rule.

    `drift = w (t_f - r/v0)`. The bracket is the *lag time* — how much longer the
    projectile took than a drag-free one would have. With no drag it is zero and
    so is the drift, which is the correct limit.
    """
    if muzzle_speed_ms <= 0:
        raise ValueError("muzzle speed must be positive")
    lag_s = flight_time_s - range_m / muzzle_speed_ms
    return crosswind_ms * lag_s


@dataclass
class WindEstimate:
    east_ms: float = 0.0
    north_ms: float = 0.0
    variance: float = 1.0

    @property
    def speed_ms(self) -> float:
        return math.hypot(self.east_ms, self.north_ms)


@dataclass
class WindFilter:
    """Two-state world-frame OU Kalman filter on splash residuals.

    Deliberately simple: two states, scalar variance, no cross-covariance. The
    measurement is a noisy indirect observation of a quantity that is not fully
    identifiable, and a more elaborate filter would express a confidence the data
    does not support.
    """

    theta: float = OU_THETA
    process_var: float = PROCESS_VAR
    measurement_var: float = MEASUREMENT_VAR
    has_anemometer: bool = False
    state: WindEstimate = field(default_factory=WindEstimate)

    def predict(self, dt: float) -> None:
        """Mean-revert toward zero, and grow the variance.

        With an anemometer the reversion is switched off: wind is being measured
        rather than inferred, so there is no ambiguity to protect against.
        """
        if dt <= 0:
            return
        if not self.has_anemometer:
            decay = math.exp(-self.theta * dt)
            self.state.east_ms *= decay
            self.state.north_ms *= decay
        self.state.variance += self.process_var * dt

    def update_from_residual(self, east_residual_ms: float, north_residual_ms: float) -> None:
        """Fold in a wind-equivalent residual inferred from a splash miss."""
        gain = self.state.variance / (self.state.variance + self.measurement_var)
        self.state.east_ms += gain * (east_residual_ms - self.state.east_ms)
        self.state.north_ms += gain * (north_residual_ms - self.state.north_ms)
        self.state.variance *= (1.0 - gain)

    def update_from_anemometer(self, east_ms: float, north_ms: float) -> None:
        """A direct measurement. Trusted far more than a splash residual, because
        it observes the quantity itself rather than one of three things that
        could explain a miss."""
        self.has_anemometer = True
        gain = self.state.variance / (self.state.variance + self.measurement_var * 0.05)
        self.state.east_ms += gain * (east_ms - self.state.east_ms)
        self.state.north_ms += gain * (north_ms - self.state.north_ms)
        self.state.variance *= (1.0 - gain)

    def wind(self) -> tuple[float, float]:
        return (self.state.east_ms, self.state.north_ms)


@dataclass
class PlantGainValidator:
    """MONITOR-ONLY comparison of observed response against prediction.

    This is the spec's replacement for online gain tuning, and the distinction is
    the entire point: online tuning would give an estimator write access to a
    safety-relevant loop, having derived its stability target from delay-free
    algebra on a loop with a measured 1-2 sample transport delay. It would raise
    gain 6-125x.

    This raises an ADVISORY instead. The operator (and SP11) learn that the model
    disagrees with reality; nothing automatically acts on it.
    """

    tolerance: float = 0.5  # fractional disagreement before complaining
    samples: int = 0
    disagreements: int = 0

    def observe(self, predicted_px: float, observed_px: float) -> bool:
        """True when the response disagreed with prediction beyond tolerance."""
        self.samples += 1
        if abs(predicted_px) < 1e-6:
            return False
        error = abs(observed_px - predicted_px) / abs(predicted_px)
        disagreed = error > self.tolerance
        if disagreed:
            self.disagreements += 1
        return disagreed

    @property
    def disagreement_rate(self) -> float:
        return self.disagreements / self.samples if self.samples else 0.0

    def advisory(self) -> str | None:
        """An operator-facing note, or None. Never a control action."""
        if self.samples >= 5 and self.disagreement_rate > 0.5:
            return (
                f"ballistic model disagrees with observed response in "
                f"{self.disagreement_rate:.0%} of cycles — check calibration"
            )
        return None
