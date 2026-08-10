# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""What the pump is actually worth, in pixels, right now.

## The measurement this exists because of

The suppression loop corrects pump from a splash-row error with one fixed gain,
`ServoConfig.range_gain_pct_per_px`. Measured across the usable pump band at the
suppression tilt (spec §3.6):

```
pump  25%  range= 6.34 m  dR/dpump=0.1148 m/%  dpx/dpump=1.134 px/%  L=0.0680
pump  40%  range= 7.78 m  dR/dpump=0.0782 m/%  dpx/dpump=0.521 px/%  L=0.0312
pump  55%  range= 8.80 m  dR/dpump=0.0591 m/%  dpx/dpump=0.309 px/%  L=0.0185
pump  70%  range= 9.60 m  dR/dpump=0.0474 m/%  dpx/dpump=0.209 px/%  L=0.0125
pump  85%  range=10.25 m  dR/dpump=0.0394 m/%  dpx/dpump=0.153 px/%  L=0.0092
```

Two collapses that compound. Ballistic authority `dR/dpump` falls **2.91x**;
loop gain `L` falls **7.43x**, because perspective foreshortening shrinks the
pixel row spacing with range, so the *observable* consequence of a pump change
dies faster than the physical one.

Range stays monotone in pump throughout — which is why a boolean monotonicity
check passes happily while control authority dies by a factor of seven. That is
the trap this module replaces with a number.

## Two things follow

**Gain scheduling.** Hold `L = gain x plant` roughly constant by scaling the gain
against the measured plant gain, instead of tuning one constant for a band where
the plant varies 7x. Everything here comes from the EXISTING ballistics model and
camera geometry: deterministic, no estimator, no persistence, nothing to poison.

**A minimum-plant-gain envelope check.** Below some pixels-per-percent the pump
simply has no observable authority, and pushing harder produces no measurable
response — the correct conclusion is "out of envelope", not "try more pump". This
subsumes the monotonicity gate rather than sitting beside it.

## Why the target loop gain is 0.3

The loop has a measured transport delay of roughly one sample: the water observed
in a cycle was launched ~1.1 s before it lands, against a cycle time of the same
order. A delay-free stability argument would allow ~0.45; that argument omits the
dominant dynamic. 0.3 leaves margin for the delay actually being there, and is
still 4-30x the effective gain the fixed constant produces across the band.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..ballistics import exit_velocity, range_of
from ..config import JetConfig
from ..geometry import CameraModel

# Target open-loop gain per correction, |d(row_px_corrected) / d(row_px_error)|.
# See the module docstring for why not 0.45.
TARGET_LOOP_GAIN = 0.3

# Below this the pump has no observable authority: a 1% change moves the splash
# less than this many pixels, which is inside the detector's own centroid noise.
# Pushing harder cannot be verified, so the honest reading is "out of envelope".
MIN_PLANT_PX_PER_PCT = 0.05

# Probe size for the numerical derivative. Large enough to clear float noise in
# an RK4 integration, small enough that the derivative is still local.
_PROBE_PCT = 1.0


@dataclass(frozen=True)
class PlantGain:
    """The local sensitivity of the observed splash row to the pump command."""

    pump_pct: float
    tilt_deg: float
    range_m: float
    d_range_per_pct: float   # metres of range per percent of pump
    d_px_per_pct: float      # IMAGE ROWS per percent of pump — what the loop sees

    @property
    def observable(self) -> bool:
        """Is a pump correction something the camera could actually verify?"""
        return abs(self.d_px_per_pct) >= MIN_PLANT_PX_PER_PCT

    def scheduled_gain(self, lo: float, hi: float) -> float:
        """Servo gain (percent of pump per pixel of row error) that puts the
        loop gain at `TARGET_LOOP_GAIN`, clamped to a sane band.

        Clamped because the schedule is only as good as the model it comes from:
        near the top of the envelope the computed gain grows without bound, and a
        safety-relevant loop should not inherit a model's asymptote.
        """
        if not self.observable:
            return lo
        return min(max(TARGET_LOOP_GAIN / abs(self.d_px_per_pct), lo), hi)


@dataclass(frozen=True)
class CycleSchedule:
    """How long one observe-then-correct cycle should be, and how hard to push.

    Both numbers come from the firing solution currently in the air, which is why
    they live together: the settle window is the flight time, and the gain is
    scheduled at the same operating point.
    """

    settle_frames: int
    cycle_frames: int
    gain_pct_per_px: float
    plant: PlantGain
    flight_time_s: float


def schedule_cycle(
    pump_pct: float,
    tilt_deg: float,
    jet: JetConfig,
    camera: CameraModel,
    *,
    base_gain: float,
    max_gain: float,
    fps: float,
    settle_margin_frames: int,
    observe_frames: int,
) -> CycleSchedule:
    """Size the next cycle from the solution currently in the air.

    Kept here rather than in `control/suppress.py` so that file stays the
    correction LAW and nothing else. Scheduling is a different concern, and it is
    the one that needs the ballistics model.
    """
    from ..ballistics import flight_time_of  # local: keeps the import graph flat

    flight_s = flight_time_of(exit_velocity(pump_pct, jet), tilt_deg, jet)
    settle = int(flight_s * fps) + settle_margin_frames
    gain = plant_gain(pump_pct, tilt_deg, jet, camera)
    return CycleSchedule(
        settle_frames=settle,
        cycle_frames=settle + observe_frames,
        gain_pct_per_px=gain.scheduled_gain(lo=base_gain, hi=max_gain),
        plant=gain,
        flight_time_s=flight_s,
    )


def plant_gain(
    pump_pct: float,
    tilt_deg: float,
    jet: JetConfig,
    camera: CameraModel,
) -> PlantGain:
    """Measure `dRange/dPump` and `dRow/dPump` at an operating point.

    A central difference where the probe fits inside the pump band, one-sided at
    the clamps — a one-sided probe against a clamp is degenerate and would report
    zero authority exactly where the loop most needs a real number. (An earlier
    draft of the spec quoted a 20x collapse; that figure included pump 100%,
    where the +1% probe is degenerate against the clamp. It is 7.43x.)
    """
    lo = max(jet.min_pump_pct, min(pump_pct - _PROBE_PCT, 100.0))
    hi = min(100.0, max(pump_pct + _PROBE_PCT, jet.min_pump_pct))
    if hi <= lo:  # a degenerate band; nothing to measure
        hi = min(100.0, lo + _PROBE_PCT)

    r_lo = range_of(exit_velocity(lo, jet), tilt_deg, jet)
    r_hi = range_of(exit_velocity(hi, jet), tilt_deg, jet)
    r_at = range_of(exit_velocity(pump_pct, jet), tilt_deg, jet)

    span = hi - lo
    d_range = (r_hi - r_lo) / span if span > 0 else 0.0

    # The same derivative expressed in the units the loop actually observes.
    # Rows, not metres: that is where the extra 2.6x of the collapse lives.
    v_lo = camera.px_row_for_ground_range(max(r_lo, 0.05))
    v_hi = camera.px_row_for_ground_range(max(r_hi, 0.05))
    d_px = (v_hi - v_lo) / span if span > 0 else 0.0

    return PlantGain(
        pump_pct=pump_pct,
        tilt_deg=tilt_deg,
        range_m=r_at,
        d_range_per_pct=d_range,
        d_px_per_pct=d_px,
    )
