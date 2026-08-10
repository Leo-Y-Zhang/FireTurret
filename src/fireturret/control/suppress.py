# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""The suppression correction law — where invariants 1, 3, 5 and 8 are actually
implemented. Kept small and isolated on purpose, and
`tests/test_import_boundaries.py` enforces the isolation.

The one worth restating here, because it is what this file must not become:
**servo in IMAGE space, never on the estimated range.** The loop drives the
observed splash *pixel* onto the fire *pixel*. Monocular range is biased, and an
earlier version that servoed on it pushed water off target. A ballistics-derived
flight time or gain is fine — it is a feedforward model of the ACTUATOR and moves
only how fast the loop converges, never where. A target-range estimate is not.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..geometry import CameraModel, clamp
from ..rig.interface import RigTelemetry
from ..vision.impact import ImpactObservation
from ..vision.tracker import Track
from .plant import schedule_cycle
from .servo import VisualServo

# Ceiling on the scheduled gain: the schedule comes from a model, and near the
# top of the envelope the computed gain grows without bound. A safety-relevant
# loop must not inherit a model's asymptote.
_MAX_SCHEDULED_GAIN = 0.60

# Legacy fixed cadence, now only the fallback before a solution is known. At
# 30 fps these put the observation window 0.30-0.70 s after a correction while
# the water observed was launched 0.94-1.28 s earlier — so the median was of the
# PREVIOUS cycle's pump, and invariant 3 was not true (spec §3.5).
SUPP_CYCLE_FRAMES = 21
SUPP_SETTLE_FRAMES = 9
SUPP_MIN_OBS = 4  # need this many clean observations to trust the median
SUPP_NO_FEEDBACK_CYCLES = 6  # cycles spraying with no splash seen => stop, hold

SETTLE_MARGIN_FRAMES = 3  # beyond flight time: actuator settle + telemetry round trip
OBSERVE_FRAMES = 12  # collecting window; must exceed SUPP_MIN_OBS comfortably
FPS = 30.0

# Reachability is judged on a DIFFERENT threshold from control, and the
# difference matters more than it looks.
#
# `range_deadband_px` exists to stop the loop dithering on per-frame noise. It is
# not a statement about reach. At long range perspective foreshortening compresses
# the image: measured on the default build, a jet landing 1.4 m short of a 10 m
# target produces a row error of only ~6.7 px — INSIDE the 12 px deadband. So a
# turret spraying at full pump, genuinely unable to reach, reads as "on target".
#
# That is why the unreachable scenario only ever reached HOLD by accident:
# renderer noise occasionally pushed the median past the deadband (measured 3 of 6
# seeds pre-SP4). Tightening the loop removed the noise and with it the accident —
# 0 of 6. The mechanism was never real.
#
# A PERSISTENT shortfall at saturated pump is meaningful even when each individual
# sample is too small to act on: there is no more pump to give, so short is short.
REACH_SHORTFALL_PX = 1.0

# The open-short offset (invariant 8): open deliberately short so the first shot
# lands in front of the fire, where it can be seen.
OPEN_SHORT_PCT = 12.0


@dataclass
class Correction:
    """What one completed observe-then-correct cycle decided. `None` means "leave
    this axis alone" — the normal case mid-cycle, when the aim is held still on
    purpose."""

    pan_deg: float | None = None
    tilt_deg: float | None = None
    pump_pct: float | None = None
    azimuth_error_deg: float | None = None
    range_error_px: float | None = None
    stop_reason: str = ""  # non-empty => give up on this target and HOLD


class SuppressionCycle:
    """Owns the observation buffer, the cycle counters and the boresight estimate.

    It does NOT own `_pan_cmd` / `_tilt_cmd` / `_pump_cmd` / `_command()` — those
    stay on `MissionController`, whose test surface reads and writes them
    directly. This class returns a `Correction` and the mission applies it.
    """

    def __init__(
        self,
        cfg,
        camera: CameraModel,
        servo: VisualServo,
        peak_tilt_deg: float,
    ) -> None:
        self.cfg = cfg
        self.camera = camera
        self.servo = servo
        self.peak_tilt_deg = peak_tilt_deg
        self.reset()

    def reset(self) -> None:
        self.frame = 0
        self.observations: list[tuple[float, float]] = []
        self.no_feedback_cycles = 0
        self.short_saturated_cycles = 0
        self.stuck_reason = ""
        self.boresight_est = 0.0
        self.settle_frames = SUPP_SETTLE_FRAMES
        self.cycle_frames = SUPP_CYCLE_FRAMES
        self.scheduled_gain = self.cfg.servo.range_gain_pct_per_px

    def retime(self, pump_pct: float, tilt_deg: float) -> None:
        """Size this cycle from the firing solution currently in the air.

        The fix for spec §3.5. Water launched the instant a correction takes
        effect is airborne for ~1 s; observing before it lands measures the
        PREVIOUS pump setting and credits it to the new one. So the settle window
        is the flight time, recomputed each cycle because flight time varies with
        pump. The arithmetic lives in `control/plant.py` — this file is the law.
        """
        sched = schedule_cycle(
            pump_pct, tilt_deg, self.cfg.jet, self.camera,
            base_gain=self.cfg.servo.range_gain_pct_per_px,
            max_gain=_MAX_SCHEDULED_GAIN,
            fps=FPS,
            settle_margin_frames=SETTLE_MARGIN_FRAMES,
            observe_frames=OBSERVE_FRAMES,
        )
        self.settle_frames = sched.settle_frames
        self.cycle_frames = sched.cycle_frames
        self.scheduled_gain = sched.gain_pct_per_px
        self.plant = sched.plant

    # ---------------------------------------------------------------- per tick

    def step(
        self,
        target: Track,
        impact: ImpactObservation | None,
        telemetry: RigTelemetry,
        tilt_cmd: float,
    ) -> Correction:
        """One tick. Returns an all-None `Correction` except on the tick that
        closes a cycle, because holding the aim still IS the behaviour."""
        if self.frame == 0:
            # size the cycle from the solution currently in the air
            self.retime(telemetry.pump_pct, tilt_cmd)
        self.frame += 1
        # Collect clean observations only after the post-correction settle
        # window — which is now the FLIGHT TIME, so what is measured is water
        # launched under the current command rather than the previous one.
        if impact is not None and self.frame > self.settle_frames:
            self.observations.append((impact.cx, impact.cy))

        if self.frame < self.cycle_frames:
            return Correction()

        had_feedback = len(self.observations) >= SUPP_MIN_OBS
        correction = self._correct(target, telemetry, tilt_cmd)
        self.frame = 0
        self.observations = []
        # Spraying but never seeing the splash (e.g. a too-close fire whose
        # overshoot lands occluded behind the flame): stop hosing blind.
        self.no_feedback_cycles = 0 if had_feedback else self.no_feedback_cycles + 1

        if self.short_saturated_cycles >= self.cfg.mission.unreachable_cycles:
            correction.stop_reason = self.stuck_reason or "target out of reach"
        elif self.no_feedback_cycles >= SUPP_NO_FEEDBACK_CYCLES:
            correction.stop_reason = "no splash / check aim"
        return correction

    # -------------------------------------------------------------- the law

    def _correct(
        self, target: Track, telemetry: RigTelemetry, tilt_cmd: float
    ) -> Correction:
        fire_px = _fire_px(target)
        if len(self.observations) < SUPP_MIN_OBS:
            return Correction()  # not enough evidence this cycle; keep holding

        xs = sorted(o[0] for o in self.observations)
        ys = sorted(o[1] for o in self.observations)
        mid = len(xs) // 2
        splash_px = (xs[mid], ys[mid])  # median: robust to outlier blobs
        step = self.servo.suppress_step(
            fire_px, splash_px, telemetry.pan_deg, telemetry.pump_pct,
            range_gain_pct_per_px=self.scheduled_gain,
        )

        # Azimuth: the water's camera bearing IS the boresight, so learn it from
        # where the splash actually lands, then FULLY track the fire to that
        # column. Full tracking rather than a partial-gain nudge, because a
        # partial correction lags a drifting or spreading fire.
        az_fire = self.camera.px_to_angles(fire_px[0], self.camera.cy).azimuth_deg
        az_splash = self.camera.px_to_angles(splash_px[0], self.camera.cy).azimuth_deg
        self.boresight_est = 0.6 * self.boresight_est + 0.4 * az_splash
        pan = clamp(
            telemetry.pan_deg + az_fire - self.boresight_est,
            self.cfg.turret.pan_min_deg,
            self.cfg.turret.pan_max_deg,
        )

        # Range: pump is the primary control (invariant 5). If it saturates while
        # the splash is still short — a receding fire — extend reach by raising
        # tilt toward the range peak. Only when BOTH pump and tilt are maxed and
        # still short is the target genuinely out of reach.
        tilt: float | None = None
        # "Out of envelope" is no longer only "the pump hit its stop". A pump
        # with no OBSERVABLE authority left — below MIN_PLANT_PX_PER_PCT, where a
        # 1% change moves the splash less than the detector's own noise — is
        # equally stuck, and a boolean monotonicity check cannot tell: range stays
        # monotone in pump the whole way while authority dies 7.4x (spec §3.6).
        no_authority = not self.plant.observable
        maxed = step.pump_pct >= 99.0 or no_authority
        # Judged on REACH_SHORTFALL_PX, not the control deadband — see the
        # constant for why the deadband cannot see a genuinely unreachable target.
        saturated_short = maxed and step.range_error_px > REACH_SHORTFALL_PX
        floored_long = (
            step.pump_pct <= self.cfg.jet.min_pump_pct + 1.0
            and step.range_error_px < -self.cfg.servo.range_deadband_px
        )
        if saturated_short and tilt_cmd < self.peak_tilt_deg - 0.5:
            tilt = min(tilt_cmd + 2.5, self.peak_tilt_deg)  # extend reach
            self.short_saturated_cycles = 0  # progress via tilt, not stuck
        elif saturated_short:
            self.short_saturated_cycles += 1  # pump AND tilt maxed, still short
            self.stuck_reason = "target out of reach"
        elif floored_long:
            self.short_saturated_cycles += 1  # pump floored, still overshooting
            self.stuck_reason = "target too close"
        else:
            self.short_saturated_cycles = 0

        return Correction(
            pan_deg=pan,
            tilt_deg=tilt,
            pump_pct=step.pump_pct,
            azimuth_error_deg=az_fire - self.boresight_est,
            range_error_px=step.range_error_px,
        )


def _fire_px(target: Track) -> tuple[float, float]:
    """The flame's ground-contact PIXEL — not a range. Duplicated from
    `ranging.target_ground_px` on purpose: importing that module would drag the
    range estimator into this file's import graph and the boundary test would
    (correctly) fail. Four lines is a cheap price for a checkable invariant."""
    if target.last_blob is not None:
        _x, y, _w, h = target.last_blob.bbox
        return (target.cx, float(y + h))
    return (target.cx, target.cy)
