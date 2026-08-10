# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""Mission state machine: turns tracked fire targets and splash observations
into rig commands.

SEARCH → ACQUIRE → RANGE → ALIGN → SUPPRESS → CONFIRM → SEARCH, with SAFE
overriding everything on E-stop or telemetry loss. Safety rules the mission
enforces (the firmware independently enforces its own):

- valve is only ever open in SUPPRESS/CONFIRM-soak,
- continuous spray is time-limited, then the solution is recomputed,
- no confirmed target ⇒ valve closed,
- SAFE closes everything and holds until the condition clears.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from .. import ballistics
from ..config import FireTurretConfig
from ..geometry import CameraModel, clamp
from ..rig.interface import RigCommand, RigTelemetry
from ..spray.envelope import LEGACY_ENVELOPE
from ..spray.keepout import check_deposition
from ..vision.impact import ImpactObservation
from ..vision.tracker import Track
from .advisories import (
    _RANK,
    CRITICAL,
    WARN,
    Advisory,
    reachability_advisory,
    traverse_advisory,
)
from .gate import GateContext, standard_gate
from .ranging import RangeEstimator, target_ground_px
from .servo import VisualServo
from .suppress import (
    OPEN_SHORT_PCT,
    SUPP_CYCLE_FRAMES,
    SUPP_MIN_OBS,
    SUPP_NO_FEEDBACK_CYCLES,
    SUPP_SETTLE_FRAMES,
    SuppressionCycle,
)

__all__ = [
    "SUPP_CYCLE_FRAMES",
    "SUPP_MIN_OBS",
    "SUPP_NO_FEEDBACK_CYCLES",
    "SUPP_SETTLE_FRAMES",
    "MissionController",
    "MissionDebug",
    "pan_in_keepout",
    "pan_touches_keepout",
    "swept_intersects_keepout",
]

State = str  # SEARCH|ACQUIRE|RANGE|ALIGN|SUPPRESS|CONFIRM|HOLD|SAFE

PARK_TILT_DEG = 20.0

# Suppression runs an observe-then-correct cycle: hold the aim dead still while
# collecting splash observations (frame-differencing needs a still camera),
# then apply ONE bounded correction and let it settle. This is what tames the
# per-frame vision noise that would otherwise drive the servo into oscillation.
# The observe-then-correct cadence now lives with the law it governs, in
# control/suppress.py. Re-exported here because this is where callers and tests
# have always found it, and moving a constant is not worth breaking them over.

# Margin added to the worst-case flight time when holding water off after the
# nozzle has swept through a keep-out sector: covers actuator settle and the
# telemetry round trip, so the hold genuinely outlasts the airborne water.
KEEPOUT_HOLD_MARGIN_S = 0.4

# Pan slew rate above which water is held off while a keep-out sector is
# configured. Water launched during a fast slew lands along an arc the aim never
# settled on, so "where the nozzle points" stops describing where water goes —
# and near a protected sector that arc is exactly what must not stray.
#
# Sits between the two rates this machine actually exhibits: closed-loop tracking
# of a moving fire runs at ~4 deg/s (docs/ARCHITECTURE.md), while a repositioning
# traverse runs at the actuator limit, TurretConfig.pan_rate_dps = 90 deg/s. So
# ordinary tracking sprays normally and a traverse does not.
#
# ONLY consulted when a keep-out is configured, which no golden or e2e scenario
# sets — this is behaviour-neutral for every existing fixture, by construction.
KEEPOUT_SLEW_HOLD_DPS = 20.0


def pan_in_keepout(pan: float, keepout: tuple[float, float] | None) -> bool:
    """Is this aim angle STRICTLY inside the protected sector?

    The boundary is excluded, and that is not an oversight. `_apply_keepout`
    resolves a forbidden command by clamping it TO the nearer edge, so the edge
    is the closest legal aim. A predicate that called the edge "inside" would
    declare the geofence's own output a violation.

    Use this for a single aim angle. For the path a slew sweeps, use
    `swept_intersects_keepout`, whose interval is CLOSED — reaching the boundary
    during a transit is a transit — or `pan_touches_keepout` for the one-angle
    version of that conservative reading.
    """
    if keepout is None:
        return False
    lo, hi = keepout
    return lo < pan < hi


def pan_touches_keepout(pan: float, keepout: tuple[float, float] | None) -> bool:
    """Is this aim angle inside the protected sector, boundary INCLUDED?

    The conservative reading, for callers that drive the rig directly rather than
    through `_apply_keepout` — the hardware self-test, a manual jog. There is an
    operator standing next to the machine during bring-up, and "pointed exactly
    at the edge of where people are" is not a reassuring place to park.
    """
    return swept_intersects_keepout(pan, pan, keepout)


def swept_intersects_keepout(
    pan_a: float, pan_b: float, keepout: tuple[float, float] | None
) -> bool:
    """Does the closed pan interval swept between two angles touch the keep-out?

    `_apply_keepout` only constrains where the turret is *aimed*. A command that
    lies outside the sector passes through unchanged, so a slew from one side to
    the other sweeps the nozzle straight across the protected sector. This is the
    predicate that catches that transit.

    Edges count as intersecting: a keep-out exists to protect something, and the
    conservative reading of "the nozzle reached the boundary" is that it was
    pointed at it.
    """
    if keepout is None:
        return False
    lo, hi = keepout
    a, b = (pan_a, pan_b) if pan_a <= pan_b else (pan_b, pan_a)
    return a <= hi and lo <= b


@dataclass
class MissionDebug:
    """Introspection for the overlay and tests."""

    state: State = "SEARCH"
    target_range_m: float | None = None
    solution: ballistics.FiringSolution | None = None
    azimuth_error_deg: float | None = None
    range_error_px: float | None = None
    spray_elapsed_s: float = 0.0
    note: str = ""  # human-readable status, e.g. "target out of reach"


class MissionController:
    def __init__(self, cfg: FireTurretConfig, camera: CameraModel, armed: bool = True) -> None:
        self.cfg = cfg
        self.camera = camera
        self.servo = VisualServo(cfg.servo, cfg.turret, camera)
        self.state: State = "SEARCH"
        self.debug = MissionDebug()

        # Water arming: when disarmed, the controller NEVER commands pump/valve,
        # regardless of state. The simulator/CLI arm it; real hardware stays
        # disarmed (dry-aim) until the operator explicitly arms water.
        self.armed = armed
        # A fault (E-stop / telemetry loss) LATCHES SAFE: it does not auto-resume
        # when the fault clears; the operator must rearm(). Safer than the mission
        # re-arming autonomy the instant a blip passes.
        self._safe_latched = False

        self._pan_cmd = 0.0
        self._tilt_cmd = PARK_TILT_DEG
        self._pump_cmd = 0.0
        self._valve = False
        self._search_dir = 1.0
        self._state_elapsed = 0.0
        self._spray_elapsed = 0.0
        self._clear_elapsed = 0.0
        self._unreachable_streak = 0  # backs off retries on a persistently unreachable target
        # Candidates that already burned a full ACQUIRE window without confirming.
        # Without this the sweep re-stops on the same object forever: see
        # _do_search. Track ids are never reused, so a genuinely new object is
        # always re-examined, and a track that later confirms arrives as `target`
        # rather than `candidate` and is unaffected.
        self._rejected: set[int] = set()
        self._align_ok = 0  # consecutive settled-and-centred frames in ALIGN
        self._ranger = RangeEstimator(camera, cfg.mission)
        # elevation of maximum reach (full pump): the ceiling for tilt when the
        # pump has saturated but the fire is still short (e.g. receding)
        self._peak_tilt = ballistics.peak_elevation(
            ballistics.exit_velocity(100.0, cfg.jet),
            cfg.jet, cfg.turret.tilt_min_deg, cfg.turret.tilt_max_deg,
        )
        # the jet's reachable range envelope, for operator advisories
        self._min_reach, self._max_reach = ballistics.reach_bounds(
            cfg.jet, cfg.turret.tilt_min_deg, cfg.turret.tilt_max_deg
        )
        self.advisories: list[Advisory] = []

        # Keep-out transit hold: water launched while the nozzle was sweeping
        # across a protected sector is still in the air after the sweep ends, so
        # clearing the sector does not immediately make spraying safe. Sized from
        # the WORST-CASE flight time (full pump on the suppression arc) so the
        # hold always outlasts the airborne water.
        self.keepout_hold_s = (
            ballistics.flight_time_of(
                ballistics.exit_velocity(100.0, cfg.jet),
                cfg.servo.suppress_tilt_deg,
                cfg.jet,
            )
            + KEEPOUT_HOLD_MARGIN_S
        )
        self._keepout_hold_remaining = 0.0
        # Previous commanded pan, for the slew-rate arm. None until the first
        # gated command, so the very first tick can never look like a 0 -> pan
        # step at infinite rate.
        self._last_pan_cmd: float | None = None

        # THE water gate. SP9 registers async safety vetoes into this same chain
        # rather than adding another gate elsewhere.
        self.gate = standard_gate(cfg)
        self.water_denials: tuple[str, ...] = ()

        # The suppression correction law (invariants 1, 3, 5, 8). Owns the
        # observation buffer, the cycle counters and the boresight estimate; the
        # state machine and the _pan/_tilt/_pump command surface stay here.
        self._cycle = SuppressionCycle(cfg, camera, self.servo, self._peak_tilt)

        # The spray's shape, as far as control is concerned — four numbers.
        # Defaults to a solid stream (a point footprint), which is what the fire
        # profile's nozzle actually is and what every existing fixture assumes.
        # A profile with a fan or fog nozzle replaces this.
        self.spray_envelope = LEGACY_ENVELOPE

    # --------------------------------------------------------------- arming

    def arm(self) -> None:
        """Enable water. The operator's deliberate act before any spray."""
        self.armed = True

    def disarm(self) -> None:
        self.armed = False

    def rearm(self) -> None:
        """Acknowledge a latched SAFE fault so autonomy can resume once the fault
        has cleared. Does NOT re-enable water — call arm() for that."""
        self._safe_latched = False

    # ------------------------------------------------------------------ util

    def _enter(self, state: State) -> None:
        self.state = state
        self._state_elapsed = 0.0
        if state in ("SEARCH", "SUPPRESS"):
            self.debug.note = ""
        if state == "ALIGN":
            self._align_ok = 0
        if state == "SEARCH":
            self._pump_cmd = 0.0
            self._valve = False
            self._tilt_cmd = PARK_TILT_DEG
            self._unreachable_streak = 0  # fresh search: forget past unreachability
            self.servo.reset()
        if state == "ALIGN":
            self.servo.reset()
        if state == "SUPPRESS":
            self._spray_elapsed = 0.0
            self._cycle.reset()
            self.servo.reset()
        if state == "CONFIRM":
            self._clear_elapsed = 0.0
        if state == "SAFE":
            self._pump_cmd = 0.0
            self._valve = False

    def _command(self) -> RigCommand:
        turret = self.cfg.turret
        pan = clamp(self._pan_cmd, turret.pan_min_deg, turret.pan_max_deg)
        return RigCommand(
            pan_deg=self._apply_keepout(pan),
            tilt_deg=clamp(self._tilt_cmd, turret.tilt_min_deg, turret.tilt_max_deg),
            pump_pct=clamp(self._pump_cmd, 0.0, 100.0),
            valve=self._valve,
        )

    def tick_keepout_hold(self, dt: float) -> None:
        """Advance the post-transit water hold. Separate from `_apply_swept_keepout`
        so the gate itself stays a pure decision on current state."""
        self._keepout_hold_remaining = max(0.0, self._keepout_hold_remaining - dt)

    def _apply_swept_keepout(
        self, cmd: RigCommand, telemetry: RigTelemetry, dt: float = 0.0
    ) -> RigCommand:
        """Close the valve whenever the nozzle's swept path touches a keep-out.

        DENY-ONLY: this may remove water, never re-aim. A safety layer that
        commands actuator motion acquires its own failure surface, and the aim is
        already constrained by `_apply_keepout` and the geofence clamp.

        Two things arm the hold, and it outlives both, because water launched
        during a sweep is still airborne when the sweep ends:
          1. a TRANSIT — commanded and actual pan straddle the sector, so the
             nozzle crosses it however briefly;
          2. a fast SLEW — above `KEEPOUT_SLEW_HOLD_DPS` the water lands along an
             arc rather than at the aim point, so the straddle test in (1) stops
             describing where the water actually goes.
        """
        keepout = self.cfg.turret.pan_keepout_deg
        if keepout is None:
            return cmd
        if swept_intersects_keepout(telemetry.pan_deg, cmd.pan_deg, keepout):
            self._keepout_hold_remaining = self.keepout_hold_s
        elif dt > 0.0 and self._last_pan_cmd is not None:
            slew_dps = abs(cmd.pan_deg - self._last_pan_cmd) / dt
            if slew_dps > KEEPOUT_SLEW_HOLD_DPS:
                self._keepout_hold_remaining = self.keepout_hold_s
        self._last_pan_cmd = cmd.pan_deg
        if self._keepout_hold_remaining <= 0.0:
            return cmd
        return replace(cmd, pump_pct=0.0, valve=False)

    def _apply_deposition_keepout(self, cmd: RigCommand) -> RigCommand:
        """Withhold water when the spray FOOTPRINT would wet a protected sector,
        even though the AIM is legal.

        `_apply_keepout` constrains where the nozzle points; this constrains where
        the water lands. For the default solid-stream envelope the footprint is a
        point and this is exactly equivalent to the aim check — which is why it is
        behaviour-neutral today. For a fan pattern it is the whole problem: at 7 m
        a 10 deg fan reaches 0.61 m either side, so an aim 3 deg outside a sector
        edge still wets 0.24 m inside it.

        Routed through the SAME note/HOLD path as an unreachable target, so it
        inherits invariant 9's backoff rather than inventing its own.
        """
        keepout = self.cfg.turret.pan_keepout_deg
        if keepout is None or self.spray_envelope.is_point:
            return cmd
        if not (cmd.valve or cmd.pump_pct > 0.0):
            return cmd
        range_m = self._ranger.ema or self.cfg.mission.default_range_m
        violation = check_deposition(cmd.pan_deg, range_m, keepout, self.spray_envelope)
        if violation is None:
            return cmd
        self.debug.note = violation.reason
        if self.state == "SUPPRESS":
            self._valve = False
            self._pump_cmd = 0.0
            self._unreachable_streak += 1
            self._enter("HOLD")
        return replace(cmd, pump_pct=0.0, valve=False)

    def _apply_keepout(self, pan: float) -> float:
        """Keep the nozzle out of the configured keep-out sector by pushing any
        command inside it to the nearer edge (never aim across people)."""
        ko = self.cfg.turret.pan_keepout_deg
        if ko is None:
            return pan
        lo, hi = ko
        if lo < pan < hi:
            return lo if (pan - lo) <= (hi - pan) else hi
        return pan

    # `target_ground_px` and the range EMA now live in control/ranging.py, so
    # control/suppress.py can be AST-tested against ever reaching for them
    # (invariant 1: servo in image space, never on the range estimate).
    def _target_ground_px(self, target: Track) -> tuple[float, float]:
        return target_ground_px(target)

    def _estimate_target_range(self, target: Track) -> float:
        return self._ranger.update(target)

    # ------------------------------------------------------------------ tick

    def update(
        self,
        dt: float,
        target: Track | None,
        candidate: Track | None,
        impact: ImpactObservation | None,
        telemetry: RigTelemetry,
    ) -> RigCommand:
        self._state_elapsed += dt
        self.tick_keepout_hold(dt)

        # SAFE dominates everything. A fault LATCHES SAFE and disarms water; it
        # does not auto-resume — the operator must rearm() once the fault clears.
        fault = (not telemetry.ok) or telemetry.estop
        if fault:
            if self.state != "SAFE":
                self._enter("SAFE")
            self._safe_latched = True
            self.armed = False
        elif self.state == "SAFE" and not self._safe_latched:
            self._enter("SEARCH")

        handler = {
            "SEARCH": self._do_search,
            "ACQUIRE": self._do_acquire,
            "RANGE": self._do_range,
            "ALIGN": self._do_align,
            "SUPPRESS": self._do_suppress,
            "CONFIRM": self._do_confirm,
            "HOLD": self._do_hold,
            "SAFE": self._do_safe,
        }[self.state]
        cmd = handler(dt, target, candidate, impact, telemetry)

        # THE water gate (control/gate.py). Every rule that can withhold water
        # lives in one AND-chain now, so the mission, the CLI and the web console
        # cannot drift apart — and SP9's async vetoes plug into the same chain
        # rather than adding a fourth gate.
        cmd, self.water_denials = self.gate.apply(
            cmd,
            GateContext(
                command=cmd, telemetry=telemetry, dt=dt,
                armed=self.armed, spray_elapsed_s=self._spray_elapsed,
            ),
        )
        # The keep-out transit hold stays here rather than in the gate: it is
        # STATEFUL (it arms a timer from the swept path and outlives the transit),
        # and gate vetoes are pure predicates over the context they are handed.
        # Behaviour-neutral when no keep-out is configured.
        cmd = self._apply_swept_keepout(cmd, telemetry, dt)
        cmd = self._apply_deposition_keepout(cmd)

        self.debug.state = self.state
        self.debug.spray_elapsed_s = self._spray_elapsed
        self._update_advisories(target, telemetry)
        # drive the hardware warning indicator when anything needs the operator
        warn = any(a.rank >= _RANK[WARN] for a in self.advisories)
        return replace(cmd, warn=warn)

    def _update_advisories(self, target: Track | None, telemetry: RigTelemetry) -> None:
        """Build the operator advisory list from the current situation."""
        adv: list[Advisory] = []
        if self.state == "SAFE":
            self.advisories = [
                Advisory("SAFETY", CRITICAL, "E-stop engaged or telemetry lost",
                         "Clear the fault and check the link / E-stop")
            ]
            return

        rng: float | None = None
        if target is not None:
            est = self.camera.ground_range_from_px(*self._target_ground_px(target))
            if est is not None:
                rng = clamp(est, 1.0, 60.0)

        # 1) a concrete engagement failure (we tried and HELD) is high-confidence
        note = self.debug.note if self.state == "HOLD" else ""
        range_str = f" (~{rng:.1f} m)" if rng is not None else ""
        if "out of reach" in note:
            adv.append(Advisory("TOO_FAR", CRITICAL, f"Fire out of reach{range_str}",
                                "Move the turret closer to the fire"))
        elif "too close" in note:
            adv.append(Advisory("TOO_CLOSE", CRITICAL, f"Fire too close{range_str}",
                                "Move the turret back from the fire"))
        elif "no splash" in note or "check aim" in note:
            adv.append(Advisory("OBSTRUCTED", WARN,
                                "Cannot see where the water lands (obstructed or too close)",
                                "Clear the line of fire or reposition the turret"))

        # 2) proactive geometry advisory (before we even fail), if not already flagged
        if rng is not None and not any(a.kind in ("TOO_FAR", "TOO_CLOSE") for a in adv):
            r = reachability_advisory(rng, self._min_reach, self._max_reach)
            if r is not None:
                adv.append(r)

        # 3) traverse-limit advisory
        if target is not None:
            t = traverse_advisory(
                self._absolute_azimuth(target, telemetry),
                self.cfg.turret.pan_min_deg, self.cfg.turret.pan_max_deg,
            )
            if t is not None:
                adv.append(t)

        self.advisories = adv

    # ---------------------------------------------------------------- states

    def _do_search(
        self, dt: float, target: Track | None, candidate: Track | None,
        impact: ImpactObservation | None, telemetry: RigTelemetry,
    ) -> RigCommand:
        turret = self.cfg.turret
        self._pan_cmd += self._search_dir * self.cfg.mission.search_pan_rate_dps * dt
        if self._pan_cmd >= turret.pan_max_deg:
            self._pan_cmd = turret.pan_max_deg
            self._search_dir = -1.0
        elif self._pan_cmd <= turret.pan_min_deg:
            self._pan_cmd = turret.pan_min_deg
            self._search_dir = 1.0
        # A candidate that has already failed to confirm must not stop the sweep
        # again. It used to: _do_search steps the sweep, then throws that step
        # away by resetting to the rig's actual angle, and enters ACQUIRE. ACQUIRE
        # times out because the flicker gate correctly refuses to confirm a static
        # fire-coloured object, and the very next frame does the same thing. Net
        # pan movement per cycle: zero, forever. The turret is then permanently
        # blind to any real fire outside its current field of view while sitting
        # in ACQUIRE looking busy - and the two tests covering the decoy scenario
        # assert only that the valve never opened, so both pass while it happens.
        # A CONFIRMED target ends the sweep immediately, exactly as it does in
        # ACQUIRE. Without this, the _rejected set introduced above could strand a
        # real fire: a track that burns its acquire window is remembered, and if
        # it later confirms it is still skipped as a candidate here while nothing
        # looks at `target` — so the turret sweeps straight past a confirmed fire
        # with the valve shut, for as long as it stays in frame. Confirmation is
        # also the moment the rejection stops being true, so the id is forgotten.
        if target is not None:
            self._rejected.discard(target.track_id)
            self._enter("RANGE")
            return self._command()

        if candidate is not None and candidate.track_id not in self._rejected:
            self._pan_cmd = telemetry.pan_deg  # stop the sweep where we are
            self._enter("ACQUIRE")
        return self._command()

    def _do_acquire(
        self, dt: float, target: Track | None, candidate: Track | None,
        impact: ImpactObservation | None, telemetry: RigTelemetry,
    ) -> RigCommand:
        if target is not None:
            self._enter("RANGE")
        elif candidate is None:
            self._enter("SEARCH")
        elif self._state_elapsed > self.cfg.mission.acquire_timeout_s:
            # It had a full acquire window and did not confirm. Remember that, so
            # the sweep gets past it instead of stopping on it again next frame.
            self._rejected.add(candidate.track_id)
            self._enter("SEARCH")
        return self._command()

    def _do_range(
        self, dt: float, target: Track | None, candidate: Track | None,
        impact: ImpactObservation | None, telemetry: RigTelemetry,
    ) -> RigCommand:
        if target is None:
            self._enter("SEARCH")
            return self._command()

        target_range = self._estimate_target_range(target)
        self.debug.target_range_m = target_range

        # Fix tilt on the low arc so the target sits mid pump-envelope; pump
        # becomes the range control (monotonic, no peak ambiguity). The closed
        # loop then trims pump both ways from the observed splash.
        solution = ballistics.plan_suppression(
            target_range,
            self.cfg.jet,
            self.cfg.turret.tilt_min_deg,
            self.cfg.turret.tilt_max_deg,
        )
        self.debug.solution = solution
        if solution is None:
            # No pump/elevation lands the estimated range. This catches BOTH
            # out-of-envelope cases: a target beyond max reach AND one nearer than
            # the low-arc minimum. For both we adopt the max-reach posture (full
            # pump at the peak-range elevation) and let the closed loop judge —
            # deliberately, because the monocular range estimate is unreliable, so
            # we never refuse to fire on the estimate alone. For a too-close fire
            # this overshoots and the splash lands occluded behind the flame; the
            # resulting no-splash-feedback path (see _do_suppress) then recognises
            # it and HOLDs rather than hosing blind (tested end-to-end).
            speed = ballistics.exit_velocity(100.0, self.cfg.jet)
            self._tilt_cmd = ballistics.peak_elevation(
                speed, self.cfg.jet, self.cfg.turret.tilt_min_deg, self.cfg.turret.tilt_max_deg
            )
            self._pump_cmd = 100.0
        else:
            self._tilt_cmd = solution.elevation_deg
            # Open deliberately SHORT and let SUPPRESS walk the pump up: a short
            # splash lands in front of the fire and is always visible, whereas a
            # long splash lands behind the flame and is occluded — so the servo
            # would lose the feedback exactly when it overshoots. Approach from
            # below and the splash stays observable the whole way in.
            self._pump_cmd = clamp(
                solution.pump_pct - OPEN_SHORT_PCT, self.cfg.jet.min_pump_pct, 100.0
            )
        # aim pan at the fire's ABSOLUTE world azimuth (camera-relative bearing
        # plus where the turret currently points); ALIGN refines once settled
        self._pan_cmd = self._absolute_azimuth(target, telemetry)
        self._enter("ALIGN")
        return self._command()

    def _absolute_azimuth(self, target: Track, telemetry: RigTelemetry) -> float:
        err = self.servo.azimuth_error_deg(target.cx)
        return clamp(
            telemetry.pan_deg + err, self.cfg.turret.pan_min_deg, self.cfg.turret.pan_max_deg
        )

    def _do_align(
        self, dt: float, target: Track | None, candidate: Track | None,
        impact: ImpactObservation | None, telemetry: RigTelemetry,
    ) -> RigCommand:
        if target is None:
            self._enter("SEARCH")
            return self._command()
        err = self.servo.azimuth_error_deg(target.cx)
        self.debug.azimuth_error_deg = err

        # Move → settle → re-measure: only trust the (stationary) reading and
        # issue a new absolute command once the pan has reached the last one.
        # Servoing every frame while slewing off laggy detection oscillates.
        settled = abs(self._pan_cmd - telemetry.pan_deg) <= 1.0
        if settled:
            if abs(err) <= self.cfg.servo.pan_deadband_deg:
                self._align_ok += 1
            else:
                self._pan_cmd = self._absolute_azimuth(target, telemetry)
                self._align_ok = 0

        pump_ready = telemetry.pump_pct >= self._pump_cmd - 8.0
        if (self._align_ok >= self.cfg.servo.settle_frames and pump_ready) or (
            self._state_elapsed > self.cfg.mission.align_timeout_s
        ):
            self._valve = True
            self._enter("SUPPRESS")
        return self._command()

    def _do_suppress(
        self, dt: float, target: Track | None, candidate: Track | None,
        impact: ImpactObservation | None, telemetry: RigTelemetry,
    ) -> RigCommand:
        self._spray_elapsed += dt
        if target is None:
            self._enter("CONFIRM")
            return self._command()
        if self._spray_elapsed > self.cfg.mission.max_spray_s:
            # re-evaluate from scratch rather than hose forever
            self._valve = False
            self._pump_cmd = 0.0
            self._enter("RANGE")
            return self._command()

        # The correction law itself lives in control/suppress.py — the one file
        # where invariants 1, 3, 5 and 8 are implemented, AST-tested against ever
        # touching a range estimate. The state machine stays here.
        correction = self._cycle.step(target, impact, telemetry, self._tilt_cmd)
        if correction.pan_deg is not None:
            self._pan_cmd = correction.pan_deg
        if correction.tilt_deg is not None:
            self._tilt_cmd = correction.tilt_deg
        if correction.pump_pct is not None:
            self._pump_cmd = correction.pump_pct
        if correction.azimuth_error_deg is not None:
            self.debug.azimuth_error_deg = correction.azimuth_error_deg
        if correction.range_error_px is not None:
            self.debug.range_error_px = correction.range_error_px

        if correction.stop_reason:
            self.debug.note = correction.stop_reason
            self._valve = False
            self._pump_cmd = 0.0
            self._unreachable_streak += 1
            self._enter("HOLD")
        # otherwise HOLD the aim (do not touch pan/tilt/pump) so the camera
        # stays still and frame-differencing keeps working
        return self._command()

    def _do_confirm(
        self, dt: float, target: Track | None, candidate: Track | None,
        impact: ImpactObservation | None, telemetry: RigTelemetry,
    ) -> RigCommand:
        if target is not None:
            # it flared back up
            self._valve = True
            self._enter("SUPPRESS")
            return self._command()
        if self._state_elapsed <= self.cfg.mission.soak_s:
            self._valve = True  # soak the embers briefly
        else:
            self._valve = False
            self._pump_cmd = 0.0
            self._clear_elapsed += dt
            if self._clear_elapsed >= self.cfg.mission.confirm_clear_s:
                self._enter("SEARCH")
        return self._command()

    def _do_hold(
        self, dt: float, target: Track | None, candidate: Track | None,
        impact: ImpactObservation | None, telemetry: RigTelemetry,
    ) -> RigCommand:
        # Target is out of reach: valve shut, no water wasted. Keep the camera
        # on it and periodically retry (it may spread closer, or be a bad read).
        self._valve = False
        self._pump_cmd = 0.0
        if target is None:
            self.debug.note = ""
            self._enter("SEARCH")
            return self._command()
        self._pan_cmd = self.servo.azimuth_step(target.cx, telemetry.pan_deg)
        # exponential backoff: a target that stays unreachable is rechecked ever
        # less often, so we stop test-spraying water we cannot land
        dwell = min(self.cfg.mission.hold_retry_s * self._unreachable_streak, 45.0)
        if self._state_elapsed >= dwell:
            self.debug.note = ""
            self._enter("RANGE")
        return self._command()

    def _do_safe(
        self, dt: float, target: Track | None, candidate: Track | None,
        impact: ImpactObservation | None, telemetry: RigTelemetry,
    ) -> RigCommand:
        self._pump_cmd = 0.0
        self._valve = False
        return self._command()
