# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""Pipeline wiring and run loops.

`Pipeline.tick(frame, dt, telemetry)` is the whole system for one frame:
detect → track → observe splash → mission update → rig command. The run
loops feed it from the simulator, a webcam, or a video file.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import cv2
import numpy as np

from .config import FireTurretConfig
from .control.mission import MissionController
from .geometry import CameraModel
from .rig.interface import RigCommand, RigTelemetry, TurretRig
from .ui.overlay import draw_overlay
from .vision.egomotion import pan_warp
from .vision.firedetect import FireDetector
from .vision.impact import ImpactObservation, SplashDetector
from .vision.tracker import FireTracker, Track

if TYPE_CHECKING:  # the engine must not import the simulator (see §3.9)
    # `SimScenario` appears here only as a type annotation, and
    # `from __future__ import annotations` makes those strings. `run_sim`
    # imports the simulator inside its own body, where the dependency is real.
    from .rig.sim_rig import SimScenario


@dataclass
class TickResult:
    command: RigCommand
    tracks: list[Track]
    target: Track | None
    impact: ImpactObservation | None


class Pipeline:
    def __init__(self, cfg: FireTurretConfig, detector: FireDetector | None = None,
                 armed: bool = True, profile=None) -> None:
        """`profile` supplies the domain-specific wiring (spray envelope, image
        setpoint, advisory vocabulary). `None` is the legacy path and
        `FireProfile()` must be **bit-identical** to it — asserted by
        `tests/test_profile_identity.py`. An abstraction whose flagship case is
        not provably identical to the code it replaced has not been extracted."""
        self.cfg = cfg
        self.profile = profile
        self.camera = CameraModel(cfg.camera)
        # the detector is injectable: anything with .detect(frame, camera_moving)
        # -> list[FireBlob] works, so a learned model can replace the classical
        # one without touching the tracker, control, or rig layers
        self.detector = detector if detector is not None else FireDetector(cfg.detector)
        self.tracker = FireTracker(cfg.detector)
        self.splash = SplashDetector()
        # armed=False keeps the control layer from ever commanding water (dry-aim)
        self.mission = MissionController(cfg, self.camera, armed=armed)
        if profile is not None:
            # A profile may only ever NARROW: the spray envelope changes what the
            # deposition check considers wetted, and the constraints intersect
            # (strictest wins). Nothing here can widen a limit.
            self.mission.spray_envelope = profile.spray_envelope
        self._prev_pan: float | None = None

    def tick(self, frame: np.ndarray, dt: float, telemetry: RigTelemetry) -> TickResult:
        # pan motion since last frame: drives flicker gating AND the tracker's
        # motion compensation (a world-fixed fire shifts -Δpan in image angle)
        pan_delta = 0.0 if self._prev_pan is None else telemetry.pan_deg - self._prev_pan
        self._prev_pan = telemetry.pan_deg
        moving = dt > 0 and abs(pan_delta) / dt > 3.0

        # Ego-motion prediction. The exact yaw warp, not the small-angle scalar
        # it replaces: the pinhole map is u = cx + fpx*tan(az), so how far a
        # track moves depends on where in the frame it is. The old single scalar
        # under-corrected by ~25% at 30 deg off-axis, which `match_dist_px = 80`
        # was quietly absorbing. See vision/egomotion.py.
        def _warp(u: float, v: float) -> tuple[float, float]:
            return pan_warp(u, pan_delta, self.camera), v

        blobs = self.detector.detect(frame, camera_moving=moving)
        tracks = self.tracker.update(blobs, warp=_warp if pan_delta else None)
        target = self.tracker.primary()
        candidate = None
        if target is None:
            live = [t for t in tracks if t.hits >= 3 and t.misses <= 2]
            candidate = max(live, key=lambda t: t.confidence * t.area, default=None)

        impact: ImpactObservation | None = None
        if self.mission.state in ("SUPPRESS", "CONFIRM") and target is not None:
            exclude = target.last_blob.bbox if target.last_blob else None
            impact = self.splash.observe(frame, (target.cx, target.cy), exclude)
        else:
            self.splash.reset()

        command = self.mission.update(dt, target, candidate, impact, telemetry)
        return TickResult(command=command, tracks=tracks, target=target, impact=impact)


@dataclass(frozen=True)
class FireOutcome:
    """Per-fire result, for multi-fire scenarios."""

    extinguished: bool
    extinguish_time_s: float | None  # sim seconds when it first went out (None if not)
    residual_intensity: float  # remaining intensity at the end (0 if extinguished)


@dataclass
class SimReport:
    extinguished: bool
    frames: int
    sim_seconds: float
    states_visited: set[str] = field(default_factory=set)
    water_used_l: float = 0.0
    final_intensity: float = 1.0
    # performance metrics
    time_to_first_suppress_s: float | None = None  # acquisition + aim time
    mean_miss_m: float = 0.0  # avg splash-to-fire distance while spraying
    peak_miss_m: float = 0.0
    advisories_raised: set[str] = field(default_factory=set)  # advisory kinds seen
    per_fire: list = field(default_factory=list)  # FireOutcome per fire (multi-fire)
    telemetry: list = field(default_factory=list)  # per-frame TelemetrySample (if recorded)


def run_sim(
    cfg: FireTurretConfig,
    scenario: SimScenario | None = None,
    seed: int = 7,
    headless: bool = False,
    max_frames: int = 6000,
    record_path: str | None = None,
    stop_when_extinguished: bool = True,
    record_telemetry: bool = False,
) -> SimReport:
    """Drive the full pipeline against the physics simulator (CLI/interactive).

    Physics, metrics, and the extinguish-tail stop policy live in
    ``simcore.drive``; this wrapper only adds the CLI-side overlay/imshow/record.
    ``simcore`` is imported lazily to avoid an app<->simcore import cycle.
    """
    from .simcore import SimStep, drive

    state: dict = {"writer": None, "quit": False}

    def on_step(step: SimStep) -> None:
        # match the original gate: overlay only when a window or a recording is wanted
        if headless and not record_path:
            return
        shown = draw_overlay(
            step.frame, cfg, step.result.tracks, step.result.target, step.result.impact,
            step.mission_debug, step.result.command, step.rig_telemetry,
            list(step.advisories),
        )
        if record_path:
            if state["writer"] is None:
                state["writer"] = cv2.VideoWriter(
                    record_path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    30,
                    (shown.shape[1], shown.shape[0]),
                )
            state["writer"].write(shown)
        if not headless:
            cv2.imshow("FireTurret (sim)", shown)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                state["quit"] = True

    try:
        report = drive(
            cfg, scenario, seed=seed, max_frames=max_frames,
            stop_when_extinguished=stop_when_extinguished,
            record_telemetry=record_telemetry, on_step=on_step,
            should_stop=lambda: state["quit"],
        )
    finally:
        if state["writer"] is not None:
            state["writer"].release()
        if not headless:
            cv2.destroyAllWindows()
    return report


@dataclass
class CaptureReport:
    frames: int = 0
    target_frames: int = 0  # frames with a confirmed fire target
    sprayed: bool = False  # did the mission ever open the valve


def run_capture(
    cfg: FireTurretConfig,
    source: int | str,
    rig: TurretRig,
    headless: bool = False,
    record_path: str | None = None,
    max_frames: int | None = None,
    water_enabled: bool = True,
    detector=None,
) -> CaptureReport:
    """Drive the pipeline from a webcam (int index) or a video file (path).

    With a SerialRig this is the real system; with a NullRig it is a dry-run
    aim-preview over footage the turret cannot influence. Returns a small
    summary of what was seen.

    ``water_enabled=False`` forces pump/valve off on every command (dry-aim on
    real hardware: pan/tilt track the fire but no water is ever released) — the
    safe default for first bring-up until the operator explicitly arms water.
    """
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"could not open video source: {source!r}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    dt = 1.0 / (fps if fps > 1 else 30.0)
    # the control layer itself is disarmed in dry-aim; the per-command neuter
    # below is a second, independent layer of the same guarantee. `detector` lets
    # a learned model replace the classical detector (e.g. an ONNX/GPU detector).
    pipeline = Pipeline(cfg, detector=detector, armed=water_enabled)
    writer = None
    report = CaptureReport()

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if (frame.shape[1], frame.shape[0]) != (cfg.camera.width, cfg.camera.height):
                frame = cv2.resize(frame, (cfg.camera.width, cfg.camera.height))
            telemetry = rig.telemetry()
            result = pipeline.tick(frame, dt, telemetry)
            command = result.command
            if not water_enabled:  # dry-aim: never actuate pump/valve
                command = replace(command, pump_pct=0.0, valve=False)
            rig.command(command)

            report.frames += 1
            if result.target is not None:
                report.target_frames += 1
            report.sprayed = report.sprayed or command.valve

            shown = draw_overlay(
                frame, cfg, result.tracks, result.target, result.impact,
                pipeline.mission.debug, command, telemetry,
                pipeline.mission.advisories,
            )
            if record_path:
                if writer is None:
                    writer = cv2.VideoWriter(
                        record_path, cv2.VideoWriter_fourcc(*"mp4v"),
                        int(round(1 / dt)), (shown.shape[1], shown.shape[0]),
                    )
                writer.write(shown)
            if not headless:
                cv2.imshow("FireTurret", shown)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
            if max_frames is not None and report.frames >= max_frames:
                break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if not headless:
            cv2.destroyAllWindows()
        rig.close()
    return report
