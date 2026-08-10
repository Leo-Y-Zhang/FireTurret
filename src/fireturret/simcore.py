# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""Qt-free simulation core: a pure per-tick generator (`simulate`) and the shared
accumulation + stop-policy loop (`drive`). `fireturret.app.run_sim` is a thin wrapper
over `drive` that adds overlay/imshow/record.

Extracted verbatim from the old `run_sim` loop so behaviour is preserved (guarded
by `tests/test_golden_behavior.py`). Do NOT "clean up" the accumulation order, the
`sim_seconds += dt` fold, the miss-sample gating, or the extinguish-tail policy —
they are behaviour-defining.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np

from .analysis import TelemetrySample
from .app import FireOutcome, Pipeline, SimReport, TickResult
from .config import FireTurretConfig
from .control.advisories import Advisory, most_urgent
from .rig.interface import RigTelemetry
from .rig.sim_rig import SimRig, SimScenario

_DT = 1.0 / 30.0


@dataclass
class SimStep:
    """Everything one tick produces. Holds a mutable `TickResult` (live Tracks) and
    a raw ndarray, so a SimStep must be consumed IN-THREAD — never emitted across a
    Qt thread boundary. Only the frozen `sample` and a copied frame may cross."""

    frame: np.ndarray            # raw rendered frame (fresh per tick)
    result: TickResult           # command, tracks, target, impact (mutable Tracks)
    sample: TelemetrySample      # frozen 14-field telemetry row
    advisories: tuple[Advisory, ...]  # FULL advisory objects raised this tick
    mission_debug: object        # snapshot for draw_overlay
    rig_telemetry: RigTelemetry  # snapshot for draw_overlay + parity
    primary_intensity: float     # fires[0].intensity -> final_intensity
    fire_intensities: tuple[float, ...]  # per-fire intensities this tick (multi-fire)
    miss_recorded: bool          # rig.splash is not None AND a live fire exists
    extinguished: bool
    water_used_l: float
    frame_index: int
    sim_seconds: float           # the running += dt fold (NOT frame_index * dt)


def simulate(
    cfg: FireTurretConfig,
    scenario: SimScenario | None = None,
    *,
    seed: int = 7,
    max_frames: int = 6000,
) -> Iterator[SimStep]:
    """Pure engine. Per tick: rig.step -> render -> pipeline.tick -> rig.command
    -> build SimStep -> yield. rig.command() (the visual-servo feedback) stays
    INSIDE; the consumer decides cadence/stop/retain and never touches the rig."""
    rig = SimRig(cfg, scenario or SimScenario(), seed=seed)
    pipeline = Pipeline(cfg)
    sim_seconds = 0.0

    for i in range(max_frames):
        rig.step(_DT)
        frame = rig.render()
        result = pipeline.tick(frame, _DT, rig.telemetry())
        rig.command(result.command)

        sim_seconds += _DT
        state = pipeline.mission.state
        advisories = tuple(pipeline.mission.advisories)

        miss = 0.0
        miss_recorded = False
        if rig.splash is not None:
            live = [f for f in rig.fires if f.intensity > 0.02]
            if live:
                miss = min(
                    ((rig.splash.x - f.position.x) ** 2 + (rig.splash.z - f.position.z) ** 2) ** 0.5
                    for f in live
                )
                miss_recorded = True

        d = pipeline.mission.debug
        top = most_urgent(pipeline.mission.advisories)
        sample = TelemetrySample(
            t=sim_seconds, state=state,
            pan_cmd=result.command.pan_deg, pan_act=rig.pan, tilt=rig.tilt,
            pump=rig.pump, valve=rig.valve,
            range_est_m=d.target_range_m or 0.0,
            az_err_deg=d.azimuth_error_deg or 0.0,
            range_err_px=d.range_error_px or 0.0,
            fire_intensity=sum(f.intensity for f in rig.fires),
            water_l=rig.water_used_l, splash_miss_m=miss,
            warning=top.kind if top else "",
        )

        yield SimStep(
            frame=frame, result=result, sample=sample, advisories=advisories,
            mission_debug=d, rig_telemetry=rig.telemetry(),
            primary_intensity=rig.fire.intensity,
            fire_intensities=tuple(f.intensity for f in rig.fires),
            miss_recorded=miss_recorded,
            extinguished=rig.extinguished, water_used_l=rig.water_used_l,
            frame_index=i, sim_seconds=sim_seconds,
        )


def drive(
    cfg: FireTurretConfig,
    scenario: SimScenario | None = None,
    *,
    seed: int = 7,
    max_frames: int = 6000,
    stop_when_extinguished: bool = True,
    record_telemetry: bool = False,
    on_step: Callable[[SimStep], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
    steps: Iterator[SimStep] | None = None,
) -> SimReport:
    """The single behaviour-defining loop shared by `run_sim` and the Studio worker.
    Order per tick: fold the report -> call `on_step` -> evaluate the extinguish-tail
    break. Metrics are ALWAYS computed; `record_telemetry` only gates whether the
    TelemetrySample list is retained. `should_stop` cooperatively cancels. `steps`
    overrides the SimStep source (defaults to `simulate(...)`); it exists so the
    Studio worker and its tests can inject a source without re-implementing this loop."""
    report = SimReport(extinguished=False, frames=0, sim_seconds=0.0)
    miss_samples: list[float] = []
    per_fire_time: list[float | None] = []  # when each fire first went out
    per_fire_residual: list[float] = []     # each fire's last intensity
    tail_frames = int((cfg.mission.soak_s + cfg.mission.confirm_clear_s + 1.5) / _DT)
    stop_at: int | None = None

    source = steps if steps is not None else simulate(cfg, scenario, seed=seed, max_frames=max_frames)
    for step in source:
        if should_stop is not None and should_stop():
            break
        report.frames = step.frame_index + 1
        report.sim_seconds = step.sim_seconds
        report.states_visited.add(step.sample.state)
        report.extinguished = step.extinguished
        report.water_used_l = step.water_used_l
        report.final_intensity = step.primary_intensity
        report.advisories_raised.update(a.kind for a in step.advisories)
        if report.time_to_first_suppress_s is None and step.sample.state == "SUPPRESS":
            report.time_to_first_suppress_s = step.sim_seconds
        if step.miss_recorded:
            miss_samples.append(step.sample.splash_miss_m)
        if not per_fire_time:  # first step sizes the per-fire trackers
            per_fire_time = [None] * len(step.fire_intensities)
            per_fire_residual = [1.0] * len(step.fire_intensities)
        for i, intensity in enumerate(step.fire_intensities):
            per_fire_residual[i] = intensity
            if intensity <= 0.0 and per_fire_time[i] is None:
                per_fire_time[i] = step.sim_seconds
        if record_telemetry:
            report.telemetry.append(step.sample)
        if on_step is not None:
            on_step(step)
        if step.extinguished and stop_when_extinguished:
            if stop_at is None:
                stop_at = step.frame_index + tail_frames
            if step.frame_index >= stop_at:
                break

    if miss_samples:
        report.mean_miss_m = sum(miss_samples) / len(miss_samples)
        report.peak_miss_m = max(miss_samples)
    report.per_fire = [
        FireOutcome(extinguished=t is not None, extinguish_time_s=t, residual_intensity=r)
        for t, r in zip(per_fire_time, per_fire_residual, strict=True)
    ]
    return report
