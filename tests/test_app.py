# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
import numpy as np

from fireturret.app import Pipeline, TickResult, run_sim
from fireturret.config import DEFAULT_CONFIG
from fireturret.rig.interface import RigTelemetry
from fireturret.rig.sim_rig import SimScenario


def idle_telemetry() -> RigTelemetry:
    return RigTelemetry(0.0, 20.0, 0.0, False, estop=False, ok=True)


def test_pipeline_tick_on_blank_frame_is_safe() -> None:
    pipe = Pipeline(DEFAULT_CONFIG)
    frame = np.zeros((DEFAULT_CONFIG.camera.height, DEFAULT_CONFIG.camera.width, 3), dtype=np.uint8)
    result = pipe.tick(frame, 1 / 30, idle_telemetry())
    assert isinstance(result, TickResult)
    assert result.command is not None
    assert result.target is None  # nothing to see on a black frame
    # with no fire the controller searches, valve shut
    assert result.command.valve is False


def test_pipeline_tick_returns_valid_ranges() -> None:
    pipe = Pipeline(DEFAULT_CONFIG)
    frame = np.zeros((DEFAULT_CONFIG.camera.height, DEFAULT_CONFIG.camera.width, 3), dtype=np.uint8)
    for _ in range(20):
        cmd = pipe.tick(frame, 1 / 30, idle_telemetry()).command
        assert DEFAULT_CONFIG.turret.pan_min_deg <= cmd.pan_deg <= DEFAULT_CONFIG.turret.pan_max_deg
        assert 0.0 <= cmd.pump_pct <= 100.0


def test_detector_is_injectable() -> None:
    # any object with .detect(frame, camera_moving) -> list[FireBlob] drops in;
    # here a stub that reports one fixed fire blob every frame
    from fireturret.vision.firedetect import FireBlob

    class StubDetector:
        def __init__(self) -> None:
            self.calls = 0

        def detect(self, frame, camera_moving=False):
            self.calls += 1
            return [FireBlob(cx=480.0, cy=270.0, area=900.0, bbox=(460, 250, 40, 40),
                             colour_score=0.9, flicker_score=0.3, confidence=0.9)]

    stub = StubDetector()
    pipe = Pipeline(DEFAULT_CONFIG, detector=stub)
    frame = np.zeros((DEFAULT_CONFIG.camera.height, DEFAULT_CONFIG.camera.width, 3), dtype=np.uint8)
    for _ in range(12):
        result = pipe.tick(frame, 1 / 30, idle_telemetry())
    assert stub.calls == 12
    assert result.target is not None  # the injected detections drive tracking


def test_sim_report_has_performance_metrics() -> None:
    rep = run_sim(DEFAULT_CONFIG, SimScenario(), seed=7, headless=True, max_frames=2400)
    assert rep.extinguished
    assert rep.time_to_first_suppress_s is not None
    assert rep.time_to_first_suppress_s > 0
    assert rep.peak_miss_m >= rep.mean_miss_m >= 0.0


def test_run_sim_is_deterministic() -> None:
    a = run_sim(DEFAULT_CONFIG, SimScenario(), seed=11, headless=True, max_frames=400)
    b = run_sim(DEFAULT_CONFIG, SimScenario(), seed=11, headless=True, max_frames=400)
    assert a.frames == b.frames
    assert a.extinguished == b.extinguished
    assert a.final_intensity == b.final_intensity
    assert a.states_visited == b.states_visited
