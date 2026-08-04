# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""Exercise the real-footage entry point (`run_capture`) end to end: render a
synthetic fire video, decode it back through cv2.VideoCapture, and run the full
pipeline over it with a NullRig (dry-run). Skips if the environment has no
working mp4 codec."""

import os
import tempfile

import cv2
import pytest

from triton.app import run_capture
from triton.config import DEFAULT_CONFIG
from triton.rig.interface import NullRig
from triton.rig.sim_rig import SimRig, SimScenario


def _write_fire_video(path: str, frames: int = 90) -> bool:
    """Render a stationary fire to an mp4; return False if encoding is unavailable."""
    cfg = DEFAULT_CONFIG
    rig = SimRig(cfg, SimScenario(fire_azimuth_deg=0.0, fire_range_m=6.0), seed=3)
    writer = cv2.VideoWriter(
        path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (cfg.camera.width, cfg.camera.height)
    )
    if not writer.isOpened():
        return False
    for _ in range(frames):
        rig.step(1 / 30)  # no commands ⇒ turret idle, just a burning fire in view
        writer.write(rig.render())
    writer.release()
    return os.path.getsize(path) > 0


def test_run_capture_over_video_detects_fire() -> None:
    path = os.path.join(tempfile.mkdtemp(), "fire.mp4")
    if not _write_fire_video(path):
        pytest.skip("no working mp4 encoder in this environment")
    cap = cv2.VideoCapture(path)
    ok = cap.isOpened() and cap.read()[0]
    cap.release()
    if not ok:
        pytest.skip("mp4 written but not decodable in this environment")

    report = run_capture(DEFAULT_CONFIG, path, NullRig(), headless=True, max_frames=90)
    os.remove(path)

    assert report.frames > 0  # the video decoded and drove the pipeline
    # the fire is dead ahead of an idle camera, so it should be confirmed
    assert report.target_frames > 0


class _RecordingRig(NullRig):
    def __init__(self) -> None:
        super().__init__()
        self.commands: list = []

    def command(self, cmd) -> None:
        self.commands.append(cmd)
        super().command(cmd)


def test_run_capture_dry_aim_never_actuates_water() -> None:
    path = os.path.join(tempfile.mkdtemp(), "fire.mp4")
    if not _write_fire_video(path):
        pytest.skip("no working mp4 encoder in this environment")
    cap = cv2.VideoCapture(path)
    ok = cap.isOpened() and cap.read()[0]
    cap.release()
    if not ok:
        pytest.skip("mp4 written but not decodable in this environment")

    rig = _RecordingRig()
    report = run_capture(DEFAULT_CONFIG, path, rig, headless=True, max_frames=90,
                         water_enabled=False)
    os.remove(path)

    assert rig.commands  # the pipeline ran and issued commands
    # dry-aim guarantee: pump/valve are OFF on every single command
    assert all(c.pump_pct == 0.0 and c.valve is False for c in rig.commands)
    assert report.sprayed is False


def test_run_capture_uses_injected_detector() -> None:
    path = os.path.join(tempfile.mkdtemp(), "fire.mp4")
    if not _write_fire_video(path):
        pytest.skip("no working mp4 encoder in this environment")
    cap = cv2.VideoCapture(path)
    ok = cap.isOpened() and cap.read()[0]
    cap.release()
    if not ok:
        pytest.skip("mp4 written but not decodable in this environment")

    class _CountingDetector:  # duck-types the injectable detector interface
        def __init__(self):
            self.calls = 0

        def detect(self, frame, camera_moving=False):
            self.calls += 1
            return []

        def reset(self):
            pass

    det = _CountingDetector()
    run_capture(DEFAULT_CONFIG, path, NullRig(), headless=True, max_frames=30, detector=det)
    os.remove(path)
    assert det.calls > 0  # the pipeline used the injected detector, not the classical one


def test_run_capture_bad_source_raises() -> None:
    with pytest.raises(RuntimeError):
        run_capture(DEFAULT_CONFIG, "does_not_exist.mp4", NullRig(), headless=True)
