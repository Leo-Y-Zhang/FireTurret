# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
import numpy as np

from fireturret.config import CameraConfig, DetectorConfig
from fireturret.rng import RngBundle
from fireturret.vision.firedetect import FireDetector, fire_colour_mask
from fireturret.vision.synthetic import FireState, SceneCamera, WorldPoint, render_scene

CFG = DetectorConfig()


def synthetic_fire_frames(n: int, seed: int = 3, intensity: float = 1.0):
    cam = SceneCamera(CameraConfig())
    # render_scene draws the fire from "fire0" and the splash from "splash";
    # both streams must be declared even though this scene has no splash.
    rng = RngBundle(seed, ["splash", "fire0"])
    fire = FireState(position=WorldPoint.from_polar(0.0, 7.0), intensity=intensity)
    return [render_scene(cam, 0.0, fire, None, rng) for _ in range(n)], cam


def test_detects_rendered_fire_with_flicker_confidence() -> None:
    frames, cam = synthetic_fire_frames(12)
    det = FireDetector(CFG)
    blobs = []
    for f in frames:
        blobs = det.detect(f)
    assert blobs, "rendered fire not detected"
    best = blobs[0]
    # fire should appear near the image centre column (azimuth 0)
    assert abs(best.cx - cam.cx) < 60
    assert best.flicker_score >= CFG.flicker_threshold
    assert best.confidence > 0.5


def test_static_orange_object_rejected_by_flicker() -> None:
    frame = np.full((540, 960, 3), (30, 40, 36), dtype=np.uint8)
    # a fire-coloured but perfectly static rectangle
    frame[300:360, 400:480] = (0, 120, 250)
    det = FireDetector(CFG)
    blobs = []
    for _ in range(12):
        blobs = det.detect(frame.copy())
    assert blobs, "colour rules should still fire"
    assert blobs[0].flicker_score < CFG.flicker_threshold
    assert blobs[0].confidence < 0.4


def test_grey_scene_has_no_detections() -> None:
    frame = np.full((540, 960, 3), (90, 90, 90), dtype=np.uint8)
    det = FireDetector(CFG)
    for _ in range(6):
        blobs = det.detect(frame.copy())
    assert blobs == []


def test_colour_mask_hits_fire_pixels_not_ground() -> None:
    frames, _ = synthetic_fire_frames(1)
    mask = fire_colour_mask(frames[0], CFG)
    assert mask.max() == 255
    # ground area (bottom-left corner) must be clean
    assert mask[-40:, :200].max() == 0


def test_camera_moving_lowers_confidence_and_clears_history() -> None:
    frames, _ = synthetic_fire_frames(12)
    det = FireDetector(CFG)
    for f in frames[:8]:
        det.detect(f)
    moving = det.detect(frames[8], camera_moving=True)
    assert moving
    assert moving[0].confidence < 0.5
    # history was cleared: next static detect has no flicker yet
    after = det.detect(frames[9])
    assert after[0].flicker_score == 0.0
