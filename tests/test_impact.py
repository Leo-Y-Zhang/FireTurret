# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
import numpy as np

from fireturret.vision.impact import SplashDetector


def scene() -> np.ndarray:
    return np.full((540, 960, 3), (40, 50, 44), dtype=np.uint8)


def with_splash(frame: np.ndarray, cx: int, cy: int, jitter: int) -> np.ndarray:
    out = frame.copy()
    rng = np.random.default_rng(jitter)
    for _ in range(20):
        ox = int((rng.random() - 0.5) * 40)
        oy = int((rng.random() - 0.5) * 16)
        shade = 190 + int(rng.random() * 60)
        out[
            max(0, cy + oy - 3) : cy + oy + 3,
            max(0, cx + ox - 3) : cx + ox + 3,
        ] = (shade, shade, shade)
    return out


def test_first_frame_has_no_observation() -> None:
    det = SplashDetector()
    assert det.observe(scene(), (480, 300), None) is None


def test_detects_moving_bright_splash_near_target() -> None:
    det = SplashDetector()
    det.observe(with_splash(scene(), 480, 350, jitter=1), (480, 300), None)
    obs = det.observe(with_splash(scene(), 480, 350, jitter=2), (480, 300), None)
    assert obs is not None
    assert abs(obs.cx - 480) < 60
    assert abs(obs.cy - 350) < 40


def test_ignores_splash_outside_corridor() -> None:
    det = SplashDetector(corridor_halfwidth_px=100)
    det.observe(with_splash(scene(), 100, 350, jitter=1), (700, 300), None)
    obs = det.observe(with_splash(scene(), 100, 350, jitter=2), (700, 300), None)
    assert obs is None


def test_detects_splash_overlapping_fire_bbox() -> None:
    # Colour (desaturated water vs saturated fire) separates them, so the
    # splash is found even when it lands on the fire — the on-target case an
    # exclusion mask would wrongly blank out. The bbox arg is accepted but ignored.
    det = SplashDetector()
    bbox = (440, 320, 80, 60)  # overlaps the splash location
    det.observe(with_splash(scene(), 480, 350, jitter=1), (480, 300), bbox)
    obs = det.observe(with_splash(scene(), 480, 350, jitter=2), (480, 300), bbox)
    assert obs is not None
    assert abs(obs.cx - 480) < 60


def test_static_scene_yields_nothing() -> None:
    det = SplashDetector()
    frame = with_splash(scene(), 480, 350, jitter=1)
    det.observe(frame, (480, 300), None)
    obs = det.observe(frame.copy(), (480, 300), None)  # identical: no motion
    assert obs is None
