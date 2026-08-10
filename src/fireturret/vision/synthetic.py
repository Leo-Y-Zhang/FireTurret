# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""Synthetic scene renderer for the simulator and tests.

Renders the view from the pan-stage camera (fixed pitch, yaw = pan angle):
ground plane, a flickering fire at a world position, and the water splash.
The fire is drawn with real fire colours and per-frame flicker so the actual
FireDetector — not a shortcut — must find it. That makes the simulator a true
end-to-end test of the perception stack.

World frame: turret at origin, X right, Z forward at pan 0, Y up.
Azimuth is clockwise from +Z (matches pan).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from ..config import CameraConfig
from ..rng import RngBundle

SKY = (38, 30, 24)  # BGR: dark slate
GROUND = (34, 44, 38)
GRID = (52, 66, 56)

# Draws each renderer's stream advances by, PER FRAME, unconditionally.
#
# The caller allocates the whole block before testing any branch, so a stream's
# position after N frames is exactly N * BLOCK — independent of whether the fire
# was alive, whether the valve was open, or which flicker tongues appeared. That
# is what stops the renderer's randomness from being a function of the control
# decisions (see `fireturret.rng`).
#
# Sized to the longest path through each renderer; `tests/test_rng.py` asserts
# no path can consume more, so an overrun is a caught bug rather than a silent
# re-roll of everything downstream.
FIRE_DRAWS = 31
SPLASH_DRAWS = 48


class Draws:
    """A fixed-length block of random values, consumed in order.

    Renderers take one of these instead of a `Generator` so that the number of
    values pulled from the underlying stream is fixed by the CALLER and cannot
    depend on the branches the renderer takes.
    """

    __slots__ = ("_values", "_index")

    def __init__(self, values: np.ndarray) -> None:
        self._values = values
        self._index = 0

    def next(self) -> float:
        """The next value. Raises IndexError past the end — deliberately: a block
        too small for its renderer's longest path would silently desynchronise
        every later frame, which is the failure this whole design prevents."""
        value = self._values[self._index]
        self._index += 1
        return float(value)

    @property
    def used(self) -> int:
        return self._index


@dataclass(frozen=True)
class WorldPoint:
    x: float
    z: float

    @staticmethod
    def from_polar(azimuth_deg: float, range_m: float) -> WorldPoint:
        a = math.radians(azimuth_deg)
        return WorldPoint(x=range_m * math.sin(a), z=range_m * math.cos(a))

    def range(self) -> float:
        return math.hypot(self.x, self.z)

    def azimuth_deg(self) -> float:
        return math.degrees(math.atan2(self.x, self.z))


class SceneCamera:
    """Projects world points into the pan-stage camera."""

    def __init__(self, cfg: CameraConfig) -> None:
        self.cfg = cfg
        self.fpx = (cfg.width / 2.0) / math.tan(math.radians(cfg.hfov_deg) / 2.0)
        self.cx = cfg.width / 2.0
        self.cy = cfg.height / 2.0

    def project(
        self, p: WorldPoint, height: float, pan_deg: float
    ) -> tuple[float, float, float] | None:
        """→ (u, v, camera-frame depth) or None if behind the camera."""
        yaw = math.radians(pan_deg)
        # yaw: align camera forward with world azimuth `pan`
        x1 = p.x * math.cos(yaw) - p.z * math.sin(yaw)
        z1 = p.x * math.sin(yaw) + p.z * math.cos(yaw)
        y1 = height - self.cfg.mount_height_m
        # pitch about the camera x-axis (mount_pitch negative = down)
        pitch = math.radians(self.cfg.mount_pitch_deg)
        y2 = y1 * math.cos(pitch) - z1 * math.sin(pitch)
        z2 = y1 * math.sin(pitch) + z1 * math.cos(pitch)
        if z2 < 0.15:
            return None
        u = self.cx + self.fpx * (x1 / z2)
        v = self.cy - self.fpx * (y2 / z2)
        return u, v, z2

    def horizon_v(self) -> int:
        return int(round(self.cy - self.fpx * math.tan(-math.radians(self.cfg.mount_pitch_deg))))


@dataclass
class FireState:
    position: WorldPoint
    radius_m: float = 0.45
    intensity: float = 1.0  # 0..1; 0 = out


def render_scene(
    cam: SceneCamera,
    pan_deg: float,
    fire: FireState | None,
    splash: WorldPoint | None,
    rng: RngBundle,
) -> np.ndarray:
    """Ground plane, the splash, and ONE fire (stream `fire0`).

    A multi-fire simulator draws its remaining fires itself, one named stream
    each — see `SimRig.render`. Both blocks are allocated below before any
    branch is tested, so `splash` and `fire0` each advance by exactly their
    block size every frame whether or not anything is drawn from them.
    """
    cfg = cam.cfg
    frame = np.empty((cfg.height, cfg.width, 3), dtype=np.uint8)
    hv = int(np.clip(cam.horizon_v(), 0, cfg.height))
    frame[:hv] = SKY
    frame[hv:] = GROUND

    # depth-cue grid: transverse ground lines at fixed ranges
    for r in (3, 5, 8, 12, 17, 25):
        pts = []
        for az in np.linspace(pan_deg - 45, pan_deg + 45, 24):
            proj = cam.project(WorldPoint.from_polar(float(az), float(r)), 0.0, pan_deg)
            if proj is not None:
                pts.append((int(proj[0]), int(proj[1])))
        if len(pts) >= 2:
            cv2.polylines(frame, [np.array(pts, dtype=np.int32)], False, GRID, 1)

    # Allocate BOTH blocks first, unconditionally — see the module constants.
    splash_draws = Draws(rng["splash"].random(SPLASH_DRAWS))
    fire_draws = Draws(rng["fire0"].random(FIRE_DRAWS))
    if splash is not None:
        _draw_splash(frame, cam, splash, pan_deg, splash_draws)
    if fire is not None and fire.intensity > 0.02:
        _draw_fire(frame, cam, fire, pan_deg, fire_draws)
    return frame


def _draw_fire(
    frame: np.ndarray,
    cam: SceneCamera,
    fire: FireState,
    pan_deg: float,
    rng: Draws,
) -> None:
    base = cam.project(fire.position, 0.0, pan_deg)
    if base is None:
        return
    u, v, depth = base
    if not (-100 <= u <= frame.shape[1] + 100):
        return
    scale = cam.fpx / depth
    r_px = max(3.0, fire.radius_m * scale * (0.85 + 0.3 * fire.intensity))

    def jitter(f: float) -> float:
        return float(1.0 + f * (rng.next() - 0.5) * 2.0)

    cx, cy = int(round(u)), int(round(v))
    # Cohesive flame teardrop built from overlapping filled circles stacked from
    # the ground point (cy) upward: one connected blob whose BOTTOM is the ground
    # contact the controller ranges from. Hottest (near-white-yellow) at the
    # base, orange in the middle, red at the tip — all kept SATURATED so the
    # splash detector (which keys on desaturated water) never mistakes fire for
    # water. Colours in BGR.
    flame_h = r_px * 2.2 * jitter(0.15)
    base_r = max(3, int(r_px * 1.05 * jitter(0.1)))
    steps = 7
    for i in range(steps):
        t = i / (steps - 1)  # 0 = base, 1 = tip
        yc = cy - base_r - int(t * flame_h) + int((rng.next() - 0.5) * r_px * 0.25)
        xoff = int((rng.next() - 0.5) * r_px * 0.55 * t)  # wavers more toward the tip
        rad = max(1, int(base_r * (1.0 - 0.82 * t)))
        if t < 0.34:
            colour = (40, 200, 255)  # yellow (hot)
        elif t < 0.67:
            colour = (20, 140, 250)  # orange
        else:
            colour = (30, 45, 215)  # red (cooler tip)
        cv2.circle(frame, (cx + xoff, yc), rad, colour, -1)
    # bright hot core at the base, still saturated
    cv2.circle(frame, (cx, cy - base_r), max(2, int(base_r * 0.6)), (55, 220, 255), -1)
    # flicker tongues above the tip
    for _ in range(3):
        if rng.next() < 0.7:
            ox = int((rng.next() - 0.5) * base_r * 1.6)
            oy = -int(base_r + rng.next() * flame_h * 1.05)
            rr = max(1, int(base_r * 0.32 * jitter(0.4)))
            colour = (25, 160, 252) if rng.next() < 0.6 else (30, 60, 220)
            cv2.circle(frame, (cx + ox, cy + oy), rr, colour, -1)


def draw_static_blob(
    frame: np.ndarray,
    cam: SceneCamera,
    point: WorldPoint,
    pan_deg: float,
    radius_m: float = 0.5,
    colour: tuple[int, int, int] = (20, 140, 250),
) -> None:
    """A fixed, fire-COLOURED but non-flickering object (a decoy: red cloth, a
    warning light). Identical every frame → the flicker gate must reject it, so
    the turret never sprays it."""
    proj = cam.project(point, 0.0, pan_deg)
    if proj is None:
        return
    u, v, depth = proj
    if not (-100 <= u <= frame.shape[1] + 100):
        return
    r = max(3, int(radius_m * cam.fpx / depth))
    cv2.ellipse(frame, (int(u), int(v) - r), (r, int(r * 1.3)), 0, 0, 360, colour, -1)


def _draw_splash(
    frame: np.ndarray,
    cam: SceneCamera,
    splash: WorldPoint,
    pan_deg: float,
    rng: Draws,
) -> None:
    base = cam.project(splash, 0.0, pan_deg)
    if base is None:
        return
    u, v, depth = base
    scale = cam.fpx / depth
    r_px = max(3, int(0.32 * scale))
    cx, cy = int(round(u)), int(round(v))
    # bright unsaturated core + churning droplets (per-frame noise → motion),
    # kept compact so the detected centroid tracks the true impact point
    cv2.ellipse(frame, (cx, cy), (int(r_px * 1.5), max(2, int(r_px * 0.7))), 0, 0, 360, (205, 205, 200), -1)
    for _ in range(12):
        ox = int((rng.next() - 0.5) * r_px * 1.8)
        oy = int((rng.next() - 0.5) * r_px * 1.2)
        rr = max(1, int(r_px * 0.3 * (0.5 + rng.next())))
        shade = 185 + int(rng.next() * 55)
        cv2.circle(frame, (cx + ox, cy + oy), rr, (shade, shade, min(255, shade + 8)), -1)
