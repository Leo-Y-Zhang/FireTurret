# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""Pinhole camera geometry: pixel ↔ angle mapping and the ground-plane range
bootstrap used for the opening ballistic solution.

Conventions: image x grows right, y grows down. Azimuth grows to the right
(clockwise from above), elevation grows upward. Angles in degrees at the API
surface, radians internally.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .config import CameraConfig


@dataclass(frozen=True)
class PixelAngles:
    azimuth_deg: float
    elevation_deg: float


class CameraModel:
    def __init__(self, cfg: CameraConfig) -> None:
        self.cfg = cfg
        self.cx = cfg.width / 2.0
        self.cy = cfg.height / 2.0
        # square pixels: one focal length from the horizontal FOV
        self.fpx = (cfg.width / 2.0) / math.tan(math.radians(cfg.hfov_deg) / 2.0)

    def px_to_angles(self, u: float, v: float) -> PixelAngles:
        az = math.degrees(math.atan((u - self.cx) / self.fpx))
        el = math.degrees(math.atan((self.cy - v) / self.fpx))
        return PixelAngles(azimuth_deg=az, elevation_deg=el)

    def angles_to_px(self, azimuth_deg: float, elevation_deg: float) -> tuple[float, float]:
        u = self.cx + math.tan(math.radians(azimuth_deg)) * self.fpx
        v = self.cy - math.tan(math.radians(elevation_deg)) * self.fpx
        return u, v

    def px_row_for_ground_range(self, range_m: float) -> float:
        """The image row a point at this ground range projects to.

        The forward direction of `ground_range_from_px`. It exists because the
        suppression loop's plant gain is `d(row) / d(pump)`: pump changes range,
        and perspective foreshortening decides how many PIXELS that is worth.
        Measuring that in pixels rather than metres is the whole point — the
        loop observes rows, so its authority collapses with the row spacing
        (spec §3.6: `dR/dpump` falls 2.9x across the band while the observable
        consequence falls 7.4x).
        """
        if range_m <= 0.0:
            raise ValueError("ground range must be positive")
        depression_deg = math.degrees(math.atan(self.cfg.mount_height_m / range_m))
        elevation_deg = -self.cfg.mount_pitch_deg - depression_deg
        return self.cy - math.tan(math.radians(elevation_deg)) * self.fpx

    def ground_range_from_px(self, u: float, v: float) -> float | None:
        """Range along the ground to the point under pixel (u, v), assuming the
        point sits on flat ground and the camera pitch/height are calibrated.
        Returns None for pixels at or above the horizon (no intersection).
        """
        ang = self.px_to_angles(u, v)
        # depression below horizontal, positive downward
        depression_deg = -(self.cfg.mount_pitch_deg + ang.elevation_deg)
        if depression_deg <= 0.5:
            return None
        return self.cfg.mount_height_m / math.tan(math.radians(depression_deg))


def wrap_deg(angle: float) -> float:
    """Normalize an angle to (-180, 180]."""
    a = math.fmod(angle, 360.0)
    if a > 180.0:
        a -= 360.0
    if a <= -180.0:
        a += 360.0
    return a


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))
