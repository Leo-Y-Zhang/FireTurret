# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""Predicting where a world-fixed point moves in the image when the camera moves.

The camera rides the pan stage (invariant 2), so every slew slides a stationary
fire across the frame. The tracker predicts that motion before association, or it
spawns a stale trail instead of following one fire through the slew.

## What was wrong with the old prediction

`app.py` used a single scalar for every track:

    shift_px = -pan_delta_deg * radians(1) * fpx

That is the tangent line to the projection at `u = cx` — exact on the optical
axis, and increasingly wrong away from it, because the pinhole map is
`u = cx + fpx·tan(az)`, not `u = cx + fpx·az`.

Measured on the default build (hfov 66 deg, fpx 739.1), for a 1 deg pan:

    on axis        exact -12.90 px   small-angle -12.90 px    0.0% error
    20 deg off     exact -14.61 px   small-angle -12.90 px   11.7% under
    30 deg off     exact -17.09 px   small-angle -12.90 px   24.5% under
    33 deg (edge)  exact -18.11 px   small-angle -12.90 px   28.8% under

The under-correction was absorbed by `tracker.match_dist_px = 80` — the
association radius was wide enough to hide it. That works until the platform
moves, at which point the same slack has to absorb real ego-motion as well.

(The spec quotes ~33% at 30 deg off-axis. The figure measured here is 24.5% at
30 deg and 28.8% at the image edge; the shape of the claim is right and the
magnitude is a little smaller. The measurement is in `tests/test_egomotion.py`
so the number in the docs is the number the code produces.)

## The general form

For a camera undergoing rotation R and translation t observing a plane at
distance d with normal n, image points map by the plane-induced homography

    H = K (R - t·nᵀ / d) K⁻¹

`plane_homography` implements that. With `t = 0` it reduces to the pure-rotation
case, which is the fixed turret, and `pan_warp` is the closed form of exactly
that for a yaw-only rotation — kept separate because it is the hot path and does
not need a 3x3 solve per track.

**Honest caveat.** The homography is exact for points ON the plane — the fire
base, which sits on the ground. It is NOT exact for the splash, which is an
airborne plume elevated above the plane, and the residual grows with plume
height. Nothing here corrects that, and nothing should pretend to.
"""

from __future__ import annotations

import math

import numpy as np

from ..geometry import CameraModel


def pan_warp(u: float, pan_delta_deg: float, camera: CameraModel) -> float:
    """Where image column `u` moves when the camera yaws by `pan_delta_deg`.

    Exact for a yaw-only rotation, which is the fixed turret's entire ego-motion.
    A world-fixed point's bearing relative to the camera decreases by the pan
    delta, so its column follows the tangent map rather than a linear one.
    """
    az = math.atan((u - camera.cx) / camera.fpx)
    return camera.cx + camera.fpx * math.tan(az - math.radians(pan_delta_deg))


def pan_shift_px(u: float, pan_delta_deg: float, camera: CameraModel) -> float:
    """The displacement `pan_warp` implies, as a delta. Same units as the scalar
    it replaces, so callers can compare like for like."""
    return pan_warp(u, pan_delta_deg, camera) - u


def small_angle_shift_px(pan_delta_deg: float, camera: CameraModel) -> float:
    """The OLD approximation, kept so the error can be measured rather than
    asserted. Not used in the pipeline."""
    return -pan_delta_deg * math.radians(1.0) * camera.fpx


def camera_matrix(camera: CameraModel) -> np.ndarray:
    """The pinhole intrinsic matrix K."""
    return np.array(
        [[camera.fpx, 0.0, camera.cx],
         [0.0, camera.fpx, camera.cy],
         [0.0, 0.0, 1.0]],
        dtype=float,
    )


def yaw_matrix(deg: float) -> np.ndarray:
    """Rotation about the camera's Y (down) axis — a pan."""
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=float)


def plane_homography(
    camera: CameraModel,
    rotation: np.ndarray,
    translation: np.ndarray | None = None,
    plane_normal: np.ndarray | None = None,
    plane_distance_m: float = 1.0,
) -> np.ndarray:
    """`H = K (R - t·nᵀ / d) K⁻¹`.

    With `translation=None` this is the pure-rotation homography and
    `plane_distance_m` is irrelevant — which is the fixed-turret case, and why
    the identity pose costs nothing.
    """
    if plane_distance_m <= 0:
        raise ValueError("plane distance must be positive")
    K = camera_matrix(camera)
    K_inv = np.linalg.inv(K)
    M = np.asarray(rotation, dtype=float)
    if translation is not None:
        t = np.asarray(translation, dtype=float).reshape(3, 1)
        n = np.asarray(
            plane_normal if plane_normal is not None else (0.0, 1.0, 0.0), dtype=float
        ).reshape(3, 1)
        M = M - (t @ n.T) / plane_distance_m
    return K @ M @ K_inv


def warp_point(H: np.ndarray, u: float, v: float) -> tuple[float, float]:
    """Apply a homography to an image point."""
    p = H @ np.array([u, v, 1.0], dtype=float)
    if abs(p[2]) < 1e-12:
        return u, v  # degenerate (the point maps to infinity); leave it be
    return float(p[0] / p[2]), float(p[1] / p[2])
