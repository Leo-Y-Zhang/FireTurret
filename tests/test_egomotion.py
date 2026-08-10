# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""SP6 — frames, pose and ego-motion.

Two things are proven here. First, that the exact yaw warp is exact and the
scalar it replaces is not — with the error MEASURED across the frame rather than
quoted, so the number in the docstring is the number the code produces. Second,
that the kinematic chain encodes invariant 2 (the camera rides the pan stage) in
a form a test can check, because that invariant is load-bearing for the entire
servo argument and a comment cannot enforce it.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from fireturret.config import DEFAULT_CONFIG
from fireturret.frames import (
    CAMERA,
    CHAIN,
    NOZZLE,
    PAN,
    PLATFORM,
    TILT,
    WORLD,
    Quaternion,
    Transform,
    camera_parent,
    chain_to,
)
from fireturret.geometry import CameraModel
from fireturret.vision.egomotion import (
    camera_matrix,
    pan_shift_px,
    pan_warp,
    plane_homography,
    small_angle_shift_px,
    warp_point,
    yaw_matrix,
)

CAM = CameraModel(DEFAULT_CONFIG.camera)


# ------------------------------------------------------------- the exact warp

def test_the_warp_is_exact_against_the_projection_it_inverts() -> None:
    """A world point at bearing `az` must land where the camera model says it
    does, before and after a pan. This is the definition of correct."""
    for az_deg in (-30.0, -10.0, 0.0, 12.0, 30.0):
        for pan_delta in (-3.0, -0.5, 0.5, 3.0):
            u_before = CAM.cx + CAM.fpx * math.tan(math.radians(az_deg))
            u_after = CAM.cx + CAM.fpx * math.tan(math.radians(az_deg - pan_delta))
            assert pan_warp(u_before, pan_delta, CAM) == pytest.approx(u_after, abs=1e-9)


def test_a_zero_pan_moves_nothing() -> None:
    for u in (0.0, CAM.cx, 959.0):
        assert pan_warp(u, 0.0, CAM) == pytest.approx(u)


def test_the_warp_is_invertible() -> None:
    for u in (100.0, CAM.cx, 800.0):
        assert pan_warp(pan_warp(u, 2.0, CAM), -2.0, CAM) == pytest.approx(u, abs=1e-9)


def test_on_axis_the_old_scalar_was_essentially_right() -> None:
    """The small-angle form is the tangent line at u = cx, so on-axis it is right
    to a hundredth of a pixel — which is why the error stayed hidden.

    Not *exactly* right, even there: tan(x) = x + x^3/3 + ..., so a 1 deg pan
    leaves a ~0.0013 px cubic residual. Far below anything observable, and worth
    stating precisely rather than rounding to "exact"."""
    exact = pan_shift_px(CAM.cx, 1.0, CAM)
    approx = small_angle_shift_px(1.0, CAM)
    assert exact == pytest.approx(approx, abs=0.01)
    assert exact != approx, "the cubic term is real, just tiny"


@pytest.mark.parametrize("az_deg,min_error_pct", [(20.0, 8.0), (30.0, 20.0), (33.0, 25.0)])
def test_the_old_scalar_under_corrects_off_axis(az_deg, min_error_pct) -> None:
    """MEASURED, not quoted. The under-correction grows with the tangent, and it
    was being absorbed by the tracker's 80 px association radius — which works
    until the platform moves and that slack has to absorb real ego-motion too."""
    u = CAM.cx + CAM.fpx * math.tan(math.radians(az_deg))
    exact = pan_shift_px(u, 1.0, CAM)
    approx = small_angle_shift_px(1.0, CAM)
    error_pct = abs(exact - approx) / abs(exact) * 100.0
    assert error_pct >= min_error_pct, (
        f"at {az_deg} deg off-axis the old scalar was only {error_pct:.1f}% wrong"
    )


def test_the_under_correction_stays_inside_the_association_radius() -> None:
    """Why this was survivable rather than fatal, stated as a number: the worst
    single-frame error is far below `match_dist_px`, so association still
    succeeded. It is a latent defect, not an active one."""
    worst = 0.0
    for az_deg in (-33.0, -20.0, 0.0, 20.0, 33.0):
        u = CAM.cx + CAM.fpx * math.tan(math.radians(az_deg))
        worst = max(worst, abs(pan_shift_px(u, 3.0, CAM) - small_angle_shift_px(3.0, CAM)))
    assert worst < 80.0


# ----------------------------------------------------------------- homography

def test_the_rotation_homography_agrees_with_the_closed_form() -> None:
    """`plane_homography` with no translation must reproduce `pan_warp`. Two
    implementations of the same thing that disagree would be worse than one."""
    H = plane_homography(CAM, yaw_matrix(-2.0))
    for u in (150.0, CAM.cx, 820.0):
        warped_u, _ = warp_point(H, u, CAM.cy)
        assert warped_u == pytest.approx(pan_warp(u, 2.0, CAM), abs=1e-6)


def test_the_identity_pose_is_the_identity_homography() -> None:
    """The property that makes SP8 affordable: with no motion, the general
    machinery costs exactly nothing and changes exactly nothing."""
    H = plane_homography(CAM, np.eye(3))
    assert np.allclose(H, np.eye(3), atol=1e-9)
    assert warp_point(H, 123.0, 456.0) == pytest.approx((123.0, 456.0))


def test_translation_enters_only_through_the_plane_term() -> None:
    H_rot = plane_homography(CAM, np.eye(3))
    H_trans = plane_homography(
        CAM, np.eye(3), translation=(0.5, 0.0, 0.0),
        plane_normal=(0.0, 1.0, 0.0), plane_distance_m=7.0,
    )
    assert not np.allclose(H_rot, H_trans)


def test_a_degenerate_plane_distance_is_rejected() -> None:
    with pytest.raises(ValueError):
        plane_homography(CAM, np.eye(3), translation=(1.0, 0.0, 0.0), plane_distance_m=0.0)


def test_camera_matrix_matches_the_camera_model() -> None:
    K = camera_matrix(CAM)
    assert K[0, 0] == pytest.approx(CAM.fpx)
    assert K[0, 2] == pytest.approx(CAM.cx)
    assert K[1, 2] == pytest.approx(CAM.cy)


# --------------------------------------------------------- invariant 2, checked

def test_the_camera_hangs_off_the_pan_stage() -> None:
    """**Invariant 2, machine-checkable for the first time.**

    The camera turning with the pan axis is exactly what lets the azimuth loop
    correct the boresight offset from the observed splash: panning moves the
    fire's camera bearing but not the splash's, so the pan that makes their
    columns meet also points the nozzle at the fire. Hang the camera off the base
    instead and that argument collapses — silently, with every test still green.
    """
    assert camera_parent() == PAN
    assert CHAIN[CAMERA] == PAN


def test_the_chain_is_the_documented_one() -> None:
    assert chain_to(NOZZLE) == [WORLD, PLATFORM, PAN, TILT, NOZZLE]
    assert chain_to(CAMERA) == [WORLD, PLATFORM, PAN, CAMERA]


def test_an_unknown_frame_raises_rather_than_resolving() -> None:
    """A typo'd frame name that silently resolved is a sign error waiting to
    happen, and sign errors here are the expensive kind."""
    with pytest.raises(KeyError, match="unknown frame"):
        chain_to("ELBOW")


# ---------------------------------------------------------------- quaternions

def test_quaternion_rotation_matches_a_hand_calculation() -> None:
    q = Quaternion.from_yaw(90.0)
    x, y, z = q.rotate((0.0, 0.0, 1.0))  # forward -> right
    assert (x, y, z) == pytest.approx((1.0, 0.0, 0.0), abs=1e-9)


def test_rotations_compose() -> None:
    q = Quaternion.from_yaw(30.0) * Quaternion.from_yaw(60.0)
    expected = Quaternion.from_yaw(90.0)
    assert q.matrix() == pytest.approx(expected.matrix(), abs=1e-9)


def test_conjugate_undoes_a_rotation() -> None:
    q = Quaternion.from_yaw(37.0) * Quaternion.from_pitch(-12.0)
    point = (0.3, -0.7, 1.1)
    assert q.conjugate().rotate(q.rotate(point)) == pytest.approx(point, abs=1e-9)


def test_quaternions_stay_unit_length() -> None:
    q = Quaternion.from_yaw(31.0) * Quaternion.from_pitch(17.0) * Quaternion.from_roll(-5.0)
    assert q.norm() == pytest.approx(1.0, abs=1e-12)


def test_a_zero_axis_is_rejected() -> None:
    with pytest.raises(ValueError):
        Quaternion.from_axis_angle((0.0, 0.0, 0.0), 45.0)


# ----------------------------------------------------------------- transforms

def test_transform_composition_and_inverse_round_trip() -> None:
    a = Transform(Quaternion.from_yaw(25.0), (1.0, 0.0, 2.0))
    b = Transform(Quaternion.from_pitch(-10.0), (0.0, 0.5, 0.0))
    composed = a.compose(b)
    point = (0.4, 1.2, -0.8)
    assert composed.apply(point) == pytest.approx(a.apply(b.apply(point)), abs=1e-9)
    assert composed.inverse().apply(composed.apply(point)) == pytest.approx(point, abs=1e-9)


def test_the_identity_transform_moves_nothing() -> None:
    assert Transform.identity().apply((1.0, 2.0, 3.0)) == pytest.approx((1.0, 2.0, 3.0))
