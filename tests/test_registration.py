# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""SP8 / spec §3.10 — keeping the flicker gate alive on a moving platform.

The defect: `FireDetector` cleared its flicker history on every moving frame. For
a fixed turret that is harmless — slews are brief and the history refills. On a
platform that moves permanently the history never accumulates, the detector
degrades to colour-only, and **the flicker gate stops existing**. That gate is
the entire defence against spraying a red object that is not a fire, so this is a
prerequisite for the mobile tier rather than a refinement of it.

The acid test is at the bottom: a decoy, under motion, must never be sprayed.
"""

from __future__ import annotations

import numpy as np
import pytest

from fireturret.config import DEFAULT_CONFIG
from fireturret.vision.firedetect import FireDetector
from fireturret.vision.registration import StabilisedFlicker, stabilise_mask

CFG = DEFAULT_CONFIG.detector
H, W = 120, 200


def _blob_mask(x: int, y: int = 40, w: int = 30, h: int = 30) -> np.ndarray:
    mask = np.zeros((H, W), dtype=bool)
    mask[y : y + h, x : x + w] = True
    return mask


# ------------------------------------------------------------- stabilisation

def test_stabilising_by_the_true_shift_aligns_consecutive_frames() -> None:
    """A stationary object, seen from a camera that panned, must land in the same
    place once the known ego-motion is undone."""
    first = _blob_mask(60)
    moved = _blob_mask(70)  # the camera panned; the object appears 10 px right
    assert np.array_equal(stabilise_mask(moved, -10.0), first)


def test_a_zero_shift_is_the_identity() -> None:
    mask = _blob_mask(60)
    assert stabilise_mask(mask, 0.0) is mask


def test_wrapped_in_columns_are_marked_unknown_not_black() -> None:
    """They sit at the frame edge, outside any eroded blob interior, so they cost
    nothing — rather than pretending to be real observations."""
    mask = np.ones((H, W), dtype=bool)
    assert not stabilise_mask(mask, 5.0)[:, :5].any()
    assert not stabilise_mask(mask, -5.0)[:, -5:].any()


# ------------------------------------------------- interior vs edge toggling

def test_a_stationary_object_perfectly_registered_scores_no_flicker() -> None:
    flicker = StabilisedFlicker(length=6)
    for i in range(6):
        flicker.append(_blob_mask(60 + i * 10), shift_px=10.0)
    assert flicker.flicker_score((110, 40, 30, 30)) == pytest.approx(0.0, abs=1e-9)


def test_registration_error_within_the_margin_manufactures_no_flicker() -> None:
    """**The trap this design exists to avoid.** Imperfect registration makes a
    rigid object shimmer at its EDGES. A detector that counted that would rate a
    decoy as more fire-like the rougher the ride got — precisely backwards.

    Measuring the eroded INTERIOR removes it: a mask that slides by less than
    `EDGE_MARGIN_PX` flips only a rim, and the rim is outside the region
    measured.
    """
    flicker = StabilisedFlicker(length=6)
    jitter = [0.0, 2.0, -2.0, 1.0, -1.0, 2.0]  # bounded error, never accumulating
    for i in range(6):
        flicker.append(_blob_mask(60 + i * 10), shift_px=10.0 + jitter[i])
    score = flicker.flicker_score((110, 40, 30, 30))
    assert score < 0.02, f"registration error manufactured {score:.3f} of flicker"


def test_registration_error_LARGER_than_the_margin_is_NOT_rejected() -> None:
    """The honest limit of the mechanism, stated rather than hidden.

    An error bigger than the erosion margin moves the interior too, and no amount
    of eroding distinguishes that from a fire. `EDGE_MARGIN_PX` is therefore a
    statement about how good the ego-motion prediction has to be: if the
    predictor degrades past a few pixels, the flicker gate degrades with it.
    """
    flicker = StabilisedFlicker(length=6)
    for i in range(6):
        flicker.append(_blob_mask(60 + i * 10), shift_px=18.0)  # 8 px/frame wrong
    assert flicker.flicker_score((110, 40, 30, 30)) > 0.02


def test_a_genuinely_flickering_object_still_scores() -> None:
    """The other half: erosion must not remove real signal. A flame changes shape
    throughout its body, so its interior toggles."""
    flicker = StabilisedFlicker(length=6)
    rng = np.random.default_rng(3)
    for i in range(6):
        mask = _blob_mask(60 + i * 10)
        # shimmer the blob's own pixels
        noise = rng.random((30, 30)) < 0.4
        mask[40:70, 60 + i * 10 : 90 + i * 10] = noise
        flicker.append(mask, shift_px=10.0)
    assert flicker.flicker_score((110, 40, 30, 30)) > 0.1


def test_the_score_is_never_negative() -> None:
    """Toggle rates are non-negative by construction; a steady patch scores zero."""
    flicker = StabilisedFlicker(length=4)
    rng = np.random.default_rng(1)
    for _ in range(4):
        mask = rng.random((H, W)) < 0.5
        mask[40:70, 60:90] = True  # a perfectly steady patch in a noisy frame
        flicker.append(mask)
    assert flicker.flicker_score((60, 40, 30, 30)) == 0.0


def test_too_short_a_history_scores_zero_rather_than_guessing() -> None:
    flicker = StabilisedFlicker(length=6)
    flicker.append(_blob_mask(60))
    assert flicker.flicker_score((60, 40, 30, 30)) == 0.0


# ------------------------------------------------------------ the detector

def test_the_fixed_turret_detector_is_unchanged() -> None:
    """Opt-in, and off by default. Enabling it for a fixed turret would change
    detection during its brief slews and move every golden fixture, for no
    benefit — the history refills as soon as it settles."""
    det = FireDetector(CFG)
    assert det.stabilised_flicker is False
    assert det._stabilised is None


def test_the_moving_detector_keeps_a_history_while_the_camera_moves() -> None:
    """The defect, directly: with the old behaviour this history is empty."""
    det = FireDetector(CFG, stabilised_flicker=True)
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    frame[40:70, 60:90] = (20, 140, 250)
    for _ in range(6):
        det.detect(frame, camera_moving=True, shift_px=1.0)
    assert len(det._stabilised) == 6
    assert len(det._history) == 0, "the legacy history still clears, as before"


# --------------------------------------------------------------- THE acid test

def test_a_decoy_is_never_sprayed_while_the_platform_moves() -> None:
    """**The acceptance criterion the spec names.**

    A fire-coloured object that does not flicker, seen from a moving platform,
    must never be confirmed. Under the old behaviour the flicker gate was absent
    entirely while moving, so this was decided by colour alone — which a red
    cloth passes.
    """
    det = FireDetector(CFG, stabilised_flicker=True)
    decoy = np.zeros((H, W, 3), dtype=np.uint8)

    blobs_seen = []
    for i in range(12):
        frame = decoy.copy()
        # a rigid, non-flickering red object, sliding across the frame as the
        # platform moves
        x = 40 + i * 4
        frame[40:70, x : x + 30] = (20, 140, 250)
        blobs_seen.extend(det.detect(frame, camera_moving=True, shift_px=4.0))

    assert blobs_seen, "the decoy was not detected at all, which tests nothing"
    # The flicker gate must REJECT it — that is the defence, and under the old
    # behaviour it was simply absent while moving.
    assert max(b.flicker_score for b in blobs_seen) < CFG.flicker_threshold, (
        "a rigid decoy passed the flicker gate while the platform moved"
    )
    # ...and it must stay below the tracker's confirmation threshold (0.3), so it
    # can never latch `ever_confirmed` and become a target.
    assert max(b.confidence for b in blobs_seen) <= 0.3, (
        f"a non-flickering decoy reached confidence "
        f"{max(b.confidence for b in blobs_seen):.2f} while the platform moved"
    )


def test_a_real_flickering_fire_is_still_confirmable_while_moving() -> None:
    """The other side of the acid test: rejecting everything would also pass the
    decoy check, and would make the mobile tier useless."""
    det = FireDetector(CFG, stabilised_flicker=True)
    rng = np.random.default_rng(11)

    scores = []
    for i in range(12):
        frame = np.zeros((H, W, 3), dtype=np.uint8)
        x = 40 + i * 4
        patch = np.zeros((30, 30, 3), dtype=np.uint8)
        # Alternate fire-colour with NON-fire colour, so the colour MASK itself
        # toggles. An earlier version of this test alternated two shades that
        # both passed the colour gate, so the mask never changed and the
        # "flickering fire" was, to the detector, perfectly steady.
        alight = rng.random((30, 30)) < 0.5
        patch[alight] = (20, 140, 250)   # fire-coloured
        patch[~alight] = (40, 40, 40)    # dark: fails the colour gate
        frame[40:70, x : x + 30] = patch
        scores.extend(b.flicker_score for b in det.detect(frame, camera_moving=True,
                                                          shift_px=4.0))
    assert max(scores) > CFG.flicker_threshold, (
        "a genuinely flickering fire could not pass the gate while moving"
    )
