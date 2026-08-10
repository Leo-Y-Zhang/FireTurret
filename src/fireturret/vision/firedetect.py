# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""Classical fire detection: HSV ∧ YCbCr colour rules, morphological cleanup,
connected components, and a temporal flicker score.

Colour rules follow the widely used fire-pixel criteria: hue in the red-orange
band with high saturation/value (HSV), intersected with the YCbCr rules
Y > Cb, Cr > Cb, and above-mean chrominance separation — robust to many
lighting conditions without a learned model. Real fire also flickers; a static
orange object passes the colour rules but fails the flicker gate.

The detector is stateful (mask history for flicker). When the turret is
slewing, flicker is unreliable, so callers pass `camera_moving=True` and
persistence in the tracker carries the target through motion instead.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from ..config import DetectorConfig
from .registration import StabilisedFlicker


@dataclass(frozen=True)
class FireBlob:
    cx: float
    cy: float
    area: float
    bbox: tuple[int, int, int, int]  # x, y, w, h
    colour_score: float
    flicker_score: float
    confidence: float
    # Whether flicker_score is a MEASUREMENT or merely absent. A semantic mask
    # from a learned detector has no temporal history, so its 0.0 means "not
    # measured", while the classical detector's 0.0 means "measured, and this
    # thing does not flicker". Those are opposite claims and the tracker must not
    # confuse them: requiring flicker evidence from a detector that cannot
    # produce any would mean nothing is ever confirmed and the turret never
    # fires. See FireTracker.update.
    flicker_measured: bool = True


def fire_colour_mask(frame_bgr: np.ndarray, cfg: DetectorConfig) -> np.ndarray:
    """Binary mask of fire-coloured pixels (uint8, 0/255)."""
    blurred = cv2.GaussianBlur(frame_bgr, (5, 5), 0)

    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    hsv_mask = cv2.inRange(
        hsv,
        np.array([0, cfg.s_min, cfg.v_min], dtype=np.uint8),
        np.array([cfg.h_max, 255, 255], dtype=np.uint8),
    )

    ycrcb = cv2.cvtColor(blurred, cv2.COLOR_BGR2YCrCb).astype(np.int16)
    y, cr, cb = ycrcb[..., 0], ycrcb[..., 1], ycrcb[..., 2]
    y_mean = float(y.mean())
    cr_mean = float(cr.mean())
    cb_mean = float(cb.mean())
    ycc = (
        (y > cb)
        & (cr > cb)
        & (y > y_mean)
        & (cb < cb_mean)
        & (cr > cr_mean)
    ).astype(np.uint8) * 255

    mask = cv2.bitwise_and(hsv_mask, ycc)
    # small open to drop speckle, larger close to fill the flame body without
    # eroding the base (which anchors the ground-contact estimate)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    return mask


def blobs_from_mask(mask: np.ndarray, min_blob_area_px: int,
                    base_confidence: float = 0.9) -> list[FireBlob]:
    """Extract FireBlobs from a binary fire mask (connected components). Shared by
    the classical detector's geometry and any learned detector that produces a
    fire-probability mask — no flicker (a semantic mask has no temporal history)."""
    count, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    blobs: list[FireBlob] = []
    for i in range(1, count):
        area = float(stats[i, cv2.CC_STAT_AREA])
        if area < min_blob_area_px:
            continue
        bbox = (
            int(stats[i, cv2.CC_STAT_LEFT]),
            int(stats[i, cv2.CC_STAT_TOP]),
            int(stats[i, cv2.CC_STAT_WIDTH]),
            int(stats[i, cv2.CC_STAT_HEIGHT]),
        )
        colour_score = min(1.0, area / 2000.0) * 0.4 + 0.6
        blobs.append(FireBlob(
            cx=float(centroids[i][0]), cy=float(centroids[i][1]), area=area, bbox=bbox,
            colour_score=colour_score, flicker_score=0.0,
            confidence=base_confidence * colour_score,
            # No temporal history exists here, so this is "not measured", not
            # "measured zero". A learned detector's own confidence is its
            # evidence: it was trained to tell fire from a red object, which is
            # the job the flicker gate does for the colour heuristic.
            flicker_measured=False,
        ))
    blobs.sort(key=lambda b: b.confidence * b.area, reverse=True)
    return blobs


class FireDetector:
    def __init__(self, cfg: DetectorConfig, stabilised_flicker: bool = False) -> None:
        """`stabilised_flicker` keeps the flicker gate alive while the camera is
        moving, by comparing masks in a scene-stabilised reference and regressing
        out the ego-motion-synchronous component (`vision/registration.py`).

        **Opt-in, and off for a fixed turret.** A fixed turret's slews are brief
        and its history refills the moment it settles, so it does not need this —
        and enabling it there would change detection during those slews and move
        every golden fixture, for no benefit. A platform that moves permanently
        does need it: without it the history never accumulates, the detector
        degrades to colour-only, and the decoy defence stops existing (spec
        §3.10). SP8 turns it on.
        """
        self.cfg = cfg
        self._history: deque[np.ndarray] = deque(maxlen=cfg.flicker_window)
        self.stabilised_flicker = stabilised_flicker
        self._stabilised = (
            StabilisedFlicker(length=cfg.flicker_window) if stabilised_flicker else None
        )

    def reset(self) -> None:
        self._history.clear()
        if self._stabilised is not None:
            self._stabilised.clear()

    def _flicker_score(self, bbox: tuple[int, int, int, int]) -> float:
        """Mean per-pixel toggle rate of the fire mask inside the bbox across
        the history window. Fire edges shimmer; static objects do not.
        """
        if len(self._history) < 3:
            return 0.0
        x, y, w, h = bbox
        toggles = 0.0
        pairs = 0
        prev = None
        for mask in self._history:
            roi = mask[y : y + h, x : x + w]
            if prev is not None:
                toggles += float(np.mean(prev != roi))
                pairs += 1
            prev = roi
        return toggles / pairs if pairs else 0.0

    def detect(
        self,
        frame_bgr: np.ndarray,
        camera_moving: bool = False,
        shift_px: float = 0.0,
    ) -> list[FireBlob]:
        mask = fire_colour_mask(frame_bgr, self.cfg)
        if self._stabilised is not None:
            # Moving platform: keep measuring flicker, in a stabilised reference.
            self._stabilised.append(mask > 0, shift_px)
        # While slewing, frame-to-frame comparison is motion, not flicker —
        # keep the history clean so it recovers quickly once stationary.
        if camera_moving:
            self._history.clear()
        else:
            self._history.append(mask > 0)

        count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
        blobs: list[FireBlob] = []
        for i in range(1, count):
            area = float(stats[i, cv2.CC_STAT_AREA])
            if area < self.cfg.min_blob_area_px:
                continue
            bbox = (
                int(stats[i, cv2.CC_STAT_LEFT]),
                int(stats[i, cv2.CC_STAT_TOP]),
                int(stats[i, cv2.CC_STAT_WIDTH]),
                int(stats[i, cv2.CC_STAT_HEIGHT]),
            )
            colour_score = min(1.0, area / 2000.0) * 0.4 + 0.6
            if camera_moving and self._stabilised is not None:
                # The mobile path: flicker is still measurable, because the masks
                # were stabilised and the frame-wide registration baseline has
                # been subtracted out (see vision/registration.py).
                flicker = self._stabilised.flicker_score(bbox)
            elif camera_moving:
                flicker = 0.0
            else:
                flicker = self._flicker_score(bbox)
            flicker_ok = flicker >= self.cfg.flicker_threshold
            if camera_moving and self._stabilised is not None:
                # Real evidence, so real trust — but still below the stationary
                # case, because a stabilised measurement is a warped one.
                confidence = colour_score * (0.75 if flicker_ok else 0.2)
            elif camera_moving:
                # Colour only: flicker was forced to 0.0 just above and the
                # history cleared, so there is NO flicker evidence here at all.
                # This used to be 0.45 * colour_score, and colour_score is at
                # least 0.6 by construction, so a large blob reached 0.45 - above
                # the tracker's 0.3 confirmation threshold. A red cloth or a
                # warning light that stayed in frame through a slew therefore
                # latched ever_confirmed on colour alone, permanently, and the
                # latch is never cleared. The stabilised branch above caps exactly
                # this case (evidence gathered, flicker gate failed) at 0.2, and
                # tests/test_registration.py asserts that contract in so many
                # words. Having no evidence cannot deserve more trust than having
                # evidence that failed. A genuine fire is unaffected: it confirms
                # while the camera is still and the latch carries it through the
                # slew, which is what Track.ever_confirmed exists for.
                confidence = colour_score * 0.2
            else:
                confidence = colour_score * (0.9 if flicker_ok else 0.25)
            blobs.append(
                FireBlob(
                    cx=float(centroids[i][0]),
                    cy=float(centroids[i][1]),
                    area=area,
                    bbox=bbox,
                    colour_score=colour_score,
                    flicker_score=flicker,
                    confidence=confidence,
                )
            )
        blobs.sort(key=lambda b: b.confidence * b.area, reverse=True)
        return blobs
