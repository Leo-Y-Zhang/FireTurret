# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""Splash (water-impact) detection for the closed servo loop.

While spraying, the landing water is the brightest fast-changing low-colour
region near the predicted impact point: frame differencing, restricted to a
vertical corridor around the target's column, excluding the fire's own bbox,
filtered to low-saturation (water/spray is white/grey, fire is saturated).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class ImpactObservation:
    """Where the water landed.

    `cx`/`cy`/`area` are the centroid and size, unchanged, and the centroid
    remains the servo signal — always. The extra fields are defaulted so every
    existing construction and comparison behaves identically.

    **`base_y` is the point of this.** The splash plume is airborne and sits ABOVE
    the ground contact, so the centroid is biased high by an amount that depends
    on plume height. An earlier design proposed correcting that with an asserted
    bias constant; the spec rejects it, and rightly — such a scalar moves the
    loop's fixed point by exactly the asserted bias and leaves ZERO residual to
    detect the assertion being wrong. The loop would converge confidently
    off-target while reporting `on_target=True`.

    `base_y` is the same information as a MEASUREMENT instead: the bottom row of
    the detected blob is where the water meets the ground. A wrong value still
    produces a visible residual, which is the entire difference.
    """

    cx: float
    cy: float
    area: float
    base_y: float | None = None       # bottom row of the blob = ground contact
    major_sigma_px: float = 0.0       # second-moment axes of the footprint
    minor_sigma_px: float = 0.0
    axis_angle_deg: float = 0.0


class SplashDetector:
    def __init__(self, min_area_px: int = 40, corridor_halfwidth_px: int = 130) -> None:
        self.min_area = min_area_px
        self.corridor = corridor_halfwidth_px
        self._prev: np.ndarray | None = None

    def reset(self) -> None:
        self._prev = None

    def observe(
        self,
        frame_bgr: np.ndarray,
        target_px: tuple[float, float],
        exclude_bbox: tuple[int, int, int, int] | None = None,
    ) -> ImpactObservation | None:
        """Locate the water impact. Colour separates water (bright, DESATURATED)
        from fire (saturated), so the splash is found even when it lands right
        on top of the fire — no bbox exclusion, which would mask exactly the
        on-target case. `exclude_bbox` is accepted for API stability but unused.
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        prev = self._prev
        self._prev = gray
        if prev is None or prev.shape != gray.shape:
            return None

        diff = cv2.absdiff(gray, prev)
        _, motion = cv2.threshold(diff, 22, 255, cv2.THRESH_BINARY)

        # water is bright and unsaturated; fire is saturated — separate by colour
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        watery = cv2.inRange(
            hsv,
            np.array([0, 0, 150], dtype=np.uint8),
            np.array([179, 90, 255], dtype=np.uint8),
        )
        mask = cv2.bitwise_and(motion, watery)

        # restrict to the corridor around the target's column
        h, w = mask.shape
        tx = int(round(target_px[0]))
        x0 = max(0, tx - self.corridor)
        x1 = min(w, tx + self.corridor)
        corridor_mask = np.zeros_like(mask)
        corridor_mask[:, x0:x1] = 255
        mask = cv2.bitwise_and(mask, corridor_mask)

        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        )
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
        best: ImpactObservation | None = None
        best_area = self.min_area
        for i in range(1, count):
            area = float(stats[i, cv2.CC_STAT_AREA])
            # the splash is the largest coherent desaturated moving region
            if area >= best_area:
                best_area = area
                top = float(stats[i, cv2.CC_STAT_TOP])
                height = float(stats[i, cv2.CC_STAT_HEIGHT])
                major, minor, angle = _second_moments(labels, i)
                best = ImpactObservation(
                    cx=float(centroids[i][0]), cy=float(centroids[i][1]), area=area,
                    # the bottom row of the blob: a MEASUREMENT of ground contact,
                    # not an assumed offset from the (airborne) centroid
                    base_y=top + height,
                    major_sigma_px=major, minor_sigma_px=minor, axis_angle_deg=angle,
                )
        return best


def _second_moments(labels: np.ndarray, label: int) -> tuple[float, float, float]:
    """Principal axes of a labelled blob, as (major sigma, minor sigma, angle).

    The footprint's shape, which SP5's dispersion model needs and which a bare
    area cannot express: a wide flat fan and a compact stream can cover the same
    number of pixels while wetting very different ground.
    """
    ys, xs = np.nonzero(labels == label)
    if xs.size < 2:
        return 0.0, 0.0, 0.0
    cov = np.cov(np.vstack((xs.astype(float), ys.astype(float))))
    if not np.all(np.isfinite(cov)):
        return 0.0, 0.0, 0.0
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.clip(eigenvalues[order], 0.0, None)
    major_vec = eigenvectors[:, order[0]]
    angle = float(np.degrees(np.arctan2(major_vec[1], major_vec[0])))
    return float(np.sqrt(eigenvalues[0])), float(np.sqrt(eigenvalues[1])), angle
