# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""Turning a detection into a protected sector — widened by wind, never narrowed.

A detection is a box in the image. The keep-out machinery works in world azimuth.
Bridging the two is easy to do carelessly and the careless version is unsafe, so
two rules govern it:

## Range is used only to WIDEN, never to narrow

The projected sector's width depends on how far away the person is, and that
distance comes from the same biased monocular estimate the whole architecture
distrusts (invariant 1). So the estimate is allowed to make the protected sector
BIGGER — where being wrong is merely wasteful — and never smaller, where being
wrong means spraying someone. When range is unknown, the widest plausible sector
is used rather than a default guess.

## The sector grows with wind

Wind blows spray toward people. A sector sized for still air is the wrong size on
a gusty day, and it is wrong in the dangerous direction. So gust RMS widens it,
using the same wind estimate SP7 produces — and if that estimate is uncertain,
uncertainty widens it too.

Merged with the static configured sector by UNION, never intersection: two
reasons to protect a region are not a reason to protect less of it.
"""

from __future__ import annotations

import math

from .detectors import Detection

# When range is unknown, assume the person is close enough to subtend a wide
# sector. Deliberately pessimistic — see "widen, never narrow".
UNKNOWN_RANGE_HALF_WIDTH_DEG = 25.0

# Degrees of extra sector per m/s of gust RMS. Wind blows spray toward people, so
# a sector sized for still air is wrong in the dangerous direction.
GUST_WIDENING_DEG_PER_MS = 1.5

# Never smaller than this, whatever the geometry says.
MIN_HALF_WIDTH_DEG = 3.0


def detection_to_sector(
    detection: Detection,
    image_width: int,
    hfov_deg: float,
    turret_pan_deg: float = 0.0,
    range_m: float | None = None,
    gust_rms_ms: float = 0.0,
) -> tuple[float, float]:
    """Project a detection to a world-azimuth sector, widened for wind and doubt."""
    x, _y, w, _h = detection.bbox
    if image_width <= 0:
        raise ValueError("image width must be positive")

    # Bearing of the box's centre, relative to the camera axis.
    centre_px = x + w / 2.0
    fraction = (centre_px - image_width / 2.0) / (image_width / 2.0)
    bearing_deg = fraction * (hfov_deg / 2.0)

    # Angular half-width of the box itself.
    box_half_deg = (w / image_width) * hfov_deg / 2.0

    if range_m is None:
        # Unknown range: assume the worst rather than guessing a middle value.
        half = max(box_half_deg, UNKNOWN_RANGE_HALF_WIDTH_DEG)
    else:
        # A person is ~0.6 m wide; at close range that subtends a lot of sector.
        person_half_deg = math.degrees(math.atan(0.6 / max(range_m, 0.5)))
        half = max(box_half_deg, person_half_deg)

    half += GUST_WIDENING_DEG_PER_MS * max(0.0, gust_rms_ms)
    half = max(half, MIN_HALF_WIDTH_DEG)

    centre = turret_pan_deg + bearing_deg
    return (centre - half, centre + half)


def merge_sectors(
    a: tuple[float, float] | None, b: tuple[float, float] | None
) -> tuple[float, float] | None:
    """Union, never intersection.

    Two reasons to protect a region are not a reason to protect less of it. This
    is the same rule as `SafetyConstraints`' keep-out handling, for the same
    reason, and it is worth having in both places rather than shared — a caller
    reaching for "merge" should get the safe answer without knowing why.
    """
    if a is None:
        return b
    if b is None:
        return a
    return (min(a[0], b[0]), max(a[1], b[1]))


def sectors_for(
    detections: list[Detection],
    image_width: int,
    hfov_deg: float,
    turret_pan_deg: float = 0.0,
    range_m: float | None = None,
    gust_rms_ms: float = 0.0,
    static_sector: tuple[float, float] | None = None,
) -> tuple[float, float] | None:
    """Every detection plus the configured sector, merged into one keep-out."""
    merged = static_sector
    for detection in detections:
        merged = merge_sectors(
            merged,
            detection_to_sector(
                detection, image_width, hfov_deg, turret_pan_deg, range_m, gust_rms_ms
            ),
        )
    return merged
