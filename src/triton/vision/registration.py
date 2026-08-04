# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""Measuring flicker while the camera is moving — spec §3.10.

## The defect

`FireDetector.detect` clears its flicker history on every frame where the camera
is moving:

    if camera_moving:
        self._history.clear()

For a fixed turret that is fine — slews are brief and the history refills the
moment the turret settles. **On a platform that is permanently moving it is
fatal.** The history never accumulates, the detector silently degrades to
colour-only, and the flicker gate — the entire defence against spraying a red
object that is not a fire — stops existing.

That makes this a prerequisite for the mobile tier, not an improvement to it.

## The fix, and the trap inside it

Compare masks in a **scene-stabilised** reference: warp each mask by the known
ego-motion before differencing, so a stationary object lands in the same place in
consecutive frames and does not read as change.

The trap is that registration is never perfect, and imperfect registration
manufactures *fake flicker* — a rigid red object appears to shimmer at its edges
purely because the warp was a pixel out. A detector that fell for that would rate
a decoy as more fire-like the rougher the ride got, which is precisely backwards.

## What removes it — and what did not

The first design here subtracted a frame-wide baseline, on the theory that
registration error affects the whole frame while a fire flickers more than its
surroundings. **That does not work on this input, and the failure is instructive.**
The history holds a *fire-colour mask*, and a colour mask has essentially no
background structure — everything set in it is the candidate. So the "global"
rate outside the blob measures nothing. Measured: registration error scored
0.055, a genuinely flickering fire scored 0.057. Indistinguishable.

(The same trap the spec flags for SP6 — a flat synthetic background makes
image-based mechanisms look vacuously perfect. Here it made one look vacuously
useless, which was easier to notice.)

What actually separates them is WHERE the toggling happens. A rigid object
mis-registered by a few pixels toggles at its **edges**: the mask slides, so a
rim of width δ flips. A fire toggles throughout its **interior**, because the
flame changes shape. So the score is the toggle rate inside the blob eroded by
`EDGE_MARGIN_PX` — registration error smaller than that margin contributes
nothing, and flicker contributes fully.

The honest limit: an error LARGER than the margin moves the interior too, and
nothing here distinguishes that from a fire. `EDGE_MARGIN_PX` is therefore a
statement about how accurate the ego-motion prediction has to be.

## Why this is opt-in

Enabling it unconditionally would change detection during the fixed turret's own
brief slews, which moves every golden fixture — and the programme's three planned
regenerations are spent. It is also unnecessary there: a fixed turret's history
refills as soon as it settles. So it is enabled for platforms that actually move,
which is exactly the case it exists for.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

# How far in from a blob's edge the interior starts. Sized to swallow the
# registration error a real ego-motion prediction leaves behind — a few pixels —
# so a rigid object mis-registered by less than this contributes no flicker at
# all, while a flame that changes shape throughout its body contributes fully.
EDGE_MARGIN_PX = 5


def stabilise_mask(mask: np.ndarray, shift_px: float) -> np.ndarray:
    """Shift a boolean mask into a common reference by the known ego-motion.

    Integer column roll rather than an interpolating warp: the input is boolean,
    so interpolation would invent intermediate values, and sub-pixel accuracy is
    exactly what the interior-erosion margin exists to tolerate.
    """
    columns = int(round(shift_px))
    if columns == 0:
        return mask
    rolled = np.roll(mask, columns, axis=1)
    # Wrapped-in columns are unknown, not black. Mark them False; they sit at the
    # frame edge, outside any eroded blob interior, so they cost nothing.
    if columns > 0:
        rolled[:, :columns] = False
    else:
        rolled[:, columns:] = False
    return rolled


@dataclass
class StabilisedFlicker:
    """Flicker history that survives a moving camera.

    Holds masks already warped into a common reference, plus the global toggle
    rate per adjacent pair so the ego-synchronous component can be removed.
    """

    length: int = 6
    _history: deque = field(default_factory=lambda: deque(maxlen=6))
    _cumulative_px: float = 0.0

    def __post_init__(self) -> None:
        self._history = deque(maxlen=self.length)

    def clear(self) -> None:
        self._history.clear()
        self._cumulative_px = 0.0

    def append(self, mask: np.ndarray, shift_px: float = 0.0) -> None:
        """Add a frame, warped back into the history's common reference.

        `shift_px` is the PER-FRAME image displacement of a world-fixed point —
        what `egomotion.pan_shift_px` returns. It is accumulated here, and the
        mask is rolled by the *negative* of that total, which is what actually
        puts a stationary object in the same columns in every stored frame.

        Applying the per-frame delta directly instead (an easy mistake, and one
        this code made first) offsets every frame by the same amount and aligns
        nothing — the object still marches across the stored history and reads as
        maximal flicker.
        """
        self._cumulative_px += shift_px
        self._history.append(
            stabilise_mask(np.asarray(mask, dtype=bool), -self._cumulative_px)
        )

    def __len__(self) -> int:
        return len(self._history)

    def global_toggle_rate(self, exclude_bbox: tuple[int, int, int, int] | None = None) -> float:
        """The registration-error baseline: how much the frame changed OUTSIDE
        the candidate.

        Excluding the candidate matters. A fire is a real part of the frame, so
        leaving it in means it contributes to the very baseline it is being
        compared against — it dilutes its own signal, and the more of the frame
        it fills the more it dilutes. The baseline is supposed to answer "how
        much did registration error move things", and the candidate region is
        precisely where that question is not being asked.
        """
        if len(self._history) < 2:
            return 0.0

        keep = None
        if exclude_bbox is not None:
            x, y, w, h = exclude_bbox
            x = int(round(x - self._cumulative_px))
            keep = np.ones(self._history[0].shape, dtype=bool)
            keep[max(0, y) : y + h, max(0, x) : x + w] = False
            if not keep.any():
                return 0.0

        toggles, pairs = 0.0, 0
        prev = None
        for mask in self._history:
            if prev is not None:
                changed = prev != mask
                toggles += float(np.mean(changed[keep]) if keep is not None
                                 else np.mean(changed))
                pairs += 1
            prev = mask
        return toggles / pairs if pairs else 0.0

    def local_toggle_rate(self, bbox: tuple[int, int, int, int]) -> float:
        """Toggle rate inside `bbox`, which is in CURRENT-FRAME coordinates.

        The stored masks are aligned to the history's reference, not to the
        current frame, so the bbox has to be mapped back by the same cumulative
        shift. Sampling a current-frame bbox directly against stabilised masks
        reads a patch of empty background — the object is not there any more —
        and returns almost no flicker for a vigorously flickering fire. (It did
        exactly that first.)
        """
        if len(self._history) < 2:
            return 0.0
        x, y, w, h = bbox
        x = int(round(x - self._cumulative_px))
        if x < 0 or y < 0:
            return 0.0
        toggles, pairs = 0.0, 0
        prev = None
        for mask in self._history:
            roi = mask[y : y + h, x : x + w]
            if prev is not None and prev.shape == roi.shape and roi.size:
                toggles += float(np.mean(prev != roi))
                pairs += 1
            prev = roi
        return toggles / pairs if pairs else 0.0

    def flicker_score(self, bbox: tuple[int, int, int, int]) -> float:
        """Toggling in the blob's INTERIOR, which is what separates a fire from a
        mis-registered rigid object.

        ## Why not a frame-wide baseline

        The obvious design — subtract the global toggle rate, on the grounds that
        registration error affects the whole frame — does not survive contact with
        the actual input. This history holds a **fire-colour mask**, and a colour
        mask has almost no background structure: everything that is set is the
        candidate. So the "global" rate outside the blob measures nothing, and
        subtracting it accomplishes nothing. Measured, when this module tried it:
        registration error scored 0.055 and a genuinely flickering fire scored
        0.057 — indistinguishable.

        (This is the same trap the spec flags for SP6: a flat synthetic background
        makes every image-based mechanism look vacuously perfect. Here it made one
        look vacuously useless, which was more informative.)

        ## What actually separates them

        A rigid object that is mis-registered by a few pixels toggles at its
        EDGES — the mask slides, so a rim of width δ flips. A fire toggles
        throughout its INTERIOR, because the flame itself changes shape.

        So the score is the toggle rate inside the blob eroded by
        `EDGE_MARGIN_PX`. Registration error smaller than that margin contributes
        nothing at all; flicker contributes fully.
        """
        x, y, w, h = bbox
        m = EDGE_MARGIN_PX
        if w <= 2 * m or h <= 2 * m:
            # Too small to have an interior. Fall back to the whole box rather
            # than reporting zero, which would silently exempt small blobs from
            # the flicker gate entirely.
            return self.local_toggle_rate(bbox)
        return self.local_toggle_rate((x + m, y + m, w - 2 * m, h - 2 * m))
