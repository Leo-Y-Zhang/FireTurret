# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""Persistence tracker: fire blobs must be seen repeatedly before they become
confirmed targets, and confirmed targets survive brief dropouts (occlusion,
turret motion, smoke). Simple nearest-centroid association — targets here are
few and slow-moving.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import DetectorConfig
from .firedetect import FireBlob


@dataclass
class Track:
    track_id: int
    cx: float
    cy: float
    area: float
    hits: int = 1
    misses: int = 0
    confidence: float = 0.0
    last_blob: FireBlob | None = None
    # Latched once the track has passed the flicker/persistence test while the
    # camera was still. It then STAYS confirmed through camera motion (which
    # depresses confidence), so aiming does not "lose" a real fire mid-slew.
    ever_confirmed: bool = False

    def confirmed(self, cfg: DetectorConfig) -> bool:
        return self.ever_confirmed


@dataclass
class FireTracker:
    cfg: DetectorConfig
    match_dist_px: float = 80.0
    tracks: list[Track] = field(default_factory=list)
    _next_id: int = 1

    def reset(self) -> None:
        self.tracks.clear()

    def update(
        self,
        blobs: list[FireBlob],
        shift_px: float = 0.0,
        warp=None,
    ) -> list[Track]:
        """Associate blobs to tracks, predicting ego-motion first.

        The camera rides the pan stage, so a slew slides a world-fixed fire
        across the image. Predicting that motion BEFORE association is what makes
        one track follow the fire through a slew instead of spawning a stale
        trail.

        `warp` is a callable `(u, v) -> (u, v)` and is the accurate path: the
        pinhole map is `u = cx + fpx·tan(az)`, so the displacement depends on
        WHERE in the frame the track is, and a single scalar under-corrects by
        ~25% at 30 deg off-axis. `shift_px` is the old scalar, kept for callers
        (and tests) that genuinely want a uniform translation.
        """
        if warp is not None:
            for track in self.tracks:
                track.cx, track.cy = warp(track.cx, track.cy)
        elif shift_px:
            for track in self.tracks:
                track.cx += shift_px
        unmatched = list(blobs)
        for track in self.tracks:
            best = None
            best_d = self.match_dist_px
            for blob in unmatched:
                d = ((blob.cx - track.cx) ** 2 + (blob.cy - track.cy) ** 2) ** 0.5
                if d < best_d:
                    best_d = d
                    best = blob
            if best is not None:
                unmatched.remove(best)
                # EMA position so noise does not shake the aim point
                track.cx = 0.7 * track.cx + 0.3 * best.cx
                track.cy = 0.7 * track.cy + 0.3 * best.cy
                track.area = best.area
                track.hits += 1
                track.misses = 0
                track.confidence = 0.8 * track.confidence + 0.2 * best.confidence
                track.last_blob = best
                # ever_confirmed is latched forever and nothing clears it, so the
                # bar for setting it is the one that really matters. Trusting a
                # smoothed confidence alone meant any path returning a high enough
                # number could latch a target without one flicker measurement ever
                # supporting it. Require the evidence directly: the matched blob
                # must have passed the flicker gate on the frame that latches it.
                # Real fire flickers, so it still confirms, stationary or through
                # a stabilised slew. A rigid fire-coloured object never can,
                # whatever its colour score.
                # The flicker requirement applies only where flicker is actually
                # MEASURABLE. A learned detector returns a semantic mask with no
                # temporal history (blobs_from_mask sets flicker_measured=False),
                # so demanding flicker evidence from it would mean no track is
                # ever confirmed, primary() always returns None, and the turret
                # never fires on the --detector onnx path. A gate that cannot be
                # satisfied is not a safety feature, it is an outage; that
                # regression shipped in 0544204 and every test passed, because
                # the suite exercises only the classical detector.
                #
                # Where flicker IS measured, 0.0 means "measured, and this does
                # not flicker", and the requirement stands: that is what stops a
                # red cloth latching a permanent target on colour alone.
                flicker_ok = (
                    not best.flicker_measured
                    or best.flicker_score >= self.cfg.flicker_threshold
                )
                if (
                    track.hits >= self.cfg.confirm_hits
                    and track.confidence > 0.3
                    and flicker_ok
                ):
                    track.ever_confirmed = True
            else:
                track.misses += 1
                track.confidence *= 0.92
        self.tracks = [t for t in self.tracks if t.misses <= self.cfg.max_misses]
        for blob in unmatched:
            self.tracks.append(
                Track(
                    track_id=self._next_id,
                    cx=blob.cx,
                    cy=blob.cy,
                    area=blob.area,
                    confidence=blob.confidence,
                    last_blob=blob,
                )
            )
            self._next_id += 1
        return self.tracks

    def primary(self) -> Track | None:
        """The target to fight: highest confidence-weighted area among
        confirmed tracks."""
        confirmed = [t for t in self.tracks if t.confirmed(self.cfg)]
        if not confirmed:
            return None
        return max(confirmed, key=lambda t: t.confidence * t.area)
