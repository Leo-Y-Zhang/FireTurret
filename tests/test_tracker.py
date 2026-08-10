# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
from fireturret.config import DetectorConfig
from fireturret.vision.firedetect import FireBlob
from fireturret.vision.tracker import FireTracker

CFG = DetectorConfig()


def blob(cx=100.0, cy=100.0, area=500.0, conf=0.8) -> FireBlob:
    return FireBlob(
        cx=cx, cy=cy, area=area,
        bbox=(int(cx) - 10, int(cy) - 10, 20, 20),
        colour_score=0.8, flicker_score=0.2, confidence=conf,
    )


def test_persistence_confirms_after_enough_hits() -> None:
    tracker = FireTracker(CFG)
    for _ in range(CFG.confirm_hits - 1):
        tracker.update([blob()])
    assert tracker.primary() is None
    tracker.update([blob()])
    assert tracker.primary() is not None


def test_track_survives_brief_dropout() -> None:
    tracker = FireTracker(CFG)
    for _ in range(CFG.confirm_hits + 2):
        tracker.update([blob()])
    for _ in range(3):
        tracker.update([])  # dropout shorter than max_misses
    assert tracker.primary() is not None
    for _ in range(CFG.max_misses + 1):
        tracker.update([])
    assert tracker.primary() is None


def test_distinct_fires_get_distinct_tracks() -> None:
    tracker = FireTracker(CFG)
    for _ in range(CFG.confirm_hits + 1):
        tracker.update([blob(cx=100), blob(cx=700)])
    assert len(tracker.tracks) == 2
    ids = {t.track_id for t in tracker.tracks}
    assert len(ids) == 2


def test_primary_prefers_confident_large_fire() -> None:
    tracker = FireTracker(CFG)
    for _ in range(CFG.confirm_hits + 3):
        tracker.update([blob(cx=100, area=300, conf=0.5), blob(cx=700, area=2000, conf=0.9)])
    primary = tracker.primary()
    assert primary is not None
    assert abs(primary.cx - 700) < 30


def test_position_smoothing_tracks_movement() -> None:
    tracker = FireTracker(CFG)
    tracker.update([blob(cx=100)])
    tracker.update([blob(cx=140)])
    t = tracker.tracks[0]
    assert 100 < t.cx < 140  # EMA between the two sightings


# ever_confirmed is latched forever and nothing in src/ clears it, so what is
# allowed to set it is the safety-critical decision in this file.

def test_a_rigid_decoy_never_latches_confirmation_on_colour_alone() -> None:
    """A fire-coloured object that does not flicker must never become a target.

    The shipped (non-stabilised) detector path returns colour-only confidence
    while the camera slews, and it used to return 0.45 * colour_score. colour_score
    is at least 0.6 by construction, so a large blob reached 0.45 - above the 0.3
    confirmation threshold. A red cloth or a warning light that stayed in frame
    through a slew therefore latched ever_confirmed on colour alone, permanently.
    Once the real fire was out it was then the only confirmed track, so the turret
    would hose it indefinitely.
    """
    tracker = FireTracker(CFG)
    rigid = blob(conf=0.45, area=2000.0)          # what the old path produced
    rigid = FireBlob(**{**rigid.__dict__, "flicker_score": 0.0})

    for _ in range(CFG.confirm_hits * 3):
        tracker.update([rigid])

    assert tracker.tracks, "the decoy should still be TRACKED, just never confirmed"
    assert not any(t.ever_confirmed for t in tracker.tracks), (
        "a non-flickering object latched confirmation on colour alone"
    )
    assert tracker.primary() is None, "an unconfirmed decoy must not become the target"


def test_a_flickering_fire_still_confirms() -> None:
    """The liveness half: the guard must not make real fire unconfirmable."""
    tracker = FireTracker(CFG)
    fire = blob(conf=0.8, area=2000.0)
    assert fire.flicker_score >= CFG.flicker_threshold  # the helper's default

    for _ in range(CFG.confirm_hits * 2):
        tracker.update([fire])

    assert any(t.ever_confirmed for t in tracker.tracks)
    assert tracker.primary() is not None


def test_a_learned_detector_can_still_confirm_without_flicker_evidence() -> None:
    """A gate that cannot be satisfied is an outage, not a safety feature.

    blobs_from_mask sets flicker_score=0.0 BY DESIGN - a semantic mask has no
    temporal history - and the ONNX detector uses it exclusively. Requiring
    flicker evidence outright meant no track could ever be confirmed on
    `--detector onnx`: primary() always None, mission never leaves SEARCH, the
    turret never fires. That regression shipped in 0544204 and the whole suite
    passed, because every other test exercises the classical detector.

    The distinction the tracker needs is between "measured, and it does not
    flicker" and "no measurement exists". Only the first is evidence against.
    """
    import numpy as np

    from fireturret.vision.firedetect import blobs_from_mask

    mask = np.zeros((240, 320), np.uint8)
    mask[100:140, 140:180] = 255
    blobs = blobs_from_mask(mask, CFG.min_blob_area_px, base_confidence=0.9)
    assert blobs and blobs[0].flicker_measured is False
    assert blobs[0].flicker_score == 0.0

    tracker = FireTracker(CFG)
    for _ in range(CFG.confirm_hits * 2):
        tracker.update(blobs)

    assert any(t.ever_confirmed for t in tracker.tracks), (
        "a learned detector must still be able to confirm a target"
    )
    assert tracker.primary() is not None


def test_a_measured_zero_flicker_still_blocks_confirmation() -> None:
    """The other half: where flicker IS measured, 0.0 is evidence AGAINST.

    Without this pair the fix above could be satisfied by dropping the flicker
    requirement altogether, which would reopen the decoy hole.
    """
    rigid = blob(conf=0.45, area=2000.0)
    assert rigid.flicker_measured is True
    rigid = FireBlob(**{**rigid.__dict__, "flicker_score": 0.0})

    tracker = FireTracker(CFG)
    for _ in range(CFG.confirm_hits * 3):
        tracker.update([rigid])

    assert not any(t.ever_confirmed for t in tracker.tracks)
