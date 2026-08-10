# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""SP10 — the person/animal detection predicate.

The distinction this sub-project exists to make is between **"I looked and nobody
is there"** and **"nothing is looking"**. An empty detection list means the first.
A system that cannot express the second will arm water on hardware with no
detector installed, because absence of detections looks like absence of people.

Everything else here is about the projection being safe in the direction it is
wrong: range widens the protected sector and never narrows it, wind widens it,
uncertainty widens it, and sectors merge by union.
"""

from __future__ import annotations

import numpy as np
import pytest

from fireturret.safety.detectors import (
    Detection,
    HogPersonDetector,
    LatencyMeasurement,
    NullSafetyDetector,
    StubSafetyDetector,
    hog_available,
    may_arm_water_on_hardware,
    measure_latency,
)
from fireturret.safety.keepout import (
    GUST_WIDENING_DEG_PER_MS,
    MIN_HALF_WIDTH_DEG,
    UNKNOWN_RANGE_HALF_WIDTH_DEG,
    detection_to_sector,
    merge_sectors,
    sectors_for,
)

WIDTH, HFOV = 960, 66.0


def _centre_detection(w: int = 100) -> Detection:
    return Detection(bbox=(WIDTH // 2 - w // 2, 200, w, 200), confidence=1.0)


# ---------------------------------------- "nobody there" vs "nothing looking"

def test_the_null_detector_declares_that_it_cannot_see() -> None:
    """`capability='none'`, not an empty list. An empty list is a CLAIM."""
    detector = NullSafetyDetector()
    assert detector.capability == "none"
    assert detector.detect(np.zeros((10, 10, 3), dtype=np.uint8)) == []


def test_water_may_not_be_armed_on_hardware_with_no_detector() -> None:
    """**The reason `capability` exists.** Without it, "no detections" and "no
    detector" are the same value, and the governor arms water in both cases."""
    allowed, reason = may_arm_water_on_hardware(NullSafetyDetector())
    assert allowed is False
    assert "nothing is looking" in reason


def test_water_may_be_armed_when_something_is_actually_looking() -> None:
    allowed, reason = may_arm_water_on_hardware(StubSafetyDetector())
    assert allowed is True
    assert "person" in reason


def test_an_object_with_no_capability_attribute_is_treated_as_blind() -> None:
    """Fail closed: a third-party detector that forgot to declare capability is
    not thereby trusted."""
    class Undeclared:
        def detect(self, frame):
            return []

    assert may_arm_water_on_hardware(Undeclared())[0] is False


# -------------------------------------------------------- sector projection

def test_a_centred_detection_produces_a_sector_around_the_aim() -> None:
    lo, hi = detection_to_sector(_centre_detection(), WIDTH, HFOV, turret_pan_deg=0.0,
                                 range_m=5.0)
    assert lo < 0.0 < hi
    assert (lo + hi) / 2 == pytest.approx(0.0, abs=1.0)


def test_the_sector_follows_the_turret_bearing() -> None:
    lo, hi = detection_to_sector(_centre_detection(), WIDTH, HFOV, turret_pan_deg=30.0,
                                 range_m=5.0)
    assert (lo + hi) / 2 == pytest.approx(30.0, abs=1.0)


def test_a_detection_off_to_the_side_maps_off_to_the_side() -> None:
    right = Detection(bbox=(WIDTH - 150, 200, 100, 200), confidence=1.0)
    lo, hi = detection_to_sector(right, WIDTH, HFOV, range_m=5.0)
    assert (lo + hi) / 2 > 10.0


def test_an_unknown_range_produces_a_WIDE_sector_not_a_guess() -> None:
    """Range comes from the same biased monocular estimate the architecture
    distrusts, so unknown range assumes the worst rather than a middle value."""
    lo, hi = detection_to_sector(_centre_detection(), WIDTH, HFOV, range_m=None)
    assert (hi - lo) / 2 >= UNKNOWN_RANGE_HALF_WIDTH_DEG


def test_range_only_ever_widens_the_sector() -> None:
    """A closer person subtends more sector. The estimate may make the protected
    region BIGGER — where being wrong is wasteful — never smaller, where being
    wrong means spraying someone."""
    near = detection_to_sector(_centre_detection(), WIDTH, HFOV, range_m=1.5)
    far = detection_to_sector(_centre_detection(), WIDTH, HFOV, range_m=15.0)
    assert (near[1] - near[0]) > (far[1] - far[0])


def test_gusts_widen_the_sector() -> None:
    """Wind blows spray toward people, so a sector sized for still air is wrong
    in the dangerous direction."""
    still = detection_to_sector(_centre_detection(), WIDTH, HFOV, range_m=5.0,
                                gust_rms_ms=0.0)
    gusty = detection_to_sector(_centre_detection(), WIDTH, HFOV, range_m=5.0,
                                gust_rms_ms=4.0)
    widening = ((gusty[1] - gusty[0]) - (still[1] - still[0])) / 2
    assert widening == pytest.approx(4.0 * GUST_WIDENING_DEG_PER_MS, abs=0.01)


def test_a_tiny_far_detection_still_gets_a_minimum_sector() -> None:
    tiny = Detection(bbox=(WIDTH // 2, 200, 4, 8), confidence=1.0)
    lo, hi = detection_to_sector(tiny, WIDTH, HFOV, range_m=40.0)
    assert (hi - lo) / 2 >= MIN_HALF_WIDTH_DEG


def test_a_degenerate_image_width_is_rejected() -> None:
    with pytest.raises(ValueError):
        detection_to_sector(_centre_detection(), 0, HFOV)


# ------------------------------------------------------------------ merging

def test_sectors_merge_by_UNION_never_intersection() -> None:
    """Two reasons to protect a region are not a reason to protect less of it."""
    assert merge_sectors((-10.0, 10.0), (5.0, 25.0)) == (-10.0, 25.0)
    assert merge_sectors(None, (5.0, 25.0)) == (5.0, 25.0)
    assert merge_sectors((-10.0, 10.0), None) == (-10.0, 10.0)
    assert merge_sectors(None, None) is None


def test_detections_merge_with_the_configured_static_sector() -> None:
    merged = sectors_for(
        [_centre_detection()], WIDTH, HFOV, range_m=5.0, static_sector=(40.0, 60.0)
    )
    assert merged[0] < 0.0
    assert merged[1] == 60.0


def test_no_detections_leaves_the_static_sector_alone() -> None:
    assert sectors_for([], WIDTH, HFOV, static_sector=(-5.0, 5.0)) == (-5.0, 5.0)


# ------------------------------------------------------------ measured, not assumed

def test_latency_is_measured_with_an_injected_clock() -> None:
    """This number sets SP9's staleness threshold, so it is a safety input rather
    than a benchmark. An assumed value produces a veto that either chatters or
    trusts stale answers."""
    ticks = iter([0.0, 0.1, 1.0, 1.3, 2.0, 2.05])
    measurement = measure_latency(
        StubSafetyDetector(), np.zeros((4, 4, 3), dtype=np.uint8),
        runs=3, clock=lambda: next(ticks),
    )
    assert measurement.count == 3
    assert measurement.mean_s == pytest.approx((0.1 + 0.3 + 0.05) / 3)
    assert measurement.p99_s == pytest.approx(0.3)


def test_an_empty_measurement_reports_zero_rather_than_dividing_by_zero() -> None:
    empty = LatencyMeasurement()
    assert empty.mean_s == 0.0 and empty.p99_s == 0.0


def test_the_measurement_summary_is_human_readable() -> None:
    m = LatencyMeasurement(samples=[0.12, 0.3, 0.25])
    assert "ms" in m.summary() and "p99" in m.summary()


def test_the_derived_staleness_threshold_uses_the_measured_p99() -> None:
    """SP9 and SP10 meeting: the measurement feeds the threshold directly."""
    from fireturret.safety.veto import derive_max_age_s

    m = LatencyMeasurement(samples=[0.12, 0.18, 0.30])
    assert derive_max_age_s(m.p99_s, submit_period_s=0.2) > 0.2 + m.p99_s


# ------------------------------------------------------------------ the HOG

def test_hog_is_unavailable_on_opencv_5_and_says_so_clearly() -> None:
    """**A real portability finding, not a footnote.** `cv2.HOGDescriptor` was
    REMOVED in OpenCV 5 — this environment runs 5.0.0 and has no `HOG*` symbol at
    all. So the "no extra dependency" argument for HOG holds on OpenCV 4 and does
    not on 5, where the ONNX detector becomes the only built-in option.

    Failing loudly at construction is right for a SAFETY component: a detector
    that silently degraded to "finds nothing" would be indistinguishable from a
    clear scene, which is the exact confusion `capability` exists to prevent.
    """
    if hog_available():
        pytest.skip("this OpenCV build has HOGDescriptor")
    with pytest.raises(RuntimeError, match="OpenCV 5"):
        HogPersonDetector()


@pytest.mark.skipif(not hog_available(), reason="cv2.HOGDescriptor removed in OpenCV 5")
def test_the_hog_detector_constructs_and_declares_its_capability() -> None:
    assert HogPersonDetector().capability == "person"


@pytest.mark.skipif(not hog_available(), reason="cv2.HOGDescriptor removed in OpenCV 5")
def test_the_hog_detector_finds_nothing_in_an_empty_scene() -> None:
    """Not a capability claim — just that it does not hallucinate. What it CANNOT
    do (seated, prone, occluded people; animals) is stated in the module
    docstring and in docs/SAFETY.md rather than tested into existence."""
    assert HogPersonDetector().detect(np.zeros((128, 64, 3), dtype=np.uint8)) == []


def test_the_stub_detector_returns_ground_truth_for_scenario_tests() -> None:
    """The bystander e2e test uses this rather than HOG, so it measures the
    SYSTEM's response to a detection rather than HOG's accuracy."""
    person = Detection(bbox=(10, 10, 20, 40), confidence=0.9)
    assert StubSafetyDetector([person]).detect(np.zeros((4, 4, 3), np.uint8)) == [person]
