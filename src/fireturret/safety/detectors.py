# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""Safety detectors — and a capability model that admits when nothing is looking.

## `capability='none'` is the point of this module

The obvious `SafetyDetector` returns a list of detections, and a null
implementation returns `[]`. That conflates two completely different situations:

    "I looked, and nobody is there"
    "nothing is looking"

An empty list means the first. A system that cannot tell them apart will happily
arm water on hardware with no safety detector installed, because the absence of
detections looks like the absence of people.

So every detector declares a `capability`, and `NullSafetyDetector` declares
`'none'`. The governor refuses to arm water on hardware when the only detector
present says it cannot see — which is the correct response to "nothing is
looking", and is not expressible without this distinction.

## What HOG is, and is not

`HogPersonDetector` uses OpenCV's built-in HOG people detector. It needs no extra
dependency, runs on a Pi, and is **poor at exactly the cases that matter**:

  - seated people
  - prone people
  - partially occluded people
  - anything not upright and person-shaped
  - animals — it is near-useless on animals, having been trained on pedestrians

None of that is a reason to omit it; a mitigation that catches the standing
bystander is worth having. It IS a reason to state it plainly rather than let a
reader infer that "person detection" means people are detected. `docs/SAFETY.md`
carries the same statement next to the supervised-demonstrator claim.

## Measured, not assumed

`measure_latency` exists so the numbers published beside the safety claim are
produced by running the thing, on the target, rather than quoted from a paper.
An unmeasured latency claim is worse than none: SP9's staleness threshold is
derived from p99 inference time, so a wrong number there produces a veto that
either chatters or trusts stale answers.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class Detection:
    """One detected thing, as an image-space box plus a confidence."""

    bbox: tuple[int, int, int, int]  # x, y, w, h
    confidence: float
    label: str = "person"


@runtime_checkable
class SafetyDetector(Protocol):
    #: 'none' | 'person' | 'person+animal'. See the module docstring — this is
    #: what lets the governor distinguish "nobody there" from "nothing looking".
    capability: str

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]: ...


class NullSafetyDetector:
    """No detector. Declares `capability='none'` rather than returning `[]`.

    Returning an empty list would be a claim: "I looked and nobody is there".
    This makes no claim at all, which is the truth.
    """

    capability = "none"

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        return []


class StubSafetyDetector:
    """Returns whatever it was told to. For tests, and for the e2e bystander
    scenario — which uses rendered ground truth so the test measures the SYSTEM's
    response to a detection rather than HOG's accuracy."""

    def __init__(self, detections: list[Detection] | None = None,
                 capability: str = "person") -> None:
        self.capability = capability
        self.detections = list(detections or [])

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        return list(self.detections)


def hog_available() -> bool:
    """Is OpenCV's HOG people detector present in this build?

    **It is not, on OpenCV 5.** `cv2.HOGDescriptor` was removed — this repo's own
    environment runs 5.0.0 and has no `HOG*` symbol at all. That is a real
    portability fact rather than a footnote: the "no extra dependency" argument
    for HOG holds on OpenCV 4 and simply does not on 5, where the ONNX detector
    behind the `[ml]` extra becomes the only built-in option.
    """
    import cv2

    return hasattr(cv2, "HOGDescriptor")


class HogPersonDetector:
    """OpenCV's HOG pedestrian detector. Core dependencies only — on OpenCV 4.

    See the module docstring for what it cannot do. Briefly: upright pedestrians
    reasonably, seated/prone/occluded people poorly, animals essentially not.

    Raises a clear, actionable error on OpenCV 5, where HOG no longer exists.
    Failing loudly at construction is the right behaviour for a SAFETY component:
    a detector that silently degraded to "finds nothing" would be indistinguishable
    from a clear scene, which is the exact confusion `capability` exists to prevent.
    """

    capability = "person"

    def __init__(self, hit_threshold: float = 0.0, win_stride: int = 8) -> None:
        import cv2

        if not hog_available():
            raise RuntimeError(
                f"cv2.HOGDescriptor is not available in OpenCV {cv2.__version__}. "
                "It was removed in OpenCV 5. Use OnnxSafetyDetector (the [ml] "
                "extra) or pin opencv-python<5. Refusing to construct a safety "
                "detector that cannot detect."
            )
        self._hog = cv2.HOGDescriptor()
        self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self._hit_threshold = hit_threshold
        self._win_stride = (win_stride, win_stride)

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        rects, weights = self._hog.detectMultiScale(
            frame_bgr, winStride=self._win_stride, hitThreshold=self._hit_threshold
        )
        out: list[Detection] = []
        for (x, y, w, h), weight in zip(rects, weights, strict=True):
            out.append(
                Detection(bbox=(int(x), int(y), int(w), int(h)),
                          confidence=float(weight), label="person")
            )
        return out


@dataclass
class LatencyMeasurement:
    """What `measure_latency` produced. Published, not assumed."""

    samples: list[float] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.samples)

    @property
    def mean_s(self) -> float:
        return sum(self.samples) / len(self.samples) if self.samples else 0.0

    @property
    def p99_s(self) -> float:
        """The number SP9's staleness threshold is derived from."""
        if not self.samples:
            return 0.0
        ordered = sorted(self.samples)
        index = min(len(ordered) - 1, int(round(0.99 * (len(ordered) - 1))))
        return ordered[index]

    def summary(self) -> str:
        return (
            f"{self.count} samples: mean {self.mean_s * 1000:.0f} ms, "
            f"p99 {self.p99_s * 1000:.0f} ms"
        )


def measure_latency(
    detector: SafetyDetector,
    frame_bgr: np.ndarray,
    runs: int = 20,
    clock=time.perf_counter,
) -> LatencyMeasurement:
    """Time a detector on the target hardware.

    The result feeds `safety.veto.derive_max_age_s`, so it is not a benchmark for
    a README — it sets a safety threshold. An assumed value there produces a veto
    that either chatters at the detector's beat frequency or silently trusts
    stale answers.
    """
    measurement = LatencyMeasurement()
    for _ in range(max(1, runs)):
        start = clock()
        detector.detect(frame_bgr)
        measurement.samples.append(clock() - start)
    return measurement


def may_arm_water_on_hardware(detector: SafetyDetector) -> tuple[bool, str]:
    """The governor's question, and the reason `capability='none'` exists.

    Returns (allowed, reason). A detector that cannot see is not the same as a
    clear scene, and on real hardware that difference decides whether water may
    be armed at all.
    """
    capability = getattr(detector, "capability", "none")
    if capability == "none":
        return False, (
            "no safety detector is installed, so 'no detections' means 'nothing is "
            "looking' rather than 'nobody is there'. Water stays disarmed."
        )
    return True, f"safety detector present (capability={capability!r})"
