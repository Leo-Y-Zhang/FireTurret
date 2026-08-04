# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""Learned (ONNX) fire detector — exercised with a fake inference session so no
model, onnxruntime, or GPU is required."""
from __future__ import annotations

import numpy as np

from triton.config import DEFAULT_CONFIG
from triton.vision.onnx_detector import OnnxFireDetector, select_providers


class _FakeSession:
    """Returns a canned probability map, ignoring the input (like a real model)."""

    def __init__(self, prob_map: np.ndarray) -> None:
        self._prob = prob_map

    def get_inputs(self):
        class _In:
            name = "input"

        return [_In()]

    def run(self, _outputs, _feeds):
        return [self._prob]


def test_probability_map_becomes_blobs():
    prob = np.zeros((54, 96), np.float32)
    prob[20:40, 40:70] = 0.9  # a hot region -> one fire blob
    det = OnnxFireDetector(None, DEFAULT_CONFIG.detector, threshold=0.5,
                           input_size=(96, 54), session=_FakeSession(prob))
    frame = np.zeros((540, 960, 3), np.uint8)
    blobs = det.detect(frame)
    assert len(blobs) >= 1
    b = blobs[0]
    assert 0 < b.cx < 960 and 0 < b.cy < 540  # mapped back to frame coordinates
    assert b.area > DEFAULT_CONFIG.detector.min_blob_area_px


def test_below_threshold_gives_no_blobs():
    prob = np.full((54, 96), 0.2, np.float32)  # all below threshold
    det = OnnxFireDetector(None, DEFAULT_CONFIG.detector, threshold=0.5,
                           session=_FakeSession(prob))
    assert det.detect(np.zeros((54, 96, 3), np.uint8)) == []


def test_provider_preference_prefers_gpu():
    # TensorRT/CUDA are chosen ahead of CPU when present (NVIDIA GPU / Jetson)
    assert select_providers(["CPUExecutionProvider", "CUDAExecutionProvider"]) == [
        "CUDAExecutionProvider", "CPUExecutionProvider"]
    assert select_providers(["CPUExecutionProvider"]) == ["CPUExecutionProvider"]
    assert select_providers([]) is None


def test_detect_matches_injectable_interface():
    # duck-types the classical FireDetector: .detect(frame, camera_moving) + .reset()
    det = OnnxFireDetector(None, DEFAULT_CONFIG.detector,
                           session=_FakeSession(np.zeros((10, 10), np.float32)))
    det.reset()
    assert det.detect(np.zeros((10, 10, 3), np.uint8), camera_moving=True) == []
