# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""Learned fire detector (ONNX), implementing the same injectable interface as the
classical FireDetector — ``detect(frame_bgr, camera_moving) -> list[FireBlob]``.

It runs a segmentation model that outputs a per-pixel fire-probability map,
thresholds it, and reuses ``blobs_from_mask`` for geometry — so a trained model
drops straight into the pipeline (``Pipeline(cfg, detector=OnnxFireDetector(...))``
or ``triton run --detector onnx --model fire.onnx``) with no other change.

Execution providers are chosen in order of preference — TensorRT, then CUDA, then
CPU — so on an NVIDIA GPU or a Jetson it runs accelerated, and falls back to CPU
elsewhere. Bring your own trained model; ``onnxruntime`` is the ``[ml]`` extra.
"""
from __future__ import annotations

import cv2
import numpy as np

from ..config import DetectorConfig
from .firedetect import FireBlob, blobs_from_mask

# most-accelerated first; the ones actually present are used (NVIDIA GPU/Jetson)
_PREFERRED_PROVIDERS = [
    "TensorrtExecutionProvider",
    "CUDAExecutionProvider",
    "CPUExecutionProvider",
]


def select_providers(available: list[str]) -> list[str] | None:
    """Preferred execution providers that are actually available (None = default)."""
    chosen = [p for p in _PREFERRED_PROVIDERS if p in available]
    return chosen or None


class OnnxFireDetector:
    def __init__(self, model_path, cfg: DetectorConfig, *, threshold: float = 0.5,
                 input_size: tuple[int, int] | None = None, session=None) -> None:
        self.cfg = cfg
        self.threshold = threshold
        self._input_size = input_size  # (w, h) the model expects; None = frame size
        self._session = session if session is not None else self._make_session(model_path)
        self._input_name = self._session.get_inputs()[0].name

    @staticmethod
    def _make_session(model_path):
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - depends on the [ml] extra
            raise ImportError(
                "OnnxFireDetector needs onnxruntime — install the ML extra: "
                "pip install triton[ml]"
            ) from exc
        providers = select_providers(ort.get_available_providers())
        return ort.InferenceSession(str(model_path), providers=providers)

    def reset(self) -> None:  # parity with FireDetector
        pass

    def detect(self, frame_bgr: np.ndarray, camera_moving: bool = False) -> list[FireBlob]:
        height, width = frame_bgr.shape[:2]
        outputs = self._session.run(None, {self._input_name: self._preprocess(frame_bgr)})
        prob = np.asarray(outputs[0]).squeeze()
        mask = (prob >= self.threshold).astype(np.uint8) * 255
        if mask.shape[:2] != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        confidence = 0.45 if camera_moving else 0.9  # colour-only trust while slewing
        return blobs_from_mask(mask, self.cfg.min_blob_area_px, base_confidence=confidence)

    def _preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        width, height = self._input_size or (frame_bgr.shape[1], frame_bgr.shape[0])
        resized = cv2.resize(frame_bgr, (width, height))
        rgb = resized[:, :, ::-1].astype(np.float32) / 255.0  # BGR->RGB, normalised
        nchw = np.transpose(rgb, (2, 0, 1))[None, ...]  # NCHW batch of 1
        return np.ascontiguousarray(nchw)
