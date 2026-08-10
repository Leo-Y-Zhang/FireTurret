# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""Live camera view: displays the annotated frames streamed from the SimWorker.

Frames arrive as BGR ndarrays (OpenCV convention); we convert to RGB and show a
size-fitted pixmap. The worker already throttles frames to ~15 Hz (latest-wins).
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


class CameraView(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._label = QLabel("No frames yet — press Run (F5)")
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setMinimumSize(240, 160)
        self._pixmap: QPixmap | None = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._label)

    def set_frame(self, frame_bgr: np.ndarray) -> None:
        rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])  # BGR -> RGB, contiguous
        h, w = rgb.shape[:2]
        image = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        # QPixmap.fromImage copies the pixels, so `rgb` may be freed afterwards.
        self._pixmap = QPixmap.fromImage(image)
        self._rescale()

    def clear(self) -> None:
        self._pixmap = None
        self._label.setText("No frames yet — press Run (F5)")

    def _rescale(self) -> None:
        if self._pixmap is None:
            return
        self._label.setPixmap(self._pixmap.scaled(
            self._label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        ))

    def resizeEvent(self, event) -> None:
        self._rescale()
        super().resizeEvent(event)
