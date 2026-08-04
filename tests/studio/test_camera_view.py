# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""P2: the live camera view accepts BGR frames and renders a pixmap."""
from __future__ import annotations

import numpy as np

from triton.studio.views.camera_view import CameraView


def test_set_frame_produces_pixmap(qapp):
    w = CameraView()
    w.resize(320, 240)
    frame = np.zeros((100, 160, 3), np.uint8)
    frame[:, :, 2] = 255  # red (BGR)
    w.set_frame(frame)
    pm = w._label.pixmap()
    assert pm is not None and not pm.isNull()


def test_clear_resets(qapp):
    w = CameraView()
    w.set_frame(np.zeros((10, 10, 3), np.uint8))
    w.clear()
    assert w._pixmap is None
