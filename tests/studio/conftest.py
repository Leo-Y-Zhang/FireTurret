# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""Headless Qt test setup for the FireTurret Studio suite.

Force the offscreen platform BEFORE any QApplication is created, and disable
pyqtgraph OpenGL (raster only) — matching the raster-only runtime story.
pytest-qt provides the `qtbot` and `qapp` fixtures.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _pyqtgraph_raster():
    import pyqtgraph as pg

    pg.setConfigOptions(useOpenGL=False, antialias=False)
    yield
