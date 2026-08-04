# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""P1a Task 5: guarded import, app bootstrap, and the `triton studio` entry point."""
from __future__ import annotations

import subprocess
import sys


def test_core_import_does_not_load_pyside():
    """`import triton.__main__` must not import PySide6 (core install has no Qt)."""
    code = (
        "import sys, triton.__main__; "
        "print(any(m == 'PySide6' or m.startswith('PySide6.') for m in sys.modules))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert out == "False"


def test_studio_help_exits_zero():
    """`triton studio --help` works without importing Qt (argparse handles --help)."""
    r = subprocess.run(
        [sys.executable, "-m", "triton", "studio", "--help"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0
    assert "studio" in (r.stdout + r.stderr).lower()


def test_build_app_returns_singleton_qapplication(qapp):
    from PySide6.QtWidgets import QApplication

    from triton.studio.app import build_app

    app = build_app()
    assert isinstance(app, QApplication)
    assert QApplication.instance() is app
    assert app.applicationName() == "Triton Studio"


def test_require_pyside_returns_qtwidgets():
    from triton.studio import require_pyside

    qtw = require_pyside()
    assert hasattr(qtw, "QApplication")
