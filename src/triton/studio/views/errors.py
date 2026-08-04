# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""Graceful error surfacing: route uncaught exceptions to a callback instead of a
bare crash. The MainWindow installs a callback that shows a non-modal dialog and
logs; tests install a plain collector.
"""
from __future__ import annotations

import sys
import traceback
from collections.abc import Callable


def install_excepthook(report: Callable[[str, str], None]):
    """Install a `sys.excepthook` that calls ``report(summary, details)`` on any
    uncaught exception. Returns the installed hook (also callable directly in tests)."""

    def hook(exc_type, exc, tb) -> None:
        details = "".join(traceback.format_exception(exc_type, exc, tb))
        report(f"{exc_type.__name__}: {exc}", details)

    sys.excepthook = hook
    return hook
