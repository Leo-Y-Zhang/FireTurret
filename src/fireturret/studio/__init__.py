# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""FireTurret Studio — the desktop engineering workbench.

Optional: requires the ``fireturret[desktop]`` extra (PySide6-Essentials + pyqtgraph).
Nothing here is imported by the core package or the CLI unless the ``studio``
subcommand is invoked, so ``pip install fireturret`` never needs Qt.
"""
from __future__ import annotations

_INSTALL_HINT = (
    "FireTurret Studio needs the desktop extra. Install it with:\n"
    "    pip install fireturret[desktop]\n"
    "(PySide6-Essentials + pyqtgraph)."
)


def require_pyside():
    """Import and return the PySide6 ``QtWidgets`` module, or raise a friendly error."""
    try:
        from PySide6 import QtWidgets
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(_INSTALL_HINT) from exc
    return QtWidgets
