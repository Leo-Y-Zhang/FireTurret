# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""QApplication bootstrap, dark engineering theme, and the Studio entry point."""
from __future__ import annotations

import sys

from . import require_pyside

# Engineering palette, shared with analysis.write_report so plots/overlay/UI
# read as one system.
BG = "#10151c"
PANEL = "#141b24"
INK = "#e8edf4"
ACCENT = "#4cc3e8"
GRID = "#253141"


def build_app():
    """Return the QApplication singleton (creating it if needed), themed + raster."""
    QtWidgets = require_pyside()
    import pyqtgraph as pg

    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv[:1])
    app.setApplicationName("FireTurret Studio")
    app.setOrganizationName("Leo-Y-Zhang")
    pg.setConfigOptions(useOpenGL=False, antialias=True, background=BG, foreground=INK)
    _apply_dark_theme(app)
    return app


def _apply_dark_theme(app) -> None:
    from PySide6 import QtGui

    pal = QtGui.QPalette()
    role = QtGui.QPalette.ColorRole
    pal.setColor(role.Window, QtGui.QColor(BG))
    pal.setColor(role.Base, QtGui.QColor(PANEL))
    pal.setColor(role.AlternateBase, QtGui.QColor(BG))
    pal.setColor(role.WindowText, QtGui.QColor(INK))
    pal.setColor(role.Text, QtGui.QColor(INK))
    pal.setColor(role.Button, QtGui.QColor(PANEL))
    pal.setColor(role.ButtonText, QtGui.QColor(INK))
    pal.setColor(role.ToolTipBase, QtGui.QColor(PANEL))
    pal.setColor(role.ToolTipText, QtGui.QColor(INK))
    pal.setColor(role.Highlight, QtGui.QColor(ACCENT))
    pal.setColor(role.HighlightedText, QtGui.QColor(BG))
    app.setPalette(pal)


def main(argv: list[str] | None = None) -> int:
    """Launch the Studio main window (blocks in the Qt event loop)."""
    app = build_app()
    from .main_window import MainWindow

    win = MainWindow()
    win.show()
    return app.exec()
