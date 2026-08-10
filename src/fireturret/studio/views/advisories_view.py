# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""Advisories panel: shows the operator advisories a run raised (fire out of
reach, too close, obstructed, out of traverse, safety) and how long each was
active. Reads the SimReport — no engine coupling.
"""
from __future__ import annotations

from PySide6.QtWidgets import QListWidget, QVBoxLayout, QWidget


class AdvisoriesView(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._list = QListWidget()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._list)
        self.kinds: list[str] = []
        self.set_report(None)

    def set_report(self, report) -> None:
        self._list.clear()
        if report is None or not report.advisories_raised:
            self._list.addItem("No advisories raised.")
            self.kinds = []
            return
        counts: dict[str, int] = {}
        for sample in report.telemetry:
            if sample.warning:
                counts[sample.warning] = counts.get(sample.warning, 0) + 1
        self.kinds = sorted(report.advisories_raised)
        for kind in self.kinds:
            self._list.addItem(f"{kind}  —  {counts.get(kind, 0)} ticks")
