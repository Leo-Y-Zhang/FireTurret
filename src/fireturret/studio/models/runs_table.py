# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""The Runs table model — a first-class list of completed simulation runs.

This is the strongest "engineering-app" signal (cf. OpenRocket's Flight
Simulations tab): every finished run lands here as a row the user can inspect,
rename, and delete. Overlay/compare/export come in later phases.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt

_ROOT = QModelIndex()  # the invalid root index (module singleton; avoids B008)


@dataclass
class RunRow:
    name: str
    extinguished: bool
    water_l: float
    acquire_s: float | None
    mean_miss_m: float
    peak_miss_m: float
    seed: int
    scenario: str
    timestamp: str
    stale: bool = False  # the config/scenario changed since this run
    n_fires: int = 1
    fires_out: int = 0

    @classmethod
    def from_report(cls, name: str, report, seed: int, scenario: str,
                    timestamp: str) -> RunRow:
        n_fires = len(report.per_fire)
        fires_out = sum(1 for f in report.per_fire if f.extinguished)
        return cls(
            name=name, extinguished=report.extinguished, water_l=report.water_used_l,
            acquire_s=report.time_to_first_suppress_s, mean_miss_m=report.mean_miss_m,
            peak_miss_m=report.peak_miss_m, seed=seed, scenario=scenario,
            timestamp=timestamp, n_fires=n_fires or 1, fires_out=fires_out,
        )


def _acq(r: RunRow) -> str:
    return f"{r.acquire_s:.1f}" if r.acquire_s is not None else "-"


def _status(r: RunRow) -> str:
    if r.n_fires > 1:  # multi-fire: show how many are out
        base = f"{r.fires_out}/{r.n_fires} out"
    else:
        base = "extinguished" if r.extinguished else "incomplete"
    return f"stale · {base}" if r.stale else base


_COLUMNS: list[tuple[str, Callable[[RunRow], str]]] = [
    ("Name", lambda r: r.name),
    ("Status", _status),
    ("Water (L)", lambda r: f"{r.water_l:.2f}"),
    ("Acquire (s)", _acq),
    ("Mean miss (m)", lambda r: f"{r.mean_miss_m:.2f}"),
    ("Peak miss (m)", lambda r: f"{r.peak_miss_m:.2f}"),
    ("Seed", lambda r: str(r.seed)),
    ("Scenario", lambda r: r.scenario),
    ("Time", lambda r: r.timestamp),
]


class RunsTableModel(QAbstractTableModel):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.rows: list[RunRow] = []

    def rowCount(self, parent=_ROOT) -> int:
        return 0 if parent.isValid() else len(self.rows)

    def columnCount(self, parent=_ROOT) -> int:
        return 0 if parent.isValid() else len(_COLUMNS)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or role != Qt.ItemDataRole.DisplayRole:
            return None
        return _COLUMNS[index.column()][1](self.rows[index.row()])

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            return _COLUMNS[section][0]
        return section + 1

    # -- mutation ------------------------------------------------------------

    def add_run(self, row: RunRow) -> None:
        n = len(self.rows)
        self.beginInsertRows(QModelIndex(), n, n)
        self.rows.append(row)
        self.endInsertRows()

    def remove(self, i: int) -> None:
        if 0 <= i < len(self.rows):
            self.beginRemoveRows(QModelIndex(), i, i)
            del self.rows[i]
            self.endRemoveRows()

    def rename(self, i: int, name: str) -> None:
        if 0 <= i < len(self.rows):
            self.rows[i].name = name
            idx = self.index(i, 0)
            self.dataChanged.emit(idx, idx)

    def clear(self) -> None:
        self.beginResetModel()
        self.rows = []
        self.endResetModel()

    def mark_all_stale(self) -> None:
        """Flag every existing run out of date (the model changed)."""
        if not self.rows:
            return
        for row in self.rows:
            row.stale = True
        self.dataChanged.emit(self.index(0, 0),
                              self.index(len(self.rows) - 1, len(_COLUMNS) - 1))
