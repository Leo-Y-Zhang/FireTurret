# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""Batch parameter sweeps: run the simulation across a grid of parameter values
(1-2 axes) x seeds, in parallel over a QThreadPool.

Each cell is independent — its own SimRig(seed) inside simcore.drive, over FROZEN
config/scenario snapshots — so parallelism is safe and deterministic. Cells return
only metrics; NO matplotlib is used on pool threads (pyplot is not thread-safe).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, replace

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal

from ..config import TritonConfig
from ..rig.sim_rig import SimScenario
from ..simcore import drive

_CONFIG_GROUPS = ("camera", "turret", "jet", "detector", "servo", "mission")


def with_field(config: TritonConfig, scenario: SimScenario, group: str, name: str, value):
    """Return (config, scenario) with one field replaced (validated via __post_init__)."""
    if group == "scenario":
        return config, replace(scenario, **{name: value})
    if group in _CONFIG_GROUPS:
        sub = replace(getattr(config, group), **{name: value})
        return replace(config, **{group: sub}), scenario
    raise ValueError(f"unknown group: {group!r}")


@dataclass
class SweepAxis:
    group: str
    name: str
    values: list


@dataclass
class SweepSpec:
    axes: list          # 1 or 2 SweepAxis
    seeds: list         # >= 1 seed
    max_frames: int = 3000


@dataclass
class SweepCell:
    index: int
    coords: dict
    config: TritonConfig
    scenario: SimScenario
    seed: int
    max_frames: int


@dataclass
class SweepResult:
    index: int
    coords: dict
    extinguished: bool
    water_l: float
    acquire_s: float | None
    mean_miss_m: float
    peak_miss_m: float
    cancelled: bool = False
    failed: bool = False   # the cell raised during drive (isolated, not hung)
    error: str = ""        # short description of the failure


def sweep_grid(base_config: TritonConfig, base_scenario: SimScenario,
               spec: SweepSpec) -> list[SweepCell]:
    axis_choices = [[(a.group, a.name, v) for v in a.values] for a in spec.axes]
    cells: list[SweepCell] = []
    idx = 0
    for combo in itertools.product(*axis_choices):
        for seed in spec.seeds:
            cfg, scn = base_config, base_scenario
            coords: dict = {"seed": seed}
            for group, name, value in combo:
                cfg, scn = with_field(cfg, scn, group, name, value)
                coords[f"{group}.{name}"] = value
            cells.append(SweepCell(idx, coords, cfg, scn, seed, spec.max_frames))
            idx += 1
    return cells


def run_cell(cell: SweepCell, should_stop=None) -> SweepResult:
    if should_stop is not None and should_stop():
        return SweepResult(cell.index, cell.coords, False, 0.0, None, 0.0, 0.0, cancelled=True)
    report = drive(cell.config, cell.scenario, seed=cell.seed,
                   max_frames=cell.max_frames, should_stop=should_stop)
    return SweepResult(
        cell.index, cell.coords, report.extinguished, report.water_used_l,
        report.time_to_first_suppress_s, report.mean_miss_m, report.peak_miss_m,
    )


class SweepRunner(QObject):
    cell_done = Signal(object)   # SweepResult
    progress = Signal(int, int)  # done, total
    finished = Signal(object)    # list[SweepResult], sorted by index

    _internal = Signal(object)   # pool-thread -> main-thread hand-off

    def __init__(self, cells, pool: QThreadPool | None = None, parent=None) -> None:
        super().__init__(parent)
        self._cells = list(cells)
        self._results: list[SweepResult] = []
        self._done = 0
        self._cancel = False
        self._pool = pool or QThreadPool(self)
        self._internal.connect(self._report_cell)

    def cancel(self) -> None:
        self._cancel = True

    def start(self) -> None:
        if not self._cells:
            self.finished.emit([])
            return
        for cell in self._cells:
            self._pool.start(_CellRunnable(cell, self))

    def _report_cell(self, result: SweepResult) -> None:
        self._results.append(result)
        self._done += 1
        self.cell_done.emit(result)
        self.progress.emit(self._done, len(self._cells))
        if self._done >= len(self._cells):
            self._results.sort(key=lambda r: r.index)
            self.finished.emit(self._results)


class _CellRunnable(QRunnable):
    def __init__(self, cell: SweepCell, runner: SweepRunner) -> None:
        super().__init__()
        self._cell = cell
        self._runner = runner

    def run(self) -> None:
        try:
            result = run_cell(self._cell, should_stop=lambda: self._runner._cancel)
        except Exception as exc:  # noqa: BLE001 — one bad cell must not hang the batch
            # Without this, an exception in run_cell/drive would skip the emit
            # below, so the runner never counts this cell and `finished` never
            # fires — the whole sweep hangs with no error. Emit a failed result
            # so the pool always drains and the UI can flag the cell.
            result = SweepResult(
                self._cell.index, self._cell.coords, False, 0.0, None, 0.0, 0.0,
                failed=True, error=f"{type(exc).__name__}: {exc}",
            )
        # delivered to the runner's (main) thread via a queued connection
        self._runner._internal.emit(result)
