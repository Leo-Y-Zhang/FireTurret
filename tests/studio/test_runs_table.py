# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""P1b: the Runs table model (contract-checked) + add/remove/rename."""
from __future__ import annotations

from PySide6.QtCore import Qt

from triton.config import DEFAULT_CONFIG
from triton.simcore import drive
from triton.studio.models.runs_table import RunRow, RunsTableModel


def _row(name="run", seed=7, extinguished=True):
    return RunRow(name, extinguished, 6.3, 0.7, 0.1, 0.2, seed, "default", "12:00")


def test_model_contract(qtmodeltester, qapp):
    m = RunsTableModel()
    m.add_run(_row("a"))
    m.add_run(_row("b", seed=3, extinguished=False))
    qtmodeltester.check(m)


def test_columns_and_values(qapp):
    m = RunsTableModel()
    m.add_run(_row("run1", seed=7))
    assert m.rowCount() == 1
    assert m.columnCount() == 9
    assert m.data(m.index(0, 0)) == "run1"
    assert m.data(m.index(0, 1)) == "extinguished"
    assert m.data(m.index(0, 6)) == "7"  # seed column present
    assert m.headerData(6, Qt.Orientation.Horizontal) == "Seed"


def test_remove_and_rename(qapp):
    m = RunsTableModel()
    m.add_run(_row("a"))
    m.add_run(_row("b", seed=3))
    m.rename(0, "renamed")
    assert m.data(m.index(0, 0)) == "renamed"
    m.remove(0)
    assert m.rowCount() == 1
    assert m.data(m.index(0, 0)) == "b"


def test_mark_all_stale(qapp):
    m = RunsTableModel()
    m.add_run(_row("a"))
    m.add_run(_row("b"))
    assert m.data(m.index(0, 1)) == "extinguished"
    m.mark_all_stale()
    assert m.data(m.index(0, 1)).startswith("stale")
    assert m.rows[1].stale is True


def test_multi_fire_status(qapp):
    from triton.__main__ import SCENARIOS
    from triton.simcore import drive

    rep = drive(DEFAULT_CONFIG, SCENARIOS["multi"], seed=7, max_frames=3000)
    row = RunRow.from_report("m", rep, 7, "multi", "t")
    assert row.n_fires == 2
    m = RunsTableModel()
    m.add_run(row)
    assert "/2" in m.data(m.index(0, 1))  # e.g. "2/2 out"


def test_from_report(qapp):
    rep = drive(DEFAULT_CONFIG, seed=7, max_frames=120, record_telemetry=True)
    row = RunRow.from_report("r", rep, 7, "default", "t")
    assert row.name == "r"
    assert row.seed == 7
    assert row.extinguished == rep.extinguished
    assert row.acquire_s == rep.time_to_first_suppress_s
