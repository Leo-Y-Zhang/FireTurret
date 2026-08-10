# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
import csv
import os
import tempfile

import pytest

from fireturret import ballistics
from fireturret.analysis import TelemetrySample, read_csv, write_csv, write_report
from fireturret.app import run_sim
from fireturret.config import DEFAULT_CONFIG
from fireturret.rig.sim_rig import SimScenario


def _samples(n: int = 5) -> list[TelemetrySample]:
    return [
        TelemetrySample(t=i * 0.1, state="SUPPRESS", pan_cmd=1.0, pan_act=1.0, tilt=15.0,
                        pump=40.0 + i, valve=True, range_est_m=7.0, az_err_deg=0.1 * i,
                        range_err_px=2.0, fire_intensity=1.0 - 0.1 * i, water_l=0.05 * i,
                        splash_miss_m=1.5 - 0.2 * i, warning="")
        for i in range(n)
    ]


def test_run_sim_records_telemetry() -> None:
    report = run_sim(DEFAULT_CONFIG, SimScenario(), seed=7, headless=True,
                     max_frames=600, record_telemetry=True)
    assert len(report.telemetry) == report.frames
    s = report.telemetry[0]
    assert isinstance(s, TelemetrySample)
    assert s.t >= 0.0
    # default: no telemetry recorded unless asked
    assert run_sim(DEFAULT_CONFIG, SimScenario(), seed=7, headless=True, max_frames=100).telemetry == []


def test_write_csv_roundtrip() -> None:
    path = os.path.join(tempfile.mkdtemp(), "run.csv")
    write_csv(_samples(4), path)
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    os.remove(path)
    assert len(rows) == 4
    assert {"t", "state", "pump", "fire_intensity", "splash_miss_m"} <= set(rows[0])
    assert rows[0]["state"] == "SUPPRESS"


def test_read_csv_roundtrips_samples() -> None:
    path = os.path.join(tempfile.mkdtemp(), "run.csv")
    original = _samples(6)
    write_csv(original, path)
    restored = read_csv(path)
    os.remove(path)
    assert restored == original  # frozen dataclass equality, types coerced back


def test_write_report_creates_png() -> None:
    pytest.importorskip("matplotlib")
    path = os.path.join(tempfile.mkdtemp(), "report.png")
    lo, hi = ballistics.reach_bounds(DEFAULT_CONFIG.jet, DEFAULT_CONFIG.turret.tilt_min_deg,
                                     DEFAULT_CONFIG.turret.tilt_max_deg)
    write_report(_samples(8), path, lo, hi)
    assert os.path.getsize(path) > 1000  # a real image was written
    os.remove(path)


def test_write_report_empty_raises() -> None:
    pytest.importorskip("matplotlib")
    with pytest.raises(ValueError):
        write_report([], os.path.join(tempfile.mkdtemp(), "x.png"), 4.0, 11.0)


def test_reach_bounds_are_ordered_and_sane() -> None:
    lo, hi = ballistics.reach_bounds(DEFAULT_CONFIG.jet, DEFAULT_CONFIG.turret.tilt_min_deg,
                                     DEFAULT_CONFIG.turret.tilt_max_deg)
    assert 2.0 < lo < hi < 20.0
