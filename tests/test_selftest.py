# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""Hardware self-test logic, exercised against the echo NullRig (no hardware)."""
from __future__ import annotations

from fireturret.config import DEFAULT_CONFIG
from fireturret.rig.interface import NullRig, RigTelemetry
from fireturret.selftest import run_selftest

_NOSLEEP = lambda _s: None  # noqa: E731 - trivial test stub


def test_selftest_passes_against_echo_rig():
    report = run_selftest(NullRig(), DEFAULT_CONFIG, dwell=0.0, sleep=_NOSLEEP)
    assert report.passed
    names = [n for n, _, _ in report.checks]
    assert names[0] == "telemetry link"
    assert "pan +5 deg" in names and "tilt to min" in names
    # water is skipped by default
    assert any("SKIPPED" in detail for _, _, detail in report.checks)


def test_selftest_includes_water_when_enabled():
    report = run_selftest(NullRig(), DEFAULT_CONFIG, water=True, dwell=0.0, sleep=_NOSLEEP)
    assert report.passed
    assert any(name.startswith("water pulse") for name, _, _ in report.checks)


def test_selftest_aborts_when_unhomed():
    class UnhomedRig:
        def command(self, cmd):
            pass

        def telemetry(self):
            return RigTelemetry(0.0, 0.0, 0.0, False, estop=False, ok=True, homed=False)

        def close(self):
            pass

    report = run_selftest(UnhomedRig(), DEFAULT_CONFIG, dwell=0.0, sleep=_NOSLEEP)
    assert not report.passed
    assert any("not homed" in detail for _, _, detail in report.checks)


def test_selftest_aborts_without_telemetry():
    class DeadRig:
        def command(self, cmd):
            pass

        def telemetry(self):
            return RigTelemetry(0.0, 0.0, 0.0, False, estop=False, ok=False)

        def close(self):
            pass

    report = run_selftest(DeadRig(), DEFAULT_CONFIG, dwell=0.0, sleep=_NOSLEEP)
    assert not report.passed
    assert report.checks[0][0] == "telemetry link"
    assert report.checks[0][1] is False
