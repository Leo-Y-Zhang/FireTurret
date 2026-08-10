# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""SP9 — the gate at the last possible point.

`MissionController` gates every command it issues. That covers the mission and
nothing else: FireTurret Studio drives the rig directly, a manual jog would, and any
future profile or plugin might. Each is a path that never constructs a `Pipeline`
and therefore never meets the gate.

`GuardedRig` moves the gate to the one place all of them must pass through.
"""

from __future__ import annotations

import inspect

import pytest

from fireturret.config import DEFAULT_CONFIG
from fireturret.control.gate import standard_gate
from fireturret.rig.guarded import GuardedRig
from fireturret.rig.interface import NullRig, RigCommand


class RecordingRig(NullRig):
    """A rig that remembers exactly what reached it."""

    def __init__(self) -> None:
        super().__init__()
        self.received: list[RigCommand] = []

    def command(self, cmd: RigCommand) -> None:
        self.received.append(cmd)
        super().command(cmd)


WET = RigCommand(pan_deg=30.0, tilt_deg=22.0, pump_pct=55.0, valve=True)


def _guarded(armed: bool = False) -> tuple[GuardedRig, RecordingRig]:
    inner = RecordingRig()
    return GuardedRig(inner, standard_gate(DEFAULT_CONFIG), armed=armed), inner


def test_a_disarmed_guarded_rig_strips_water() -> None:
    """The case that matters: a caller that never met the mission's gate still
    cannot spray."""
    rig, inner = _guarded(armed=False)
    rig.command(WET)
    assert inner.received[-1].valve is False
    assert inner.received[-1].pump_pct == 0.0


def test_an_armed_guarded_rig_passes_water_through() -> None:
    rig, inner = _guarded(armed=True)
    rig.command(WET)
    assert inner.received[-1] == WET


def test_the_guard_never_re_aims() -> None:
    rig, inner = _guarded(armed=False)
    rig.command(WET)
    assert inner.received[-1].pan_deg == WET.pan_deg
    assert inner.received[-1].tilt_deg == WET.tilt_deg


def test_denials_are_reported_for_the_operator_surface() -> None:
    rig, _ = _guarded(armed=False)
    rig.command(WET)
    assert any(d.startswith("armed") for d in rig.last_denials)


def test_arming_and_disarming_take_effect_immediately() -> None:
    rig, inner = _guarded(armed=False)
    rig.arm()
    rig.command(WET)
    assert inner.received[-1].valve is True
    rig.disarm()
    rig.command(WET)
    assert inner.received[-1].valve is False


def test_the_guard_delegates_telemetry_and_close() -> None:
    rig, inner = _guarded(armed=True)
    assert rig.telemetry() == inner.telemetry()
    rig.close()  # must not raise
    assert rig.inner is inner


def test_a_veto_added_later_applies_to_the_guarded_rig() -> None:
    """The gate is shared, so registering a safety veto protects every path at
    once rather than only the ones someone remembered to update."""
    from fireturret.safety.veto import AsyncVeto, NeverStaleClock, VetoAnswer

    gate = standard_gate(DEFAULT_CONFIG)
    inner = RecordingRig()
    rig = GuardedRig(inner, gate, armed=True)
    rig.command(WET)
    assert inner.received[-1].valve is True

    clock = NeverStaleClock()
    veto = AsyncVeto(max_age_s=10.0, clock=clock)
    veto.submit(VetoAnswer(allowed=False, computed_at=clock.now, reason="person"))
    gate.add(veto)

    rig.command(WET)
    assert inner.received[-1].valve is False


def test_spray_time_backstop_reaches_the_guarded_rig() -> None:
    rig, inner = _guarded(armed=True)
    rig.note_spray_time(10_000.0)
    rig.command(WET)
    assert inner.received[-1].valve is False


# ------------------------------------------------------- one way to get a rig

def test_every_cli_hardware_path_returns_a_guarded_rig() -> None:
    """SP1 made `_open_serial_rig` the single factory. SP9 makes that factory the
    thing that applies the gate, so "the CLI cannot bypass it" is structural
    rather than a habit."""
    import fireturret.__main__ as cli

    source = inspect.getsource(cli._open_serial_rig)
    assert "GuardedRig" in source, "the rig factory no longer guards its rig"

    for name in ("_cmd_run", "_cmd_web", "_cmd_selftest"):
        body = inspect.getsource(getattr(cli, name))
        assert "SerialRig(" not in body, f"{name} constructs an unguarded rig"


def test_the_factory_hands_back_a_disarmed_rig() -> None:
    """Dry-aim by default has to survive the wrapper: a guarded rig that arrived
    armed would quietly undo the CLI's safe default."""
    import fireturret.__main__ as cli

    assert inspect.signature(GuardedRig.__init__).parameters["armed"].default is False
    assert "arm()" in inspect.getsource(cli._cmd_run) or "rig.arm" in inspect.getsource(cli._cmd_run)


@pytest.mark.parametrize("water_armed", [False, True])
def test_selftest_water_still_works_through_the_guard(water_armed: bool) -> None:
    """The guard must not break the legitimate armed path — an over-eager safety
    layer that blocks the self-test's pump pulse would just get removed."""
    from fireturret.selftest import run_selftest

    inner = RecordingRig()
    rig = GuardedRig(inner, standard_gate(DEFAULT_CONFIG), armed=water_armed)
    report = run_selftest(rig, DEFAULT_CONFIG, water=water_armed,
                          dwell=0.0, sleep=lambda _s: None)
    assert report.passed is True
    if water_armed:
        assert any(c.valve for c in inner.received), "armed water pulse was blocked"
    else:
        assert not any(c.valve for c in inner.received)
