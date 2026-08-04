# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""SP3 — the one water gate.

The load-bearing property is negative: **no veto, and no combination of vetoes,
can turn a denial into an allowance.** That is proven exhaustively below over the
boolean product rather than argued, because "deny-only" is the kind of claim that
is easy to state, easy to believe, and easy to break with one well-meaning
`return cmd` in the wrong branch.

The second property is that failure is denial. A veto that raises must deny, not
be skipped — an exception is exactly the situation where you least want the
safety layer to abstain.
"""

from __future__ import annotations

import itertools
from dataclasses import replace

import pytest

from triton.config import DEFAULT_CONFIG
from triton.control.gate import (
    SPRAY_BACKSTOP_FACTOR,
    ArmedVeto,
    GateContext,
    HomedVeto,
    LinkVeto,
    SprayTimeVeto,
    Verdict,
    WaterGate,
    standard_gate,
)
from triton.rig.interface import RigCommand, RigTelemetry

WET = RigCommand(pan_deg=30.0, tilt_deg=22.0, pump_pct=55.0, valve=True)
HEALTHY = RigTelemetry(30.0, 22.0, 55.0, True, estop=False, ok=True, homed=True)


def _ctx(**kw) -> GateContext:
    base = dict(command=WET, telemetry=HEALTHY, dt=1 / 30, armed=True, spray_elapsed_s=0.0)
    base.update(kw)
    return GateContext(**base)


class Stub:
    """A veto with a fixed answer, or one that raises."""

    def __init__(self, name: str, allowed: bool, boom: bool = False) -> None:
        self.name = name
        self._allowed = allowed
        self._boom = boom

    def evaluate(self, ctx: GateContext) -> Verdict:
        if self._boom:
            raise RuntimeError("veto exploded")
        return Verdict.allow() if self._allowed else Verdict.deny("stub denies")


# ------------------------------------------------------- the negative property

@pytest.mark.parametrize("answers", list(itertools.product([True, False], repeat=4)))
def test_water_survives_only_if_every_veto_allows(answers) -> None:
    """Exhaustive over the boolean product: water is wet if and only if ALL
    vetoes allowed. In particular no arrangement of allowing vetoes can rescue a
    command that any one veto denied."""
    gate = WaterGate([Stub(f"v{i}", a) for i, a in enumerate(answers)])
    out, denials = gate.apply(WET, _ctx())

    if all(answers):
        assert out == WET
        assert denials == ()
    else:
        assert out.valve is False
        assert out.pump_pct == 0.0
        assert len(denials) == answers.count(False)


def test_a_denial_cannot_be_reversed_by_adding_allowing_vetoes() -> None:
    """The same claim from the other direction, because this is the failure that
    would matter: piling on permissive vetoes must never restore water."""
    denied = WaterGate([Stub("no", False)]).apply(WET, _ctx())[0]
    for extra in range(1, 6):
        gate = WaterGate([Stub("no", False)] + [Stub(f"yes{i}", True) for i in range(extra)])
        assert gate.apply(WET, _ctx())[0] == denied


def test_an_erroring_veto_denies() -> None:
    """Failure is denial. An exception is the situation where abstaining is worst."""
    out, denials = WaterGate([Stub("boom", True, boom=True)]).apply(WET, _ctx())
    assert out.valve is False
    assert out.pump_pct == 0.0
    assert "boom" in denials[0] and "errored" in denials[0]


def test_one_erroring_veto_does_not_stop_the_others_being_evaluated() -> None:
    gate = WaterGate([Stub("boom", True, boom=True), Stub("also_no", False)])
    _out, denials = gate.apply(WET, _ctx())
    assert len(denials) == 2


def test_the_gate_never_re_aims() -> None:
    """Deny-only means water-only. A safety layer that commands motion acquires
    its own failure surface, and re-aiming belongs to the mission."""
    out, _ = WaterGate([Stub("no", False)]).apply(WET, _ctx())
    assert out.pan_deg == WET.pan_deg
    assert out.tilt_deg == WET.tilt_deg
    assert out.laser == WET.laser


def test_an_empty_gate_allows() -> None:
    assert WaterGate([]).apply(WET, _ctx()) == (WET, ())


def test_a_dry_command_is_unchanged_by_a_denial() -> None:
    """Denying water on a command that carries none is a no-op, not an error."""
    dry = RigCommand(pan_deg=30.0, tilt_deg=22.0, pump_pct=0.0, valve=False)
    assert WaterGate([Stub("no", False)]).apply(dry, _ctx())[0] == dry


# ------------------------------------------------------------ built-in vetoes

def test_armed_veto() -> None:
    assert ArmedVeto().evaluate(_ctx(armed=True)).allowed is True
    assert ArmedVeto().evaluate(_ctx(armed=False)).allowed is False


def test_homed_veto() -> None:
    assert HomedVeto().evaluate(_ctx()).allowed is True
    unhomed = replace(HEALTHY, homed=False)
    assert HomedVeto().evaluate(_ctx(telemetry=unhomed)).allowed is False


def test_link_veto_covers_estop_and_staleness() -> None:
    assert LinkVeto().evaluate(_ctx()).allowed is True
    assert LinkVeto().evaluate(_ctx(telemetry=replace(HEALTHY, estop=True))).allowed is False
    assert LinkVeto().evaluate(_ctx(telemetry=replace(HEALTHY, ok=False))).allowed is False


def test_spray_backstop_trips_above_the_missions_own_limit_not_at_it() -> None:
    """If the gate cut at exactly `max_spray_s` it would race the mission's own
    re-evaluation and change behaviour. It is a backstop for a controller that
    ignores the rule, so it must fire strictly later."""
    limit = DEFAULT_CONFIG.mission.max_spray_s
    veto = SprayTimeVeto(limit_s=limit * SPRAY_BACKSTOP_FACTOR)
    assert SPRAY_BACKSTOP_FACTOR > 1.0
    assert veto.evaluate(_ctx(spray_elapsed_s=limit)).allowed is True
    assert veto.evaluate(_ctx(spray_elapsed_s=limit * SPRAY_BACKSTOP_FACTOR + 0.1)).allowed is False


# ------------------------------------------------------------- the real chain

def test_standard_gate_allows_the_simulators_conditions() -> None:
    """The property that keeps every existing fixture still: under the sim's
    conditions (armed, homed, no fault) the gate is invisible."""
    gate = standard_gate(DEFAULT_CONFIG)
    out, denials = gate.apply(WET, _ctx())
    assert out == WET
    assert denials == ()


@pytest.mark.parametrize("break_it,expected", [
    ({"armed": False}, "armed"),
    ({"telemetry": replace(HEALTHY, homed=False)}, "homed"),
    ({"telemetry": replace(HEALTHY, estop=True)}, "link"),
    ({"telemetry": replace(HEALTHY, ok=False)}, "link"),
    ({"spray_elapsed_s": 10_000.0}, "spray_time"),
])
def test_standard_gate_denies_each_condition(break_it, expected) -> None:
    out, denials = standard_gate(DEFAULT_CONFIG).apply(WET, _ctx(**break_it))
    assert out.valve is False
    assert any(d.startswith(expected) for d in denials), denials
