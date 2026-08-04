# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""SP9 — asynchronous safety vetoes, proven before any detector exists.

The plumbing is the testable part. A detector's judgement is not: nobody can
write a test that says "this model correctly recognises a person". What CAN be
proven exhaustively is that no output of that model — right, wrong, late,
missing, or crashed — can ever cause water to flow when it otherwise would not.

So this file proves the fail-safe table row by row, and the deny-only property
over the whole boolean product, with a stub in place of the detector.
"""

from __future__ import annotations

import itertools
from dataclasses import replace

import pytest

from triton.config import DEFAULT_CONFIG
from triton.control.gate import GateContext, WaterGate, standard_gate
from triton.rig.interface import RigCommand, RigTelemetry
from triton.safety.veto import (
    STALENESS_MARGIN,
    AlwaysAllowVeto,
    AsyncVeto,
    NeverStaleClock,
    VetoAnswer,
    derive_max_age_s,
)

WET = RigCommand(pan_deg=30.0, tilt_deg=22.0, pump_pct=55.0, valve=True)
HEALTHY = RigTelemetry(30.0, 22.0, 55.0, True, estop=False, ok=True, homed=True)


def _ctx(**kw) -> GateContext:
    base = dict(command=WET, telemetry=HEALTHY, dt=1 / 30, armed=True, spray_elapsed_s=0.0)
    base.update(kw)
    return GateContext(**base)


def _veto(max_age_s: float = 1.0, **kw) -> tuple[AsyncVeto, NeverStaleClock]:
    clock = NeverStaleClock()
    return AsyncVeto(max_age_s=max_age_s, clock=clock, **kw), clock


# --------------------------------------------------------- the fail-safe table

def test_no_answer_yet_denies() -> None:
    """Before the worker has produced anything, "nobody is there" is unknown —
    and unknown is never yes."""
    veto, _ = _veto()
    assert veto.evaluate(_ctx()).allowed is False


def test_a_fresh_allow_allows() -> None:
    veto, clock = _veto()
    veto.submit(VetoAnswer(allowed=True, computed_at=clock.now))
    assert veto.evaluate(_ctx()).allowed is True


def test_a_stale_answer_denies() -> None:
    """A stale answer is indistinguishable from a dead worker."""
    veto, clock = _veto(max_age_s=1.0)
    veto.submit(VetoAnswer(allowed=True, computed_at=clock.now))
    clock.advance(1.01)
    verdict = veto.evaluate(_ctx())
    assert verdict.allowed is False
    assert "stale" in verdict.reason


def test_an_answer_exactly_at_the_threshold_is_still_fresh() -> None:
    """The boundary is defined, not accidental: `>` not `>=`, so a threshold
    derived to just cover the worst case does not reject the worst case."""
    veto, clock = _veto(max_age_s=1.0)
    veto.submit(VetoAnswer(allowed=True, computed_at=clock.now))
    clock.advance(1.0)
    assert veto.evaluate(_ctx()).allowed is True


def test_a_denial_denies_and_latches() -> None:
    """A safety denial does not evaporate when the next frame looks clear. The
    operator acknowledges it; the detector does not get to."""
    veto, clock = _veto()
    veto.submit(VetoAnswer(allowed=False, computed_at=clock.now, reason="person in sector"))
    assert veto.evaluate(_ctx()).allowed is False

    veto.submit(VetoAnswer(allowed=True, computed_at=clock.now))
    assert veto.evaluate(_ctx()).allowed is False, "denial did not latch"
    assert veto.latched is True


def test_clearing_a_latch_requires_a_fresh_allow_too() -> None:
    veto, clock = _veto(max_age_s=1.0)
    veto.submit(VetoAnswer(allowed=False, computed_at=clock.now, reason="person"))
    veto.clear()
    # the latch is gone, but the last answer is still a denial
    assert veto.evaluate(_ctx()).allowed is False
    veto.submit(VetoAnswer(allowed=True, computed_at=clock.now))
    assert veto.evaluate(_ctx()).allowed is True


def test_clearing_does_not_make_a_stale_answer_fresh() -> None:
    veto, clock = _veto(max_age_s=1.0)
    veto.submit(VetoAnswer(allowed=True, computed_at=clock.now))
    clock.advance(5.0)
    veto.clear()
    assert veto.evaluate(_ctx()).allowed is False


def test_a_raised_predicate_denies_and_latches() -> None:
    """A crashed detector must not read as "nobody is there"."""
    veto, _ = _veto()
    veto.submit_failure(RuntimeError("model blew up"))
    verdict = veto.evaluate(_ctx())
    assert verdict.allowed is False
    assert "RuntimeError" in verdict.reason
    assert veto.latched is True


def test_a_non_latching_veto_can_be_configured_but_is_not_the_default() -> None:
    """Non-latching exists for advisory-grade predicates. The DEFAULT latches,
    because that is the safe direction to be wrong in."""
    assert AsyncVeto().latching is True
    veto, clock = _veto(latching=False)
    veto.submit(VetoAnswer(allowed=False, computed_at=clock.now, reason="blip"))
    assert veto.evaluate(_ctx()).allowed is False
    veto.submit(VetoAnswer(allowed=True, computed_at=clock.now))
    assert veto.evaluate(_ctx()).allowed is True


def test_a_zero_or_negative_max_age_is_rejected() -> None:
    with pytest.raises(ValueError):
        AsyncVeto(max_age_s=0.0)


# ----------------------------------------------------------- deny-only, again

@pytest.mark.parametrize("answers", list(itertools.product([True, False], repeat=3)))
def test_no_combination_of_vetoes_can_widen_permission(answers) -> None:
    """Exhaustive over the boolean product, now with a real AsyncVeto in the
    chain alongside stubs. Water survives if and only if every veto allowed."""
    clock = NeverStaleClock()
    vetoes = []
    for i, allowed in enumerate(answers):
        veto = AsyncVeto(name=f"v{i}", max_age_s=10.0, clock=clock)
        veto.submit(VetoAnswer(allowed=allowed, computed_at=clock.now, reason="stub"))
        vetoes.append(veto)

    out, denials = WaterGate(vetoes).apply(WET, _ctx())
    if all(answers):
        assert out == WET and denials == ()
    else:
        assert out.valve is False and out.pump_pct == 0.0


def test_the_veto_never_alters_the_aim() -> None:
    """There is deliberately no `forced_pan_deg`. A layer that commands motion
    acquires its own failure surface, and it would have to project a detector
    bbox through the same monocular model the architecture distrusts."""
    veto, clock = _veto()
    veto.submit(VetoAnswer(allowed=False, computed_at=clock.now, reason="person"))
    out, _ = WaterGate([veto]).apply(WET, _ctx())
    assert out.pan_deg == WET.pan_deg
    assert out.tilt_deg == WET.tilt_deg


def test_always_allow_proves_the_wiring_without_an_opinion() -> None:
    out, denials = WaterGate([AlwaysAllowVeto()]).apply(WET, _ctx())
    assert out == WET and denials == ()


def test_a_veto_registers_into_the_standard_gate() -> None:
    """The point of SP3's consolidation: safety plugs into the EXISTING chain
    rather than becoming a fourth gate somewhere else."""
    gate = standard_gate(DEFAULT_CONFIG)
    before = len(gate.vetoes)
    veto, clock = _veto()
    veto.submit(VetoAnswer(allowed=False, computed_at=clock.now, reason="person in sector"))
    gate.add(veto)
    assert len(gate.vetoes) == before + 1

    out, denials = gate.apply(WET, _ctx())
    assert out.valve is False
    assert any("person in sector" in d for d in denials)


# ------------------------------------------------------- the derived threshold

def test_max_age_is_derived_from_measured_latency_not_the_heartbeat() -> None:
    """Borrowing the firmware's 500 ms heartbeat produces a layer that marks
    fresh results stale the instant they arrive, so the valve chatters at the
    detector's beat frequency — and an unusable safety layer gets switched off."""
    p99_inference_s = 0.300  # measured on a Pi 5 CPU
    submit_period_s = 0.200  # 5 Hz
    max_age = derive_max_age_s(p99_inference_s, submit_period_s)

    worst_age_on_arrival = submit_period_s + p99_inference_s
    assert worst_age_on_arrival == pytest.approx(0.5)
    assert max_age > worst_age_on_arrival, "a fresh result would be marked stale"
    assert max_age == pytest.approx(worst_age_on_arrival * STALENESS_MARGIN)
    assert max_age > 0.5, "the firmware heartbeat is the wrong threshold here"


def test_a_result_arriving_at_worst_case_age_is_accepted() -> None:
    """The derivation, exercised rather than asserted: a result that took the
    full p99 and arrived one whole period late must still be usable."""
    max_age = derive_max_age_s(0.300, 0.200)
    clock = NeverStaleClock()
    veto = AsyncVeto(max_age_s=max_age, clock=clock)

    computed_at = clock.now
    clock.advance(0.500)  # worst-case age on arrival
    veto.submit(VetoAnswer(allowed=True, computed_at=computed_at))
    assert veto.evaluate(_ctx()).allowed is True


@pytest.mark.parametrize("bad", [(-0.1, 0.2), (0.1, 0.0), (0.1, -1.0)])
def test_degenerate_latency_inputs_are_rejected(bad) -> None:
    with pytest.raises(ValueError):
        derive_max_age_s(*bad)


# ------------------------------------------------------------- fault injection

def test_the_simulator_can_now_reach_safe() -> None:
    """Before SP9 `SimRig.telemetry()` hardcoded estop=False, ok=True — so no
    golden and no end-to-end run had EVER visited SAFE. The latching failsafe was
    exercised only by unit tests driving the controller directly."""
    from triton.app import Pipeline
    from triton.rig.sim_rig import SimRig, SimScenario

    scenario = SimScenario(estop_window_s=(1.0, 2.0))
    rig = SimRig(DEFAULT_CONFIG, scenario, seed=7)
    pipe = Pipeline(DEFAULT_CONFIG)
    states = set()
    for _ in range(120):  # 4 s
        rig.step(1 / 30)
        result = pipe.tick(rig.render(), 1 / 30, rig.telemetry())
        rig.command(result.command)
        states.add(pipe.mission.state)
    assert "SAFE" in states


def test_fault_injection_is_inert_by_default() -> None:
    """Every existing fixture depends on this: no window configured means no
    fault, ever."""
    from triton.rig.sim_rig import SimRig, SimScenario

    rig = SimRig(DEFAULT_CONFIG, SimScenario(), seed=7)
    for _ in range(60):
        rig.step(1 / 30)
        tel = rig.telemetry()
        assert tel.estop is False and tel.ok is True and tel.homed is True


@pytest.mark.parametrize("field,check", [
    ("estop_window_s", lambda t: t.estop is True),
    ("telemetry_dropout_s", lambda t: t.ok is False),
    ("unhomed_window_s", lambda t: t.homed is False),
])
def test_each_injected_fault_appears_only_inside_its_window(field, check) -> None:
    from triton.rig.sim_rig import SimRig, SimScenario

    scenario = replace(SimScenario(), **{field: (1.0, 2.0)})
    rig = SimRig(DEFAULT_CONFIG, scenario, seed=7)
    seen_inside = seen_clear_after = False
    for _ in range(120):
        rig.step(1 / 30)
        tel = rig.telemetry()
        if 1.0 <= rig.time < 2.0:
            seen_inside = seen_inside or check(tel)
        elif rig.time >= 2.0:
            seen_clear_after = seen_clear_after or not check(tel)
    assert seen_inside, f"{field} never took effect inside its window"
    assert seen_clear_after, f"{field} did not clear after its window"
