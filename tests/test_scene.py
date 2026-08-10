# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""SP15 — advisory scene understanding.

The model's judgement cannot be tested and nothing here pretends to test it.
What is proven, exhaustively, is the property that makes it safe to have at all:
**no provider output, in any combination, can turn a denial into an allowance.**

Plus the privacy property, which is asserted rather than described: the default
build makes no network call, checked by making sockets raise.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from fireturret.config import DEFAULT_CONFIG
from fireturret.control.gate import GateContext, WaterGate, standard_gate
from fireturret.rig.interface import RigCommand, RigTelemetry
from fireturret.scene import (
    HazardAssessment,
    LocalSceneProvider,
    NullSceneProvider,
    RemoteSceneProvider,
    SceneVeto,
    WrongAgent,
)

FRAME = np.zeros((16, 16, 3), dtype=np.uint8)
WET = RigCommand(pan_deg=10.0, tilt_deg=22.0, pump_pct=55.0, valve=True)
HEALTHY = RigTelemetry(10.0, 22.0, 55.0, True, estop=False, ok=True, homed=True)


def _ctx(**kw) -> GateContext:
    base = dict(command=WET, telemetry=HEALTHY, dt=1 / 30, armed=True, spray_elapsed_s=0.0)
    base.update(kw)
    return GateContext(**base)


# ------------------------------------------------- THE conservative-only property

@pytest.mark.parametrize("answers", list(itertools.product([True, False], repeat=3)))
def test_no_scene_output_can_ever_widen_permission(answers) -> None:
    """Exhaustive over the boolean product. A scene provider may only narrow —
    there is no combination of assessments that makes water flow where it
    otherwise would not."""
    vetoes = []
    for i, allow in enumerate(answers):
        veto = SceneVeto(name=f"scene{i}")
        veto.submit(HazardAssessment(allow_water=allow, summary="stub"))
        vetoes.append(veto)

    out, denials = WaterGate(vetoes).apply(WET, _ctx())
    if all(answers):
        assert out == WET and denials == ()
    else:
        assert out.valve is False and out.pump_pct == 0.0


def test_an_allowing_scene_grants_nothing_on_its_own() -> None:
    """`Verdict.allow()` is the gate's neutral element. A scene provider saying
    "looks fine to me" must not override a disarmed turret."""
    veto = SceneVeto()
    veto.submit(HazardAssessment(allow_water=True, summary="looks fine"))
    gate = standard_gate(DEFAULT_CONFIG)
    gate.add(veto)
    out, denials = gate.apply(WET, _ctx(armed=False))
    assert out.valve is False
    assert any(d.startswith("armed") for d in denials)


def test_the_veto_never_alters_the_aim() -> None:
    veto = SceneVeto()
    veto.submit(HazardAssessment(allow_water=False, wrong_agent=WrongAgent.ELECTRICAL))
    out, _ = WaterGate([veto]).apply(WET, _ctx())
    assert out.pan_deg == WET.pan_deg and out.tilt_deg == WET.tilt_deg


def test_no_assessment_yet_is_permissive_here_and_that_is_deliberate() -> None:
    """Unlike the person detector, this layer is advisory and default-absent.
    Denying until an assessment arrives would disable the machine for everyone
    who never enabled scene understanding — so the DEFAULT is neutral, and the
    authority only ever runs in the deny direction."""
    assert SceneVeto().evaluate(_ctx()).allowed is True


# --------------------------------------------------------- the wrong agent

@pytest.mark.parametrize("agent", [
    WrongAgent.ELECTRICAL, WrongAgent.GREASE, WrongAgent.CHEMICAL, WrongAgent.METAL,
])
def test_a_wrong_agent_assessment_shuts_the_valve_and_reaches_the_operator(agent) -> None:
    """Water on an electrical fire is a conduction path; on a grease fire it
    causes a steam explosion that throws burning oil. Stopping and telling the
    operator is the whole point of the layer."""
    veto = SceneVeto()
    veto.submit(HazardAssessment(allow_water=False, wrong_agent=agent))

    out, denials = WaterGate([veto]).apply(WET, _ctx())
    assert out.valve is False
    assert agent.value in denials[0]
    assert agent.value in veto.operator_advisory


def test_nothing_to_say_produces_no_advisory() -> None:
    veto = SceneVeto()
    veto.submit(HazardAssessment(allow_water=True))
    assert veto.operator_advisory is None


def test_a_hazard_list_reaches_the_operator_surface() -> None:
    veto = SceneVeto()
    veto.submit(HazardAssessment(allow_water=False, hazards=("gas cylinder", "people")))
    assert "gas cylinder" in veto.operator_advisory


# ------------------------------------------------------------------ providers

def test_the_default_provider_has_no_opinion_and_no_network() -> None:
    provider = NullSceneProvider()
    assert provider.transmits_off_device is False
    assert provider.assess(FRAME).allow_water is True


def test_a_local_provider_with_no_model_is_neutral() -> None:
    assert LocalSceneProvider().assess(FRAME).allow_water is True


def test_a_local_model_that_raises_DENIES() -> None:
    """A crashed model is not a clear scene."""
    def explode(frame):
        raise RuntimeError("model died")

    assessment = LocalSceneProvider(model=explode).assess(FRAME)
    assert assessment.allow_water is False
    assert "RuntimeError" in assessment.summary


def test_a_local_model_returning_nonsense_DENIES() -> None:
    assert LocalSceneProvider(model=lambda f: "yes please").assess(FRAME).allow_water is False


def test_a_local_model_can_deny() -> None:
    def grease(frame):
        return HazardAssessment(allow_water=False, wrong_agent=WrongAgent.GREASE)

    assert LocalSceneProvider(model=grease).assess(FRAME).wrong_agent is WrongAgent.GREASE


# -------------------------------------------------------------------- privacy

def test_a_remote_provider_refuses_to_construct_without_acknowledgement() -> None:
    """The one component that can put images of somebody's kitchen on someone
    else's server. Enabling it should be a decision somebody made on purpose."""
    with pytest.raises(ValueError, match="transmit camera frames"):
        RemoteSceneProvider("https://example.invalid/assess")


def test_enabling_a_remote_provider_prints_a_data_flow_warning_naming_the_destination() -> None:
    printed: list[str] = []
    RemoteSceneProvider(
        "https://example.invalid/assess", acknowledge_data_flow=True,
        transport=lambda f: HazardAssessment(), printer=printed.append,
    )
    assert any("example.invalid" in line for line in printed)
    assert any("leave this device" in line for line in printed)


def test_a_remote_call_that_fails_DENIES() -> None:
    def broken(frame):
        raise OSError("no route to host")

    provider = RemoteSceneProvider(
        "https://example.invalid/assess", acknowledge_data_flow=True,
        transport=broken, printer=lambda _s: None,
    )
    assert provider.assess(FRAME).allow_water is False


def test_only_the_remote_provider_declares_that_it_transmits() -> None:
    assert NullSceneProvider().transmits_off_device is False
    assert LocalSceneProvider().transmits_off_device is False
    assert RemoteSceneProvider.transmits_off_device is True


def test_the_default_build_makes_NO_network_call(monkeypatch) -> None:
    """**Asserted, not described.** Sockets are made to raise, and then a full
    pipeline tick is run. Reading the code and believing it is not the same
    check."""
    import socket

    def forbidden(*args, **kwargs):
        raise AssertionError("the default build attempted a network connection")

    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)

    from fireturret.app import Pipeline
    from fireturret.rig.sim_rig import SimRig, SimScenario

    rig = SimRig(DEFAULT_CONFIG, SimScenario(), seed=7)
    pipe = Pipeline(DEFAULT_CONFIG)
    for _ in range(30):
        rig.step(1 / 30)
        rig.command(pipe.tick(rig.render(), 1 / 30, rig.telemetry()).command)

    # and the default scene provider specifically
    assert NullSceneProvider().assess(FRAME).allow_water is True
