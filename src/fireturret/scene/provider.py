# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""Scene providers and the deny-only veto that carries their output.

## Privacy

`RemoteSceneProvider` transmits **camera frames off-device**. It is the only
subsystem in this repository capable of making a network call, it is off by
default, it requires an explicit config flag to enable, and enabling it prints a
data-flow warning naming the destination. `tests/test_scene.py` asserts that the
default build makes no network call at all, by monkeypatching `socket.socket` to
raise — which is a stronger check than reading the code and believing it.

A fire-suppression turret watching a domestic or industrial space is a camera in
a place people did not choose to be filmed. Sending those frames to a third party
should be a decision somebody made on purpose.

## The vocabulary

`WrongAgent` is the assessment that matters. Water on an electrical fire is a
conduction path; water on a grease fire causes a steam explosion that throws
burning oil; water on some chemical fires is worse still. The correct response is
to stop and tell the operator, which is exactly what a deny-only veto does.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable

from ..control.gate import GateContext, Verdict


class WrongAgent(Enum):
    """Why water is the wrong thing to apply here."""

    NONE = "none"
    ELECTRICAL = "electrical"   # water conducts
    GREASE = "grease"           # steam explosion throws burning oil
    CHEMICAL = "chemical"       # reacts, or spreads the agent
    METAL = "metal"             # burning metal reacts violently with water


@dataclass(frozen=True)
class HazardAssessment:
    """A provider's structured, ADVISORY output.

    `allow_water` is the only field with any authority, and it can only ever be
    used to withhold. `summary` is for the operator surface.
    """

    allow_water: bool = True
    wrong_agent: WrongAgent = WrongAgent.NONE
    hazards: tuple[str, ...] = ()
    summary: str = ""
    confidence: float = 0.0

    @property
    def reason(self) -> str:
        if self.wrong_agent is not WrongAgent.NONE:
            return f"water is the wrong agent ({self.wrong_agent.value})"
        if self.hazards:
            return "scene hazard: " + ", ".join(self.hazards)
        return self.summary or "scene assessment withheld permission"


@runtime_checkable
class SceneProvider(Protocol):
    #: True only for providers that send data off-device.
    transmits_off_device: bool

    def assess(self, frame_bgr) -> HazardAssessment: ...


class NullSceneProvider:
    """The default. No model, no opinion, no network."""

    transmits_off_device = False

    def assess(self, frame_bgr) -> HazardAssessment:
        return HazardAssessment(allow_water=True, summary="no scene provider configured")


class LocalSceneProvider:
    """A local model. No network, but also no model shipped here.

    Deliberately a seam rather than an implementation: shipping a specific VLM
    would pin a large dependency and imply a level of validation this project has
    not done. `model` is any callable taking a frame and returning a
    `HazardAssessment`.
    """

    transmits_off_device = False

    def __init__(self, model=None) -> None:
        self._model = model

    def assess(self, frame_bgr) -> HazardAssessment:
        if self._model is None:
            return HazardAssessment(allow_water=True, summary="no local model loaded")
        try:
            result = self._model(frame_bgr)
        except Exception as exc:  # noqa: BLE001 - a failed model must DENY
            return HazardAssessment(
                allow_water=False,
                summary=f"local scene model failed ({exc.__class__.__name__})",
            )
        return result if isinstance(result, HazardAssessment) else HazardAssessment(
            allow_water=False, summary="local scene model returned an unusable result"
        )


class RemoteSceneProvider:
    """Sends camera frames to a remote service. **Off by default.**

    Construction requires `acknowledge_data_flow=True` and prints a warning
    naming the destination. That is not ceremony: this is the one component that
    can put images of somebody's kitchen on someone else's server.
    """

    transmits_off_device = True

    def __init__(
        self,
        endpoint: str,
        acknowledge_data_flow: bool = False,
        transport=None,
        printer=print,
    ) -> None:
        if not acknowledge_data_flow:
            raise ValueError(
                f"RemoteSceneProvider would transmit camera frames to {endpoint!r}. "
                "This is the only subsystem in FireTurret that sends data off-device, and "
                "a fire turret's camera watches places people did not choose to be "
                "filmed. Pass acknowledge_data_flow=True to accept that."
            )
        self.endpoint = endpoint
        self._transport = transport
        printer(
            f"WARNING: scene understanding is ENABLED and will transmit camera "
            f"frames to {endpoint}. Images of the monitored area will leave this "
            f"device."
        )

    def assess(self, frame_bgr) -> HazardAssessment:
        if self._transport is None:
            return HazardAssessment(
                allow_water=False, summary="remote scene provider has no transport"
            )
        try:
            result = self._transport(frame_bgr)
        except Exception as exc:  # noqa: BLE001 - a failed call must DENY
            return HazardAssessment(
                allow_water=False,
                summary=f"remote scene call failed ({exc.__class__.__name__})",
            )
        return result if isinstance(result, HazardAssessment) else HazardAssessment(
            allow_water=False, summary="remote scene provider returned an unusable result"
        )


@dataclass
class SceneVeto:
    """Carries a provider's assessment into the water gate. **Deny-only.**

    Structurally incapable of granting: `evaluate` returns either `Verdict.allow()`
    — which is the gate's neutral element and grants nothing on its own — or a
    denial. There is no path by which a provider's output makes water flow that
    would not otherwise have flowed.
    """

    provider: SceneProvider = field(default_factory=NullSceneProvider)
    name: str = "scene"
    latest: HazardAssessment | None = None

    def submit(self, assessment: HazardAssessment) -> None:
        self.latest = assessment

    def evaluate(self, ctx: GateContext) -> Verdict:
        if self.latest is None:
            # No assessment yet is not a denial here: unlike the person detector,
            # this layer is advisory and default-absent, and denying by default
            # would disable the machine for everyone who never enabled it.
            return Verdict.allow()
        if self.latest.allow_water:
            return Verdict.allow()
        return Verdict.deny(self.latest.reason)

    @property
    def operator_advisory(self) -> str | None:
        """What to show the operator. `None` when there is nothing to say."""
        if self.latest is None or self.latest.allow_water:
            return None
        return self.latest.reason
