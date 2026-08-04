# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""The ONE water gate. Every command that can carry water passes through here.

Before this module there were three separate gates — `MissionController.update`
(armed + homed + keep-out), `app.run_capture` (`water_enabled`), and
`webserver.rig_command_for_hardware` (E-stop + `water_enabled`) — each written
independently, each enforcing a slightly different subset. Four more workstreams
were about to add a fourth. Consolidating them first is cheaper than reconciling
five.

## The contract

A veto may only ever **subtract** permission. There is no mechanism by which any
veto, or any combination of vetoes, can turn a denial into an allowance: the gate
starts from the command it was handed and only ever removes water from it. That
is not a convention to be respected — `WaterGate.apply` literally cannot express
the opposite, and `tests/test_gate.py` proves it exhaustively over the boolean
product of every veto.

**Deny-only means water-only.** The gate never touches `pan_deg` or `tilt_deg`.
A safety layer that commands actuator motion acquires its own failure surface,
and re-aiming belongs to the mission — whose next command is gated anyway.

**Failure is denial.** A veto that raises denies. A veto whose answer is stale
denies (see `safety/veto.py` in SP9, which builds on this). The gate never
interprets "I don't know" as "yes".

## Why the spray-time backstop is a backstop

`MissionConfig.max_spray_s` is already enforced by the mission state machine,
which re-evaluates its firing solution rather than hosing forever. The gate's
own limit exists for the case the mission *cannot* cover: a third-party
controller, a plugin, or a future profile that ignores the rule. It therefore
trips at a deliberate margin ABOVE the mission's own limit, so in normal
operation the mission always acts first and the gate is invisible — which is
what keeps this module behaviour-neutral for every existing fixture.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from typing import Protocol

from ..rig.interface import RigCommand, RigTelemetry

# The gate's independent spray limit, as a multiple of `MissionConfig.max_spray_s`.
# Strictly greater than 1.0 so the mission's own re-evaluation always fires first
# in normal operation; the gate is the backstop for a controller that ignores it.
SPRAY_BACKSTOP_FACTOR = 1.5


@dataclass(frozen=True)
class GateContext:
    """Everything a veto is allowed to see. Deliberately small: a veto that needs
    more than this is probably trying to make a control decision, not a safety
    one."""

    command: RigCommand
    telemetry: RigTelemetry
    dt: float
    armed: bool
    spray_elapsed_s: float = 0.0


@dataclass(frozen=True)
class Verdict:
    allowed: bool
    reason: str = ""

    @staticmethod
    def allow() -> Verdict:
        return Verdict(True)

    @staticmethod
    def deny(reason: str) -> Verdict:
        return Verdict(False, reason)


class WaterVeto(Protocol):
    """A predicate that may withhold water. It may not grant it."""

    name: str

    def evaluate(self, ctx: GateContext) -> Verdict: ...


@dataclass
class ArmedVeto:
    """The control-layer arming gate. Disarmed means dry-aim: the turret may
    track, but nothing below this may open a valve."""

    name: str = "armed"

    def evaluate(self, ctx: GateContext) -> Verdict:
        if ctx.armed:
            return Verdict.allow()
        return Verdict.deny("water not armed (dry-aim)")


@dataclass
class HomedVeto:
    """Pan is dead-reckoned, so without a homing reference the commanded angle
    means nothing — and an angle that means nothing must not carry water. A
    belt-and-suspenders check on top of the firmware's own homing lock."""

    name: str = "homed"

    def evaluate(self, ctx: GateContext) -> Verdict:
        if ctx.telemetry.homed:
            return Verdict.allow()
        return Verdict.deny("pan axis not homed")


@dataclass
class LinkVeto:
    """An engaged E-stop or stale telemetry. The mission latches SAFE on these
    too; this exists so a path that never constructs a MissionController still
    cannot spray through a severed link."""

    name: str = "link"

    def evaluate(self, ctx: GateContext) -> Verdict:
        if ctx.telemetry.estop:
            return Verdict.deny("E-stop engaged")
        if not ctx.telemetry.ok:
            return Verdict.deny("telemetry stale")
        return Verdict.allow()


@dataclass
class SprayTimeVeto:
    """Independent endless-hosing backstop — see the module docstring for why it
    trips above the mission's own limit rather than at it."""

    limit_s: float
    name: str = "spray_time"

    def evaluate(self, ctx: GateContext) -> Verdict:
        if ctx.spray_elapsed_s > self.limit_s:
            return Verdict.deny(f"continuous spray exceeded {self.limit_s:.0f}s")
        return Verdict.allow()


class WaterGate:
    """An AND-chain of vetoes. Water survives only if every veto allows it."""

    def __init__(self, vetoes: Iterable[WaterVeto] = ()) -> None:
        self._vetoes: list[WaterVeto] = list(vetoes)

    def add(self, veto: WaterVeto) -> None:
        self._vetoes.append(veto)

    @property
    def vetoes(self) -> Sequence[WaterVeto]:
        return tuple(self._vetoes)

    def evaluate(self, ctx: GateContext) -> tuple[str, ...]:
        """Reasons water is denied, in veto order. Empty means allowed.

        Every veto is evaluated even once one has denied: the operator surface
        wants all the reasons, not just the first, and a veto is a pure predicate
        so there is nothing to short-circuit for.
        """
        denials: list[str] = []
        for veto in self._vetoes:
            try:
                verdict = veto.evaluate(ctx)
            except Exception as exc:  # noqa: BLE001 - a veto that fails must DENY
                denials.append(f"{veto.name}: veto errored ({exc.__class__.__name__})")
                continue
            if not verdict.allowed:
                denials.append(f"{veto.name}: {verdict.reason}")
        return tuple(denials)

    def apply(self, cmd: RigCommand, ctx: GateContext) -> tuple[RigCommand, tuple[str, ...]]:
        """The gated command, plus why water was withheld (empty if it was not).

        Note what this can and cannot do: it either returns `cmd` unchanged, or
        returns it with pump 0 and the valve shut. There is no path that sets
        `valve=True`, and none that touches `pan_deg` or `tilt_deg`.
        """
        denials = self.evaluate(ctx)
        if not denials:
            return cmd, ()
        return replace(cmd, pump_pct=0.0, valve=False), denials


def standard_gate(cfg) -> WaterGate:
    """The gate every built-in path uses. One definition, so the arming rules
    cannot drift apart between the mission, the CLI and the web console again."""
    return WaterGate([
        ArmedVeto(),
        HomedVeto(),
        LinkVeto(),
        SprayTimeVeto(limit_s=cfg.mission.max_spray_s * SPRAY_BACKSTOP_FACTOR),
    ])
