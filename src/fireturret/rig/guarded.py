# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""A `TurretRig` decorator that enforces the water gate at the LAST possible point.

`MissionController` runs every command through `control/gate.py`. That covers the
mission — and nothing else. Studio drives the rig directly, a manual jog
would, and any future profile or plugin might. Each of those is a path that never
constructs a `Pipeline` and therefore never meets the gate.

Rather than ask every such path to remember, the gate moves to the one place all
of them must pass through: the rig itself. `GuardedRig` wraps any `TurretRig` and
applies the gate to every command on its way out, so a caller that forgets is
still safe and a caller that is malicious is still gated.

**This is belt-and-braces, not a replacement.** The mission keeps its own gate,
because a denial there also feeds the operator advisory surface and the state
machine's own decisions. Double-gating is the intended design: the two are
independent, and neither can grant water the other denied.

Deny-only, as everywhere: the wrapper may remove water, never add it, and it
never touches `pan_deg` or `tilt_deg`.
"""

from __future__ import annotations

from ..control.gate import GateContext, WaterGate
from .interface import RigCommand, RigTelemetry, TurretRig


class GuardedRig:
    """Wraps a rig so every command passes the water gate before it is sent."""

    def __init__(self, rig: TurretRig, gate: WaterGate, *, armed: bool = False) -> None:
        self._rig = rig
        self._gate = gate
        self._armed = armed
        self._spray_elapsed_s = 0.0
        self.last_denials: tuple[str, ...] = ()

    # ------------------------------------------------------------- arming

    def arm(self) -> None:
        self._armed = True

    def disarm(self) -> None:
        self._armed = False

    @property
    def armed(self) -> bool:
        return self._armed

    # -------------------------------------------------------- TurretRig API

    def command(self, cmd: RigCommand) -> None:
        telemetry = self._rig.telemetry()
        gated, denials = self._gate.apply(
            cmd,
            GateContext(
                command=cmd, telemetry=telemetry, dt=0.0,
                armed=self._armed, spray_elapsed_s=self._spray_elapsed_s,
            ),
        )
        self.last_denials = denials
        self._rig.command(gated)

    def telemetry(self) -> RigTelemetry:
        return self._rig.telemetry()

    def close(self) -> None:
        self._rig.close()

    # ---------------------------------------------------------------- extras

    @property
    def inner(self) -> TurretRig:
        """The wrapped rig. Exposed for introspection (tests, the Studio's
        model-vs-truth view) — NOT as a way to route commands around the gate."""
        return self._rig

    def note_spray_time(self, elapsed_s: float) -> None:
        """Feed the gate's endless-hosing backstop. Optional: a caller that never
        reports spray time simply never trips that particular veto, and every
        other veto still applies."""
        self._spray_elapsed_s = elapsed_s
