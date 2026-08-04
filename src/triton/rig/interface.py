# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""Hardware abstraction: everything above this layer is pure software; every
implementation of TurretRig (serial hardware, simulator, null) is swappable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class RigCommand:
    pan_deg: float
    tilt_deg: float
    pump_pct: float
    valve: bool
    laser: bool = False  # optional accessory; unused by the core mission
    warn: bool = False  # operator-warning indicator (LED/buzzer): needs attention


@dataclass(frozen=True)
class RigTelemetry:
    pan_deg: float
    tilt_deg: float
    pump_pct: float
    valve: bool
    estop: bool
    ok: bool  # False when telemetry is stale/unavailable
    # Pan axis has a known reference; the firmware locks water while False.
    # Defaults to False so a rig that does not report homing is never ASSUMED
    # homed — this field gates water, so its default must fail closed. Every rig
    # that genuinely has no pan reference to establish says so explicitly.
    homed: bool = False


class TurretRig(Protocol):
    def command(self, cmd: RigCommand) -> None:
        """Send targets. Also serves as the heartbeat: firmware failsafes if
        commands stop arriving."""
        ...

    def telemetry(self) -> RigTelemetry:
        ...

    def close(self) -> None:
        ...


class NullRig:
    """Echoes commands as instantly-achieved telemetry. Used for dry runs over
    recorded video, where no actuator can affect the footage anyway."""

    def __init__(self) -> None:
        self._last = RigCommand(0.0, 20.0, 0.0, False)

    def command(self, cmd: RigCommand) -> None:
        self._last = cmd

    def telemetry(self) -> RigTelemetry:
        c = self._last
        # No physical pan axis exists to home, so homing is vacuously satisfied.
        # Stated explicitly rather than inherited from a default (which now fails
        # closed) so the intent survives future changes to that default.
        return RigTelemetry(
            c.pan_deg, c.tilt_deg, c.pump_pct, c.valve, estop=False, ok=True, homed=True
        )

    def close(self) -> None:
        pass
