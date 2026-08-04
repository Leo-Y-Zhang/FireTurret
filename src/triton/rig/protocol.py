# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""Line protocol shared with the microcontroller firmware.

Host → firmware (one line per command, also the heartbeat):
    C p=<pan_deg> t=<tilt_deg> w=<pump_pct> v=<0|1> l=<0|1> x=<0|1>\n
Firmware → host (10 Hz):
    S p=<pan_deg> t=<tilt_deg> w=<pump_pct> v=<0|1> e=<0|1> h=<0|1>\n
(l = laser accessory, x = operator-warning indicator, e = E-stop engaged,
 h = pan axis homed. h absent ⇒ treated as NOT homed: this field gates water, so
 an unknown value must fail closed. Firmware from this repo always sends it.)

Every host→firmware token must have a UNIQUE FIRST CHARACTER. Older deployed
boards dispatch on `tok[0]` alone, so a new token sharing a first letter with an
existing one silently overwrites that field. `tests/test_safety_debt.py` pins
this as a protocol constraint.

Keep this file and firmware/turret_firmware/turret_firmware.ino in lockstep.
"""

from __future__ import annotations

from .interface import RigCommand, RigTelemetry


def encode_command(cmd: RigCommand) -> str:
    return (
        f"C p={cmd.pan_deg:.2f} t={cmd.tilt_deg:.2f} w={cmd.pump_pct:.1f} "
        f"v={1 if cmd.valve else 0} l={1 if cmd.laser else 0} "
        f"x={1 if cmd.warn else 0}\n"
    )


def parse_telemetry(line: str) -> RigTelemetry | None:
    line = line.strip()
    if not line.startswith("S "):
        return None
    fields: dict[str, str] = {}
    for token in line[2:].split():
        if "=" in token:
            key, _, value = token.partition("=")
            fields[key] = value
    try:
        return RigTelemetry(
            pan_deg=float(fields["p"]),
            tilt_deg=float(fields["t"]),
            pump_pct=float(fields["w"]),
            valve=fields["v"] == "1",
            estop=fields.get("e", "0") == "1",
            ok=True,
            homed=fields.get("h", "0") == "1",  # absent ⇒ NOT homed (fail closed)
        )
    except (KeyError, ValueError):
        return None
