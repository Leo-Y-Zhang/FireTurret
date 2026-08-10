# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""Hardware self-test: exercise each actuator in isolation and verify the
telemetry link BEFORE ever running an autonomous mission on a real turret.

Water stays OFF unless explicitly enabled. Pan/tilt targets are kept small and
inside the configured geofence. The physical failsafe checks (E-stop, heartbeat
watchdog) require operator action and are printed as a checklist by the CLI.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .config import FireTurretConfig
from .control.mission import pan_touches_keepout
from .rig.interface import RigCommand

_ANGLE_TOL = 3.0  # deg; generous for a slewing pan axis, exact for an echo rig


@dataclass
class SelfTestReport:
    checks: list = field(default_factory=list)  # list[(name, ok, detail)]

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(ok for _, ok, _ in self.checks)


def run_selftest(rig, cfg: FireTurretConfig, *, water: bool = False,
                 dwell: float = 0.3, sleep=time.sleep) -> SelfTestReport:
    report = SelfTestReport()

    def readback():
        t = rig.telemetry()
        detail = (f"pan={t.pan_deg:.1f} tilt={t.tilt_deg:.1f} pump={t.pump_pct:.0f} "
                  f"valve={int(t.valve)} estop={int(t.estop)} homed={int(t.homed)} "
                  f"link={'ok' if t.ok else 'STALE'}")
        return t, detail

    def step(name, cmd, verify=None):
        rig.command(cmd)
        sleep(dwell)
        tel, detail = readback()
        ok = tel.ok and (verify(tel) if verify else True)
        report.checks.append((name, ok, detail))

    tel0, detail0 = readback()
    report.checks.append(("telemetry link", tel0.ok, detail0))
    if not tel0.ok:
        report.checks.append(("ABORTED", False, "no telemetry — unsafe to actuate"))
        return report
    # The firmware suppresses pan motion and water while un-homed or E-stopped, so
    # every actuation check would fail (or, for water, silently pass on a commanded
    # echo). Abort clearly instead of running misleading checks.
    if tel0.estop:
        report.checks.append(("ABORTED", False, "E-stop engaged — release it first"))
        return report
    if not tel0.homed:
        report.checks.append(("ABORTED", False, "pan axis not homed — home the turret first"))
        return report

    t = cfg.turret
    keepout = t.pan_keepout_deg
    pan_hi, pan_lo = min(5.0, t.pan_max_deg), max(-5.0, t.pan_min_deg)
    mid = (t.tilt_min_deg + t.tilt_max_deg) / 2.0

    # The self-test drives the turret directly, bypassing MissionController and
    # therefore its geofence. Nothing here may aim into the keep-out sector —
    # the sector exists to keep the nozzle off people, and a bring-up test with
    # an operator standing next to the machine is exactly when that matters.
    park = _park_pan(keepout, t.pan_min_deg, t.pan_max_deg)
    if park is None:
        report.checks.append(
            ("ABORTED", False,
             f"keep-out {keepout} leaves no reachable azimuth within "
             f"[{t.pan_min_deg}, {t.pan_max_deg}] — nothing can be actuated safely"))
        return report

    def pan_step(name: str, pan: float) -> None:
        if pan_touches_keepout(pan, keepout):
            report.checks.append(
                (name, True, f"SKIPPED (pan {pan:+.1f} is inside keep-out {keepout})"))
            return
        step(name, RigCommand(pan, 20.0, 0.0, False),
             lambda x: abs(x.pan_deg - pan) < _ANGLE_TOL)

    pan_step("pan +5 deg", pan_hi)
    pan_step("pan -5 deg", pan_lo)
    step("tilt to min", RigCommand(park, t.tilt_min_deg, 0.0, False),
         lambda x: abs(x.tilt_deg - t.tilt_min_deg) < _ANGLE_TOL)
    step("tilt to mid", RigCommand(park, mid, 0.0, False),
         lambda x: abs(x.tilt_deg - mid) < _ANGLE_TOL)
    step("warn indicator", RigCommand(park, 20.0, 0.0, False, warn=True))
    step("laser accessory", RigCommand(park, 20.0, 0.0, False, laser=True))

    if water:
        # telemetry echoes the COMMAND, not the actuated state, so this only
        # confirms the command reached a homed, un-E-stopped board (the firmware
        # would suppress water otherwise) — not that water physically flowed.
        # `park` is guaranteed clear of the keep-out sector by the abort above.
        step("water pulse (low, commanded)",
             RigCommand(park, t.tilt_min_deg, 25.0, True),
             lambda x: x.valve and x.pump_pct > 0 and x.homed and not x.estop)
    else:
        report.checks.append(
            ("water pulse", True, "SKIPPED (dry self-test; pass --arm-water to enable)"))

    rig.command(RigCommand(park, 20.0, 0.0, False))  # leave the rig safe
    return report


def _park_pan(
    keepout: tuple[float, float] | None, pan_min: float, pan_max: float
) -> float | None:
    """A resting azimuth inside the mechanical limits and outside the keep-out.

    Straight ahead when that is clear; otherwise one degree beyond the nearer
    sector edge (edges themselves count as inside, conservatively). Returns None
    when the sector and the limits between them leave nowhere safe to point —
    a real configuration, and one the caller must refuse to actuate in rather
    than quietly picking the least-bad angle.
    """
    candidates = [0.0]
    if keepout is not None:
        lo, hi = keepout
        candidates += sorted((lo - 1.0, hi + 1.0), key=abs)
    for pan in candidates:
        if pan_min <= pan <= pan_max and not pan_touches_keepout(pan, keepout):
            return pan
    return None
