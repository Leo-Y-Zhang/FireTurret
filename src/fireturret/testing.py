# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""The public conformance suite — what a third-party implementation must satisfy.

Shipped as part of the package, and **used by the built-in tests**. That is the
load-bearing detail: a conformance suite the maintainers do not run against their
own implementations is documentation, not a contract, and it drifts the first
time an interface changes.

    from fireturret.testing import check_impact_observer
    check_impact_observer(MyObserver())

## The check that matters

`check_impact_observer`'s overlap case is the executable form of **invariant 4**:
*colour separates splash from fire*. An impact rendered directly ON TOP of the
target must still be detected.

An implementation that excluded the target's bounding box would pass every
obvious test and fail this one — and it would fail in the worst possible way,
by going blind exactly when the water is on target. The loop would lose feedback
at the moment it succeeded. This project has had that bug.
"""

from __future__ import annotations

import numpy as np


class ConformanceFailure(AssertionError):
    """A third-party implementation does not satisfy the contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConformanceFailure(message)


def check_impact_observer(observer, *, size: tuple[int, int] = (200, 300)) -> None:
    """An impact observer must find water, INCLUDING water landing on the target.

    `observer` needs `.observe(frame_bgr, target_px, exclude_bbox=None)` returning
    something with `.cx`, `.cy` and `.area`, or None.
    """
    h, w = size
    blank = np.zeros((h, w, 3), dtype=np.uint8)

    # 1. a still scene with nothing in it produces no detection
    observer.reset() if hasattr(observer, "reset") else None
    observer.observe(blank, (w / 2, h / 2))
    _require(
        observer.observe(blank, (w / 2, h / 2)) is None,
        "impact observer reported a splash in an empty, static scene",
    )

    # 2. a bright desaturated blob IS a splash
    frame = blank.copy()
    frame[100:140, 130:170] = (205, 205, 200)
    found = observer.observe(frame, (150, 120))
    _require(found is not None, "impact observer missed an obvious splash")
    _require(abs(found.cx - 150) < 25, f"splash column badly wrong: {found.cx}")

    # 3. THE overlap case — invariant 4, executable.
    if hasattr(observer, "reset"):
        observer.reset()
    fire = blank.copy()
    fire[100:140, 130:170] = (20, 140, 250)  # saturated orange: a fire
    observer.observe(fire, (150, 120))

    on_target = fire.copy()
    on_target[105:135, 135:165] = (205, 205, 200)  # water landing ON the fire
    overlapped = observer.observe(on_target, (150, 120), (130, 100, 40, 40))
    _require(
        overlapped is not None,
        "impact observer went blind when the splash landed ON the target. "
        "This is invariant 4: colour separates splash from fire. An observer "
        "that excludes the target's bbox loses feedback exactly when the water "
        "is on target — the worst possible moment.",
    )


def check_rig(rig) -> None:
    """A rig must accept commands and report telemetry without actuating on
    construction."""
    from .rig.interface import RigCommand

    telemetry = rig.telemetry()
    for field in ("pan_deg", "tilt_deg", "pump_pct", "valve", "estop", "ok", "homed"):
        _require(hasattr(telemetry, field), f"telemetry is missing {field!r}")

    _require(
        telemetry.valve is False and telemetry.pump_pct == 0.0,
        "a freshly constructed rig reported water already flowing",
    )
    rig.command(RigCommand(pan_deg=0.0, tilt_deg=20.0, pump_pct=0.0, valve=False))
    rig.close()


def check_profile(profile) -> None:
    """A mission profile must be complete and must not widen any constraint."""
    from .profiles.base import SafetyConstraints

    for field in ("name", "policy", "spray_envelope", "constraints"):
        _require(hasattr(profile, field), f"profile is missing {field!r}")
    _require(bool(profile.name), "profile has no name")

    policy = profile.policy
    _require(hasattr(policy, "setpoint_offset_px"), "policy has no setpoint_offset_px")
    _require(hasattr(policy, "is_complete"), "policy has no is_complete")

    offset = policy.setpoint_offset_px(None)
    _require(len(offset) == 2, "setpoint_offset_px must return (dx, dy)")

    # narrowing only
    strict = SafetyConstraints(max_spray_s=1.0, max_pump_pct=1.0)
    merged = profile.constraints.intersect(strict)
    _require(
        merged.max_spray_s <= 1.0 and merged.max_pump_pct <= 1.0,
        "profile constraints widened when intersected with a stricter set",
    )
