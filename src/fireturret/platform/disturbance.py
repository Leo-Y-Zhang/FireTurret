# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""Bump versus reposition — told apart by DWELL, from three independent channels.

A knock and a deliberate move look identical for the first fraction of a second.
What separates them is what happens next: a bump is a transient the platform
returns from, a reposition is a step it stays at. So the discriminator is dwell,
not magnitude.

Why it matters: after a **bump** the turret should re-acquire and carry on, since
the world has not really changed. After a **reposition** its calibration — the
learned boresight, the range estimate, the keep-out sector's meaning — is stale
and must be re-established.

## Three OR-ed channels, and why the visual one must stand alone

    inertial   a jerk spike from the IMU
    attitude   a persistent step in roll/pitch
    visual     a scene shift that phase correlation cannot explain by ego-motion

They are OR-ed rather than voted, because missing a disturbance is worse than
flagging a spurious one: a false positive costs a re-acquisition, a false
negative means spraying on a stale calibration.

**The visual channel must work with no IMU at all.** The cheapest fixed build has
no inertial sensor, and it is exactly the build most likely to be knocked — a
camera on a tripod someone walks into. A design where disturbance detection
required an IMU would protect only the builds that need it least.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class DisturbanceKind(Enum):
    NONE = "none"
    BUMP = "bump"              # transient: re-acquire and carry on
    REPOSITION = "reposition"  # persistent: calibration is stale


# A bump that persists beyond this is a reposition. Chosen to be comfortably
# longer than a knock's settling time and comfortably shorter than a deliberate
# move, so neither is mistaken for the other.
REPOSITION_DWELL_S = 1.5

# Channel thresholds.
JERK_DPS2 = 400.0        # inertial: angular acceleration spike
ATTITUDE_STEP_DEG = 3.0  # attitude: a step that has not returned
SCENE_SHIFT_PX = 25.0    # visual: unexplained image motion


@dataclass
class DisturbanceMonitor:
    """Detects disturbances and classifies them once their dwell is known."""

    reposition_dwell_s: float = REPOSITION_DWELL_S
    jerk_dps2: float = JERK_DPS2
    attitude_step_deg: float = ATTITUDE_STEP_DEG
    scene_shift_px: float = SCENE_SHIFT_PX

    kind: DisturbanceKind = DisturbanceKind.NONE
    _active_s: float = 0.0
    _prev_rate_dps: float | None = None
    _reference_attitude: tuple[float, float] | None = None
    _channels: set[str] = field(default_factory=set)

    def observe(
        self,
        dt: float,
        *,
        total_rate_dps: float | None = None,
        roll_deg: float | None = None,
        pitch_deg: float | None = None,
        scene_shift_px: float | None = None,
    ) -> DisturbanceKind:
        """One tick. Every channel is optional — pass what the build has.

        A build with no IMU passes only `scene_shift_px` and gets full
        disturbance detection from it.
        """
        channels: set[str] = set()

        # --- inertial: a jerk spike
        if total_rate_dps is not None:
            if self._prev_rate_dps is not None and dt > 0:
                jerk = abs(total_rate_dps - self._prev_rate_dps) / dt
                if jerk > self.jerk_dps2:
                    channels.add("inertial")
            self._prev_rate_dps = total_rate_dps

        # --- attitude: a step away from the reference that has not returned
        if roll_deg is not None and pitch_deg is not None:
            if self._reference_attitude is None:
                self._reference_attitude = (roll_deg, pitch_deg)
            else:
                ref_roll, ref_pitch = self._reference_attitude
                if (abs(roll_deg - ref_roll) > self.attitude_step_deg
                        or abs(pitch_deg - ref_pitch) > self.attitude_step_deg):
                    channels.add("attitude")

        # --- visual: image motion ego-motion cannot explain. Stands alone.
        if scene_shift_px is not None and abs(scene_shift_px) > self.scene_shift_px:
            channels.add("visual")

        if channels:
            self._active_s += dt
            self._channels |= channels
            self.kind = (
                DisturbanceKind.REPOSITION
                if self._active_s >= self.reposition_dwell_s
                else DisturbanceKind.BUMP
            )
        else:
            if self.kind is DisturbanceKind.BUMP:
                # A transient that ended: the world did not change, so forget it
                # and let the mission re-acquire.
                self.kind = DisturbanceKind.NONE
            self._active_s = 0.0
            self._channels.clear()
        return self.kind

    @property
    def channels(self) -> frozenset[str]:
        """Which channels are currently firing — for the operator surface."""
        return frozenset(self._channels)

    def acknowledge(self) -> None:
        """Clear a latched reposition, once calibration has been re-established."""
        self.kind = DisturbanceKind.NONE
        self._active_s = 0.0
        self._channels.clear()
        self._reference_attitude = None
