# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""The simulated world, separated from the simulated rig.

`SimRig` was doing two jobs: modelling the ACTUATORS (slew limits, heartbeat
failsafe, telemetry) and modelling the WORLD (where the water lands, whether the
fire goes out). Only the second is needed by the mobile-platform tier, the spray
model, or anything else that wants to ask "what would happen if the turret were
pointed here from there".

The split is mechanical and behaviour-preserving: the arithmetic below is moved,
not rewritten, and the golden fixtures are the check.

## The platform argument

`step(...)` takes a `platform` pose that defaults to `IDENTITY`. Today every
caller uses the default, and with the identity pose the maths reduces exactly to
the fixed-turret case — no branch, no epsilon, the same numbers. The parameter
exists now so that SP8 changes the *values* flowing through a signature that
already exists, rather than changing the signature of everything at once.

The same reasoning applies to the spray hook: `step` returns an object exposing
`.centroid`, which the legacy `WorldPoint` satisfies by returning itself. SP5
replaces the point with a distribution that has a centroid *and* a covariance,
and nothing in between has to change.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..ballistics import exit_velocity, range_of
from ..config import TritonConfig
from ..geometry import clamp
from ..vision.synthetic import FireState, WorldPoint


@dataclass(frozen=True)
class PlatformPose:
    """Where the turret BASE sits in the world.

    The fixed turret is the zero-motion case of a moving one, which is why this
    exists before anything moves: it makes the fixed case a *value* rather than
    an absence, so SP8 adds trajectories instead of adding a concept.
    """

    x: float = 0.0
    z: float = 0.0
    yaw_deg: float = 0.0
    roll_deg: float = 0.0
    pitch_deg: float = 0.0

    @property
    def is_identity(self) -> bool:
        return (self.x, self.z, self.yaw_deg, self.roll_deg, self.pitch_deg) == (
            0.0, 0.0, 0.0, 0.0, 0.0
        )


IDENTITY = PlatformPose()


@dataclass
class WorldStep:
    """What one world tick produced."""

    splash: WorldPoint | None
    water_used_delta_l: float
    extinguished: bool

    @property
    def centroid(self) -> WorldPoint | None:
        """The servo signal, and the ONLY spray property control may consume.
        SP5 gives the spray a covariance too; the centroid stays the signal."""
        return self.splash


class SimWorld:
    """Fire, water and the interaction between them. No actuators, no telemetry."""

    def __init__(self, cfg: TritonConfig, scenario, fires: list[FireState]) -> None:
        self.cfg = cfg
        self.scenario = scenario
        self.fires = fires
        # The controller plans with the nominal JetConfig; water lands where the
        # TRUE physics says. The end-to-end test passing is what proves the closed
        # loop corrects that model error, so this gap is deliberate (invariant 10).
        self.true_jet = _perturbed_jet(cfg, scenario)

    def step(
        self,
        dt: float,
        t_s: float,
        *,
        pan_deg: float,
        tilt_deg: float,
        pump_pct: float,
        valve: bool,
        platform: PlatformPose = IDENTITY,
    ) -> WorldStep:
        self._drift_fire(t_s)
        splash = self._splash(t_s, pan_deg, tilt_deg, pump_pct, valve, platform)
        water = 0.0
        if splash is not None:
            water = (5.0 / 60.0) * (pump_pct / 100.0) * dt  # ~5 L/min at full pump
        self._douse(splash, dt)
        return WorldStep(
            splash=splash,
            water_used_delta_l=water,
            extinguished=all(f.intensity <= 0.0 for f in self.fires),
        )

    # ------------------------------------------------------------- internals

    def _drift_fire(self, t_s: float) -> None:
        s = self.scenario
        if s.fire_drift_az_dps or s.fire_drift_range_mps:
            az = s.fire_azimuth_deg + s.fire_drift_az_dps * t_s
            rng = clamp(s.fire_range_m + s.fire_drift_range_mps * t_s, 3.0, 11.0)
            self.fires[0].position = WorldPoint.from_polar(az, rng)

    def _splash(
        self,
        t_s: float,
        pan_deg: float,
        tilt_deg: float,
        pump_pct: float,
        valve: bool,
        platform: PlatformPose,
    ) -> WorldPoint | None:
        if not (valve and pump_pct >= self.cfg.jet.min_pump_pct):
            return None
        s = self.scenario
        speed = exit_velocity(pump_pct, self.true_jet)
        true_range = range_of(speed, tilt_deg, self.true_jet)
        # Gusting wind pushes the splash in azimuth and range, unknown to the
        # controller; only the closed loop watching the splash can compensate.
        phase = 2.0 * math.pi * t_s / s.wind_period_s
        cross = s.wind_cross_amp_deg * math.sin(phase)
        range_off = s.wind_range_amp_m * math.cos(phase)
        # With the identity pose these two terms are +0.0 and the result is
        # bit-identical to the fixed-turret arithmetic they replace.
        azimuth = pan_deg + s.azimuth_bias_deg + cross + platform.yaw_deg
        point = WorldPoint.from_polar(azimuth, max(0.5, true_range + range_off))
        if platform.is_identity:
            return point
        return WorldPoint(x=point.x + platform.x, z=point.z + platform.z)

    def _douse(self, splash: WorldPoint | None, dt: float) -> None:
        """Each live fire is doused if the splash lands inside its radius, else it
        slowly regrows. The scene is out only when every fire is."""
        s = self.scenario
        for fire in self.fires:
            if fire.intensity <= 0.0:
                continue
            if splash is not None:
                dx = splash.x - fire.position.x
                dz = splash.z - fire.position.z
                miss = (dx * dx + dz * dz) ** 0.5
                if miss < s.douse_radius_m:
                    falloff = 1.0 - 0.6 * (miss / s.douse_radius_m)
                    fire.intensity -= s.douse_rate * falloff * dt
                else:
                    fire.intensity += s.regrow_rate * dt
            else:
                fire.intensity += s.regrow_rate * dt
            fire.intensity = clamp(fire.intensity, 0.0, 1.0)
            if fire.intensity < 0.03:
                fire.intensity = 0.0


def _perturbed_jet(cfg: TritonConfig, scenario):
    from dataclasses import replace

    return replace(
        cfg.jet,
        velocity_coeff=cfg.jet.velocity_coeff * scenario.velocity_coeff_scale,
        drag_k=cfg.jet.drag_k * scenario.drag_scale,
    )
