# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""Simulated rig: slew-limited actuators, deliberately perturbed 'true' jet
physics, a dousable fire, and a rendered camera view.

The perturbation matters: the controller plans with the nominal JetConfig,
but water lands where the *true* physics says — including an azimuth bias
that models boresight error. The end-to-end test passes only because the
closed visual-servo loop corrects what the model gets wrong.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import FireTurretConfig
from ..geometry import clamp
from ..rng import RngBundle
from ..vision.synthetic import (
    FIRE_DRAWS,
    Draws,
    FireState,
    SceneCamera,
    WorldPoint,
    _draw_fire,
    draw_static_blob,
    render_scene,
)
from .interface import RigCommand, RigTelemetry
from .sim_world import SimWorld, _perturbed_jet


@dataclass
class SimScenario:
    fire_azimuth_deg: float = 24.0
    fire_range_m: float = 7.0
    fire_radius_m: float = 0.5
    # true-physics perturbations (unknown to the controller)
    velocity_coeff_scale: float = 0.88
    drag_scale: float = 1.3
    azimuth_bias_deg: float = 1.6
    douse_radius_m: float = 1.1
    douse_rate: float = 0.55  # intensity/s at splash centre
    regrow_rate: float = 0.03  # intensity/s while unhit
    # additional fires as (azimuth_deg, range_m); the mission must handle each
    # in turn (CONFIRM → SEARCH → next). Empty = single fire.
    extra_fires: tuple[tuple[float, float], ...] = ()
    # gusting wind, UNKNOWN to the controller: a slowly varying crosswind
    # (azimuthal) and head/tailwind (range) on the water. The open-loop model
    # has no idea; only the closed loop watching the splash can compensate.
    wind_cross_amp_deg: float = 0.0
    wind_range_amp_m: float = 0.0
    wind_period_s: float = 8.0
    # a static, fire-coloured but non-flickering decoy (azimuth_deg, range_m):
    # a red object the turret must NOT engage (the flicker gate rejects it)
    decoy: tuple[float, float] | None = None
    # the primary fire drifts (spreads) over time; the tracker and servo must
    # follow a moving target, not just a stationary one
    fire_drift_az_dps: float = 0.0
    fire_drift_range_mps: float = 0.0
    # Fault injection, as (start_s, end_s) windows. `SimRig.telemetry()` used to
    # hardcode estop=False, ok=True, homed=True — which means no golden and no
    # end-to-end run had EVER visited the SAFE state, and the latching failsafe
    # was exercised only by unit tests driving the controller directly.
    # All default None, so every existing fixture is untouched.
    estop_window_s: tuple[float, float] | None = None
    telemetry_dropout_s: tuple[float, float] | None = None
    unhomed_window_s: tuple[float, float] | None = None


class SimRig:
    def __init__(self, cfg: FireTurretConfig, scenario: SimScenario, seed: int = 7) -> None:
        self.cfg = cfg
        self.scenario = scenario
        self.scene_camera = SceneCamera(cfg.camera)

        # The deliberately-perturbed "true" physics (invariant 10) now belongs to
        # the world; kept as an attribute here because tests and the Studio read
        # it to show model-vs-truth.
        self.true_jet = _perturbed_jet(cfg, scenario)

        self.fire = FireState(
            position=WorldPoint.from_polar(scenario.fire_azimuth_deg, scenario.fire_range_m),
            radius_m=scenario.fire_radius_m,
            intensity=1.0,
        )
        # fires[0] is the primary (aliased as self.fire for single-fire compat)
        self.fires = [self.fire] + [
            FireState(position=WorldPoint.from_polar(az, rng), radius_m=scenario.fire_radius_m, intensity=1.0)
            for az, rng in scenario.extra_fires
        ]
        self.extinguished = False

        # The world half of the simulator. It owns the same `self.fires` list, so
        # dousing is visible here without copying state back and forth.
        self.world = SimWorld(cfg, scenario, self.fires)

        # One named stream per renderer, declared up front. Keyed by name, so
        # adding a stream later cannot renumber the existing ones — see
        # `fireturret.rng` for why that matters more than it sounds.
        self.rng = RngBundle(
            seed, ["splash", *(f"fire{i}" for i in range(len(self.fires)))]
        )

        self.pan = 0.0
        self.tilt = cfg.turret.tilt_min_deg
        self.pump = 0.0
        self.valve = False
        self._cmd = RigCommand(self.pan, self.tilt, 0.0, False)
        self._since_command = 0.0
        self.splash: WorldPoint | None = None
        self.water_used_l = 0.0
        self.time = 0.0

    # -- rig interface -------------------------------------------------------

    def command(self, cmd: RigCommand) -> None:
        turret = self.cfg.turret
        self._cmd = RigCommand(
            pan_deg=clamp(cmd.pan_deg, turret.pan_min_deg, turret.pan_max_deg),
            tilt_deg=clamp(cmd.tilt_deg, turret.tilt_min_deg, turret.tilt_max_deg),
            pump_pct=clamp(cmd.pump_pct, 0.0, 100.0),
            valve=cmd.valve,
            laser=cmd.laser,
        )
        self._since_command = 0.0

    def telemetry(self) -> RigTelemetry:
        return RigTelemetry(
            pan_deg=self.pan,
            tilt_deg=self.tilt,
            pump_pct=self.pump,
            valve=self.valve,
            estop=_in_window(self.time, self.scenario.estop_window_s),
            ok=not _in_window(self.time, self.scenario.telemetry_dropout_s),
            # The simulated pan axis starts at a known reference by construction.
            # Stated explicitly: the field defaults to False so real rigs fail
            # closed, and the simulator must opt in rather than inherit.
            homed=not _in_window(self.time, self.scenario.unhomed_window_s),
        )

    def close(self) -> None:
        pass

    # -- world stepping ------------------------------------------------------

    def step(self, dt: float) -> None:
        turret = self.cfg.turret
        self._since_command += dt
        # firmware-mirroring failsafe: no heartbeat → pump/valve off
        cmd = self._cmd
        if self._since_command > self.cfg.mission.heartbeat_timeout_s:
            cmd = RigCommand(cmd.pan_deg, cmd.tilt_deg, 0.0, False)

        self.pan = _slew(self.pan, cmd.pan_deg, turret.pan_rate_dps * dt)
        self.tilt = _slew(self.tilt, cmd.tilt_deg, turret.tilt_rate_dps * dt)
        self.pump = _slew(self.pump, cmd.pump_pct, turret.pump_rate_pct_ps * dt)
        self.valve = cmd.valve

        self.time += dt
        # The WORLD physics — fire drift, where the water lands, dousing — lives
        # in rig/sim_world.py so the mobile-platform tier and the spray model can
        # reuse it. This rig keeps what is genuinely rig-shaped: slew limits, the
        # heartbeat failsafe, telemetry and rendering.
        result = self.world.step(
            dt, self.time,
            pan_deg=self.pan, tilt_deg=self.tilt,
            pump_pct=self.pump, valve=self.valve,
        )
        self.splash = result.splash
        self.water_used_l += result.water_used_delta_l
        self.extinguished = result.extinguished

    def render(self) -> np.ndarray:
        # Fire i is ALWAYS drawn from stream "fire{i}", keyed on its index rather
        # than on which fires happen to be alive. Under the old shared generator,
        # a fire going out changed how many draws the frame consumed and thereby
        # re-rolled every later random value in the run.
        frame = render_scene(
            self.scene_camera, self.pan, self.fires[0], self.splash, self.rng
        )
        for i, extra in enumerate(self.fires[1:], start=1):
            # allocate the block before testing intensity, so the stream advances
            # by the same amount whether or not this fire is still burning
            draws = Draws(self.rng[f"fire{i}"].random(FIRE_DRAWS))
            if extra.intensity > 0.02:
                _draw_fire(frame, self.scene_camera, extra, self.pan, draws)
        if self.scenario.decoy is not None:
            az, rng = self.scenario.decoy
            draw_static_blob(frame, self.scene_camera, WorldPoint.from_polar(az, rng), self.pan)
        return frame


def _in_window(t_s: float, window: tuple[float, float] | None) -> bool:
    """Is `t_s` inside an injected-fault window? `None` means no fault ever —
    which is every existing scenario, so this is behaviour-neutral by default."""
    if window is None:
        return False
    start, end = window
    return start <= t_s < end


def _slew(value: float, target: float, max_delta: float) -> float:
    delta = target - value
    if delta > max_delta:
        delta = max_delta
    elif delta < -max_delta:
        delta = -max_delta
    return value + delta
