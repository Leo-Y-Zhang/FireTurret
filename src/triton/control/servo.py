# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""Closed-loop visual servoing.

The camera rides the pan stage, so aiming has two phases:

**ALIGN (dry, no water yet).** Centre the fire in the image by panning — this
gets the fire into the aim envelope. Camera-relative, so it ignores the
boresight offset between camera and nozzle (unknowable without water in the
air).

**SUPPRESS (wet, closed on the splash).** Now the water itself is visible, so
we servo the *splash* onto the *fire*:

- **Azimuth:** the nozzle points along the splash bearing; the fire sits at its
  own bearing. Panning changes the fire's camera bearing but not the splash's
  (the splash follows the nozzle, which follows the pan), so the pan that makes
  the fire's column meet the splash's column also makes the nozzle point at the
  fire — **correcting the boresight bias that camera-centring cannot.**
- **Range:** tilt is held on the controllable low arc and pump alone sets range
  (monotonic, no peak ambiguity). Splash short of the fire ⇒ more pump.

All model error — wrong discharge coefficient, wrong drag, boresight bias — is
observed as a splash-vs-fire offset and servoed away. That is the whole design.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import ServoConfig, TurretConfig
from ..geometry import CameraModel, clamp


@dataclass(frozen=True)
class SuppressStep:
    pan_deg: float
    pump_pct: float
    az_error_deg: float
    range_error_px: float
    on_target: bool


class VisualServo:
    def __init__(self, cfg: ServoConfig, turret: TurretConfig, camera: CameraModel) -> None:
        self.cfg = cfg
        self.turret = turret
        self.camera = camera
        self._aligned_streak = 0
        self._on_streak = 0

    def reset(self) -> None:
        self._aligned_streak = 0
        self._on_streak = 0

    # -- ALIGN: centre the fire ---------------------------------------------

    def azimuth_error_deg(self, target_px_x: float) -> float:
        ang = self.camera.px_to_angles(target_px_x, self.camera.cy)
        return ang.azimuth_deg + self.camera.cfg.boresight_offset_deg

    def azimuth_step(self, target_px_x: float, current_pan_deg: float) -> float:
        err = self.azimuth_error_deg(target_px_x)
        if abs(err) <= self.cfg.pan_deadband_deg:
            self._aligned_streak += 1
            return current_pan_deg
        self._aligned_streak = 0
        step = clamp(self.cfg.pan_gain * err, -6.0, 6.0)
        return clamp(
            current_pan_deg + step, self.turret.pan_min_deg, self.turret.pan_max_deg
        )

    @property
    def azimuth_aligned(self) -> bool:
        return self._aligned_streak >= self.cfg.settle_frames

    # -- SUPPRESS: drive the splash onto the fire ---------------------------

    def suppress_step(
        self,
        fire_px: tuple[float, float],
        splash_px: tuple[float, float],
        current_pan_deg: float,
        current_pump_pct: float,
        range_gain_pct_per_px: float | None = None,
        setpoint_offset_px: tuple[float, float] = (0.0, 0.0),
    ) -> SuppressStep:
        """One correction toward putting the splash pixel on the fire pixel.

        Both positions are directly observed in the same image, so no range
        estimate is needed. Azimuth ← the column gap; pump ← the row gap.

        `range_gain_pct_per_px` overrides the configured constant with a gain
        SCHEDULED for the current operating point (`control/plant.py`). One fixed
        constant cannot serve a plant whose observable gain varies 7.4x across
        the pump band; passing None keeps the old fixed-gain behaviour.

        `setpoint_offset_px` generalises the loop to **"drive the observed impact
        onto a commanded image setpoint"**. Fire suppression is the case where
        that offset is (0, 0) — aim the water AT the thing. A profile that wants
        to land water deliberately short, or beside a target, sets an offset
        instead of needing a parallel control law. Three lines, not an
        abstraction layer.
        """
        range_gain = (
            self.cfg.range_gain_pct_per_px
            if range_gain_pct_per_px is None
            else range_gain_pct_per_px
        )
        fire_px = (fire_px[0] + setpoint_offset_px[0], fire_px[1] + setpoint_offset_px[1])
        # Azimuth: pan so the fire's column meets the splash's column. Fire to
        # the right of the splash (az_err > 0) ⇒ increase pan to swing the
        # nozzle toward the fire.
        az_fire = self.camera.px_to_angles(fire_px[0], self.camera.cy).azimuth_deg
        az_splash = self.camera.px_to_angles(splash_px[0], self.camera.cy).azimuth_deg
        az_err = az_fire - az_splash
        if abs(az_err) > self.cfg.suppress_pan_deadband_deg:
            d_pan = clamp(self.cfg.suppress_pan_gain * az_err, -4.0, 4.0)
            pan = clamp(
                current_pan_deg + d_pan, self.turret.pan_min_deg, self.turret.pan_max_deg
            )
        else:
            pan = current_pan_deg

        # Range: the splash below the fire base in the image (row_err > 0) means
        # it landed short (nearer) ⇒ more pump; above means long ⇒ less pump.
        row_err = splash_px[1] - fire_px[1]
        if abs(row_err) > self.cfg.range_deadband_px:
            d_pump = clamp(
                range_gain * row_err,
                -self.cfg.pump_step_pct,
                self.cfg.pump_step_pct,
            )
            pump = clamp(current_pump_pct + d_pump, 0.0, 100.0)
        else:
            pump = current_pump_pct

        on = (
            abs(az_err) <= self.cfg.suppress_pan_deadband_deg
            and abs(row_err) <= self.cfg.range_deadband_px
        )
        self._on_streak = self._on_streak + 1 if on else 0
        return SuppressStep(
            pan_deg=pan,
            pump_pct=pump,
            az_error_deg=az_err,
            range_error_px=row_err,
            on_target=on,
        )

    @property
    def on_target(self) -> bool:
        return self._on_streak >= self.cfg.settle_frames
