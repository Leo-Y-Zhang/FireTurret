# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
import pytest

from fireturret.config import CameraConfig, ServoConfig, TurretConfig
from fireturret.control.servo import VisualServo
from fireturret.geometry import CameraModel

SCFG = ServoConfig()
TCFG = TurretConfig()
CAM = CameraModel(CameraConfig())


@pytest.fixture
def servo() -> VisualServo:
    return VisualServo(SCFG, TCFG, CAM)


# -- ALIGN azimuth (camera-centring) ---------------------------------------

def test_azimuth_steps_toward_target(servo: VisualServo) -> None:
    right = servo.azimuth_step(CAM.cx + 200, current_pan_deg=0.0)
    assert right > 0
    left = servo.azimuth_step(CAM.cx - 200, current_pan_deg=0.0)
    assert left < 0


def test_azimuth_deadband_and_settle(servo: VisualServo) -> None:
    for _ in range(SCFG.settle_frames):
        pan = servo.azimuth_step(CAM.cx + 1, current_pan_deg=10.0)
        assert pan == 10.0
    assert servo.azimuth_aligned


def test_azimuth_converges_iteratively(servo: VisualServo) -> None:
    target_azimuth = 20.0
    pan = 0.0
    for _ in range(40):
        err = target_azimuth - pan
        u, _ = CAM.angles_to_px(err, 0.0)
        pan = servo.azimuth_step(u, pan)
    assert pan == pytest.approx(target_azimuth, abs=SCFG.pan_deadband_deg)


# -- SUPPRESS splash-relative servo ----------------------------------------

def test_suppress_pans_toward_fire_from_splash(servo: VisualServo) -> None:
    # fire to the right of the splash column ⇒ increase pan
    step = servo.suppress_step((CAM.cx + 80, 300.0), (CAM.cx - 20, 300.0),
                               current_pan_deg=10.0, current_pump_pct=50.0)
    assert step.pan_deg > 10.0
    # fire left of splash ⇒ decrease pan
    step2 = servo.suppress_step((CAM.cx - 80, 300.0), (CAM.cx + 20, 300.0), 10.0, 50.0)
    assert step2.pan_deg < 10.0


def test_suppress_raises_pump_when_short(servo: VisualServo) -> None:
    # splash BELOW the fire base in the image (larger cy) = landed short ⇒ more pump
    short = servo.suppress_step((CAM.cx, 300.0), (CAM.cx, 360.0),
                                current_pan_deg=0.0, current_pump_pct=50.0)
    assert short.pump_pct > 50.0
    assert short.range_error_px > 0
    # splash above the fire base = landed long ⇒ less pump
    long = servo.suppress_step((CAM.cx, 300.0), (CAM.cx, 240.0), 0.0, 50.0)
    assert long.pump_pct < 50.0
    assert long.range_error_px < 0


def test_suppress_pump_step_is_bounded(servo: VisualServo) -> None:
    step = servo.suppress_step((CAM.cx, 300.0), (CAM.cx, 540.0),
                               current_pan_deg=0.0, current_pump_pct=50.0)
    assert step.pump_pct - 50.0 <= SCFG.pump_step_pct + 1e-9


def test_suppress_pump_clamped_to_100(servo: VisualServo) -> None:
    step = servo.suppress_step((CAM.cx, 300.0), (CAM.cx, 540.0),
                               current_pan_deg=0.0, current_pump_pct=99.0)
    assert step.pump_pct <= 100.0


def test_suppress_on_target_when_splash_on_fire(servo: VisualServo) -> None:
    fire = (CAM.cx, 300.0)
    step = None
    for _ in range(SCFG.settle_frames):
        step = servo.suppress_step(fire, fire, current_pan_deg=0.0, current_pump_pct=50.0)
        assert step.on_target
    assert servo.on_target


def test_suppress_converges_azimuth_and_range() -> None:
    """Closed loop in pure image space with a boresight bias: pan and pump
    corrections drive the splash pixel onto the fire pixel."""
    servo = VisualServo(SCFG, TCFG, CAM)
    fire_az = 12.0
    boresight = 1.6  # nozzle points this much right of the camera axis
    fire_cy = 300.0
    pan = 12.0  # start with the fire centred in the camera
    pump = 40.0
    for _ in range(120):
        # fire camera bearing = fire_az - pan; splash bearing = boresight (fixed)
        fire_u, _ = CAM.angles_to_px(fire_az - pan, 0.0)
        splash_u, _ = CAM.angles_to_px(boresight, 0.0)
        # more pump ⇒ splash lands further ⇒ higher in the image (smaller cy)
        splash_cy = fire_cy + (55.0 - pump) * 3.0
        step = servo.suppress_step((fire_u, fire_cy), (splash_u, splash_cy), pan, pump)
        pan = step.pan_deg
        pump = step.pump_pct
    # nozzle (pan + boresight) ends up pointing at the fire, splash row on fire
    assert pan + boresight == pytest.approx(fire_az, abs=0.8)
    assert pump == pytest.approx(55.0, abs=5.0)
