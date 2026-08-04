# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
import pytest

from triton import ballistics
from triton.config import DEFAULT_CONFIG
from triton.rig.interface import RigCommand
from triton.rig.sim_rig import SimRig, SimScenario

CFG = DEFAULT_CONFIG
DT = 1 / 30


def make_rig(**kw) -> SimRig:
    return SimRig(CFG, SimScenario(**kw), seed=5)


def test_actuators_slew_toward_commands_at_bounded_rate() -> None:
    rig = make_rig()
    rig.command(RigCommand(90.0, 40.0, 100.0, False))
    rig.step(DT)
    assert rig.pan == pytest.approx(CFG.turret.pan_rate_dps * DT)
    assert rig.pan < 90
    for _ in range(600):
        rig.step(DT)
        rig.command(RigCommand(90.0, 40.0, 100.0, False))
    assert rig.pan == pytest.approx(90.0)
    assert rig.tilt == pytest.approx(40.0)
    assert rig.pump == pytest.approx(100.0)


def test_commands_clamped_to_limits() -> None:
    rig = make_rig()
    rig.command(RigCommand(999.0, 999.0, 250.0, False))
    for _ in range(2000):
        rig.step(DT)
        rig.command(RigCommand(999.0, 999.0, 250.0, False))
    assert rig.pan <= CFG.turret.pan_max_deg
    assert rig.tilt <= CFG.turret.tilt_max_deg
    assert rig.pump <= 100.0


def test_heartbeat_loss_closes_valve_and_pump() -> None:
    rig = make_rig()
    rig.command(RigCommand(0.0, 30.0, 80.0, True))
    for _ in range(30):
        rig.step(DT)
        rig.command(RigCommand(0.0, 30.0, 80.0, True))
    assert rig.valve is True
    # stop sending commands: firmware-mirror failsafe engages after the
    # heartbeat timeout, then the pump slews down. Valve cuts as soon as the
    # failsafe trips; the pump needs time to ramp to zero.
    for _ in range(int(0.6 / DT)):
        rig.step(DT)
    assert rig.valve is False  # cut promptly once the failsafe trips
    for _ in range(int(1.5 / DT)):
        rig.step(DT)
    assert rig.pump < 5.0


def test_no_splash_below_min_pump() -> None:
    rig = make_rig()
    rig.command(RigCommand(0.0, 30.0, CFG.jet.min_pump_pct - 5, True))
    for _ in range(120):
        rig.step(DT)
        rig.command(RigCommand(0.0, 30.0, CFG.jet.min_pump_pct - 5, True))
    assert rig.splash is None


def test_accurate_spray_douses_the_fire() -> None:
    scenario = SimScenario(fire_azimuth_deg=10.0, fire_range_m=7.0,
                           azimuth_bias_deg=0.0, velocity_coeff_scale=1.0, drag_scale=1.0)
    rig = SimRig(CFG, scenario, seed=5)
    # aim exactly with the true physics (no perturbation in this scenario)
    solution = ballistics.choose_solution(7.0, CFG.jet, CFG.turret.tilt_min_deg, CFG.turret.tilt_max_deg)
    assert solution is not None
    cmd = RigCommand(10.0, solution.elevation_deg, solution.pump_pct, True)
    start = rig.fire.intensity
    for _ in range(int(20 / DT)):
        rig.step(DT)
        rig.command(cmd)
        if rig.extinguished:
            break
    assert rig.fire.intensity < start
    assert rig.extinguished


def test_missed_spray_lets_fire_regrow() -> None:
    scenario = SimScenario(fire_azimuth_deg=10.0, fire_range_m=7.0)
    rig = SimRig(CFG, scenario, seed=5)
    rig.fire.intensity = 0.5
    cmd = RigCommand(-60.0, 20.0, 80.0, True)  # spraying the wrong way
    for _ in range(int(5 / DT)):
        rig.step(DT)
        rig.command(cmd)
    assert rig.fire.intensity > 0.5


def test_render_shape_and_determinism() -> None:
    a = make_rig()
    b = make_rig()
    frame_a = a.render()
    frame_b = b.render()
    assert frame_a.shape == (CFG.camera.height, CFG.camera.width, 3)
    assert (frame_a == frame_b).all()
