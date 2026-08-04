# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
import math

import pytest

from triton.config import DEFAULT_CONFIG
from triton.control.mission import (
    SUPP_NO_FEEDBACK_CYCLES,
    MissionController,
)
from triton.geometry import CameraModel
from triton.rig.interface import RigTelemetry
from triton.vision.firedetect import FireBlob
from triton.vision.impact import ImpactObservation
from triton.vision.tracker import Track

CFG = DEFAULT_CONFIG
DT = 1 / 30
CAM = CameraModel(CFG.camera)


def make_track(cx: float = 480.0, ground_range_m: float = 8.0) -> Track:
    """A confirmed target whose flame base projects to `ground_range_m` on the
    ground plane, so the mission's range estimate is physically reachable."""
    depression = math.degrees(math.atan(CFG.camera.mount_height_m / ground_range_m))
    ground_elev = -CFG.camera.mount_pitch_deg - depression
    _, v_ground = CAM.angles_to_px(0.0, ground_elev)
    bbox_h = 60
    bbox_y = int(round(v_ground)) - bbox_h
    flame_cy = bbox_y + bbox_h / 2
    t = Track(track_id=1, cx=cx, cy=flame_cy, area=800.0, hits=20, confidence=0.8)
    t.last_blob = FireBlob(
        cx=cx, cy=flame_cy, area=800.0, bbox=(int(cx) - 20, bbox_y, 40, bbox_h),
        colour_score=0.8, flicker_score=0.2, confidence=0.8,
    )
    return t


def echo_telemetry(mission: MissionController) -> RigTelemetry:
    """Telemetry that instantly matches the last command (ideal actuators).

    `homed` is stated explicitly because it now defaults to False (a rig that
    does not report a pan reference must not be assumed to have one). These
    tests exercise the mission on a healthy, homed turret; the un-homed path has
    its own test below.
    """
    cmd = mission._command()
    return RigTelemetry(
        cmd.pan_deg, cmd.tilt_deg, cmd.pump_pct, cmd.valve, False, True, homed=True
    )


@pytest.fixture
def mission() -> MissionController:
    return MissionController(CFG, CameraModel(CFG.camera))


def idle_telemetry(pan=0.0) -> RigTelemetry:
    return RigTelemetry(pan, 20.0, 0.0, False, estop=False, ok=True, homed=True)


def test_search_sweeps_pan(mission: MissionController) -> None:
    first = mission.update(DT, None, None, None, idle_telemetry())
    for _ in range(30):
        last = mission.update(DT, None, None, None, idle_telemetry())
    assert last.pan_deg != first.pan_deg
    assert last.valve is False
    assert last.pump_pct == 0.0


def test_candidate_stops_search_then_confirm_starts_ranging(mission: MissionController) -> None:
    candidate = make_track()
    mission.update(DT, None, candidate, None, idle_telemetry(pan=15.0))
    assert mission.state == "ACQUIRE"
    mission.update(DT, make_track(), None, None, idle_telemetry(pan=15.0))
    assert mission.state in ("RANGE", "ALIGN")  # RANGE is one-shot into ALIGN


def test_range_solves_and_align_opens_valve_when_ready(mission: MissionController) -> None:
    target = make_track(cx=CFG.camera.width / 2)  # already centred
    mission.update(DT, None, target, None, idle_telemetry())  # SEARCH -> ACQUIRE
    mission.update(DT, target, None, None, idle_telemetry())  # ACQUIRE -> RANGE
    mission.update(DT, target, None, None, idle_telemetry())  # RANGE -> ALIGN
    assert mission.state == "ALIGN"
    assert mission.debug.solution is not None
    assert mission.debug.target_range_m is not None
    # loop with echo telemetry until the servo settles and pump is up
    cmd = mission._command()
    for _ in range(60):
        cmd = mission.update(DT, target, None, None, echo_telemetry(mission))
        if mission.state == "SUPPRESS":
            break
    assert mission.state == "SUPPRESS"
    assert cmd.valve is True


@pytest.mark.parametrize("ground_range_m", [3.0, 14.0])  # too close AND too far
def test_unreachable_target_opens_max_reach_posture(
    mission: MissionController, ground_range_m: float
) -> None:
    """When no firing solution exists (target beyond max reach OR nearer than the
    low-arc minimum), RANGE deliberately opens the SAME max-reach posture — full
    pump at the peak-range elevation — for BOTH cases. This is intentional, not a
    bug: the monocular range estimate is unreliable, so the mission never refuses
    to fire on the estimate alone. For a too-close fire the overshoot lands the
    splash occluded, and the no-splash-feedback path then HOLDs (covered by the
    end-to-end test_too_close_fire_holds_not_hoses). Do not "fix" this by opening
    short for the too-close case — it breaks that recognise-and-HOLD behaviour."""
    target = make_track(cx=CFG.camera.width / 2, ground_range_m=ground_range_m)
    mission.update(DT, None, target, None, idle_telemetry())  # SEARCH -> ACQUIRE
    mission.update(DT, target, None, None, idle_telemetry())  # ACQUIRE -> RANGE
    mission.update(DT, target, None, None, idle_telemetry())  # RANGE -> ALIGN
    assert mission.debug.solution is None  # on the no-solution branch
    assert mission._pump_cmd == pytest.approx(100.0)
    assert mission._tilt_cmd > CFG.turret.tilt_min_deg  # peak elevation, not the low arc


def drive_to_suppress(mission: MissionController) -> Track:
    target = make_track(cx=CFG.camera.width / 2)
    mission.update(DT, None, target, None, idle_telemetry())
    mission.update(DT, target, None, None, idle_telemetry())
    for _ in range(120):
        mission.update(DT, target, None, None, echo_telemetry(mission))
        if mission.state == "SUPPRESS":
            break
    assert mission.state == "SUPPRESS"
    return target


def test_suppress_raises_pump_when_splash_is_short(mission: MissionController) -> None:
    target = drive_to_suppress(mission)
    tilt_before = mission._command().tilt_deg
    pump_before = mission._command().pump_pct
    # a splash well below (nearer than) the fire's ground point = short range
    ground_y = target.last_blob.bbox[1] + target.last_blob.bbox[3]
    impact = ImpactObservation(cx=target.cx, cy=ground_y + 90, area=200)
    cmd = None
    # Run until the correction actually lands rather than for a fixed frame
    # count. Since SP4 the observe window is derived from the current solution's
    # FLIGHT TIME, so it varies with pump — any hard-coded number is a guess that
    # goes stale the next time the cadence changes.
    for _ in range(300):
        cmd = mission.update(DT, target, None, impact, echo_telemetry(mission))
        if mission.debug.range_error_px is not None:
            break
    # short splash ⇒ positive row error ⇒ pump pushed up
    assert mission.debug.range_error_px is not None and mission.debug.range_error_px > 0
    assert cmd.pump_pct > pump_before
    # tilt is held fixed on the low arc throughout suppression
    assert cmd.tilt_deg == pytest.approx(tilt_before, abs=1e-6)


def test_target_loss_goes_confirm_then_search(mission: MissionController) -> None:
    drive_to_suppress(mission)
    cmd = mission.update(DT, None, None, None, echo_telemetry(mission))
    assert mission.state == "CONFIRM"
    assert cmd.valve is True  # soaking
    steps = int((CFG.mission.soak_s + CFG.mission.confirm_clear_s) / DT) + 10
    for _ in range(steps):
        cmd = mission.update(DT, None, None, None, echo_telemetry(mission))
    assert mission.state == "SEARCH"
    assert cmd.valve is False
    assert cmd.pump_pct == 0.0


def test_reflare_during_confirm_resumes_suppression(mission: MissionController) -> None:
    target = drive_to_suppress(mission)
    mission.update(DT, None, None, None, echo_telemetry(mission))
    assert mission.state == "CONFIRM"
    mission.update(DT, target, None, None, echo_telemetry(mission))
    assert mission.state == "SUPPRESS"


def test_spray_time_limit_forces_reevaluation(mission: MissionController) -> None:
    target = drive_to_suppress(mission)
    # feed an on-target splash so the no-feedback failsafe does NOT trip; this
    # isolates the max-spray timeout path (feedback present, fire just won't die)
    ground_y = target.last_blob.bbox[1] + target.last_blob.bbox[3]
    impact = ImpactObservation(cx=target.cx, cy=ground_y, area=200)
    steps = int(CFG.mission.max_spray_s / DT) + 5
    for _ in range(steps):
        mission.update(DT, target, None, impact, echo_telemetry(mission))
        if mission.state != "SUPPRESS":
            break
    assert mission.state in ("RANGE", "ALIGN")


def test_no_splash_feedback_holds(mission: MissionController) -> None:
    # spraying but never seeing the splash (occluded / mis-aimed) ⇒ HOLD, not
    # endless blind spraying
    target = drive_to_suppress(mission)
    # Budget generously in frames rather than in the legacy cadence: the
    # flight-time-derived window (SP4) is about twice as long, so a count sized
    # for the old 21-frame cycle never completes enough cycles to reach the
    # failsafe. The loop breaks as soon as HOLD is reached.
    budget = 120 * (SUPP_NO_FEEDBACK_CYCLES + 2)
    for _ in range(budget):
        mission.update(DT, target, None, None, echo_telemetry(mission))
        if mission.state == "HOLD":
            break
    assert mission.state == "HOLD"
    assert mission._command().valve is False


def test_estop_latches_safe_until_rearm(mission: MissionController) -> None:
    drive_to_suppress(mission)
    stopped = RigTelemetry(0, 20, 50, True, estop=True, ok=True)
    cmd = mission.update(DT, make_track(), None, None, stopped)
    assert mission.state == "SAFE"
    assert cmd.valve is False
    assert cmd.pump_pct == 0.0
    assert mission.armed is False  # a fault disarms water

    # the fault clears, but SAFE LATCHES — no auto-resume
    mission.update(DT, None, None, None, idle_telemetry())
    assert mission.state == "SAFE"

    # operator re-arms -> autonomy resumes on the next clean tick
    mission.rearm()
    mission.update(DT, None, None, None, idle_telemetry())
    assert mission.state == "SEARCH"


def test_disarmed_mission_never_opens_water(mission: MissionController) -> None:
    mission.disarm()
    target = make_track(cx=CFG.camera.width / 2)
    mission.update(DT, None, target, None, idle_telemetry())
    mission.update(DT, target, None, None, idle_telemetry())
    commands = [
        mission.update(DT, target, None, None, echo_telemetry(mission))
        for _ in range(300)  # long enough to pass ALIGN timeout into SUPPRESS
    ]
    # the state machine may reach SUPPRESS, but water is NEVER commanded
    assert all(c.valve is False and c.pump_pct == 0.0 for c in commands)
    assert mission.state in ("SUPPRESS", "CONFIRM", "HOLD", "RANGE", "ALIGN")


def test_arm_toggles_water_authority(mission: MissionController) -> None:
    assert mission.armed is True  # armed by default (the simulator/CLI arms it)
    mission.disarm()
    assert mission.armed is False
    mission.arm()
    assert mission.armed is True


def test_unhomed_telemetry_blocks_water(mission: MissionController) -> None:
    target = make_track(cx=CFG.camera.width / 2)
    mission.update(DT, None, target, None, idle_telemetry())
    mission.update(DT, target, None, None, idle_telemetry())

    def unhomed() -> RigTelemetry:
        cmd = mission._command()
        return RigTelemetry(cmd.pan_deg, cmd.tilt_deg, cmd.pump_pct, cmd.valve,
                            estop=False, ok=True, homed=False)

    commands = [mission.update(DT, target, None, None, unhomed()) for _ in range(300)]
    assert all(c.valve is False and c.pump_pct == 0.0 for c in commands)


def test_keepout_sector_pushes_pan_to_nearer_edge() -> None:
    from dataclasses import replace

    cfg = replace(CFG, turret=replace(CFG.turret, pan_keepout_deg=(-20.0, 20.0)))
    m = MissionController(cfg, CameraModel(cfg.camera))
    m._pan_cmd = 5.0  # inside the sector, nearer the +20 edge
    assert m._command().pan_deg == 20.0
    m._pan_cmd = -10.0  # inside, nearer the -20 edge
    assert m._command().pan_deg == -20.0
    m._pan_cmd = 45.0  # outside the sector -> unchanged
    assert m._command().pan_deg == 45.0


def test_stale_telemetry_forces_safe(mission: MissionController) -> None:
    drive_to_suppress(mission)
    dead = RigTelemetry(0, 0, 0, False, estop=False, ok=False)
    cmd = mission.update(DT, make_track(), None, None, dead)
    assert mission.state == "SAFE"
    assert cmd.valve is False


def test_safe_raises_safety_advisory_and_warn_flag(mission: MissionController) -> None:
    dead = RigTelemetry(0, 0, 0, False, estop=False, ok=False)
    cmd = mission.update(DT, None, None, None, dead)
    assert mission.state == "SAFE"
    assert any(a.kind == "SAFETY" for a in mission.advisories)
    assert cmd.warn is True  # hardware warning indicator driven


def test_no_advisory_and_no_warn_when_nominal(mission: MissionController) -> None:
    cmd = mission.update(DT, None, None, None, idle_telemetry())
    assert mission.advisories == []
    assert cmd.warn is False


def test_a_rejected_track_that_later_confirms_is_engaged_not_swept_past() -> None:
    """The _rejected set must not strand a real fire.

    A track that burns its ACQUIRE window is remembered so the sweep can get past
    it. If that same track later CONFIRMS, it is still skipped as a candidate —
    and _do_search looked at nothing else, so the turret swept straight past a
    confirmed fire with the valve shut, for as long as it stayed in frame.
    _do_acquire has always handled `target`; _do_search did not.
    """
    from triton.rig.interface import RigTelemetry
    from triton.vision.tracker import Track

    mc = MissionController(CFG, CameraModel(CFG.camera))
    dt = 1 / 30
    tel = RigTelemetry(pan_deg=0.0, tilt_deg=0.0, pump_pct=0.0, valve=False,
                       estop=False, ok=True, homed=True)

    decoy = Track(track_id=7, cx=320.0, cy=240.0, area=15000.0,
                  hits=50, misses=0, confidence=0.27, ever_confirmed=False)

    # Burn the acquire window so track 7 is remembered as rejected.
    for _ in range(int(CFG.mission.acquire_timeout_s / dt) + 120):
        mc.update(dt, None, decoy, None, tel)
    assert 7 in mc._rejected, "precondition: the track was rejected"
    assert mc.state == "SEARCH"

    # It now confirms — same track id, same object, real fire after all.
    confirmed = Track(track_id=7, cx=320.0, cy=240.0, area=15000.0,
                      hits=60, misses=0, confidence=0.9, ever_confirmed=True)
    mc.update(dt, confirmed, confirmed, None, tel)

    assert mc.state != "SEARCH", "a confirmed target must end the sweep"
    assert 7 not in mc._rejected, "confirmation retires the rejection"
