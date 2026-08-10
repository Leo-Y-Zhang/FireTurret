# FireTurret — Camera-Guided Fire-Suppression Turret: Design

**Date:** 2026-07-17
**Status:** Approved (supersedes an earlier, since-deleted grid-world simulator prototype)

## Purpose

FireTurret is the complete software for a fixed, fire-truck-style water turret that takes
**real camera footage as input**, detects fire in the frames, and drives pan/tilt motors,
a pump, and a valve to arc water onto the fire — correcting its own aim by watching where
the water actually lands (**closed-loop visual servoing**). It runs identically against
real hardware (serial link to a microcontroller) or against a built-in physics + rendering
simulator, so the entire system is testable end-to-end with zero hardware.

Safety framing: this is a supervised demonstrator, not a life-safety device. Software
enforces watchdog/failsafe behaviour, but it never replaces alarms or extinguishers.

## Core decisions

- **Closed-loop visual servoing** (the chosen approach). No depth sensor required. The camera
  is mounted on the **pan stage only**, at a fixed slight downward pitch — so azimuth
  error is read directly from the target's pixel-x offset, the target never leaves frame
  when the nozzle tilts, and the water splash's pixel-y offset from the target gives the
  range error.
- **Two-axis decomposition.** Pan aligns image-x (azimuth). Tilt + pump pressure move the
  splash along image-y (range). Each axis is an independent damped P-controller with
  deadbands and settle counters.
- **Ballistics for the first shot, servoing for the hit.** A drag-corrected projectile
  model (RK4, quadratic drag) turns pump pressure into exit velocity and solves the
  low-arc elevation for a target range estimated from ground-plane geometry. The closed
  loop then absorbs all model and calibration error.
- **Classical CV fire detection** (colour + flicker + persistence): HSV and YCbCr fire
  rules intersected, morphological cleanup, connected components, temporal flicker
  scoring, and a centroid tracker that requires persistence before a target is confirmed.
  Runs in real time on a Raspberry Pi 5; the detector interface allows a learned model
  to be swapped in later.
- **Python 3.11+ / OpenCV / NumPy / pyserial; pytest.** All control, ballistics, and
  vision logic is pure and unit-tested; hardware sits behind a `TurretRig` interface with
  a serial implementation (matching the included microcontroller firmware) and a
  simulated implementation.

## Architecture

```
Frame source (webcam | video file | simulator)
      │
      ▼
vision/firedetect.py   colour rules (HSV ∧ YCbCr) + flicker history → fire blobs
vision/tracker.py      persistence tracking → confirmed primary target
vision/impact.py       splash detector (frame-difference, ROI near predicted impact)
      │
      ▼
control/mission.py     state machine: SEARCH → ACQUIRE → RANGE → ALIGN → SUPPRESS → CONFIRM → SEARCH
                                                        (SAFE on E-stop/heartbeat loss)
control/servo.py       pan ← image-x error · tilt/pump ← splash image-y error
ballistics.py          pressure → exit velocity; RK4 drag arc; elevation solver; pressure selection
geometry.py            pixel ↔ angle (pinhole), ground-plane range bootstrap
      │
      ▼
rig/interface.py       RigCommand{pan, tilt, pump, valve, laser} / RigTelemetry
rig/serial_rig.py      line protocol over USB serial → firmware (watchdog, limits, E-stop)
rig/sim_rig.py         simulated actuators (slew-limited) + perturbed "true" physics +
                       rendered scene (fire, splash, ground) → frames for the same pipeline
      │
      ▼
ui/overlay.py          operator overlay: detections, state, aim solution, arc side-view,
                       telemetry panel; app.py wires everything; __main__.py is the CLI
```

The simulator's "true" physics uses deliberately perturbed jet parameters, so the
end-to-end test proves the closed loop converges despite a wrong model — the whole point
of visual servoing.

## Mission state machine

- **SEARCH** — pump off, valve closed; slow pan sweep looking for fire.
- **ACQUIRE** — candidate found; hold; wait for tracker confirmation (persistence + flicker).
- **RANGE** — bootstrap range from ground-plane geometry (camera height/pitch); choose
  pump pressure and solve arc elevation; command pan/tilt; spin pump up.
- **ALIGN** — dry azimuth alignment: centre the target's pixel-x (bounded steps, settle).
- **SUPPRESS** — valve open; watch the splash; servo pan/tilt/pump each observation;
  monitor target confidence. Continuous-spray time limit forces re-evaluation.
- **CONFIRM** — fire signal gone: soak briefly, close valve, observe; reappearance → ACQUIRE.
- **SAFE** — E-stop, telemetry loss, or heartbeat failure: pump off, valve closed, hold.

Firmware failsafe mirrors this independently: no heartbeat for 500 ms → pump/valve off.

## Hardware (documented for the later build, in docs/BUILD_GUIDE.md)

Raspberry Pi 5 + ESP32 (or Uno) over USB serial; NEMA 17 + TMC2209 pan on a slew bearing;
DS3218 tilt for the nozzle; 12 V diaphragm pump on a MOSFET (PWM) + solenoid valve;
camera on the pan stage; normally-closed E-stop cutting motor/pump power in hardware.
The BUILD_GUIDE covers BOM, wiring, assembly, firmware flashing, calibration (boresight,
level/zero, camera FOV, pressure→range fitting via `fireturret fit`), and a safety checklist.

## Testing

Unit: geometry (pixel/angle, ground-plane), ballistics (monotonic range, solver inversion,
drag effect, pressure selection), fire detection on synthetic frames (positives; static
orange negative rejected by flicker; grey-scene negative), tracker persistence, splash
detector, servo sign/convergence/deadband, mission transitions incl. failsafes, protocol
round-trip, sim rig actuator slews and douse physics.
End-to-end: seeded simulator run, headless — the system must SEARCH → … → SUPPRESS and
drive the simulated fire's intensity to zero within a bounded frame budget, despite
perturbed true physics. CI: pytest on Python 3.11/3.12/3.13 (the suite is the
local source of truth).

## Non-goals

No mobile base (fixed turret, as specified). No claim of certified fire-safety
performance. No cloud/network dependency — everything runs locally.
