# FireTurret — architecture

How the pieces fit, and — more importantly — *why* the control is shaped the way
it is. Most of these choices were forced by one fact: **a single camera gives
direction, not distance.**

## Data flow

```
                 ┌─────────────────── Rig (hardware OR simulator) ───────────────────┐
 camera frame ──▶│                                                                    │
                 │  SimRig: renders a scene; SerialRig: reads a real camera elsewhere │
                 └───────────────────────────┬────────────────────────────────────────┘
                                             ▼
   Pipeline.tick(frame, dt, telemetry):
     firedetect ─▶ tracker(shift_px) ─▶ primary target
     splash detector (during SUPPRESS/CONFIRM)
                                             ▼
     MissionController.update ── state machine ──▶ VisualServo ──▶ RigCommand
                                             ▼
                 ┌────────── Rig.command(cmd) ─▶ actuators / firmware ──────────┐
                 └───────────────────────────────────────────────────────────────┘
```

`Pipeline` is pure perception+decision: frame in, `RigCommand` out. The rig is
the only thing that differs between simulation and hardware, behind one
interface (`rig/interface.py`).

## Why the control looks like this

**1. The camera measures direction; distance is inferred, and inference is
biased.** So the closed loop is servoed in *image space* — drive the observed
splash pixel onto the observed fire pixel — never onto an *estimated* range.
Both pixels are measured; a range estimate is not. (An earlier version servoed
`splash_range → target_range`; both were monocular estimates and the biased
target estimate pushed the water off-target. Removing that was the single most
important fix.)

**2. The camera rides the pan stage.** This is what lets azimuth correct the
boresight offset between camera and nozzle. Panning changes the fire's camera
bearing but not the splash's (the splash follows the nozzle, which follows the
pan), so the pan that lines up their columns also points the nozzle at the fire.
A fixed camera could never learn the boresight from the image alone.

**3. Frame-differencing needs a still camera.** The splash is found by motion +
colour; while the turret slews, everything moves and the signal is swamped. So
suppression is **observe-then-correct**: hold the aim dead still, collect splash
observations over a window, median-filter, apply one bounded correction, settle,
repeat. The same "move → settle → observe" discipline governs ALIGN (command the
absolute azimuth, re-measure only once arrived).

**4. A world-fixed fire slides across the image when the camera pans.** Tracking
in image space would leave a stale trail of tracks at old positions. So the
tracker is **motion-compensated**: shift each track by the pan-induced pixel
delta before association, so one track follows the fire through a slew. Tracks
also *latch* confirmed (once they pass the flicker test while still) so motion's
low confidence doesn't drop them.

**5. Range has a peak.** Under quadratic drag, landing distance rises with
elevation to a peak around 30° then falls — so tilt is a *bad* range control
(dRange/dTilt ≈ 0 at the peak, and the sign flips across it). Instead tilt is
**fixed** on the low arc and **pump** sets range (monotonic, unambiguous).
`plan_suppression` picks a tilt that places the target mid pump-envelope so the
servo has headroom both ways.

**6. A long splash is occluded; a short one is visible.** Water landing beyond
the fire is behind the flame from the camera's view. So the pump opens
deliberately *short* and the servo walks it *up* — the splash stays in front of
the fire (observable) the whole way in.

**7. Some fires are simply out of reach.** When the pump saturates and the
splash stays short for several cycles, the target is beyond the jet's envelope.
The mission enters HOLD (valve shut) with exponential backoff rather than hosing
water it can never land.

**8. The turret is fixed, so unreachable fires are the operator's problem.**
When a fire is out of reach, too close, beyond the pan traverse, or the water
impact can't be seen, `control/advisories.py` turns the condition into a
structured **operator advisory** — *what is wrong* and *what to do*
("move the turret ~N m closer", "rotate the base left"). Advisories surface as a
colour-coded overlay banner and drive a hardware warning output
(`RigCommand.warn` → `x=` → firmware `WARN_PIN`), so a human can reposition the
turret. This is guidance to a person, deliberately separate from the autonomous
control loop.

## Safety

Failsafes exist in two independent layers: the mission's SAFE state (E-stop or
telemetry loss ⇒ pump/valve off, hold) and the firmware's own E-stop + 500 ms
heartbeat watchdog (`firmware/turret_firmware/`). Aim is geofenced by the pan/
tilt limits in `TurretConfig`. This is a supervised demonstrator, not a
life-safety device.

## Performance

The perception + control pipeline (`Pipeline.tick`) runs at **~125 fps
(~8 ms/frame) at 960×540** on a desktop CPU — comfortably real-time, since the
detection is classical CV (colour thresholds + connected components + frame
differencing), not a neural network. On a Raspberry Pi 5 expect lower throughput;
drop the capture resolution in `CameraConfig` if needed. The detector interface
leaves room to swap in a learned model where a GPU (e.g. Jetson) is available.

## Known limits

Honest boundaries of the current design:

- **Range recession beyond reach.** A fire fleeing in *range* past the jet's
  maximum reach cannot be hit — the controller recognises this (HOLD) rather
  than pretending otherwise. Lateral (azimuthal) spread up to ~4°/s *is* tracked.
- **Wind faster than the correction cadence.** Suppression corrects on a ~0.7 s
  observe-then-correct cycle, so it tracks slow gusts, not turbulence.
- **A single monocular RGB camera.** No true depth; the opening range estimate
  is a ground-plane bootstrap and can be biased (the closed loop absorbs it).
  Smoke or a fire whose base is hidden degrades the ground-contact estimate.
- **Occlusion.** Deliberately mitigated by approaching from below, but a fire
  that fully hides the splash would starve the range loop.

## Determinism

The simulator uses one seeded RNG; the perception and control are deterministic
given the frames. `run_sim(seed=…)` replays identically — the basis of the
end-to-end tests, whose fires use *perturbed* true physics (different discharge
coefficient, drag, and a boresight bias) so that passing proves the closed loop
corrects model error rather than getting a lucky opening shot.
