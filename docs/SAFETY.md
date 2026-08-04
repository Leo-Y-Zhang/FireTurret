# Safety design, and what this machine does NOT claim

FireTurret is a **supervised demonstrator**. It is not a life-safety
device, it is not certified for anything, and nothing in this document should be read as
suggesting it could be left to operate unattended around people.

This file records what the safety design actually is, what it is measured to do,
and — the part that matters most — the places where it is weaker than it might
appear.

## The layers, innermost first

Each layer is independent. None of them relies on another being correct.

1. **Firmware, on the microcontroller.** A hardware E-stop on its own pin, a
   500 ms heartbeat watchdog, mechanical pan/tilt clamps, and a homing lock that
   keeps water and pan motion disabled until the pan axis has a known reference.
   These execute without the host and survive the host crashing.
2. **The wire protocol.** A line missing any control field is refused outright
   rather than acted on in part, and a refused line does **not** feed the
   heartbeat watchdog — so a corrupted "close the valve" lets the watchdog fire
   instead of silently leaving water flowing.
3. **The water gate** (`control/gate.py`). One AND-chain that every command
   passes: armed, homed, link healthy, spray-time backstop, plus any registered
   safety veto. Vetoes may only ever **subtract** permission; the code cannot
   express the opposite, and the property is proven exhaustively over the boolean
   product of every veto.
4. **The guarded rig** (`rig/guarded.py`). The gate applied at the last point
   before a command reaches hardware, so a path that never constructs a
   `Pipeline` — the desktop workbench, a manual jog, a future plugin — still
   cannot spray.
5. **The mission's own failsafes.** A fault latches SAFE and disarms water; it
   does not auto-resume when the fault clears. Aim is constrained by the keep-out
   geofence, and water is withheld while the nozzle's swept path crosses a
   protected sector *and* for the flight time afterwards, because water launched
   mid-transit is still airborne when the transit ends.
6. **The deadline watchdog** (`runner.py`) — ⚠ **BUILT BUT NOT WIRED: this layer
   is NOT active in any running path.** `TickWatchdog` is referenced nowhere in
   `src/` outside its own module, and the only importer in the repository is
   `tests/test_runner.py`. Neither `triton run` nor `triton web` measures tick
   time, and neither passes its command through `gate()`, so nothing sheds work
   and nothing cuts water. **Treat this as a recorded gap, not a mitigation**,
   until a wiring assertion exists (see the note at the top of `runner.py` for
   what closing it requires, and why it was not done blind).

   When wired it sheds optional work on a late tick and cuts water after
   sustained overruns. It exists because the firmware
   heartbeat catches a *hung* host, not a *slow* one: a loop degraded from 30 Hz
   to 4 Hz feeds the heartbeat perfectly while servoing on quarter-second-old
   observations with the valve open — which is, until this is wired, exactly what
   would happen.

## Dry-aim by default

On real hardware the pump and valve are **off** unless the operator passes
`--arm-water` *and* types `ARM`. That applies to `run`, `web` and `selftest`
alike — a self-test's pump pulse is exactly as wet as a mission's, and it used to
fire on a bare flag with no confirmation.

The guarded rig is handed back **disarmed**, so forgetting to arm fails toward
dry, never toward wet.

## The unauthenticated console

`triton web` binds to **loopback** by default. The console has no authentication
and its `POST /estop` latches the mission into SAFE with no rearm path from the
browser — so on a routable address, anyone who can reach the network can
irreversibly disable a fire-suppression device, and watch the camera. Reaching it
from another machine requires `--host <addr> --expose`, which is a deliberate
acknowledgement rather than a default. Put a VPN or an authenticating reverse
proxy in front of it on any network you do not control.

## Safety vetoes: a mitigation, NOT an interlock

`safety/veto.py` allows a slow predicate (for example a person detector) to
withhold water. Its properties are proven exhaustively: a stale answer denies, a
missing answer denies, a raised exception denies, a denial latches until an
operator clears it, and no combination of vetoes can turn a denial into an
allowance.

What it explicitly does **not** claim:

- It does not make the machine safe around people. It is a mitigation layered on
  top of dry-aim-by-default and the deterministic keep-out sector, not a
  replacement for either, and not an interlock.
- A veto may deny water and freeze the aim. It may **not** command motion. A
  layer that re-aims acquires its own failure surface, and it would have to
  project a detector's bounding box through the same monocular range model the
  rest of the architecture deliberately distrusts.
- The staleness threshold is derived from **measured** end-to-end latency, not
  borrowed from the firmware heartbeat. Borrowing 500 ms would mark results stale
  the instant they arrive and chatter the valve at the detector's beat frequency,
  and an unusable safety layer gets switched off — which is worse than a slower
  one.

## A pure-ROS rig has no independent watchdog

The firmware's E-stop and 500 ms heartbeat live on the microcontroller. A rig
driven only over ROS 2 topics has neither: if the graph stalls, nothing on the
turret notices. So a ROS-driven build must either be paired with the firmware
E-stop on its own pin, or run **dry-aim only**.

This is stated rather than designed around. Designing around it would mean
claiming a safety property the architecture does not have.

## Known gaps, recorded rather than closed

**There is no water shut-off independent of the microcontroller.** The E-stop,
the heartbeat watchdog, the homing lock and the keep-out geofence all execute in
the same `loop()` on the same chip. One firmware hang, or one stuck MOSFET,
defeats all of them at once. A genuinely independent cut — a normally-closed
solenoid in series, on its own timer — is not implemented. This is the most
significant limitation in the design and it is stated here rather than designed
around.

**The system has never run on real hardware.** No physical turret exists. Every
claim in this repository is from simulation or from unit tests, and the
simulator's physics are deliberately perturbed relative to the controller's model
precisely so that "it works in the sim" cannot be mistaken for "it works".

**Range is monocular and biased.** The architecture is built around not trusting
it — the suppression loop servos on what the camera sees, not on an estimated
distance — but every place a range estimate *is* used (the opening solution,
operator advisories) inherits that bias.

**The person detector is poor at exactly the cases that matter.** HOG handles
upright pedestrians reasonably; it is poor on seated, prone and partially
occluded people, and essentially useless on animals — it was trained on
pedestrians. That is stated here rather than left for a reader to infer from the
phrase "person detection".

**HOG is not available at all on OpenCV 5.** `cv2.HOGDescriptor` was removed;
this repository's own environment runs OpenCV 5.0.0 and has no `HOG*` symbol.
So the "no extra dependency" argument for HOG holds on OpenCV 4 and does not on
5, where the ONNX detector behind the `[ml]` extra is the only built-in option.
`HogPersonDetector` raises a clear error rather than constructing — a safety
detector that silently degraded to "finds nothing" would be indistinguishable
from a clear scene, which is precisely the confusion the `capability` field
exists to prevent.

**"No detections" and "no detector" are different.** Every detector declares a
`capability`, and the null one declares `'none'`. Water may not be armed on
hardware when the only detector present says it cannot see.

## Reporting

Software vulnerabilities: GitHub Private Vulnerability Reporting on the
repository. Anything involving a **physical incident** — someone hit by water,
an unexpected actuation — should be reported the same way and will be treated as
higher priority than any software defect.
