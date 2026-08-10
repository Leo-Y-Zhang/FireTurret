# FireTurret — what it is for, and what it is not

Written 2026-08-03, after the work. Where this describes a decision, that
decision is traceable to code or to a test; where it describes a limitation, the
limitation is real and is repeated in [SAFETY.md](SAFETY.md) rather than softened
here.

[TDD.md](TDD.md) · [APP_FLOW.md](APP_FLOW.md) · [DESIGN_BRIEF.md](DESIGN_BRIEF.md) · [SAFETY.md](SAFETY.md) · [ARCHITECTURE.md](ARCHITECTURE.md)

## It looks like a perception problem; it is a range problem

Aiming a water jet at a fire from a camera looks like perception and is really
range. A ballistic projectile needs to know how far to throw; a single camera
measures *direction* only.

Every naive build closes that gap by estimating distance — from the fire's
apparent size, or from where its base meets the ground plane — and then aims
open-loop from the estimate. The estimate is biased, the bias is invisible, and
the water lands consistently short or long with nothing in the system able to
notice.

The concrete symptom is in this repo's own history. An earlier version servoed an
estimated *splash range* onto an estimated *target range*. Both were monocular
estimates, the target estimate carried the larger bias, and the loop converged
confidently onto the wrong place. Removing that was the single most important
change the project made.

## Two gaps, stated rather than designed around

Because everything below is about a machine that throws water, the honest
limitations belong before the achievements.

**There is no water shut-off independent of the microcontroller.** One firmware
hang or one stuck MOSFET defeats the E-stop, the heartbeat watchdog, the homing
lock and the geofence simultaneously. It is the most significant known gap in the
system and it is not designed around.

**The tick-deadline watchdog in `runner.py` is built and tested but not wired
into any running path.** A host degraded from 30 Hz to 4 Hz would feed the
firmware heartbeat perfectly while servoing on quarter-second-old observations
with the valve open. That is listed as a gap rather than a mitigation, and it is
the highest-value piece of remaining work.

Neither of these is softened anywhere else in the repository, and no layer
described below should be read as compensating for them.

## Requirements

**Must**

- Servo in image space only. No control law may consume an absolute range
  estimate. Enforced by design in `control/servo.py`: the range channel drives
  *pump percent* from a pixel-row error, not from metres.
- One engine, three front-ends. The CLI, the browser console and the desktop
  workbench must all drive the same `simcore` code path.
- A rig abstraction with at least three implementations — real serial hardware, a
  physics simulator, and a null rig — swappable without touching control code.
- Water must fail closed at every layer: firmware, wire protocol, water gate,
  guarded rig, mission failsafes. No layer may depend on another being correct.
- The whole system testable with no hardware and no network.
- Honest documentation: every gap recorded in the repository rather than in
  someone's head.

**Should**

- A desktop engineering workbench for parameter exploration and calibration.
- An operator advisory vocabulary that says what to *do* rather than just what is
  wrong, and drives a physical warning output as well as the screen.
- A browser console usable from a phone, for a Raspberry Pi deployment.
- An injectable detector, so a learned ONNX model can replace the classical one.
- Batch parameter sweeps, so a claim about a gain can be checked over a grid
  rather than at one point.

## What the simulator proves

- [x] The system extinguishes a simulated fire whose **true physics differ from
      the controller's model** — discharge coefficient, drag, and a boresight
      bias — which only succeeds if the closed loop corrects the model error
      (`tests/test_end_to_end.py`, slow tier).
- [x] A crosswind and headwind the ballistic model knows nothing about are
      rejected by the loop (`windy` scenario, pinned by a test).
- [x] A static fire-coloured decoy is **never** sprayed (`decoy` scenario).
- [x] A target beyond the jet's reach is recognised and *held*, not hosed
      indefinitely, and raises an operator advisory naming the fix (`unreachable`
      scenario, `control/advisories.py`).
- [x] The entire stack runs with no hardware: `python -m fireturret sim`.
- [x] Exact engine behaviour is pinned by golden regression fixtures, so the CLI,
      the web console and the desktop workbench cannot drift apart.
- [x] Water cannot be commanded on hardware without an explicit `--arm-water`
      *and* a typed `ARM` confirmation, on every entry point including selftest.
- [x] No personal identity appears anywhere except the LICENCE authorship line,
      enforced by `scripts/verify.py`'s `identity` check over the whole git
      history.

Deliberately absent from that list: any accuracy figure in metres, any response
time, any claim about real fires. The system has never run on real hardware, so a
number of that shape would be fiction.

## Architecturally refused

**Measuring distance.** Not deferred — refused. Adding a depth sensor would
remove the problem the project exists to solve. The one place a range estimate
survives, the *opening* shot and the operator advisories, is documented as
inheriting the estimate's bias.

**Being trustworthy around people.** There is a keep-out geofence and an optional
slow person-detector veto, and neither is an interlock. `SAFETY.md` states
plainly that the person detector is poor on seated, prone and occluded people and
useless on animals.

**Unattended operation.** Every entry point assumes a supervisor within reach of
a hardware E-stop.

**A hardened network surface.** The web console has no authentication by design.
It binds to loopback and requires an explicit `--expose` acknowledgement to leave
it. Anything beyond that is a reverse proxy's job, and the README says so rather
than pretending otherwise.

**Automatic agent selection.** Water is the wrong agent for electrical and
grease fires, and the system does not attempt to tell the difference.

**Anything off the machine.** No cloud service, no account system, no telemetry
leaving the box.

**Certification.** None is held, none is sought, and no life-safety claim is
made anywhere in this repository.

**Real-fire performance.** No physical turret exists. Every figure in the
repository comes from simulation or unit tests.

## Who it is for

Three readers, in priority order, and the ordering is what keeps the scope
honest.

The author, as an engineering exercise. The project exists to build and defend a
non-obvious control idea end to end — perception, ballistics, control, embedded
firmware, and the test rig that proves them — rather than to reach a market.

A technical reviewer, an engineer or an admissions tutor, who wants to see
whether the claims survive contact with the code. That reader needs to run the
whole system in one command with no hardware, and needs the failure modes stated
rather than discovered.

A hobbyist who wants to build the physical rig. [BUILD_GUIDE.md](BUILD_GUIDE.md)
and [COMMISSIONING.md](COMMISSIONING.md) exist for that person, gated so that the
first bring-up is dry.

It is not for a building owner looking for fire protection. Nothing here is
certified, and the README, the NOTICE and the safety document all say so in their
first paragraph.

## Camera, network, and the exposed console

The camera is the only sensor and it points at a room. Frames are held in memory
and written to disk only when the operator asks: `--record out.mp4` writes
annotated video, `--log run.csv` writes numeric telemetry and no imagery. Nothing
is uploaded, and `tests/test_scene.py` asserts it by making sockets raise and
running a full pipeline to completion. The single subsystem capable of a network
call is optional scene understanding, which is off by default and refuses to
construct without an explicit data-flow acknowledgement.

On the default configuration only the person at the machine can see any of it.
`fireturret web` binds to `127.0.0.1`; binding anywhere else requires
`--host <addr> --expose`, and `--expose` exists purely so that exposure is a
typed act rather than a default.

There is no account system, so "what happens when access is revoked" resolves to
network reachability — and it has a sharp edge worth stating. The console has no
authentication, and its `POST /estop` **latches** the mission into SAFE with no
rearm path from the browser. On a routable address, anyone who can reach the port
can irreversibly disable the device and watch the camera. That is recorded in
`SAFETY.md` and in the README, and the mitigation offered is a VPN or an
authenticating reverse proxy, not a claim that the console is safe to expose.

The worst outcome, if all of this is wrong, is water aimed at something it should
not be: a person, or an electrical fire it makes worse. The design response is
layered and none of it is a claim of safety — dry-aim by default with a typed
confirmation, a deterministic pan keep-out sector that also withholds water while
the nozzle *sweeps through* the sector and for the water's flight time
afterwards, an AND-chain water gate that vetoes can only ever subtract from, and
a firmware E-stop on its own pin that executes without the host. Read all of that
against the two gaps at the top of this document.

## Designs considered and dropped

| Design | The argument against |
|---|---|
| **Stereo camera or a depth sensor** | Solves the range problem by deleting it. The project's entire thesis is that a single camera plus feedback is sufficient; buying depth would make the interesting part disappear and add a second calibration to drift. |
| **Open-loop ballistics from an estimated range** | This was the original design and it failed exactly as predicted: a biased monocular target-range estimate pushed water off-target with nothing in the loop able to detect it. Replaced by servoing observed splash pixel onto observed fire pixel. |
| **Tilt as the range control** | Under quadratic drag, landing distance peaks near 30° elevation and *falls* on either side, so tilt is non-monotonic in range and a servo on it can converge to the wrong branch. Elevation is held at a fixed low arc and pump pressure carries range instead. |
| **End-to-end learned control** | Unfalsifiable at this scale. There is no real-hardware dataset, and a learned policy would make every failure a data problem rather than a reasoned one. The detector is injectable so a learned *perceptual* model drops in; the control law stays explicit. |
| **ROS 2 as the core architecture** | Adopted as an optional bridge (`ros/translate.py`), refused as the spine. A pure-ROS rig has no independent hardware watchdog: if the graph stalls, nothing on the turret notices. `SAFETY.md` says a ROS-driven build must pair with the firmware E-stop or run dry-aim only. |
| **A safety veto that re-aims** | A layer that commands motion acquires its own failure surface, and would have to project a bounding box through the same monocular range model the architecture distrusts. Vetoes may only subtract water; the gate cannot express the opposite, and `tests/test_gate.py` proves it over the full boolean product. |
| **Borrowing the firmware's 500 ms heartbeat as the veto staleness threshold** | At 5 Hz submission with 120–300 ms inference, a result is already ~500 ms old on arrival, so a 500 ms threshold marks every result stale as it lands and chatters the valve. An unusable safety layer gets switched off, which is worse than a slower one. The threshold is derived from measured latency instead. |
| **Authentication on the web console** | Not built, and not papered over: a half-built auth system on a device that latches E-stop would invite exposure by making it *feel* safe. What ships instead is loopback-by-default, an explicit `--expose` acknowledgement, and a README and SAFETY.md paragraph saying plainly what an exposed console lets a stranger do. |
| **Renaming the Python package during the 2026-08 repo rename** | The package, the CLI, `*.fireturret.json` session files and the ROS frame ids are interfaces, not labels. A cosmetic rename is not worth breaking every import, saved calibration and TF subscriber. See "A note on the name" in the README. |

Since this document is retrospective, the provenance is worth separating. The
*open-loop range*, *ROS-as-spine*, *re-aiming veto*,
*heartbeat-as-staleness-threshold* and *tilt as range control* rows are recorded
decisions — each argued in the code or in `docs/`, and most carrying a test. The
*stereo/depth*, *end-to-end learned control* and *authentication* rows are the
author's reasoning written down here for the first time. They are honest, but
they are not evidence of a documented earlier debate.

## What is still unknown

**Does the image-space loop survive real optics and a real jet?** Unknown — no
physical turret has been built. The simulator's physics are deliberately
perturbed relative to the controller's model, so that "works in sim" cannot be
read as "works", but that is a hedge rather than an answer.

**What wires the deadline watchdog without changing behaviour?** It needs a
command path through `gate()` in `run` and `web`, and an assertion that proves it
is wired. It was left unwired rather than wired blind.

**Is HOG worth keeping at all?** `cv2.HOGDescriptor` was removed in OpenCV 5,
which this repository's own environment runs, so on the current stack the ONNX
detector behind the `[ml]` extra is the only working person detector.
