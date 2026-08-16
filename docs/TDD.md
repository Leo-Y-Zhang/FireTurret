# TDD — FireTurret

The as-built technical design: contracts, data model, failure modes and undo.
[ARCHITECTURE.md](ARCHITECTURE.md) explains *why* the control is shaped this way;
[PRD.md](PRD.md) says what the thing is for.

Derived from the code, not from the README. Every module path below is real, and
every property attributed to a test names the test file.

## One tick, one command

`Pipeline.tick(frame, dt, telemetry) -> RigCommand` is the whole system for one
frame: detect fire → track → observe the water splash → advance a mission state
machine → emit one actuator command. Above it sits nothing but I/O. Below it sits
a single `TurretRig` protocol with three implementations — serial hardware,
physics simulator, null. Because the boundary is that narrow, the same code path
is exercised by the CLI, the browser console and the desktop workbench, and the
whole stack runs with no hardware.

The control loop is servoed in **image space**. Azimuth drives the pan angle from
the splash-to-fire *column* gap; range drives **pump percent**, not tilt, from
the splash-to-fire *row* gap. No control law consumes an absolute range estimate.

There is no database, no server-side state and no user accounts.

## Water is denied five times, independently

Firmware, wire protocol, water gate, guarded rig, mission failsafes — and no
layer relies on another being correct. Three of those five are worth specifying
in detail.

### The water gate — `control/gate.py`

```python
gate.apply(cmd, ctx) -> (RigCommand, tuple[str, ...])   # command, deny reasons
```

One AND-chain that every water-carrying command passes. The contract is the
design.

**Deny-only.** `apply` starts from the command it was given and can only *remove*
water. There is no expressible path from denial to allowance, and
`tests/test_gate.py` proves it exhaustively over the boolean product of every
veto.

**Water-only.** The gate never touches `pan_deg` or `tilt_deg`. A safety layer
that commands motion acquires its own failure surface, and re-aiming belongs to
the mission, whose next command is gated anyway.

**Failure is denial.** A veto that raises denies; a veto whose answer is stale
denies; "I don't know" is never "yes".

Standard vetoes are `ArmedVeto`, `HomedVeto`, `LinkVeto` and `SprayTimeVeto`. The
spray backstop trips at `SPRAY_BACKSTOP_FACTOR = 1.5 ×` the mission's own
`max_spray_s`, deliberately *above* it, so in normal operation the mission always
re-evaluates first and the gate stays invisible — which is what keeps the gate
behaviour-neutral for every existing golden fixture.

This module exists because there were previously **three** independently-written
gates — `MissionController.update`, `app.run_capture`,
`webserver.rig_command_for_hardware` — each enforcing a slightly different
subset, with a fourth about to be added.

### Asynchronous safety vetoes — `safety/veto.py`

A person detector costs 120–300 ms per inference on a Raspberry Pi 5 CPU; the
control loop runs at 33 ms. The predicate cannot run on the loop, and the loop
must act on an answer that is always somewhat out of date. `AsyncVeto` is the
plumbing for that, built and proven *before* any detector existed.

| Situation | Result |
|---|---|
| answer says allow, and is fresh | allow |
| answer says deny | **deny, and latch until an operator clears it** |
| answer older than `max_age_s` | **deny** |
| predicate raised | **deny** |
| no answer has ever arrived | **deny** |

`max_age_s` comes from `derive_max_age_s()` over *measured* latency, not from the
firmware's 500 ms heartbeat. Borrowing that number would mark results stale on
arrival and chatter the valve at the detector's beat frequency.

### The guarded rig — `rig/guarded.py`

`GuardedRig` applies the gate at the **last point before hardware**, so a path
that never constructs a `Pipeline` — the desktop workbench, a manual jog, a
future plugin — still cannot spray. It is constructed **disarmed**, so forgetting
to arm fails toward dry.

## The rig boundary and the wire protocol

```python
class TurretRig(Protocol):
    def command(self, cmd: RigCommand) -> None: ...   # also the heartbeat
    def telemetry(self) -> RigTelemetry: ...
    def close(self) -> None: ...
```

`RigCommand(pan_deg, tilt_deg, pump_pct, valve, laser=False, warn=False)` and
`RigTelemetry(pan_deg, tilt_deg, pump_pct, valve, estop, ok, homed=False)` are
both frozen. `command()` doubling as the heartbeat is a contract rather than an
optimisation: the firmware failsafes when commands stop arriving, so a caller
that stops sending is *already* the safe case.

Implementations are `SerialRig` for real hardware, `SimRig` for physics plus a
rendered camera, `NullRig` echoing commands as instantly-achieved telemetry for
dry runs over recorded video where no actuator could affect the footage anyway,
and `GuardedRig`, the decorator above.

```
host → firmware   C p=<pan> t=<tilt> w=<pump%> v=<0|1> l=<0|1> x=<0|1>\n
firmware → host   S p=<pan> t=<tilt> w=<pump%> v=<0|1> e=<0|1> h=<0|1>\n
```

`parse_telemetry` returns `None` for any line missing a required field rather
than a partially-populated record — and a refused line does **not** feed the
heartbeat watchdog, so a corrupted "close the valve" lets the watchdog fire
instead of leaving water flowing.

One non-obvious constraint, pinned by `tests/test_safety_debt.py`: **every
host→firmware token must have a unique first character.** Deployed boards
dispatch on `tok[0]` alone, so a new token sharing a first letter silently
overwrites an existing field.

## Plugins, engine and front-ends

`plugins.discover()` reads `importlib.metadata` **metadata only** and never
imports a plugin module. Enumerating what is installed must not execute
third-party code, so merely having a package installed cannot get its code run.
The module docstring names a `fireturret --list-plugins` flag; **no such flag exists
in `build_parser`** — the property is real, the command it cites is not.

`simcore.simulate()` is a pure per-tick generator and `simcore.drive()` is the
shared accumulation and stop-policy loop, with `app.run_sim` a thin wrapper
adding overlay and recording. Studio's `SimWorker` runs `drive` on a worker
thread and communicates **only** via Qt signals: a `SimStep` holds a mutable
`TickResult` and a raw ndarray and must be consumed in-thread, so only the frozen
`TelemetrySample` and a copied frame may cross the boundary.

## Configuration is the data model

What would be a schema in a web service is here a frozen configuration dataclass
tree plus the on-disk formats derived from it.

`FireTurretConfig` (`src/fireturret/config.py`) is a frozen dataclass of six frozen
sub-configs. Frozen matters: every edit goes through `dataclasses.replace`, which
re-runs `__post_init__` validation, so an invalid value raises instead of being
stored.

| Group | Holds | Validation in `__post_init__` |
|---|---|---|
| `camera` | `width`, `height`, `hfov_deg`, `mount_height_m`, `mount_pitch_deg`, `boresight_offset_deg` | positive dimensions; `0 < hfov_deg < 180`; positive mount height |
| `turret` | pan/tilt limits, slew rates, `pan_keepout_deg: tuple \| None` | `pan_min < pan_max`; `0 <= tilt_min < tilt_max <= 90`; positive rates; **keep-out must be an ascending sub-range of the pan limits** |
| `jet` | `max_pressure_psi`, `velocity_coeff`, `drag_k`, `nozzle_height_m`, `min_pump_pct` | — (the first three are what `fireturret fit` refines) |
| `detector` | colour/flicker thresholds, tracker association distance | — |
| `servo` | `suppress_tilt_deg`, pan gain/deadband, `range_gain_pct_per_px`, `range_deadband_px`, `pump_step_pct`, `settle_frames` | — |
| `mission` | timeouts, `max_spray_s`, `soak_s`, `heartbeat_timeout_s`, `default_range_m`, `unreachable_cycles`, `hold_retry_s` | — |

A misconfigured turret is a *safety* issue, since the keep-out sector is an aim
geofence, so `TurretConfig` fails loudly rather than clamping.

There are no database migrations, and the forward-compatibility work sits here
instead, following the same rule migrations follow: additive first, and never let
an old artefact become dangerous by being old.

`SCHEMA_VERSION = 1` is written into every serialised config, and two rules make
old and new files interoperate without a migration step. **Reading a newer schema
warns but still loads**, degrading to the fields this version understands rather
than crashing. And **unknown top-level groups are carried through a load/save
round trip untouched** (`_EXTRA_GROUPS_ATTR`), because third-party plugin config
lives in such groups and silently dropping them would destroy a user's settings
on every save.

Built-in group names are registered with `configschema` so a plugin cannot shadow
them. That is a safety property rather than tidiness: a plugin able to redefine
`turret` could redefine the geofence.

| Format | Written by | Shape | Compatibility rule |
|---|---|---|---|
| Config JSON (`turret.json`) | `config.save_config`, Studio export | `{schema_version, camera:{…}, turret:{…}, …}` | interchangeable with a session file, which nests the same object under `"config"` |
| Session (`*.fireturret.json`) | Studio `File ▸ Save As` | `{config:{…}, scenario:{…}}` | the config half loads directly into the CLI via `--config` |
| Telemetry CSV (`--log`) | `analysis.TelemetrySample` | 14 columns, one row per tick | append-only record; no imagery |
| Run export | Studio runs table | `<name>.csv` + `<name>.json` sidecar | sidecar carries the metadata the CSV cannot |

The wire-protocol equivalent of a nullable column is a **missing token**, and it
has already happened once: `RigTelemetry.homed` was added after firmware existed
that did not send `h=`. Its default is `False`, and the comment in
`rig/interface.py` states why — this field gates water, so an absent value must
fail closed. A firmware binary predating the field is therefore treated as *not
homed* and gets no water, which is the correct degradation.

## No authentication anywhere

That is a decision rather than an omission. What exists instead:

| Surface | Who can reach it | Control |
|---|---|---|
| CLI | anyone with a shell on the host | OS file permissions only |
| Serial link | anyone with physical access to the USB port | none in software; the firmware E-stop is the physical control |
| Web console `GET /`, `/status`, `/stream` | anyone who can reach the bind address | **loopback by default**; `--host` + `--expose` required to leave it |
| Web console `POST /estop` | same | **unauthenticated, and it latches** — no rearm path exists from the browser |

With no accounts, "revoked access" means "can no longer reach the port". There is
no session to invalidate and no token to rotate. The consequence is stated in
`SAFETY.md` and in the README rather than mitigated: on a routable address,
anyone who can reach the console can watch the camera and irreversibly disable a
fire-suppression device. The recommended deployment is a VPN or an authenticating
reverse proxy in front.

The one privilege enforced in software is water, and the five layers that enforce
it are specified above.

## Where it fails, and what fails safe

| What breaks | Who notices | How it is detected | How it is undone |
|---|---|---|---|
| Host process crashes or hangs | firmware | commands stop arriving; 500 ms heartbeat watchdog | firmware cuts water and holds; restart the host, re-home |
| Serial line corrupts a command | firmware | a line missing any field is refused outright, and refusal does **not** feed the watchdog | watchdog fires as if the host were gone |
| Host degrades from 30 Hz to 4 Hz | ⚠ **nobody** | `TickWatchdog` exists in `runner.py` but is **not wired into any running path** | no automatic response today — recorded gap, not a mitigation |
| Telemetry goes stale (`ok=False`) | `LinkVeto` | gate denies water; mission latches SAFE | operator clears the fault; SAFE does **not** auto-resume |
| E-stop pressed | firmware and host | `estop=1` in telemetry; mission → SAFE | physical release, then an explicit rearm |
| Pan axis not homed | `HomedVeto` + firmware homing lock | `h=0`, or the field absent | run homing; absent field is treated as not-homed by design |
| Splash never observed (obstructed view) | mission | `SUPP_NO_FEEDBACK_CYCLES` cycles with no observation | `OBSTRUCTED` advisory: "clear the line of fire or reposition"; warning output asserted |
| Target beyond jet reach | mission | pump saturated and still short for `unreachable_cycles` | `TOO FAR` advisory with a distance to move; HOLD with backoff, retry after `hold_retry_s` |
| Fire-coloured static object (decoy) | detector | temporal flicker gate rejects it | never enters ACQUIRE; pinned by the `decoy` scenario |
| Spray runs on too long | mission, then gate | `max_spray_s`, then the gate's 1.5× backstop | solution recomputed; backstop only fires for a controller that ignores the mission |
| Nozzle slews across the keep-out sector | mission | `swept_intersects_keepout` on the commanded transit | water withheld for the transit **plus** worst-case flight time + `KEEPOUT_HOLD_MARGIN_S` |
| Person detector unavailable (OpenCV 5 removed `HOGDescriptor`) | construction time | `HogPersonDetector` **raises** rather than constructing | use the ONNX detector (`[ml]` extra); a detector that silently found nothing would be indistinguishable from a clear scene |
| Config file from a newer version | loader | `schema_version` comparison | warns and loads the fields it understands; unknown groups preserved |
| Qt import fails (no `[desktop]` extra) | CLI | guarded import in `__main__` | `fireturret studio --help` still works; the core CLI never imports Qt |

## Backing a change out

**Software.** Nothing is deployed and nothing is stateful, so rollback is
`git checkout <sha>` plus `pip install -e .`, under a minute. Behaviour is pinned
by golden regression fixtures, which makes a rollback *verifiable*: run
`python scripts/verify.py` and the golden tier either matches the fixtures or it
does not.

**Golden fixtures** are the one artefact where undo needs care. Regenerating them
makes the tests agree with whatever the code now does, which is exactly the
failure mode a golden suite exists to prevent. The discipline is to **budget
regenerations explicitly** — the v2 programme budgeted three, spent them, and a
fourth beyond that budget was argued for in writing before it was spent; every
regeneration's reason is recorded in `tests/golden/MANIFEST.json`. A
regeneration is reviewed as a behaviour change, because that is what it is.

**Calibration.** `--config turret.json` keeps a build's calibration in a file
rather than in source, so reverting a bad calibration is restoring a file: no
code change, no reflash.

**Firmware.** Reflashing is the rollback, and it is not instant. The protocol's
unique-first-character rule exists precisely because a deployed board may be
older than the host, so the host must stay compatible with firmware it cannot
update.

**Irreversible, and accepted:** water that has already left the nozzle. Every
timing margin in the keep-out logic — flight time plus `KEEPOUT_HOLD_MARGIN_S` —
exists because that is the one action with no undo.

## One gate, and everything it runs

`python scripts/verify.py` is **the** gate, and the GitHub Actions workflow is a
thin wrapper that calls it, deliberately, so "green locally" and "green in CI"
are the same assertion by construction rather than by discipline. It runs `ruff`,
fast pytest, slow pytest (`-m slow`), SPDX headers, git-history identity, and
build metadata.

**Positive.** `tests/test_end_to_end.py` extinguishes a fire whose true discharge
coefficient, drag and boresight differ from the model. It passes only because the
loop corrects model error.

**Negative.** `tests/test_gate.py`: no combination of vetoes turns a denial into
an allowance, over the full boolean product. `tests/test_veto.py`: stale, missing
and raised answers all deny. And the `decoy` scenario: a static fire-coloured
object is never sprayed.

**Boundary.** `tests/test_swept_path.py`: the keep-out interval is *closed* for a
swept transit but *open* for a single aim angle, because `_apply_keepout`
resolves a forbidden command by clamping **to** the edge, and a predicate calling
the edge "inside" would declare the geofence's own output a violation. Separately,
telemetry with `h=` absent parses as `homed=False`.

**Legacy.** `tests/test_profile_identity.py`: the `FireProfile` abstraction is
bit-identical to the pre-abstraction path. An abstraction whose flagship case is
not provably identical to the code it replaced has not been extracted.

**Architectural.** `tests/test_import_boundaries.py`: the engine must not import
the simulator. `tests/test_scene.py`: a full pipeline completes with sockets made
to raise.

**Anonymity.** `scripts/verify.py`'s `identity` check audits the *entire* author
history for non-noreply addresses, which is why CI checks out with
`fetch-depth: 0`.

## How the build actually ran

1. Geometry and ballistics with tests, before any vision.
2. Classical fire detection and tracking against rendered frames.
3. The rig protocol and the simulator, so control had something to drive.
4. Mission state machine and the two servo loops; the end-to-end test.
5. Firmware, and the serial rig behind the same protocol.
6. Golden fixtures, to lock the behaviour before extracting `simcore`.
7. `simcore` extraction — mechanical, guarded by the goldens.
8. Studio on top of `simcore`; web console alongside.
9. Safety consolidation: one water gate, the guarded rig, the async veto.
10. Licensing, anonymity audit, and the verify gate CI wraps.

## The gaps that are still gaps

**Wiring `TickWatchdog`** into `run` and `web` without perturbing golden
behaviour, plus an assertion that proves it is wired. Currently the largest gap
between what `SAFETY.md` describes and what executes.

**Whether the HOG person detector should be removed outright** now that OpenCV 5
has deleted `cv2.HOGDescriptor`, leaving ONNX as the only path.

*No longer a gap: `[project] name` was `triton`, which collided with a
widely-installed PyPI distribution of that name, so a user who installed both
into one environment got whichever landed last. The 2026-08-11 package rename to
`fireturret` ended it.*

## The Qt crash: found, and what it actually was

Kept in full, because two earlier attempts narrowed a gate instead of fixing the
code and the record of that is more useful than a tidy note.

**The symptom.** Through 2026-08-03 the *fast* tier died natively — on Windows 11
/ Python 3.13 / PySide6 6.11.1 after exactly 26 passing tests and about 18
seconds, with no traceback and no faulthandler dump at all, and in CI as a
bare `Fatal Python error: Aborted`. Commit
`14e820f` had earlier stopped the *slow* tier re-running the studio suite
nested; that removed one exposure and nothing else.

**The cause.** The CI dump, once the frame hoisting in `verify.py` surfaced it,
named `src/fireturret/studio/main_window.py` line 447 inside `run` — the
`self._thread = thread` rebinding, reached from
`tests/studio/test_main_window.py::test_replay_and_compare_runs`, the only test
that calls `run()` twice on one window. That assignment drops the last Python
reference to the *previous* run's `QThread`, so PySide destroys the C++ object.
`worker.finished -> thread.quit()` is a queued connection into the GUI thread, so
after `run_finished` the old thread has usually not been told to stop yet, let
alone stopped. Destroying a running `QThread` makes Qt call
`qFatal("QThread: Destroyed while thread is still running")`, which calls
`abort()` — hence an abort with no Python-level cause, and hence its dependence
on timing rather than on any one test.

**The fix.** `MainWindow._join_run_thread()` stops and joins the previous thread
before anything can release it, called from `run()` before the new thread is
built and unconditionally from `closeEvent` (a window closed after a finished run
drops the same reference). A thread that will not stop within
`_THREAD_EXIT_MS` is parked in `_stuck_threads` rather than freed: a bounded leak
is a visible bug, a freed running thread is a dead process.

**What it is not.** It is *not* the documented pytest-qt/`QApplication` teardown
family: `tests/studio/conftest.py` already forces `QT_QPA_PLATFORM=offscreen`
before any Qt import and pytest-qt's session-scoped `qapp` already gives exactly
one `QApplication`. Neither was the missing piece.

**Evidence, measured on this machine, fast tier under `pytest -q -p
no:cacheprovider`:** before the fix, 4 runs out of 4 died without ever printing a
test summary; the one captured to a file was 26 bytes long — 26 dots, nothing
else. After the fix, 4 runs out of 4 reached 100% and exited 0;
the studio suite alone ran 3 further times clean. An intermediate attempt that
*also* nulled `_thread`/`_worker` in `closeEvent` turned the abort into an access
violation — freeing a `QObject` that still belongs to the retiring thread — which
is why the join deliberately keeps both references until their replacements are
bound.

**Still open.** CI has not yet run the fix; a green Linux run is confirmation
this project has not got. And a native crash of this shape can only ever be shown
absent over a sample, never proven gone.
