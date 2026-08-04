# Security and incident reporting

## Two channels, because they are two different things

`SECURITY.md` conventionally covers software vulnerabilities. This project also
aims a stream of water at things, so it needs a channel for the case where
somebody gets hit — and that is not a vulnerability report.

Both use **GitHub Private Vulnerability Reporting** on this repository.

There is deliberately **no security email address**. An address would
de-anonymise the maintainer, and this project is maintained anonymously. Private
Vulnerability Reporting provides a confidential channel without one.

## 1. Software vulnerabilities

Use Private Vulnerability Reporting. Please include what you were running, what
you expected, and what happened.

Of particular interest, because these are the paths where a defect has real-world
consequences rather than merely wrong output:

- **Anything that opens the valve when it should not.** The water gate
  (`control/gate.py`) is an AND-chain in which vetoes may only ever subtract
  permission. A path that grants water, or that bypasses the gate entirely, is
  the most serious class of bug this project can have.
- **Anything that reaches the rig without passing `GuardedRig`.**
- **The unauthenticated web console.** It binds to loopback by default and needs
  `--expose` to do otherwise; its `POST /estop` latches SAFE irreversibly from
  the browser. Reports about that surface are welcome — its limitations are known
  and documented, but a way to reach it *without* `--expose` would be a defect.
- **Firmware command parsing.** A malformed line must be dropped without feeding
  the heartbeat watchdog.

## 2. Physical incidents

**If a person or animal has been struck by water, or the turret actuated
unexpectedly, or a failsafe did not fire — report it through the same private
channel, and say so in the first line.**

This is treated as higher priority than any software defect. Please include:

- what the machine was doing (scenario, profile, whether water was armed),
- which failsafes were configured (keep-out sector, E-stop, safety detector),
- what actually happened, and whether anyone was hurt,
- logs or a recording if you have them.

You will not be asked to justify reporting. A report that turns out to be
operator error is still worth having, because "the operator could do that" is
itself a design finding.

## What this project does not claim

FireTurret is a **supervised demonstrator**. It is not a life-safety
device, it is not certified for anything, and **it has never been run on real hardware** — no
physical turret exists. Every claim in this repository comes from simulation or
unit tests.

`docs/SAFETY.md` records the known gaps rather than glossing them. The most
significant: **there is no water shut-off independent of the microcontroller.**
The E-stop, heartbeat watchdog, homing lock and geofence all execute in the same
`loop()` on the same chip, so one firmware hang or one stuck MOSFET defeats all
of them at once. That gap is recorded, not closed.

## Scope

In scope: this repository's source, firmware, and documented behaviour.

Out of scope: third-party dependencies (report those upstream), and any model
weights you supply to the ONNX detector — none are shipped here, and their
provenance is yours.
