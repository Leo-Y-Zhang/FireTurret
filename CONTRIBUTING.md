# Contributing to FireTurret

## Before anything else: this is a supervised demonstrator

This project aims a stream of water using a camera. It is not a life-safety
device, it is not certified, and it has never run on real hardware. Read
[`docs/SAFETY.md`](docs/SAFETY.md) before proposing anything that touches the
control loop, the water gate, or the failsafes — particularly the "known gaps"
section, which is deliberately blunt.

**Never weaken a failsafe to make a test pass.** If a safety test is in the way,
that is the finding.

## DCO, not a CLA

Contributions are accepted under the [Developer Certificate of
Origin](https://developercertificate.org/). Sign off each commit:

```bash
git commit -s -m "your message"
```

which appends `Signed-off-by: Your Name <your@email>`.

**Why DCO rather than a CLA.** A contributor licence agreement requires a named
legal entity to assign rights *to*. This project is maintained anonymously, and
creating that entity would de-anonymise the maintainer — so a CLA is not
available here, and the DCO does the necessary job: you assert you have the right
to contribute what you are contributing, under the project's licence.

## Licence

Proprietary (see [`LICENSE`](LICENSE)). Every source file carries a one-line
identifier:

```python
# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
```

`python scripts/check_spdx.py --fix` adds missing ones, and `verify.py` fails if
any file is missing one. Full-file boilerplate is deliberately not used: the
LICENSE and NOTICE carry the terms, not a thirteen-line appendix on every file.
`LicenseRef-` is the SPDX convention for a licence that is not on the SPDX list,
which a bespoke proprietary licence necessarily is not.

Before adding a dependency, check [`docs/LICENCE_AUDIT.md`](docs/LICENCE_AUDIT.md).
Copyleft dependencies are not automatically disqualified, but they are not free
either — the audit records what PySide6 costs and why. **AGPL is disqualifying**,
and that includes model weights.

## The gate

```bash
pip install -e ".[dev,analysis,desktop]"   # test extras — quoted for zsh
python scripts/verify.py          # everything
python scripts/verify.py --fast   # skips the ~25 minute slow tier
```

That install line is the same one `.github/workflows/ci.yml` runs, and all three
extras are needed for a full local gate: `[dev]` for pytest, pytest-qt and a Qt
binding, `[desktop]` for pyqtgraph (imported by `tests/studio/conftest.py`), and
`[analysis]` only so the two matplotlib tests run rather than skip. On
Debian/Ubuntu, install the OpenCV and Qt system libraries first — see *Run it* in
the README.

`scripts/verify.py` is **authoritative**. CI wraps it rather than defining its
own checks, because two definitions of "green" drift apart until a CI-only
failure appears that nobody can reproduce. If `verify.py` passes locally, it
passes.

Give the slow tier the machine to itself — running it alongside another pytest
gets it killed partway through, which looks like a crash and is not.

## Architecture rules that are enforced, not suggested

`tests/test_import_boundaries.py` fails the build on each of these:

- **`control/suppress.py` may not reference any range-estimate symbol.** The
  suppression loop servos in image space — splash pixel onto fire pixel. Servoing
  on the monocular range estimate is a bug this project has already had, and it
  pushed water off target. A ballistics-derived *timing* or *gain* is fine; a
  target-range estimate is not.
- **The engine may not import the simulator.**
- **`control/` may import only the spray *envelope* and its keep-out**, never the
  droplet ensemble. Control consumes four numbers; whatever computes them stays
  on the far side of that boundary and out of the 33 ms loop.
- **Importing the core must not pull in Qt, ONNX, ROS or matplotlib.** A
  Raspberry Pi install stays small; those live behind extras.

## Golden fixtures

Three tiers, described in `tests/test_invariants.py` and
`tests/test_golden_behavior.py`:

1. **Tier 1** (`tests/test_invariants.py`) — behavioural contracts. **Never
   regenerated.** If one is red, the behaviour regressed.
2. **Tier 2** — metrics with declared tolerances.
3. **Tier 3** — a byte-exact command sequence over 120 frozen frames.

`scripts/goldens.py` is the only generator, it **refuses to run while tier 1 is
red**, and it requires a written `--reason` recorded in
`tests/golden/MANIFEST.json`. There is deliberately no override: a regression
must not be launderable into a fixture.

## Reporting problems

- **Software vulnerabilities** — GitHub Private Vulnerability Reporting on this
  repository. Not a public issue, and not an email address (an address would
  de-anonymise the maintainer).
- **A physical incident** — somebody hit by water, an unexpected actuation, a
  failsafe that did not fire — use the same private reporting channel. It will be
  treated as higher priority than any software defect. `SECURITY.md` covers
  vulnerabilities; this covers people, and the two are not the same thing.
