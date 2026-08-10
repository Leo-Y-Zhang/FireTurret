# Changelog

All notable changes to this project are documented in this file. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

This file starts at the August 2026 rename. Everything before that point was
built under the name **FireTurret**, and that history is kept with the project's private development record rather than
here; back-filling release notes for work that shipped without them would be
writing history after the fact, which is the thing this project's documents are
supposed not to do. The one exception is the summary under *Before this file*
at the bottom, which is a pointer, not a reconstruction.

The project is a supervised demonstrator and has never been tagged for release,
so there are no version headings yet - only *Unreleased*. The first tag will get
one.

## [Unreleased]

### Fixed

- **The native Qt crash in the fast test tier** (2026-08-03). `MainWindow.run()`
  rebound `self._thread`, dropping the last Python reference to the previous
  run's `QThread` while that thread was usually still running: PySide destroys
  the C++ object, Qt calls `qFatal("QThread: Destroyed while thread is still
  running")`, and the process aborts with no Python-level cause. The window is
  wide because `worker.finished -> thread.quit()` is a queued connection into
  the GUI thread, so a second Run reaches the rebinding first. `run()` and
  `closeEvent` now join the previous thread through `_join_run_thread()` before
  anything can release it, and a thread that refuses to stop is parked rather
  than freed. Two regression tests: a deterministic unit test of the join, and
  one that asserts a second run never releases the first run's live thread.
  Measured on Windows 11 / Python 3.13 / PySide6 6.11.1: 4 of 4 fast-tier runs
  died before the change, 4 of 4 completed after it.

### Safety framing

- **The README now opens with what this is not** (2026-08-03). The first
  screenful previously described a working turret and put the demonstrator
  notice below two embedded media items; a reader could reach the build guide
  believing a validated rig exists. A *Read this first* block now states, above
  everything else, that no physical turret has ever been built, that every
  figure comes from the simulator or unit tests, that no certification is held
  or sought, and that the hardware path has never been validated against a real
  turret. *Building the real turret* says the same in place.

### Changed

- **Shortened the repository name from FireSuppressionTurret to FireTurret**
  (2026-08-03). Same reasoning as the rename below, carried one step further: the
  short form says the same thing and survives being read in a URL, a badge and a
  service unit. The entry below is left as written rather than folded into this
  one - `FireSuppressionTurret` was the real name at `dcc1215`, GitHub still
  redirects the old URL, and collapsing two same-day steps into one would make
  the record tidier than the history. Carried through the same surfaces as
  before: README title and CI badge URL, `CONTRIBUTING.md`,
  `NOTICE`, `SECURITY.md`, the nine current documents under `docs/` (including
  the clone command and the systemd unit in `RASPBERRY_PI.md`), the firmware
  header, and two module docstrings. Nothing executable changed.

  The dated design records under `docs/specs/` still say FireTurret and stay that
  way. They are records of what was decided on a given day under the name it
  had; editing them would be back-dating, which is the thing the note at the
  top of this file exists to refuse.

  One thing a rename does break, and it was found broken here: an editable
  install records an **absolute** path, so
  `.venv/Lib/site-packages/__editable__.fireturret-0.1.0.pth` still recorded the
  old `FireSuppressionTurret\src` checkout path after the directory moved, and
  `python -m fireturret --help` failed with `No module named fireturret`. The tracked
  tree was correct; the environment was stale. **Renaming or moving a checkout
  requires `pip install -e .` again** - there is no src-path fallback in
  `pyproject.toml` or a root `conftest.py`, so nothing masks it.

- **Renamed the repository from FireTurret to FireSuppressionTurret** (2026-08-03,
  `dcc1215`). The new name says what the thing is instead of what it was named
  after. The rename was carried through the README,
  `CONTRIBUTING.md`, `NOTICE`, `SECURITY.md`, every file under `docs/`, the
  firmware header, five module docstrings, and the CI badge URL.

### Unchanged (deliberately)

- **The Python package and CLI are still `fireturret`**: `src/fireturret/`,
  `python -m fireturret`, the `fireturret[desktop]` extra. Also unchanged: the
  `FireTurretConfig` dataclass, the `*.fireturret.json` saved-session extension, the
  ROS 2 frame ids (`fireturret/world`, `fireturret/pan`, ...) and joint names, and the
  workbench's application name and window title, "FireTurret Studio".

  Each of those is an interface rather than a label. Renaming the package
  breaks every import in the tree and every third-party plugin entry point;
  renaming the file extension or the frame ids breaks saved calibrations and
  anything already subscribed to the TF tree. A cosmetic rename is not worth a
  compatibility break. The reasoning is set out in full in the README under
  *A note on the name*.

### Known issue

- **The distribution name `fireturret` collides with the `fireturret` package on
  PyPI** - "a language and compiler for custom Deep Learning operations", which
  PyTorch declares as a dependency. Same distribution name, same top-level
  import name, so pip treats the two as one package: installing either
  uninstalls the other, with no warning, and `import fireturret` then resolves to
  whichever survived. `pip install torch` can silently replace this package and
  `pip install -e .` can silently break a working PyTorch install.

  The fix is a per-project virtual environment, which is what this repo's own
  instructions already use. Checked on 2026-08-03 against the PyPI JSON API:
  `fireturret` is at 3.7.1 over 21 releases, all manylinux `x86_64`/`aarch64`, with
  no Windows or macOS wheel ever published - so the collision is a Linux
  concern (a Jetson or Pi deployment running a PyTorch detector), not a
  Windows-workstation one.

  Not being fixed, for the same reason the import path was kept: renaming the
  distribution renames the package. Nothing is ever published to PyPI from
  here - the licence forbids redistribution - so this can never reach a public
  index from this side. Written up in full in the README under *A note on the
  name*, and recorded here because it is a real property of the packaging
  metadata rather than a documentation gap.

### Documentation

- Added the four design documents, written from the code rather than from the
  README: `docs/PRD.md`, `docs/TDD.md`, `docs/APP_FLOW.md`,
  `docs/DESIGN_BRIEF.md`.
- Corrected two counts that had drifted: the README claimed 260 tests and the
  architecture diagram claimed 171 engine tests, against a real suite of 867.
  Both were re-measured from a collection run.
- Added this file, so that the renames above have a changelog record at all.

## Before this file

The full build history is kept with the project's private development record. The shape of it, briefly, so a reader
knows what they are looking at:

- **July 2026, v2 programme** - sixteen sequenced sub-projects (`SP1`-`SP16`)
  covering the determinism substrate and golden tiers, the single water-arming
  gate, the range loop and `RangeTable`, the mobile-platform tier, the ROS 2
  translation layer, the bounded ballistics learner, and the deny-only scene
  understanding that is off by default. The original design note is under
  `docs/specs/`.
- **Late July 2026** - relicensed from Apache-2.0 to proprietary,
  source-available for evaluation. The earlier reasoning is kept in the README
  rather than deleted, because reversing a decision is worth recording.
- **August 2026** - detection and mission fixes (a confirmed fire is engaged
  instead of swept past; confirmation never latches on colour alone), CI
  repairs, and a 60-minute job timeout.

Two things a reader should not infer from a quiet log.

**The Qt crash was narrowed twice before it was fixed.** `14e820f` stopped the
slow tier's nested run from re-executing the studio suite it had already run
minutes earlier, which removed one of two exposures and about 265 seconds from
this repository's most expensive gate - but that changed how often the gate met
the crash, not whether the crash existed. The cause was found on 2026-08-03 and
is recorded under *Fixed* above. `docs/TDD.md` keeps the whole trail, including
a wrong first attempt, because narrowing a gate twice is the more useful lesson.

**Green CI is not a claim about the hardware.** Every automated run is
simulator and headless-Qt only. Nothing in this file or the badge says the water
path, the E-stop or the firmware watchdog has been exercised on a rig. See
`docs/SAFETY.md`.
