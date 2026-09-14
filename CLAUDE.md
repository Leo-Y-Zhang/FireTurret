# FireTurret

A supervised, simulator-only engineering demonstrator for a camera-guided
fire-suppression turret: it detects fire in camera footage it has never seen
and aims a pan/tilt water cannon at it, correcting its own aim by watching
where the water actually lands (closed-loop visual servoing). The same
software drives either real turret firmware over serial or a built-in
physics-and-rendering simulator, so perception, ballistics, and control are
all testable end to end with no hardware. No physical turret has ever been
built or validated; treat this as a controls-and-vision exercise, not a
fire-safety product (see the README's "Read this first" banner).

## Directory layout

- `src/fireturret/` — the package: `vision/` (fire detection), `control/`
  (aim/servo loop), `spray/` (ballistics/water model), `scene/`, `rig/`
  (serial + sim hardware backends), `safety/`, `platform/`, `profiles/`,
  `triage/`, `adapt/`, `ros/` (optional ROS 2 bridge), and `studio/` (the
  PySide6 desktop workbench, with `studio/views` and `studio/models`).
- `tests/` — unit tests mirroring the package, plus `tests/studio/` for the
  desktop UI.
- `scripts/verify.py` — the authoritative local gate (see below).
- `scripts/check_spdx.py`, `scripts/goldens.py` — helpers verify.py calls.
- `docs/` — including `BUILD_GUIDE.md`, `SAFETY.md`, `STUDIO.md`.
- `firmware/` — microcontroller firmware for the real serial rig.

## Install

```
python3.12 -m venv .venv          # any Python >= 3.11 works; the hook picks 3.13, 3.12 or python3
.venv/bin/pip install -e '.[dev,analysis,desktop]'
```

`desktop` is not optional even for headless CI: `dev` pulls in `pytest-qt`,
which refuses to collect *any* test without a Qt binding (from `desktop`).
Also needs system libs for headless Qt/OpenCV: `libegl1 libgl1 libglib2.0-0
libxkbcommon0 libdbus-1-3` (apt). The `[ml]` (onnxruntime) and `[ros]` extras
are optional; tests skip cleanly when absent.

## Lint / format

```
.venv/bin/ruff check .
```

Ruff config (`pyproject.toml`): `line-length = 100`, `target-version = "py311"`,
`select = ["E", "F", "I", "UP", "B"]`, `ignore = ["E501"]` (long lines are
allowed deliberately). No formatter or type-checker (no `ruff format`/`black`,
no mypy) is configured or run in CI.

## Test

Fast tier (default, everyday command — ~10 min in this container):
```
QT_QPA_PLATFORM=offscreen .venv/bin/pytest -q
```
`addopts = "-q -m 'not slow'"` in `pyproject.toml` deselects the slow tier by
default. Fastest useful subset — a single module, e.g.:
```
QT_QPA_PLATFORM=offscreen .venv/bin/pytest -q tests/test_firedetect.py
```
Full suite including the slow/golden/e2e tier (`-m slow`) takes ~30-34 minutes
in CI and was not run in the audit (exceeds the 10-minute time box); run it
only when you have that much time budgeted.

## Verification gate (source of truth)

`scripts/verify.py` is the authoritative local gate — CI (`ci.yml`) is
explicitly documented as a thin wrapper around it, not a second definition of
"green". It runs, in order: ruff lint, the fast pytest suite, the slow/e2e
pytest tier (`-m slow`; skippable with `--fast`), SPDX header checks
(`scripts/check_spdx.py`), a git-identity/anonymity audit of the full commit
history, architectural import-boundary tests, and an "installed import" check
that the editable install is importable by name from outside `src/`. Run
`python scripts/verify.py --fast` for a quicker pass, or the full
`python scripts/verify.py` when time allows (~34 min).

## Environment caveats (from audit)

- Needs `QT_QPA_PLATFORM=offscreen` for any pytest-qt/desktop-studio test to
  run headless; without the apt Qt libs above, PySide6 fails to import at
  collection time (`ImportError: libEGL.so.1`), aborting the whole run before
  a single test executes.
- `verify.py`'s git-identity audit needs full history (`fetch-depth: 0` in
  CI); a shallow clone will make that check behave differently.
- `[ml]` (onnxruntime) and `[ros]` (rclpy, not on PyPI) extras are optional;
  their tests self-skip when the dependency is absent.
- Test suite has grown slightly since the README/badges were last updated
  (871 tests: 823 fast + 48 slow, matching the README after its 2026-09-14 correction) — all
  passing, not a regression.

## CI / conventions

- `ci.yml` installs `[dev,analysis,desktop]` plus the apt Qt/OpenCV libs, sets
  `QT_QPA_PLATFORM=offscreen`, and runs `python scripts/verify.py` as the one
  gate (60-min timeout; gate itself budgeted ~34 min). A separate `gitleaks`
  job scans full history for secrets.
- No coverage floor is enforced; no mypy/type-checking is configured.
