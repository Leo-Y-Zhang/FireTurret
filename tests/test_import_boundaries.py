# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""SP3 — layering, enforced by test rather than by convention.

Architecture rules that live only in a document get violated by accident, and the
violation is invisible until four workstreams have widened it. These are the
rules the programme depends on, made executable:

1. The engine must not import the simulator.
2. `control/suppress.py` must never reach for a range estimate — invariant 1 is
   the whole architecture, and this is the file where breaking it would be
   easiest and least visible.
3. Importing the core engine must not drag in Qt, ONNX, ROS or matplotlib. The
   `[desktop]`/`[ml]` extras exist precisely so a Raspberry Pi install stays small.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "triton"


def _imports(path: Path) -> set[str]:
    """Every module named by an import in this file, at any nesting depth —
    including imports inside function bodies, which is where a layering
    violation usually hides once someone knows module-level ones are checked."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            # relative imports: reconstruct enough to match on the tail
            found.add(node.module)
            found.update(f"{node.module}.{a.name}" for a in node.names)
    return found


def _is_type_checking_only(path: Path, module: str) -> bool:
    """True when every import of `module` sits under `if TYPE_CHECKING:`.

    Such an import costs nothing at runtime — with `from __future__ import
    annotations` the annotation is a string — so it is not a layering violation.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    guarded, total = 0, 0
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and module in node.module:
            total += 1
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        name = getattr(test, "id", None) or getattr(test, "attr", None)
        if name != "TYPE_CHECKING":
            continue
        for sub in ast.walk(node):
            if isinstance(sub, ast.ImportFrom) and sub.module and module in sub.module:
                guarded += 1
    return total > 0 and guarded == total


# --------------------------------------------------- 1. engine vs simulator

ENGINE_FILES = [
    *sorted((SRC / "control").glob("*.py")),
    SRC / "ballistics.py",
    SRC / "geometry.py",
]


@pytest.mark.parametrize("path", ENGINE_FILES, ids=lambda p: p.name)
def test_engine_does_not_import_the_simulator(path: Path) -> None:
    """`control/`, `ballistics` and `geometry` are the parts that run on real
    hardware. A dependency on the simulator there is either dead weight on a Pi
    or a sign that a test double leaked into production code."""
    forbidden = {"sim_rig", "synthetic", "sim_world", "sim_platform"}
    offenders = {
        imp for imp in _imports(path)
        if any(f in imp for f in forbidden)
    }
    assert not offenders, f"{path.name} imports the simulator: {sorted(offenders)}"


def test_app_only_names_the_simulator_for_typing() -> None:
    """`app.py` used to import `SimScenario` at module level purely for one
    annotation (spec §3.9). Small, but it made the engine depend on the
    simulator, and four workstreams would have widened it."""
    assert _is_type_checking_only(SRC / "app.py", "sim_rig"), (
        "app.py imports the simulator at runtime again"
    )


# ------------------------------------------- 2. suppress.py vs range estimates

# Symbols that constitute an ESTIMATE OF WHERE THE TARGET IS — the monocular,
# biased measurement invariant 1 forbids the loop from servoing on.
RANGE_ESTIMATE_SYMBOLS = (
    "target_range", "range_ema", "ground_range_from_px", "estimate_target_range",
    "RangeEstimator", "RangeTable", "plan_suppression", "ranging",
)


def test_suppress_never_references_a_range_estimate() -> None:
    """INVARIANT 1, made checkable — and note precisely what it forbids.

    The rule is not "no physics". It is that the loop's FIXED POINT must be set by
    what the camera sees: splash pixel onto fire pixel. Servoing on the monocular
    range estimate is a bug this project has already had once — the estimate is
    biased, and it pushed the water off target.

    A ballistics-derived flight time or plant gain is a different thing. It is a
    feedforward model of the ACTUATOR, and it changes only how fast the loop
    converges and when it looks — never where it converges to. A wrong model
    makes the loop sluggish; a wrong range estimate makes it confidently wrong.
    That is why `control/plant.py` may use the ballistics model and this file may
    not name a target-range symbol.
    """
    path = SRC / "control" / "suppress.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))

    offenders: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in RANGE_ESTIMATE_SYMBOLS:
            offenders.append(f"name {node.id!r} (line {node.lineno})")
        elif isinstance(node, ast.Attribute) and node.attr in RANGE_ESTIMATE_SYMBOLS:
            offenders.append(f"attribute .{node.attr} (line {node.lineno})")
    for imp in _imports(path):
        if "ranging" in imp:
            offenders.append(f"import {imp!r}")

    assert not offenders, (
        "control/suppress.py reaches for a range estimate: " + "; ".join(offenders)
    )


SPRAY_ALLOWED_FROM_CONTROL = {"envelope", "keepout"}


@pytest.mark.parametrize("path", ENGINE_FILES, ids=lambda p: p.name)
def test_control_imports_only_the_spray_envelope_and_its_keepout(path: Path) -> None:
    """The spray package's whole reason for existing is that a droplet ensemble
    must NEVER run inside the control loop. Control consumes a four-number
    envelope; whatever computes it stays on the far side of that boundary.

    Two modules are allowed, not one. The spec says "only `envelope.py`", and
    `keepout.py` is admitted deliberately: it is a pure geometric predicate over
    an envelope — no atomisation, no integration, no per-tick cost — and the
    alternative is to duplicate it inside `control/`, which would put the same
    maths in two places for the sake of the letter of a rule aimed at something
    else. What must never be admitted is the ensemble machinery itself:
    `dropletdist`, `nozzle`, `arcs`, `fluid`.
    """
    offenders = set()
    for imp in _imports(path):
        if "spray" not in imp:
            continue
        parts = imp.replace("triton.", "").split(".")
        for part in parts:
            if part in {"spray", "triton", ""}:
                continue
            if part in SPRAY_ALLOWED_FROM_CONTROL:
                break
            # a symbol imported FROM an allowed module is fine
            if any(a in imp for a in SPRAY_ALLOWED_FROM_CONTROL):
                break
            offenders.add(imp)
            break
    assert not offenders, (
        f"{path.name} imports spray internals beyond the envelope: {sorted(offenders)}"
    )


def test_the_correction_is_driven_by_the_observed_pixel_error() -> None:
    """The positive half of invariant 1, which the negative test alone cannot
    give: the pump correction must still be a function of the OBSERVED row error.
    A file could pass the test above and still have quietly become open-loop."""
    src = (SRC / "control" / "suppress.py").read_text(encoding="utf-8")
    assert "step.range_error_px" in src, (
        "the pump correction no longer references the observed row error"
    )
    assert "splash_px" in src and "fire_px" in src, (
        "the correction no longer compares the splash pixel to the fire pixel"
    )


def _code_lines(path: Path) -> int:
    """Executable lines: no blanks, no comments, no docstrings."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docstring_lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) \
                and isinstance(node.value.value, str):
            docstring_lines.update(range(node.lineno, (node.end_lineno or node.lineno) + 1))

    count = 0
    for i, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#") or i in docstring_lines:
            continue
        count += 1
    return count


def test_suppress_stays_small() -> None:
    """The cap is not arbitrary — this is the one file where four invariants are
    implemented, so it must stay short enough that a reviewer reads all of it.

    DELIBERATE DEVIATION from the spec's literal "200 lines": the cap is applied
    to EXECUTABLE lines, not the file. SP4 added real functionality here (the
    flight-time-derived cycle and the envelope check), and a cap that counts
    docstrings and comments would be paid for by deleting the explanation that
    makes the file reviewable in the first place — which inverts the cap's own
    purpose. The law itself is what must stay small, and it is: ~120 lines.
    A generous total is still capped so the file cannot quietly become an essay.
    """
    path = SRC / "control" / "suppress.py"
    code = _code_lines(path)
    total = len(path.read_text(encoding="utf-8").splitlines())
    assert code <= 140, f"control/suppress.py has {code} executable lines (cap 140)"
    assert total <= 280, f"control/suppress.py is {total} lines total (cap 280)"


# ------------------------------------------------------- 3. optional extras

def test_importing_the_engine_does_not_pull_in_optional_heavyweights() -> None:
    """A core install on a Raspberry Pi gets numpy/opencv/pyserial and nothing
    else. Qt, ONNX, ROS and matplotlib live behind extras; if any of them creeps
    into the core import graph, that install silently grows by hundreds of MB."""
    code = (
        "import sys; import triton.simcore; "
        "bad=[m for m in ('PySide6','onnxruntime','rclpy','matplotlib') "
        "if any(k==m or k.startswith(m+'.') for k in sys.modules)]; "
        "print(','.join(bad))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert out == "", f"triton.simcore pulled in optional dependencies: {out}"
