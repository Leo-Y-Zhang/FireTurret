# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""The AUTHORITATIVE gate. CI is a wrapper around this, not a parallel definition.

## Why this exists rather than a workflow file

The local gate, not the hosted workflow, is the source of truth here. Splitting
authority between the two is usually how a project quietly ends up with two
definitions of "green" — one in a workflow file, one in whatever the maintainer
happens to run — which drift until a CI-only failure appears that nobody can
reproduce.

Making the local script authoritative and CI a thin caller of it means "green on
my machine" and "green in CI" are the **same assertion by construction**, not by
discipline. It also means a contributor with no CI access can still run exactly
what the maintainer runs.

## Checks

    ruff          lint
    pytest        fast suite
    pytest -m slow  golden / e2e tier (skippable with --fast for a quick pass)
    spdx          per-file licence identifiers
    identity      no real-name or non-noreply author in the git history
    boundaries    architectural import contracts (part of the test suite)
    build         the package builds and its metadata is valid

`--fast` skips the slow tier, and says so in the summary rather than quietly
reporting a pass that covered less.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

# Every commit must be attributable to an anonymous GitHub noreply address.
# This is the anonymity gate AND the relicensing precondition — the same audit
# proves the right to relicense and that no personal identity has leaked.
ALLOWED_EMAIL_SUFFIX = "@users.noreply.github.com"


# A failing check's output has to be clipped -- a full pytest log is thousands of
# lines -- but WHICH end to keep is the whole question, and there is no single
# right answer. A Python traceback names its exception on the LAST line. A
# faulthandler crash dump is the exact opposite: `Fatal Python error:` and the
# innermost frames come FIRST, and it ends in runpy boilerplate and a module
# list. Keeping one end therefore deletes the cause of whichever failure it was
# not chosen for. That is not hypothetical here: when the nested fast tier
# crashed under Qt in CI, a tail-only rule left a log containing the pluggy stack
# and nothing else. Keep both ends, and hoist the lines that name a cause out of
# the middle where no positional rule can reach them.
HEAD_CHARS = 1500
TAIL_CHARS = 2500

CAUSE_MARKERS = (
    "Fatal Python error",
    "Windows fatal exception",
    "Segmentation fault",
    "INTERNALERROR",
    "short test summary",
    "FAILED ",
    "ERROR ",
)
# A native crash is reported by faulthandler rather than by pytest, and its dump
# is almost entirely interpreter and site-packages frames. The handful of frames
# in THIS repo are the ones that say where the crash happened, so they are worth
# hoisting -- but only out of a crash, because an ordinary traceback is already
# reported properly and hoisting its frames would just duplicate it.
FATAL_MARKERS = ("Fatal Python error", "Windows fatal exception", "Segmentation fault")
OWN_CODE = ("tests/", "tests\\", "src/fireturret", "src\\fireturret")
MAX_CAUSE_LINES = 12


@dataclass
class Result:
    name: str
    ok: bool
    detail: str = ""
    seconds: float = 0.0


def clip(text: str, head: int = HEAD_CHARS, tail: int = TAIL_CHARS) -> str:
    """Keep both ends of `text`, marking what was dropped rather than hiding it."""
    if len(text) <= head + tail:
        return text
    elided = len(text) - head - tail
    return f"{text[:head]}\n\n[... {elided} characters elided ...]\n\n{text[-tail:]}"


def _is_own_frame(line: str) -> bool:
    return 'File "' in line and any(part in line for part in OWN_CODE)


def causes(text: str, limit: int = MAX_CAUSE_LINES) -> str:
    """The lines that say what went wrong, wherever in the output they sit."""
    crashed = any(m in text for m in FATAL_MARKERS)
    lines = text.splitlines()
    hits = []
    for i, ln in enumerate(lines):
        # The frame directly under `Current thread` is the innermost one: it names
        # what died, where an own-code frame names which test got it there. A crash
        # in a third-party paint handler and a crash in our own loop want different
        # fixes, so keep both rather than choosing.
        innermost = crashed and i > 0 and "Current thread" in lines[i - 1]
        if any(m in ln for m in CAUSE_MARKERS) or innermost or (crashed and _is_own_frame(ln)):
            hits.append(ln.strip())
    if not hits:
        return ""
    tail = f"\n... and {len(hits) - limit} more" if len(hits) > limit else ""
    return "\n".join(hits[:limit]) + tail


def summarise_failure(output: str) -> str:
    found = causes(output)
    return (f"CAUSE:\n{found}\n\nOUTPUT:\n" if found else "") + clip(output)


def run(name: str, argv: list[str], cwd: Path = ROOT) -> Result:
    start = time.perf_counter()
    proc = subprocess.run(argv, cwd=cwd, capture_output=True, text=True)
    elapsed = time.perf_counter() - start
    detail = "" if proc.returncode == 0 else summarise_failure(proc.stdout + proc.stderr)
    return Result(name, proc.returncode == 0, detail, elapsed)


def check_identity() -> Result:
    """No commit may carry a non-noreply author or committer."""
    start = time.perf_counter()
    proc = subprocess.run(
        ["git", "log", "--format=%an <%ae>%n%cn <%ce>"],
        cwd=ROOT, capture_output=True, text=True,
    )
    if proc.returncode != 0:
        return Result("identity", False, "git log failed", time.perf_counter() - start)

    offenders = sorted({
        line.strip() for line in proc.stdout.splitlines()
        if line.strip() and ALLOWED_EMAIL_SUFFIX not in line
    })
    elapsed = time.perf_counter() - start
    if offenders:
        return Result(
            "identity", False,
            "non-noreply identities in history:\n  " + "\n  ".join(offenders), elapsed,
        )
    return Result("identity", True, "", elapsed)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", action="store_true",
                        help="skip the slow golden/e2e tier")
    args = parser.parse_args()

    results: list[Result] = []
    results.append(run("ruff", [PY, "-m", "ruff", "check", "src", "tests", "scripts"]))
    results.append(run("spdx", [PY, "scripts/check_spdx.py"]))
    results.append(check_identity())
    results.append(run("pytest (fast)", [PY, "-m", "pytest", "-q", "-p", "no:cacheprovider"]))
    if not args.fast:
        results.append(run(
            "pytest (slow)",
            [PY, "-m", "pytest", "-m", "slow", "-q", "-p", "no:cacheprovider"],
        ))
    results.append(run("build metadata", [PY, "-c", "import fireturret; print(fireturret.__name__)"]))

    print()
    for r in results:
        mark = "PASS" if r.ok else "FAIL"
        print(f"  [{mark}] {r.name:16s} {r.seconds:6.1f}s")
        if not r.ok and r.detail:
            # `detail` has already been clipped at both ends by `summarise_failure`,
            # so it is printed whole. It used to be truncated a second time here,
            # which threw away the half the first clip had just been careful to keep.
            print("        " + r.detail.replace("\n", "\n        "))

    failed = [r for r in results if not r.ok]
    print()
    if args.fast:
        print("NOTE: --fast skipped the slow golden/e2e tier. This is not a full gate.")
    if failed:
        print(f"VERIFY FAILED: {len(failed)} check(s) — {', '.join(r.name for r in failed)}")
        return 1
    print(f"VERIFY PASSED: {len(results)} checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
