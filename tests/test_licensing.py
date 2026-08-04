# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""SP13 — licence, contributor readiness, and the gates that keep them true.

The point of testing this is that licence hygiene decays silently. Nobody
notices a missing SPDX header, a CI job that quietly checks less than the local
gate, or a personal email address in one commit — until the moment those things
matter, which is exactly the moment they cannot be fixed retroactively.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOREPLY = "@users.noreply.github.com"


# ------------------------------------------------------------------- licence

def test_the_licence_grants_no_reuse_rights() -> None:
    """Triton is proprietary and source-available: readable for evaluation, not
    reusable.

    This replaces a test that asserted Apache-2.0 and its patent grant. That
    choice was deliberate at the time -- a patent grant is worth having on a
    hardware/control project -- and was given up deliberately too, because
    Apache-2.0 grants commercial reuse that is not intended for a portfolio
    piece. The reasoning is preserved in README.md rather than deleted.

    What is pinned here is the part that would be dangerous to lose silently: an
    accidental revert to a permissive licence would hand away reuse rights
    without anyone noticing, so the absence of a permissive grant is asserted
    explicitly rather than assumed."""
    text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert "no reuse rights granted" in text.lower()
    assert "Grant of Patent License" not in text, "permissive licence reintroduced"
    assert "Apache License" not in text, "permissive licence reintroduced"
    for forbidden in ("modify", "distribute", "training data"):
        assert forbidden in text.lower(), f"restriction on {forbidden} missing"


def copyright_holder() -> str:
    """Read the holder out of LICENSE, asserting the shape on the way through.

    This is a plain helper rather than a test that returns its finding, because
    a test used as a fixture is one pytest release away from breaking: returning
    anything but None already warns, and is documented to become an error. The
    two tests below keep their own names and their own reasons to fail.
    """
    text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    m = re.search(r"Copyright \(c\) (\d{4}) (.+?)\. All rights reserved", text)
    assert m, "LICENSE needs a 'Copyright (c) <year> <holder>. All rights reserved' line"
    holder = m.group(2).strip()
    assert len(holder) > 2, f"copyright holder looks empty or placeholder: {holder!r}"
    return holder


def test_the_licence_carries_the_copyright_line() -> None:
    """LICENSE is the SINGLE place the copyright holder is named.

    This deliberately does not hardcode the name. A name written into a source
    file travels with every copied snippet, so keeping it in LICENSE alone means
    copying code does not copy an identity. The test therefore asserts the SHAPE
    of the notice and reads the holder out of it, rather than restating it here
    and creating a second place that can drift or leak.
    """
    assert copyright_holder()


def test_a_notice_file_exists_and_names_the_copyleft_dependency() -> None:
    """The NOTICE has to travel with the work and carry the PySide6 warning,
    because that is the obligation a downstream bundler inherits without being
    told otherwise.

    This used to assert "Apache License" appeared here, back when Triton itself
    shipped under Apache-2.0. It no longer does -- the project is proprietary and
    source-available -- so that assertion was checking a fact that had changed
    rather than a rule that still held. The obligations below are the part that
    was always the point: they come from DEPENDENCIES, not from Triton's own
    licence, so relicensing Triton does not discharge them."""
    notice = (ROOT / "NOTICE").read_text(encoding="utf-8")
    # Derived from LICENSE, not restated: NOTICE and LICENSE must agree on who
    # holds the copyright, and neither this file nor any other source file needs
    # to know the answer.
    assert copyright_holder() in notice, "NOTICE must name the same holder as LICENSE"
    assert "Apache License" not in notice, (
        "stale self-licence claim: Triton is no longer Apache-2.0"
    )
    assert "PySide6" in notice
    assert "LGPL" in notice
    assert "model weights" in notice.lower()


def test_no_model_weights_are_checked_in() -> None:
    """Many popular detection weights are AGPL-3.0, which would be incompatible
    with the Apache-2.0 promise if shipped — or even if implied by a default
    download. There is neither."""
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.split()
    weights = [f for f in tracked if f.lower().endswith((".onnx", ".pt", ".pth", ".weights"))]
    assert not weights, f"model weights are checked in: {weights}"


# ---------------------------------------------------------------------- SPDX

def test_every_source_file_carries_an_spdx_identifier() -> None:
    """One line, not a thirteen-line block. Apache-2.0 requires the LICENSE and
    NOTICE to travel, not the per-file appendix."""
    proc = subprocess.run(
        [__import__("sys").executable, "scripts/check_spdx.py"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_the_spdx_checker_actually_detects_a_missing_header(tmp_path) -> None:
    """A checker nobody has seen fail is a checker nobody knows works."""
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    import check_spdx  # noqa: PLC0415

    bare = tmp_path / "bare.py"
    bare.write_text("x = 1\n", encoding="utf-8")
    assert check_spdx.has_identifier(bare) is False

    check_spdx.insert_identifier(bare)
    assert check_spdx.has_identifier(bare) is True
    assert bare.read_text(encoding="utf-8").startswith("# SPDX-License-Identifier")


def test_the_spdx_inserter_respects_a_shebang(tmp_path) -> None:
    """Putting a comment above `#!` silently breaks an executable script."""
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    import check_spdx  # noqa: PLC0415

    script = tmp_path / "s.py"
    script.write_text("#!/usr/bin/env python\nx = 1\n", encoding="utf-8")
    check_spdx.insert_identifier(script)
    lines = script.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("#!")
    assert "SPDX" in lines[1]


# ------------------------------------------------------------- the anonymity gate

def test_every_commit_is_authored_under_an_anonymous_noreply_identity() -> None:
    """**The relicensing precondition and the anonymity gate are the same audit.**

    Relicensing requires the right to relicense, which requires knowing who
    contributed. Verifying that also verifies no personal identity leaked — and
    a leak cannot be fixed retroactively without rewriting published history.
    """
    proc = subprocess.run(
        ["git", "log", "--format=%an <%ae>%n%cn <%ce>"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    )
    offenders = sorted({
        line.strip() for line in proc.stdout.splitlines()
        if line.strip() and NOREPLY not in line
    })
    assert not offenders, f"non-noreply identities in history: {offenders}"


# ------------------------------------------------------- contributor readiness

def test_contributing_requires_dco_and_explains_why_not_a_cla() -> None:
    """A CLA needs a named legal entity to assign rights to, which would
    de-anonymise the maintainer. The reasoning belongs in the document, not only
    in someone's head."""
    # Normalised, because markdown wraps and a phrase split across a line break
    # is still the phrase. A test that failed on rewrapping would just get the
    # prose reflowed to suit it, which is the tail wagging the dog.
    text = " ".join((ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8").split())
    assert "Developer Certificate of Origin" in text
    assert "git commit -s" in text
    assert "de-anonymise" in text


def test_security_covers_physical_incidents_not_only_vulnerabilities() -> None:
    """`SECURITY.md` conventionally covers software. This machine aims water at
    things, so somebody being hit is a different report and needs a channel."""
    text = (ROOT / "SECURITY.md").read_text(encoding="utf-8")
    assert "Private Vulnerability Reporting" in text
    assert "physical" in text.lower()
    assert "struck by water" in text.lower() or "hit by water" in text.lower()


def test_there_is_no_security_email_address_to_deanonymise_anyone() -> None:
    """An address would undo the anonymity the DCO decision protects."""
    text = (ROOT / "SECURITY.md").read_text(encoding="utf-8")
    addresses = re.findall(r"[\w.+-]+@[\w-]+\.[\w.]+", text)
    real = [a for a in addresses if NOREPLY not in a]
    assert not real, f"SECURITY.md exposes an email address: {real}"


def test_the_licence_audit_records_the_copyleft_finding() -> None:
    text = " ".join((ROOT / "docs" / "LICENCE_AUDIT.md").read_text(encoding="utf-8").split())
    assert "PySide6" in text
    assert "GPL-2.0-only is incompatible" in text.replace("**", "")
    assert "AGPL" in text
    assert "101 commits" in text


# ------------------------------------------------------------------ the gate

def test_ci_is_a_wrapper_around_the_local_gate() -> None:
    """**The point of SP13's CI change.** The local gate is the source of truth,
    and two definitions of "green" drift until a CI-only failure appears that
    nobody can reproduce. One command, both places.
    """
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "scripts/verify.py" in workflow
    assert "fetch-depth: 0" in workflow, (
        "the identity audit needs full history, not a shallow clone"
    )
    # and CI must not have grown its own parallel checks
    assert "pytest -q" not in workflow
    assert "ruff check" not in workflow


def test_verify_runs_every_gate_including_the_slow_tier() -> None:
    source = (ROOT / "scripts" / "verify.py").read_text(encoding="utf-8")
    for gate in ("ruff", "check_spdx.py", "pytest", "-m\", \"slow", "check_identity"):
        assert gate in source, f"verify.py is missing the {gate} gate"


def test_verify_says_so_when_fast_mode_skipped_the_slow_tier() -> None:
    """A partial pass reported as a pass is worse than a failure."""
    source = (ROOT / "scripts" / "verify.py").read_text(encoding="utf-8")
    assert "not a full gate" in source


def _verify_module():
    """`scripts/` is not a package, so the gate is loaded by path."""
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("_verify", ROOT / "scripts" / "verify.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # Registered before execution: `@dataclass` resolves annotations through
    # `sys.modules[cls.__module__]`, which is None until the module is there.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_a_failing_check_keeps_both_ends_of_its_output() -> None:
    """A traceback names its exception last; a crash dump names it first. A gate
    that keeps one end reports one of them with the cause deleted."""
    verify = _verify_module()
    text = "FIRST-LINE\n" + ("x" * 20_000) + "\nLAST-LINE"

    clipped = verify.clip(text)

    assert "FIRST-LINE" in clipped
    assert "LAST-LINE" in clipped
    assert "elided" in clipped, "silent truncation reads as complete output"
    assert len(clipped) < len(text)


def test_short_output_is_not_clipped_at_all() -> None:
    verify = _verify_module()
    assert verify.clip("brief") == "brief"


def test_the_line_naming_a_crash_survives_from_the_middle() -> None:
    """The real CI failure: the fatal line sat in the middle of a 30k-line log,
    where no head-or-tail rule reaches it."""
    verify = _verify_module()
    output = "\n".join(
        ["." * 80] * 400
        + ["Fatal Python error: Segmentation fault"]
        + ["  File \"x.py\", line 1 in f"] * 400
    )

    summary = verify.summarise_failure(output)

    assert "Fatal Python error: Segmentation fault" in summary


def test_a_crash_dump_names_the_frame_in_this_repo() -> None:
    """Knowing it segfaulted is half an answer. A faulthandler dump is nearly all
    site-packages frames, and the one line that locates the crash is a needle in
    it -- which is how the first fix arrived with the cause but not the site."""
    verify = _verify_module()
    noise = ['  File "/usr/lib/python3/site-packages/_pytest/main.py", line 12 in x'] * 200
    output = "\n".join(
        ["Fatal Python error: Segmentation fault", "", "Current thread 0x1 (most recent call first):"]
        + noise[:100]
        + ['  File "/home/runner/work/Triton/Triton/tests/studio/test_plots.py", line 42 in test_it']
        + noise[100:]
    )

    summary = verify.summarise_failure(output)

    assert "tests/studio/test_plots.py" in summary


def test_an_ordinary_traceback_does_not_get_its_frames_hoisted() -> None:
    """Hoisting is for crashes. A normal failure is already reported properly and
    duplicating its frames on top would only bury the assertion."""
    verify = _verify_module()
    output = '\n'.join([
        'tests/test_thing.py:4: in test_it',
        '  File "tests/test_thing.py", line 4 in test_it',
        "E   AssertionError: nope",
    ])

    assert "CAUSE" not in verify.summarise_failure(output)


@pytest.mark.slow
def test_the_verifier_runs_and_passes_in_fast_mode() -> None:
    """Exercise the gate itself. `--fast` because the slow tier is already
    running around this test.

    The nested run includes tests/studio again. It was narrowed out at 14e820f
    after four consecutive red runs from a Qt segfault; that crash was then
    root-caused (a QThread destroyed while running, from rebinding in
    MainWindow.run) and fixed in b72482d, and the exact removed configuration
    passed 4/4 locally post-fix. Re-including it restores the gate's original
    end-to-end claim; any ambient PYTEST_ADDOPTS is still stripped so the
    nested run is exactly what verify.py spawns.
    """
    import os
    import sys

    verify = _verify_module()
    env = {k: v for k, v in os.environ.items() if k != "PYTEST_ADDOPTS"}
    proc = subprocess.run(
        [sys.executable, "scripts/verify.py", "--fast"],
        cwd=ROOT, capture_output=True, text=True,
        env=env,
    )
    assert proc.returncode == 0, verify.summarise_failure(proc.stdout + proc.stderr)
    assert "VERIFY PASSED" in proc.stdout
