# SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary
"""SPDX header check — one line per file, not a thirteen-line block.

LicenseRef-Leo-Y-Zhang-Proprietary requires that the LICENSE and NOTICE travel with the work. It does
**not** require the per-file appendix boilerplate, and pasting thirteen lines of
legal text onto the top of every source file makes every file's first screenful
identical and unread. A single machine-readable identifier does the same job:

    # SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary

Run with `--fix` to insert missing headers.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IDENTIFIER = "SPDX-License-Identifier: LicenseRef-Leo-Y-Zhang-Proprietary"

# Directories whose contents are not ours to label.
SKIP_DIRS = {".venv", ".git", "__pycache__", ".pytest_cache", ".ruff_cache",
             "build", "dist", "golden"}


def source_files() -> list[Path]:
    out: list[Path] = []
    for pattern in ("src/**/*.py", "tests/**/*.py", "scripts/**/*.py"):
        for path in ROOT.glob(pattern):
            if any(part in SKIP_DIRS for part in path.parts):
                continue
            out.append(path)
    return sorted(out)


def has_identifier(path: Path) -> bool:
    # Only the first few lines: an identifier buried in the middle of a file is
    # not a header, and would not survive an automated relicence.
    head = path.read_text(encoding="utf-8", errors="replace").splitlines()[:5]
    return any(IDENTIFIER in line for line in head)


def insert_identifier(path: Path) -> None:
    """Add the header, respecting an existing shebang or encoding line."""
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    at = 0
    if lines and lines[0].startswith("#!"):
        at = 1
    if len(lines) > at and "coding" in lines[at] and lines[at].startswith("#"):
        at += 1
    lines.insert(at, f"# {IDENTIFIER}\n")
    path.write_text("".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fix", action="store_true", help="insert missing headers")
    args = parser.parse_args()

    missing = [p for p in source_files() if not has_identifier(p)]
    if args.fix:
        for path in missing:
            insert_identifier(path)
            print(f"  added header: {path.relative_to(ROOT)}")
        print(f"added {len(missing)} header(s)")
        return 0

    if missing:
        print(f"{len(missing)} file(s) missing '{IDENTIFIER}':", file=sys.stderr)
        for path in missing:
            print(f"  {path.relative_to(ROOT)}", file=sys.stderr)
        print("\nrun: python scripts/check_spdx.py --fix", file=sys.stderr)
        return 1

    print(f"SPDX: all {len(source_files())} source files carry an identifier")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
