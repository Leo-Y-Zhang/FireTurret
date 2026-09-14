#!/bin/bash
set -euo pipefail

# Only do work in Claude Code on the web; local sessions are untouched.
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

cd "$CLAUDE_PROJECT_DIR"

# Create the venv only if it doesn't already exist (idempotent, and lets the
# container cache the result across a session).
if [ ! -x ".venv/bin/python" ]; then
  PY=$(command -v python3.13 || command -v python3.12 || command -v python3)
  "$PY" -m venv .venv
fi

# desktop is required (not optional) even headlessly: pytest-qt (from dev)
# refuses to collect any test without a Qt binding, which lives in desktop.
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e '.[dev,analysis,desktop]'

# Make the venv's binaries the default python/pytest/ruff for the session,
# and force Qt into offscreen mode so pytest-qt and the studio app can run
# without a display.
{
  echo "export PATH=\"$CLAUDE_PROJECT_DIR/.venv/bin:\$PATH\""
  echo "export QT_QPA_PLATFORM=offscreen"
} >> "$CLAUDE_ENV_FILE"
