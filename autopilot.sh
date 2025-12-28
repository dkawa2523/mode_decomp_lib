#!/usr/bin/env bash
set -euo pipefail
# Convenience wrapper: run autopilot from repo root.
chmod +x tools/autopilot.sh >/dev/null 2>&1 || true
exec ./tools/autopilot.sh "${@:-30}"
