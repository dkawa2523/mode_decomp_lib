#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN="${PYTHON_BIN:-python3}"
LIVE_TEE="${LIVE_TEE:-1}"
python tools/apply_addon_after_p2_300.py
LIVE_TEE="$LIVE_TEE" PYTHON_BIN="$PYTHON_BIN" ./autopilot.sh 80
