#!/usr/bin/env bash
set -euo pipefail

# Portable Python launcher:
# - macOS では `python` が存在しないことがあるため `python3` を優先
PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "$PYTHON_BIN" ]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  else
    echo "ERROR: Python が見つかりません。" >&2
    echo "  - python3 をインストールしてください (例: brew install python)" >&2
    echo "  - もしくは環境変数 PYTHON_BIN に python 実行パスを指定してください" >&2
    exit 127
  fi
fi

"$PYTHON_BIN" tools/codex_prompt.py doctor
