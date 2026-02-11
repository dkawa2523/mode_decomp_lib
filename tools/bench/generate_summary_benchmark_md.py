#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Allow importing `mode_decomp_ml` without installation (shim package lives at repo root).
for _p in (str(PROJECT_ROOT), str(PROJECT_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tools.bench.report.render_md import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
