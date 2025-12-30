"""Shim package to expose src/processes when running from repo root."""
from __future__ import annotations

from pathlib import Path

# CONTRACT: extend package path to include src/processes.
ROOT = Path(__file__).resolve().parent.parent
SRC_PKG = ROOT / "src" / "processes"
if SRC_PKG.is_dir():
    __path__.append(str(SRC_PKG))
