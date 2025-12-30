"""Shim package to make src/ importable without installation."""
from __future__ import annotations

from pathlib import Path

# CONTRACT: extend package path to include src/mode_decomp_ml.
ROOT = Path(__file__).resolve().parent.parent
SRC_PKG = ROOT / "src" / "mode_decomp_ml"
if SRC_PKG.is_dir():
    __path__.append(str(SRC_PKG))
