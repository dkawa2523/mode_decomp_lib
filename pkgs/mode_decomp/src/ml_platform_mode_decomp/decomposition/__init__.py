"""Numerical decomposers; concrete backends are loaded by the catalog."""

from ml_platform_mode_decomp.catalog import build_decomposer, list_decomposers

from .base import BaseDecomposer

__all__ = ["BaseDecomposer", "build_decomposer", "list_decomposers"]
