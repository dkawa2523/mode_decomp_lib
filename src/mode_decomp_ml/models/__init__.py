"""Deprecated models package (shim)."""
from __future__ import annotations

import warnings as _warnings

_warnings.warn(
    "mode_decomp_ml.models is deprecated; use mode_decomp_ml.plugins.models",
    DeprecationWarning,
    stacklevel=2,
)

from mode_decomp_ml.plugins.models import *  # noqa: F401,F403
