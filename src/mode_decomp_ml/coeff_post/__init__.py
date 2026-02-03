"""Deprecated coeff_post package (shim)."""
from __future__ import annotations

import warnings as _warnings

_warnings.warn(
    "mode_decomp_ml.coeff_post is deprecated; use mode_decomp_ml.plugins.coeff_post",
    DeprecationWarning,
    stacklevel=2,
)

from mode_decomp_ml.plugins.coeff_post import *  # noqa: F401,F403
