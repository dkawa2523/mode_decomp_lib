"""Plugin entrypoints (registries + implementations)."""
from __future__ import annotations

# Import preprocess implementations to populate the central registry.
# These classes must live in `mode_decomp_ml.preprocess` for pickle compatibility.
import mode_decomp_ml.preprocess as _preprocess  # noqa: F401

from .registry import (
    build_coeff_codec,
    build_coeff_post,
    build_decomposer,
    build_preprocess,
    build_regressor,
    list_coeff_codecs,
    list_coeff_posts,
    list_decomposers,
    list_preprocess,
    list_regressors,
    register_coeff_codec,
    register_coeff_post,
    register_decomposer,
    register_preprocess,
    register_regressor,
)

__all__ = [
    "build_coeff_post",
    "build_decomposer",
    "build_regressor",
    "build_coeff_codec",
    "build_preprocess",
    "list_coeff_codecs",
    "list_coeff_posts",
    "list_decomposers",
    "list_preprocess",
    "list_regressors",
    "register_coeff_codec",
    "register_coeff_post",
    "register_decomposer",
    "register_preprocess",
    "register_regressor",
]
