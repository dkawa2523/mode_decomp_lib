"""Plugin entrypoints (registries + implementations)."""
from __future__ import annotations

from .registry import (
    build_coeff_codec,
    build_coeff_post,
    build_decomposer,
    build_regressor,
    list_coeff_codecs,
    list_coeff_posts,
    list_decomposers,
    list_regressors,
    register_coeff_codec,
    register_coeff_post,
    register_decomposer,
    register_regressor,
)

__all__ = [
    "build_coeff_post",
    "build_decomposer",
    "build_regressor",
    "build_coeff_codec",
    "list_coeff_codecs",
    "list_coeff_posts",
    "list_decomposers",
    "list_regressors",
    "register_coeff_codec",
    "register_coeff_post",
    "register_decomposer",
    "register_regressor",
]
