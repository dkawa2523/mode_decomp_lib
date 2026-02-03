"""Coeff post plugins."""
from __future__ import annotations

from mode_decomp_ml.plugins.registry import build_coeff_post, list_coeff_posts, register_coeff_post

from .basic import (
    BaseCoeffPost,
    DictLearningCoeffPost,
    NoOpCoeffPost,
    PCACoeffPost,
    QuantileCoeffPost,
    StandardizeCoeffPost,
)

__all__ = [
    "BaseCoeffPost",
    "NoOpCoeffPost",
    "StandardizeCoeffPost",
    "QuantileCoeffPost",
    "PCACoeffPost",
    "DictLearningCoeffPost",
    "register_coeff_post",
    "list_coeff_posts",
    "build_coeff_post",
]
