"""Regressor interfaces and lazy construction."""

from ml_platform_mode_decomp.catalog import build_regressor, list_regressors
from .base import BaseRegressor, IndependentMultiOutputWrapper

__all__ = ["BaseRegressor", "IndependentMultiOutputWrapper", "build_regressor", "list_regressors"]
