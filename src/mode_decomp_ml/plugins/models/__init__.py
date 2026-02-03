"""Model plugins."""
from __future__ import annotations

from mode_decomp_ml.plugins.registry import build_regressor, list_regressors, register_regressor

from .base import BaseRegressor, IndependentMultiOutputWrapper
from .gbdt import CatBoostRegressor, LGBMRegressor, XGBRegressor
from .mtgp import MTGPRegressor
from .sklearn import (
    ElasticNetRegressor,
    GPRRegressor,
    MultiTaskElasticNetRegressor,
    MultiTaskLassoRegressor,
    RidgeRegressor,
)

__all__ = [
    "BaseRegressor",
    "IndependentMultiOutputWrapper",
    "CatBoostRegressor",
    "ElasticNetRegressor",
    "GPRRegressor",
    "LGBMRegressor",
    "MTGPRegressor",
    "MultiTaskElasticNetRegressor",
    "MultiTaskLassoRegressor",
    "RidgeRegressor",
    "XGBRegressor",
    "register_regressor",
    "list_regressors",
    "build_regressor",
]
