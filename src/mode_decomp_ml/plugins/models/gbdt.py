"""Optional GBDT regressors (XGBoost / LightGBM / CatBoost)."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from sklearn.preprocessing import StandardScaler

from mode_decomp_ml.optional import require_dependency
from mode_decomp_ml.plugins.models.base import (
    BaseRegressor,
    IndependentMultiOutputWrapper,
    cfg_get,
    _ensure_2d_features,
    _ensure_2d_predictions,
    _ensure_2d_targets,
)
from mode_decomp_ml.plugins.registry import register_regressor

try:  # optional dependency
    import xgboost as xgb
except Exception:  # pragma: no cover - optional dependency
    xgb = None

try:  # optional dependency
    import lightgbm as lgbm
except Exception:  # pragma: no cover - optional dependency
    lgbm = None

try:  # optional dependency
    import catboost
except Exception:  # pragma: no cover - optional dependency
    catboost = None

_VALID_TARGET_SPACES = {"a", "z"}
_VALID_SCALERS = {"none", "standardize"}


def _resolve_target_space(cfg: Mapping[str, Any]) -> str:
    target_space = str(cfg_get(cfg, "target_space", "")).strip().lower()
    if not target_space:
        raise ValueError("model.target_space is required")
    if target_space not in _VALID_TARGET_SPACES:
        raise ValueError(f"model.target_space must be one of {_VALID_TARGET_SPACES}, got {target_space}")
    return target_space


def _resolve_scaler(cfg: Mapping[str, Any]) -> str:
    scaler = cfg_get(cfg, "cond_scaler", "none")
    if scaler is None:
        return "none"
    scaler_name = str(scaler).strip().lower()
    if not scaler_name:
        return "none"
    if scaler_name not in _VALID_SCALERS:
        raise ValueError(f"cond_scaler must be one of {_VALID_SCALERS}, got {scaler_name}")
    return scaler_name


def _resolve_random_state(cfg: Mapping[str, Any]) -> int | None:
    seed = cfg_get(cfg, "seed", None)
    random_state = cfg_get(cfg, "random_state", None)
    if seed is not None:
        seed = int(seed)
    if random_state is not None:
        random_state = int(random_state)
    if seed is not None and random_state is not None and seed != random_state:
        raise ValueError("seed and random_state must match when both are provided")
    return seed if seed is not None else random_state


def _coerce_fraction(value: Any, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0 or value > 1:
        raise ValueError(f"{name} must be in (0, 1], got {value}")
    return value


def _validate_positive(value: float, name: str) -> float:
    if value <= 0:
        raise ValueError(f"{name} must be > 0")
    return value


def _validate_non_negative(value: float, name: str) -> float:
    if value < 0:
        raise ValueError(f"{name} must be >= 0")
    return value


@register_regressor("xgb")
class XGBRegressor(BaseRegressor):
    """XGBoost regressor (wrapped for multi-output)."""

    def __init__(self, *, cfg: Mapping[str, Any]) -> None:
        super().__init__(cfg=cfg)
        require_dependency(xgb, name="xgb regressor", pip_name="xgboost")
        self.name = "xgb"
        self.target_space = _resolve_target_space(cfg)
        self._cond_scaler_name = _resolve_scaler(cfg)
        self._scaler = StandardScaler() if self._cond_scaler_name == "standardize" else None
        self._random_state = _resolve_random_state(cfg)

        self._n_estimators = int(cfg_get(cfg, "n_estimators", 200))
        if self._n_estimators <= 0:
            raise ValueError("n_estimators must be > 0")
        self._max_depth = int(cfg_get(cfg, "max_depth", 6))
        if self._max_depth <= 0:
            raise ValueError("max_depth must be > 0")
        self._learning_rate = _validate_positive(float(cfg_get(cfg, "learning_rate", 0.1)), "learning_rate")
        self._subsample = _coerce_fraction(cfg_get(cfg, "subsample", 1.0), "subsample")
        self._colsample_bytree = _coerce_fraction(cfg_get(cfg, "colsample_bytree", 1.0), "colsample_bytree")
        self._min_child_weight = _validate_non_negative(
            float(cfg_get(cfg, "min_child_weight", 1.0)),
            "min_child_weight",
        )
        self._gamma = _validate_non_negative(float(cfg_get(cfg, "gamma", 0.0)), "gamma")
        self._reg_lambda = _validate_non_negative(float(cfg_get(cfg, "reg_lambda", 1.0)), "reg_lambda")
        self._tree_method = str(cfg_get(cfg, "tree_method", "auto")).strip() or "auto"
        self._n_jobs = cfg_get(cfg, "n_jobs", None)
        if self._n_jobs is not None:
            self._n_jobs = int(self._n_jobs)

        self._model: IndependentMultiOutputWrapper | None = None
        self._target_dim: int | None = None

    def _base_factory(self) -> Any:
        return xgb.XGBRegressor(
            n_estimators=self._n_estimators,
            max_depth=self._max_depth,
            learning_rate=self._learning_rate,
            subsample=self._subsample,
            colsample_bytree=self._colsample_bytree,
            min_child_weight=self._min_child_weight,
            gamma=self._gamma,
            reg_lambda=self._reg_lambda,
            tree_method=self._tree_method,
            objective="reg:squarederror",
            random_state=self._random_state,
            n_jobs=self._n_jobs,
        )

    def fit(self, X_cond: np.ndarray, Y: np.ndarray) -> "XGBRegressor":
        X_cond = _ensure_2d_features(X_cond, "X_cond")
        Y = _ensure_2d_targets(Y, X_cond.shape[0])
        if self._scaler is not None:
            X_cond = self._scaler.fit_transform(X_cond)

        wrapper = IndependentMultiOutputWrapper(self._base_factory, name=self.name)
        wrapper.fit(X_cond, Y)
        self._model = wrapper
        self._target_dim = wrapper.target_dim
        self._state = {
            "method": self.name,
            "target_space": self.target_space,
            "cond_scaler": self._cond_scaler_name,
            "n_estimators": self._n_estimators,
            "max_depth": self._max_depth,
            "learning_rate": self._learning_rate,
            "subsample": self._subsample,
            "colsample_bytree": self._colsample_bytree,
            "min_child_weight": self._min_child_weight,
            "gamma": self._gamma,
            "reg_lambda": self._reg_lambda,
            "tree_method": self._tree_method,
            "random_state": self._random_state,
            "library": "xgboost",
        }
        self._mark_fitted()
        return self

    def predict(self, X_cond: np.ndarray) -> np.ndarray:
        self._require_fitted()
        if self._model is None:
            raise ValueError("model state is missing")
        X_cond = _ensure_2d_features(X_cond, "X_cond")
        if self._scaler is not None:
            X_cond = self._scaler.transform(X_cond)
        pred = self._model.predict(X_cond)
        return _ensure_2d_predictions(pred, n_samples=X_cond.shape[0], target_dim=self._target_dim)


@register_regressor("lgbm")
class LGBMRegressor(BaseRegressor):
    """LightGBM regressor (wrapped for multi-output)."""

    def __init__(self, *, cfg: Mapping[str, Any]) -> None:
        super().__init__(cfg=cfg)
        require_dependency(lgbm, name="lgbm regressor", pip_name="lightgbm")
        self.name = "lgbm"
        self.target_space = _resolve_target_space(cfg)
        self._cond_scaler_name = _resolve_scaler(cfg)
        self._scaler = StandardScaler() if self._cond_scaler_name == "standardize" else None
        self._random_state = _resolve_random_state(cfg)

        self._n_estimators = int(cfg_get(cfg, "n_estimators", 200))
        if self._n_estimators <= 0:
            raise ValueError("n_estimators must be > 0")
        self._learning_rate = _validate_positive(float(cfg_get(cfg, "learning_rate", 0.1)), "learning_rate")
        self._max_depth = int(cfg_get(cfg, "max_depth", -1))
        if self._max_depth == 0 or self._max_depth < -1:
            raise ValueError("max_depth must be -1 or > 0")
        self._num_leaves = int(cfg_get(cfg, "num_leaves", 31))
        if self._num_leaves < 2:
            raise ValueError("num_leaves must be >= 2")
        self._subsample = _coerce_fraction(cfg_get(cfg, "subsample", 1.0), "subsample")
        self._colsample_bytree = _coerce_fraction(cfg_get(cfg, "colsample_bytree", 1.0), "colsample_bytree")
        self._min_child_samples = int(cfg_get(cfg, "min_child_samples", 20))
        if self._min_child_samples <= 0:
            raise ValueError("min_child_samples must be > 0")
        self._reg_lambda = _validate_non_negative(float(cfg_get(cfg, "reg_lambda", 0.0)), "reg_lambda")
        self._boosting_type = str(cfg_get(cfg, "boosting_type", "gbdt")).strip() or "gbdt"
        self._n_jobs = cfg_get(cfg, "n_jobs", None)
        if self._n_jobs is not None:
            self._n_jobs = int(self._n_jobs)
        self._verbosity = int(cfg_get(cfg, "verbosity", -1))

        self._model: IndependentMultiOutputWrapper | None = None
        self._target_dim: int | None = None

    def _base_factory(self) -> Any:
        return lgbm.LGBMRegressor(
            n_estimators=self._n_estimators,
            learning_rate=self._learning_rate,
            max_depth=self._max_depth,
            num_leaves=self._num_leaves,
            subsample=self._subsample,
            colsample_bytree=self._colsample_bytree,
            min_child_samples=self._min_child_samples,
            reg_lambda=self._reg_lambda,
            boosting_type=self._boosting_type,
            objective="regression",
            random_state=self._random_state,
            n_jobs=self._n_jobs,
            verbosity=self._verbosity,
        )

    def fit(self, X_cond: np.ndarray, Y: np.ndarray) -> "LGBMRegressor":
        X_cond = _ensure_2d_features(X_cond, "X_cond")
        Y = _ensure_2d_targets(Y, X_cond.shape[0])
        if self._scaler is not None:
            X_cond = self._scaler.fit_transform(X_cond)

        wrapper = IndependentMultiOutputWrapper(self._base_factory, name=self.name)
        wrapper.fit(X_cond, Y)
        self._model = wrapper
        self._target_dim = wrapper.target_dim
        self._state = {
            "method": self.name,
            "target_space": self.target_space,
            "cond_scaler": self._cond_scaler_name,
            "n_estimators": self._n_estimators,
            "learning_rate": self._learning_rate,
            "max_depth": self._max_depth,
            "num_leaves": self._num_leaves,
            "subsample": self._subsample,
            "colsample_bytree": self._colsample_bytree,
            "min_child_samples": self._min_child_samples,
            "reg_lambda": self._reg_lambda,
            "boosting_type": self._boosting_type,
            "random_state": self._random_state,
            "library": "lightgbm",
        }
        self._mark_fitted()
        return self

    def predict(self, X_cond: np.ndarray) -> np.ndarray:
        self._require_fitted()
        if self._model is None:
            raise ValueError("model state is missing")
        X_cond = _ensure_2d_features(X_cond, "X_cond")
        if self._scaler is not None:
            X_cond = self._scaler.transform(X_cond)
        pred = self._model.predict(X_cond)
        return _ensure_2d_predictions(pred, n_samples=X_cond.shape[0], target_dim=self._target_dim)


@register_regressor("catboost")
class CatBoostRegressor(BaseRegressor):
    """CatBoost regressor (wrapped for multi-output)."""

    def __init__(self, *, cfg: Mapping[str, Any]) -> None:
        super().__init__(cfg=cfg)
        require_dependency(catboost, name="catboost regressor", pip_name="catboost")
        self.name = "catboost"
        self.target_space = _resolve_target_space(cfg)
        self._cond_scaler_name = _resolve_scaler(cfg)
        self._scaler = StandardScaler() if self._cond_scaler_name == "standardize" else None
        self._random_state = _resolve_random_state(cfg)

        self._n_estimators = int(cfg_get(cfg, "n_estimators", 200))
        if self._n_estimators <= 0:
            raise ValueError("n_estimators must be > 0")
        depth_value = cfg_get(cfg, "depth", None)
        if depth_value is None:
            depth_value = cfg_get(cfg, "max_depth", 6)
        self._depth = int(depth_value)
        if self._depth <= 0:
            raise ValueError("depth must be > 0")
        self._learning_rate = _validate_positive(float(cfg_get(cfg, "learning_rate", 0.1)), "learning_rate")
        self._l2_leaf_reg = _validate_non_negative(float(cfg_get(cfg, "l2_leaf_reg", 3.0)), "l2_leaf_reg")
        self._loss_function = str(cfg_get(cfg, "loss_function", "RMSE")).strip() or "RMSE"
        self._allow_writing_files = bool(cfg_get(cfg, "allow_writing_files", False))
        self._verbose = bool(cfg_get(cfg, "verbose", False))
        self._thread_count = cfg_get(cfg, "thread_count", None)
        if self._thread_count is not None:
            self._thread_count = int(self._thread_count)

        self._model: IndependentMultiOutputWrapper | None = None
        self._target_dim: int | None = None

    def _base_factory(self) -> Any:
        return catboost.CatBoostRegressor(
            iterations=self._n_estimators,
            depth=self._depth,
            learning_rate=self._learning_rate,
            loss_function=self._loss_function,
            l2_leaf_reg=self._l2_leaf_reg,
            random_seed=self._random_state,
            verbose=self._verbose,
            allow_writing_files=self._allow_writing_files,
            thread_count=self._thread_count,
        )

    def fit(self, X_cond: np.ndarray, Y: np.ndarray) -> "CatBoostRegressor":
        X_cond = _ensure_2d_features(X_cond, "X_cond")
        Y = _ensure_2d_targets(Y, X_cond.shape[0])
        if self._scaler is not None:
            X_cond = self._scaler.fit_transform(X_cond)

        wrapper = IndependentMultiOutputWrapper(self._base_factory, name=self.name)
        wrapper.fit(X_cond, Y)
        self._model = wrapper
        self._target_dim = wrapper.target_dim
        self._state = {
            "method": self.name,
            "target_space": self.target_space,
            "cond_scaler": self._cond_scaler_name,
            "n_estimators": self._n_estimators,
            "depth": self._depth,
            "learning_rate": self._learning_rate,
            "l2_leaf_reg": self._l2_leaf_reg,
            "loss_function": self._loss_function,
            "random_state": self._random_state,
            "library": "catboost",
        }
        self._mark_fitted()
        return self

    def predict(self, X_cond: np.ndarray) -> np.ndarray:
        self._require_fitted()
        if self._model is None:
            raise ValueError("model state is missing")
        X_cond = _ensure_2d_features(X_cond, "X_cond")
        if self._scaler is not None:
            X_cond = self._scaler.transform(X_cond)
        pred = self._model.predict(X_cond)
        return _ensure_2d_predictions(pred, n_samples=X_cond.shape[0], target_dim=self._target_dim)


__all__ = [
    "XGBRegressor",
    "LGBMRegressor",
    "CatBoostRegressor",
]
