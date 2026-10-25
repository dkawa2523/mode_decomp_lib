"""Optional GBDT regressors (XGBoost / LightGBM / CatBoost)."""

from __future__ import annotations

from ml_platform_mode_decomp.config import constructor_config, integer_parameter

from typing import Any, Mapping

import numpy as np
from sklearn.preprocessing import StandardScaler

from ml_platform_mode_decomp.optional import import_optional
from ml_platform_mode_decomp.regressors.base import (
    BaseRegressor,
    IndependentMultiOutputWrapper,
    _ensure_2d_features,
    _ensure_2d_predictions,
    _ensure_2d_targets,
)


_VALID_SCALERS = {"none", "standardize"}


def _resolve_scaler(cfg: Mapping[str, Any]) -> str:
    scaler = cfg["cond_scaler"]
    if scaler is None:
        return "none"
    scaler_name = str(scaler).strip().lower()
    if not scaler_name:
        return "none"
    if scaler_name not in _VALID_SCALERS:
        raise ValueError(f"cond_scaler must be one of {_VALID_SCALERS}, got {scaler_name}")
    return scaler_name


def _resolve_random_state(cfg: Mapping[str, Any]) -> int | None:
    seed = cfg["seed"]
    random_state = cfg["random_state"]
    if seed is not None:
        seed = integer_parameter(seed, "seed")
    if random_state is not None:
        random_state = integer_parameter(random_state, "random_state")
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


class XGBRegressor(BaseRegressor):
    """XGBoost regressor (wrapped for multi-output)."""

    def __init__(
        self,
        *,
        cfg: Mapping[str, Any] | None = None,
        n_jobs: Any = None,
        n_estimators: Any = 200,
        max_depth: Any = 6,
        subsample: Any = 1.0,
        colsample_bytree: Any = 1.0,
        learning_rate: Any = 0.1,
        min_child_weight: Any = 1.0,
        gamma: Any = 0.0,
        reg_lambda: Any = 1.0,
        tree_method: Any = "auto",
        seed: Any = None,
        random_state: Any = None,
        cond_scaler: Any = "standardize",
    ) -> None:
        cfg = constructor_config(cfg, locals(), name="xgb")
        super().__init__(cfg=cfg)
        import_optional("xgboost", component="xgb regressor")
        self.name = "xgb"
        self._cond_scaler_name = _resolve_scaler(cfg)
        self._scaler = StandardScaler() if self._cond_scaler_name == "standardize" else None
        self._random_state = _resolve_random_state(cfg)

        self._n_estimators = integer_parameter(cfg["n_estimators"], "n_estimators")
        if self._n_estimators <= 0:
            raise ValueError("n_estimators must be > 0")
        self._max_depth = integer_parameter(cfg["max_depth"], "max_depth")
        if self._max_depth <= 0:
            raise ValueError("max_depth must be > 0")
        self._learning_rate = _validate_positive(float(cfg["learning_rate"]), "learning_rate")
        self._subsample = _coerce_fraction(cfg["subsample"], "subsample")
        self._colsample_bytree = _coerce_fraction(cfg["colsample_bytree"], "colsample_bytree")
        self._min_child_weight = _validate_non_negative(
            float(cfg["min_child_weight"]),
            "min_child_weight",
        )
        self._gamma = _validate_non_negative(float(cfg["gamma"]), "gamma")
        self._reg_lambda = _validate_non_negative(float(cfg["reg_lambda"]), "reg_lambda")
        self._tree_method = str(cfg["tree_method"]).strip() or "auto"
        self._n_jobs = cfg["n_jobs"]
        if self._n_jobs is not None:
            self._n_jobs = integer_parameter(self._n_jobs, "n_jobs")

        self._model: IndependentMultiOutputWrapper | None = None
        self._target_dim: int | None = None

    def _base_factory(self) -> Any:
        return import_optional("xgboost", component="xgb regressor").XGBRegressor(
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
        self._begin_fit()
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


class LGBMRegressor(BaseRegressor):
    """LightGBM regressor (wrapped for multi-output)."""

    def __init__(
        self,
        *,
        cfg: Mapping[str, Any] | None = None,
        n_jobs: Any = None,
        n_estimators: Any = 200,
        max_depth: Any = -1,
        num_leaves: Any = 31,
        subsample: Any = 1.0,
        subsample_freq: Any = None,
        colsample_bytree: Any = 1.0,
        min_child_samples: Any = 20,
        verbosity: Any = -1,
        learning_rate: Any = 0.1,
        reg_lambda: Any = 0.0,
        boosting_type: Any = "gbdt",
        seed: Any = None,
        random_state: Any = None,
        cond_scaler: Any = "standardize",
    ) -> None:
        cfg = constructor_config(cfg, locals(), name="lgbm")
        super().__init__(cfg=cfg)
        import_optional("lightgbm", component="lgbm regressor")
        self.name = "lgbm"
        self._cond_scaler_name = _resolve_scaler(cfg)
        self._scaler = StandardScaler() if self._cond_scaler_name == "standardize" else None
        self._random_state = _resolve_random_state(cfg)

        self._n_estimators = integer_parameter(cfg["n_estimators"], "n_estimators")
        if self._n_estimators <= 0:
            raise ValueError("n_estimators must be > 0")
        self._learning_rate = _validate_positive(float(cfg["learning_rate"]), "learning_rate")
        self._max_depth = integer_parameter(cfg["max_depth"], "max_depth")
        if self._max_depth == 0 or self._max_depth < -1:
            raise ValueError("max_depth must be -1 or > 0")
        self._num_leaves = integer_parameter(cfg["num_leaves"], "num_leaves")
        if self._num_leaves < 2:
            raise ValueError("num_leaves must be >= 2")
        self._subsample = _coerce_fraction(cfg["subsample"], "subsample")
        self._subsample_freq = integer_parameter(
            (1 if self._subsample < 1.0 else 0) if cfg["subsample_freq"] is None else cfg["subsample_freq"],
            "subsample_freq",
        )
        if self._subsample_freq < 0:
            raise ValueError("subsample_freq must be >= 0")
        if self._subsample < 1.0 and self._subsample_freq == 0:
            raise ValueError("subsample < 1 requires subsample_freq > 0")
        self._colsample_bytree = _coerce_fraction(cfg["colsample_bytree"], "colsample_bytree")
        self._min_child_samples = integer_parameter(cfg["min_child_samples"], "min_child_samples")
        if self._min_child_samples <= 0:
            raise ValueError("min_child_samples must be > 0")
        self._reg_lambda = _validate_non_negative(float(cfg["reg_lambda"]), "reg_lambda")
        self._boosting_type = str(cfg["boosting_type"]).strip() or "gbdt"
        self._n_jobs = cfg["n_jobs"]
        if self._n_jobs is not None:
            self._n_jobs = integer_parameter(self._n_jobs, "n_jobs")
        self._verbosity = integer_parameter(cfg["verbosity"], "verbosity")

        self._model: IndependentMultiOutputWrapper | None = None
        self._target_dim: int | None = None

    def _base_factory(self) -> Any:
        return import_optional("lightgbm", component="lgbm regressor").LGBMRegressor(
            n_estimators=self._n_estimators,
            learning_rate=self._learning_rate,
            max_depth=self._max_depth,
            num_leaves=self._num_leaves,
            subsample=self._subsample,
            subsample_freq=self._subsample_freq,
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
        self._begin_fit()
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
            "cond_scaler": self._cond_scaler_name,
            "n_estimators": self._n_estimators,
            "learning_rate": self._learning_rate,
            "max_depth": self._max_depth,
            "num_leaves": self._num_leaves,
            "subsample": self._subsample,
            "subsample_freq": self._subsample_freq,
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


class CatBoostRegressor(BaseRegressor):
    """CatBoost regressor (wrapped for multi-output)."""

    def __init__(
        self,
        *,
        cfg: Mapping[str, Any] | None = None,
        depth: Any = None,
        thread_count: Any = None,
        n_estimators: Any = 200,
        max_depth: Any = 6,
        allow_writing_files: Any = False,
        verbose: Any = False,
        learning_rate: Any = 0.1,
        l2_leaf_reg: Any = 3.0,
        loss_function: Any = "RMSE",
        seed: Any = None,
        random_state: Any = None,
        cond_scaler: Any = "standardize",
    ) -> None:
        cfg = constructor_config(cfg, locals(), name="catboost")
        super().__init__(cfg=cfg)
        import_optional("catboost", component="catboost regressor")
        self.name = "catboost"
        self._cond_scaler_name = _resolve_scaler(cfg)
        self._scaler = StandardScaler() if self._cond_scaler_name == "standardize" else None
        self._random_state = _resolve_random_state(cfg)

        self._n_estimators = integer_parameter(cfg["n_estimators"], "n_estimators")
        if self._n_estimators <= 0:
            raise ValueError("n_estimators must be > 0")
        depth_value = cfg["depth"]
        if depth_value is None:
            depth_value = cfg["max_depth"]
        self._depth = integer_parameter(depth_value, "depth_value")
        if self._depth <= 0:
            raise ValueError("depth must be > 0")
        self._learning_rate = _validate_positive(float(cfg["learning_rate"]), "learning_rate")
        self._l2_leaf_reg = _validate_non_negative(float(cfg["l2_leaf_reg"]), "l2_leaf_reg")
        self._loss_function = str(cfg["loss_function"]).strip() or "RMSE"
        self._allow_writing_files = bool(cfg["allow_writing_files"])
        self._verbose = bool(cfg["verbose"])
        self._thread_count = cfg["thread_count"]
        if self._thread_count is not None:
            self._thread_count = integer_parameter(self._thread_count, "thread_count")

        self._model: IndependentMultiOutputWrapper | None = None
        self._target_dim: int | None = None

    def _base_factory(self) -> Any:
        return import_optional("catboost", component="catboost regressor").CatBoostRegressor(
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
        self._begin_fit()
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
