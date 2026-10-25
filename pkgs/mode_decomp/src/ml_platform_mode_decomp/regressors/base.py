"""Shared model base utilities."""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping

import numpy as np


def _ensure_2d_features(array: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim == 1:
        return array[None, :]
    if array.ndim == 2:
        return array
    raise ValueError(f"{name} must be 1D or 2D, got shape {array.shape}")


def _ensure_2d_targets(array: np.ndarray, n_samples: int) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim == 1:
        if array.shape[0] != n_samples:
            raise ValueError("Y must be 1D with length matching X_cond samples")
        return array[:, None]
    if array.ndim == 2:
        if array.shape[0] != n_samples:
            raise ValueError("Y sample count does not match X_cond")
        return array
    raise ValueError(f"Y must be 1D or 2D, got shape {array.shape}")


def _ensure_2d_predictions(
    array: np.ndarray,
    *,
    n_samples: int | None = None,
    target_dim: int | None = None,
) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim == 1:
        if n_samples is not None and array.shape[0] != n_samples:
            raise ValueError("prediction sample count does not match X_cond")
        array = array[:, None]
    elif array.ndim == 2:
        if n_samples is not None and array.shape[0] != n_samples:
            raise ValueError("prediction sample count does not match X_cond")
    else:
        raise ValueError(f"prediction must be 1D or 2D, got shape {array.shape}")
    if target_dim is not None and array.shape[1] != target_dim:
        raise ValueError("predicted target dimension does not match fitted target_dim")
    return array


class BaseRegressor:
    """Minimal regressor interface."""

    name: str

    def __init__(self, *, cfg: Mapping[str, Any]) -> None:
        self._cfg = dict(cfg)
        self._fitted = False
        self._state: Dict[str, Any] | None = None

    def fit(self, X_cond: np.ndarray, Y: np.ndarray) -> "BaseRegressor":
        raise NotImplementedError

    def predict(self, X_cond: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def predict_std(self, X_cond: np.ndarray) -> np.ndarray | None:
        return None

    def predict_with_std(self, X_cond: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        return self.predict(X_cond), self.predict_std(X_cond)

    def state(self) -> Mapping[str, Any]:
        if self._state is None:
            raise ValueError("state is not available before fit")
        return self._state

    def _begin_fit(self) -> None:
        self._fitted = False
        self._state = None

    def _mark_fitted(self) -> None:
        self._fitted = True

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise ValueError("model must be fit before predict")


class IndependentMultiOutputWrapper:
    """Fit independent single-output estimators per target."""

    def __init__(self, base_factory: Callable[[], Any], *, name: str = "base") -> None:
        if not callable(base_factory):
            raise ValueError("base_factory must be callable")
        self._base_factory: Callable[[], Any] | None = base_factory
        self._name = name
        self._models: list[Any] | None = None
        self._target_dim: int | None = None

    @property
    def target_dim(self) -> int:
        if self._target_dim is None:
            raise ValueError("target_dim is not available before fit")
        return self._target_dim

    def fit(self, X_cond: np.ndarray, Y: np.ndarray) -> "IndependentMultiOutputWrapper":
        if self._base_factory is None:
            raise ValueError("A fitted independent wrapper is inference-only; construct a new wrapper to refit")
        X_cond = _ensure_2d_features(X_cond, "X_cond")
        Y = _ensure_2d_targets(Y, X_cond.shape[0])
        models: list[Any] = []
        for idx in range(Y.shape[1]):
            model = self._base_factory()
            if not hasattr(model, "fit"):
                raise TypeError(f"{self._name} base model must implement fit")
            model.fit(X_cond, Y[:, idx])
            models.append(model)
        self._models = models
        self._target_dim = Y.shape[1]
        # Avoid pickling closures after fit; wrapper is inference-only post-train.
        self._base_factory = None
        return self

    def _require_fitted(self) -> None:
        if self._models is None or self._target_dim is None:
            raise ValueError("wrapper must be fit before predict")

    def predict(self, X_cond: np.ndarray) -> np.ndarray:
        self._require_fitted()
        X_cond = _ensure_2d_features(X_cond, "X_cond")
        preds: list[np.ndarray] = []
        for model in self._models or []:
            pred = np.asarray(model.predict(X_cond))
            if pred.ndim == 2 and pred.shape[1] == 1:
                pred = pred[:, 0]
            if pred.ndim != 1:
                raise ValueError("base model predict must return 1D per output")
            preds.append(pred)
        stacked = np.stack(preds, axis=1)
        if stacked.shape[1] != self._target_dim:
            raise ValueError("predicted target dimension does not match fitted target_dim")
        return stacked

    def predict_std(self, X_cond: np.ndarray) -> np.ndarray | None:
        self._require_fitted()
        X_cond = _ensure_2d_features(X_cond, "X_cond")
        stds: list[np.ndarray] = []
        for model in self._models or []:
            std = None
            if hasattr(model, "predict_with_std"):
                _, std = model.predict_with_std(X_cond)
            elif hasattr(model, "predict_std"):
                std = model.predict_std(X_cond)
            if std is None:
                return None
            std = np.asarray(std)
            if std.ndim == 2 and std.shape[1] == 1:
                std = std[:, 0]
            if std.ndim != 1:
                raise ValueError("base model predict_std must return 1D per output")
            stds.append(std)
        stacked = np.stack(stds, axis=1)
        if stacked.shape[1] != self._target_dim:
            raise ValueError("predict_std dimension does not match fitted target_dim")
        return stacked


__all__ = [
    "BaseRegressor",
    "IndependentMultiOutputWrapper",
    "_ensure_2d_features",
    "_ensure_2d_targets",
    "_ensure_2d_predictions",
]
