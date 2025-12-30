"""Model registry and regression implementations."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Callable, Dict, Mapping

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.linear_model import ElasticNet, MultiTaskElasticNet, Ridge
from sklearn.preprocessing import StandardScaler

_REGRESSOR_REGISTRY: Dict[str, Callable[..., "BaseRegressor"]] = {}
_VALID_TARGET_SPACES = {"a", "z"}
_VALID_SCALERS = {"none", "standardize"}
_VALID_GPR_KERNELS = {"rbf"}
_VALID_SELECTION = {"cyclic", "random"}


def _cfg_get(cfg: Mapping[str, Any] | None, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        try:
            return cfg.get(key, default)
        except TypeError:
            pass
    return getattr(cfg, key, default)


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


def _resolve_target_space(cfg: Mapping[str, Any]) -> str:
    target_space = str(_cfg_get(cfg, "target_space", "")).strip().lower()
    # CONTRACT: target_space must be explicit to keep a/z artifacts comparable.
    if not target_space:
        raise ValueError("model.target_space is required")
    if target_space not in _VALID_TARGET_SPACES:
        raise ValueError(f"model.target_space must be one of {_VALID_TARGET_SPACES}, got {target_space}")
    return target_space


def _resolve_scaler(cfg: Mapping[str, Any]) -> str:
    scaler = _cfg_get(cfg, "cond_scaler", "none")
    if scaler is None:
        return "none"
    scaler_name = str(scaler).strip().lower()
    if not scaler_name:
        return "none"
    if scaler_name not in _VALID_SCALERS:
        raise ValueError(f"cond_scaler must be one of {_VALID_SCALERS}, got {scaler_name}")
    return scaler_name


def _resolve_random_state(cfg: Mapping[str, Any]) -> int | None:
    seed = _cfg_get(cfg, "seed", None)
    random_state = _cfg_get(cfg, "random_state", None)
    if seed is not None:
        seed = int(seed)
    if random_state is not None:
        random_state = int(random_state)
    if seed is not None and random_state is not None and seed != random_state:
        raise ValueError("seed and random_state must match when both are provided")
    return seed if seed is not None else random_state


def _resolve_optimizer(cfg: Mapping[str, Any]) -> str | None:
    optimizer = _cfg_get(cfg, "optimizer", "fmin_l_bfgs_b")
    if optimizer is None:
        return None
    optimizer_name = str(optimizer).strip()
    if not optimizer_name:
        return None
    return optimizer_name


def _resolve_selection(cfg: Mapping[str, Any]) -> str:
    selection = str(_cfg_get(cfg, "selection", "cyclic")).strip().lower()
    if not selection:
        selection = "cyclic"
    if selection not in _VALID_SELECTION:
        raise ValueError(f"selection must be one of {_VALID_SELECTION}, got {selection}")
    return selection


def _coerce_bounds(value: Any, name: str) -> tuple[float, float]:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)) and len(value) == 2:
        low = float(value[0])
        high = float(value[1])
        if not np.isfinite([low, high]).all():
            raise ValueError(f"{name} must be finite, got {value}")
        if low <= 0 or high <= 0 or low >= high:
            raise ValueError(f"{name} must be (low, high) with 0 < low < high, got {value}")
        return (low, high)
    raise ValueError(f"{name} must be a length-2 sequence, got {value}")


def _build_gpr_kernel(cfg: Mapping[str, Any]) -> tuple[Any, Dict[str, Any]]:
    kernel_name = str(_cfg_get(cfg, "kernel", "rbf")).strip().lower()
    if kernel_name not in _VALID_GPR_KERNELS:
        raise ValueError(f"kernel must be one of {_VALID_GPR_KERNELS}, got {kernel_name}")

    # CONTRACT: kernel configuration is sourced from config for reproducibility.
    length_scale = np.asarray(_cfg_get(cfg, "length_scale", 1.0), dtype=float)
    if length_scale.ndim == 0:
        length_scale_value: float | list[float] = float(length_scale)
        if length_scale_value <= 0:
            raise ValueError("length_scale must be > 0")
    else:
        length_scale_value = [float(val) for val in length_scale.tolist()]
        if not np.isfinite(length_scale).all() or np.any(length_scale <= 0):
            raise ValueError("length_scale values must be finite and > 0")

    length_scale_bounds = _coerce_bounds(_cfg_get(cfg, "length_scale_bounds", (1e-2, 1e2)), "length_scale_bounds")
    kernel_constant = float(_cfg_get(cfg, "kernel_constant", 1.0))
    if kernel_constant <= 0:
        raise ValueError("kernel_constant must be > 0")
    kernel_constant_bounds = _coerce_bounds(
        _cfg_get(cfg, "kernel_constant_bounds", (1e-3, 1e3)),
        "kernel_constant_bounds",
    )

    kernel = ConstantKernel(kernel_constant, kernel_constant_bounds) * RBF(length_scale, length_scale_bounds)

    white_noise = bool(_cfg_get(cfg, "white_noise", True))
    noise_level = float(_cfg_get(cfg, "noise_level", 1e-6))
    if noise_level < 0:
        raise ValueError("noise_level must be >= 0")
    noise_level_bounds = _coerce_bounds(_cfg_get(cfg, "noise_level_bounds", (1e-8, 1e1)), "noise_level_bounds")
    if white_noise:
        kernel += WhiteKernel(noise_level=noise_level, noise_level_bounds=noise_level_bounds)

    kernel_cfg = {
        "kernel": kernel_name,
        "kernel_constant": kernel_constant,
        "kernel_constant_bounds": kernel_constant_bounds,
        "length_scale": length_scale_value,
        "length_scale_bounds": length_scale_bounds,
        "white_noise": white_noise,
        "noise_level": noise_level,
        "noise_level_bounds": noise_level_bounds,
    }
    return kernel, kernel_cfg


def register_regressor(name: str) -> Callable[[Callable[..., "BaseRegressor"]], Callable[..., "BaseRegressor"]]:
    def _wrapper(cls: Callable[..., "BaseRegressor"]) -> Callable[..., "BaseRegressor"]:
        if name in _REGRESSOR_REGISTRY:
            raise KeyError(f"Regressor already registered: {name}")
        _REGRESSOR_REGISTRY[name] = cls
        return cls

    return _wrapper


def list_regressors() -> tuple[str, ...]:
    return tuple(sorted(_REGRESSOR_REGISTRY.keys()))


def build_regressor(cfg: Mapping[str, Any]) -> "BaseRegressor":
    name = str(_cfg_get(cfg, "name", _cfg_get(cfg, "method", ""))).strip()
    if not name:
        raise ValueError("model.name is required")
    cls = _REGRESSOR_REGISTRY.get(name)
    if cls is None:
        raise KeyError(f"Unknown regressor: {name}. Available: {list_regressors()}")
    return cls(cfg=cfg)


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

    def predict_with_std(self, X_cond: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        return self.predict(X_cond), None

    def state(self) -> Mapping[str, Any]:
        if self._state is None:
            raise ValueError("state is not available before fit")
        return self._state

    def save_state(self, run_dir: str | Path) -> Path:
        out_dir = Path(run_dir) / "artifacts" / "model"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "model.pkl"
        # CONTRACT: model state is persisted under artifacts/model for reproducibility.
        with path.open("wb") as fh:
            pickle.dump(self, fh)
        return path

    @staticmethod
    def load_state(path: str | Path) -> "BaseRegressor":
        with Path(path).open("rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, BaseRegressor):
            raise TypeError("Loaded model state is not a BaseRegressor")
        return obj

    def _mark_fitted(self) -> None:
        self._fitted = True

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise ValueError("model must be fit before predict")


@register_regressor("ridge")
class RidgeRegressor(BaseRegressor):
    """Multi-output Ridge regressor for cond -> target."""

    def __init__(self, *, cfg: Mapping[str, Any]) -> None:
        super().__init__(cfg=cfg)
        self.name = "ridge"
        self.target_space = _resolve_target_space(cfg)
        self._alpha = float(_cfg_get(cfg, "alpha", 1.0))
        self._fit_intercept = bool(_cfg_get(cfg, "fit_intercept", True))
        self._solver = str(_cfg_get(cfg, "solver", "auto"))
        self._max_iter = _cfg_get(cfg, "max_iter", None)
        self._tol = _cfg_get(cfg, "tol", 1e-4)
        self._random_state = _resolve_random_state(cfg)
        self._cond_scaler_name = _resolve_scaler(cfg)
        self._scaler = StandardScaler() if self._cond_scaler_name == "standardize" else None
        self._model: Ridge | None = None
        self._target_dim: int | None = None

        if self._alpha < 0:
            raise ValueError("alpha must be >= 0")
        if not self._solver:
            raise ValueError("solver must be a non-empty string")
        if self._max_iter is not None:
            self._max_iter = int(self._max_iter)
        if self._tol is None:
            raise ValueError("tol must be set to a non-negative float")
        self._tol = float(self._tol)

    def fit(self, X_cond: np.ndarray, Y: np.ndarray) -> "RidgeRegressor":
        X_cond = _ensure_2d_features(X_cond, "X_cond")
        Y = _ensure_2d_targets(Y, X_cond.shape[0])

        if self._scaler is not None:
            X_cond = self._scaler.fit_transform(X_cond)

        model = Ridge(
            alpha=self._alpha,
            fit_intercept=self._fit_intercept,
            solver=self._solver,
            max_iter=self._max_iter,
            tol=self._tol,
            random_state=self._random_state,
        )
        model.fit(X_cond, Y)
        self._model = model
        self._target_dim = Y.shape[1]
        # REVIEW: state captures config + target_space for downstream artifact checks.
        self._state = {
            "method": self.name,
            "target_space": self.target_space,
            "alpha": self._alpha,
            "fit_intercept": self._fit_intercept,
            "solver": self._solver,
            "max_iter": self._max_iter,
            "tol": self._tol,
            "cond_scaler": self._cond_scaler_name,
            "random_state": self._random_state,
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
        pred = np.asarray(self._model.predict(X_cond))
        if pred.ndim == 1:
            pred = pred[:, None]
        if self._target_dim is not None and pred.shape[1] != self._target_dim:
            raise ValueError("predicted target dimension does not match fitted target_dim")
        return pred


@register_regressor("elasticnet")
class ElasticNetRegressor(BaseRegressor):
    """ElasticNet regressor for cond -> target."""

    def __init__(self, *, cfg: Mapping[str, Any]) -> None:
        super().__init__(cfg=cfg)
        self.name = "elasticnet"
        self.target_space = _resolve_target_space(cfg)
        self._alpha = float(_cfg_get(cfg, "alpha", 1.0))
        self._l1_ratio = float(_cfg_get(cfg, "l1_ratio", 0.5))
        self._fit_intercept = bool(_cfg_get(cfg, "fit_intercept", True))
        self._max_iter = _cfg_get(cfg, "max_iter", 1000)
        self._tol = _cfg_get(cfg, "tol", 1e-4)
        self._selection = _resolve_selection(cfg)
        self._positive = bool(_cfg_get(cfg, "positive", False))
        self._random_state = _resolve_random_state(cfg)
        self._cond_scaler_name = _resolve_scaler(cfg)
        self._scaler = StandardScaler() if self._cond_scaler_name == "standardize" else None
        self._model: ElasticNet | None = None
        self._target_dim: int | None = None

        if self._alpha < 0:
            raise ValueError("alpha must be >= 0")
        if not (0 <= self._l1_ratio <= 1):
            raise ValueError("l1_ratio must be between 0 and 1")
        if self._max_iter is None:
            self._max_iter = 1000
        self._max_iter = int(self._max_iter)
        if self._max_iter <= 0:
            raise ValueError("max_iter must be > 0")
        if self._tol is None:
            raise ValueError("tol must be set to a non-negative float")
        self._tol = float(self._tol)
        if self._tol < 0:
            raise ValueError("tol must be >= 0")

    def fit(self, X_cond: np.ndarray, Y: np.ndarray) -> "ElasticNetRegressor":
        X_cond = _ensure_2d_features(X_cond, "X_cond")
        Y = _ensure_2d_targets(Y, X_cond.shape[0])

        if self._scaler is not None:
            X_cond = self._scaler.fit_transform(X_cond)

        model = ElasticNet(
            alpha=self._alpha,
            l1_ratio=self._l1_ratio,
            fit_intercept=self._fit_intercept,
            max_iter=self._max_iter,
            tol=self._tol,
            positive=self._positive,
            random_state=self._random_state,
            selection=self._selection,
        )
        model.fit(X_cond, Y)
        self._model = model
        self._target_dim = Y.shape[1]
        # REVIEW: state captures ElasticNet hyperparameters for run comparison.
        self._state = {
            "method": self.name,
            "target_space": self.target_space,
            "alpha": self._alpha,
            "l1_ratio": self._l1_ratio,
            "fit_intercept": self._fit_intercept,
            "max_iter": self._max_iter,
            "tol": self._tol,
            "selection": self._selection,
            "positive": self._positive,
            "cond_scaler": self._cond_scaler_name,
            "random_state": self._random_state,
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
        pred = np.asarray(self._model.predict(X_cond))
        if pred.ndim == 1:
            pred = pred[:, None]
        if self._target_dim is not None and pred.shape[1] != self._target_dim:
            raise ValueError("predicted target dimension does not match fitted target_dim")
        return pred


@register_regressor("multitask_elasticnet")
class MultiTaskElasticNetRegressor(BaseRegressor):
    """MultiTask ElasticNet regressor for cond -> target."""

    def __init__(self, *, cfg: Mapping[str, Any]) -> None:
        super().__init__(cfg=cfg)
        self.name = "multitask_elasticnet"
        self.target_space = _resolve_target_space(cfg)
        self._alpha = float(_cfg_get(cfg, "alpha", 1.0))
        self._l1_ratio = float(_cfg_get(cfg, "l1_ratio", 0.5))
        self._fit_intercept = bool(_cfg_get(cfg, "fit_intercept", True))
        self._max_iter = _cfg_get(cfg, "max_iter", 1000)
        self._tol = _cfg_get(cfg, "tol", 1e-4)
        self._selection = _resolve_selection(cfg)
        self._random_state = _resolve_random_state(cfg)
        self._cond_scaler_name = _resolve_scaler(cfg)
        self._scaler = StandardScaler() if self._cond_scaler_name == "standardize" else None
        self._model: MultiTaskElasticNet | None = None
        self._target_dim: int | None = None

        if self._alpha < 0:
            raise ValueError("alpha must be >= 0")
        if not (0 <= self._l1_ratio <= 1):
            raise ValueError("l1_ratio must be between 0 and 1")
        if self._max_iter is None:
            self._max_iter = 1000
        self._max_iter = int(self._max_iter)
        if self._max_iter <= 0:
            raise ValueError("max_iter must be > 0")
        if self._tol is None:
            raise ValueError("tol must be set to a non-negative float")
        self._tol = float(self._tol)
        if self._tol < 0:
            raise ValueError("tol must be >= 0")

    def fit(self, X_cond: np.ndarray, Y: np.ndarray) -> "MultiTaskElasticNetRegressor":
        X_cond = _ensure_2d_features(X_cond, "X_cond")
        Y = _ensure_2d_targets(Y, X_cond.shape[0])

        if self._scaler is not None:
            X_cond = self._scaler.fit_transform(X_cond)

        model = MultiTaskElasticNet(
            alpha=self._alpha,
            l1_ratio=self._l1_ratio,
            fit_intercept=self._fit_intercept,
            max_iter=self._max_iter,
            tol=self._tol,
            random_state=self._random_state,
            selection=self._selection,
        )
        model.fit(X_cond, Y)
        self._model = model
        self._target_dim = Y.shape[1]
        # REVIEW: state captures shared-sparsity ElasticNet params for inspection.
        self._state = {
            "method": self.name,
            "target_space": self.target_space,
            "alpha": self._alpha,
            "l1_ratio": self._l1_ratio,
            "fit_intercept": self._fit_intercept,
            "max_iter": self._max_iter,
            "tol": self._tol,
            "selection": self._selection,
            "cond_scaler": self._cond_scaler_name,
            "random_state": self._random_state,
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
        pred = np.asarray(self._model.predict(X_cond))
        if pred.ndim == 1:
            pred = pred[:, None]
        if self._target_dim is not None and pred.shape[1] != self._target_dim:
            raise ValueError("predicted target dimension does not match fitted target_dim")
        return pred


@register_regressor("gpr")
class GPRRegressor(BaseRegressor):
    """Gaussian Process regressor for small-data cond -> target."""

    def __init__(self, *, cfg: Mapping[str, Any]) -> None:
        super().__init__(cfg=cfg)
        self.name = "gpr"
        self.target_space = _resolve_target_space(cfg)
        self._cond_scaler_name = _resolve_scaler(cfg)
        self._scaler = StandardScaler() if self._cond_scaler_name == "standardize" else None
        self._kernel, self._kernel_cfg = _build_gpr_kernel(cfg)
        self._alpha = float(_cfg_get(cfg, "alpha", 1e-10))
        if self._alpha < 0:
            raise ValueError("alpha must be >= 0")
        self._normalize_y = bool(_cfg_get(cfg, "normalize_y", True))
        self._optimizer = _resolve_optimizer(cfg)
        self._n_restarts_optimizer = int(_cfg_get(cfg, "n_restarts_optimizer", 0))
        if self._n_restarts_optimizer < 0:
            raise ValueError("n_restarts_optimizer must be >= 0")
        self._random_state = _resolve_random_state(cfg)
        self._model: GaussianProcessRegressor | None = None
        self._target_dim: int | None = None

    def fit(self, X_cond: np.ndarray, Y: np.ndarray) -> "GPRRegressor":
        X_cond = _ensure_2d_features(X_cond, "X_cond")
        Y = _ensure_2d_targets(Y, X_cond.shape[0])

        if self._scaler is not None:
            X_cond = self._scaler.fit_transform(X_cond)

        model = GaussianProcessRegressor(
            kernel=self._kernel,
            alpha=self._alpha,
            optimizer=self._optimizer,
            normalize_y=self._normalize_y,
            n_restarts_optimizer=self._n_restarts_optimizer,
            random_state=self._random_state,
        )
        model.fit(X_cond, Y)
        self._model = model
        self._target_dim = Y.shape[1]
        # REVIEW: capture kernel config + target_space for artifact inspection.
        self._state = {
            "method": self.name,
            "target_space": self.target_space,
            "cond_scaler": self._cond_scaler_name,
            "kernel": self._kernel_cfg["kernel"],
            "kernel_constant": self._kernel_cfg["kernel_constant"],
            "kernel_constant_bounds": self._kernel_cfg["kernel_constant_bounds"],
            "length_scale": self._kernel_cfg["length_scale"],
            "length_scale_bounds": self._kernel_cfg["length_scale_bounds"],
            "white_noise": self._kernel_cfg["white_noise"],
            "noise_level": self._kernel_cfg["noise_level"],
            "noise_level_bounds": self._kernel_cfg["noise_level_bounds"],
            "alpha": self._alpha,
            "normalize_y": self._normalize_y,
            "optimizer": self._optimizer,
            "n_restarts_optimizer": self._n_restarts_optimizer,
            "random_state": self._random_state,
            "kernel_repr": str(model.kernel_),
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
        pred = np.asarray(self._model.predict(X_cond))
        if pred.ndim == 1:
            pred = pred[:, None]
        if self._target_dim is not None and pred.shape[1] != self._target_dim:
            raise ValueError("predicted target dimension does not match fitted target_dim")
        return pred

    def predict_with_std(self, X_cond: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        self._require_fitted()
        if self._model is None:
            raise ValueError("model state is missing")
        X_cond = _ensure_2d_features(X_cond, "X_cond")
        if self._scaler is not None:
            X_cond = self._scaler.transform(X_cond)
        mean, std = self._model.predict(X_cond, return_std=True)
        mean = np.asarray(mean)
        std = np.asarray(std)
        if mean.ndim == 1:
            mean = mean[:, None]
        if std.ndim == 1:
            std = std[:, None]
        if self._target_dim is not None and mean.shape[1] != self._target_dim:
            raise ValueError("predicted target dimension does not match fitted target_dim")
        if std.shape != mean.shape:
            raise ValueError("predict_std shape does not match predicted mean")
        return mean, std


__all__ = [
    "BaseRegressor",
    "ElasticNetRegressor",
    "GPRRegressor",
    "MultiTaskElasticNetRegressor",
    "RidgeRegressor",
    "build_regressor",
    "list_regressors",
    "register_regressor",
]
