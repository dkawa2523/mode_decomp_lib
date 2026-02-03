"""Optional multi-task Gaussian Process regressor (gpytorch)."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from sklearn.preprocessing import StandardScaler

from mode_decomp_ml.optional import require_dependency
from mode_decomp_ml.plugins.models.base import (
    BaseRegressor,
    cfg_get,
    _ensure_2d_features,
    _ensure_2d_predictions,
    _ensure_2d_targets,
)
from mode_decomp_ml.plugins.registry import register_regressor

try:  # optional dependency
    import torch
    import gpytorch
except Exception:  # pragma: no cover - gpytorch is optional
    torch = None
    gpytorch = None

_VALID_TARGET_SPACES = {"a", "z"}
_VALID_SCALERS = {"none", "standardize"}
_VALID_KERNELS = {"rbf"}


def _require_dependency() -> None:
    if torch is None or gpytorch is None:
        require_dependency(
            None,
            name="mtgp regressor",
            pip_name="torch gpytorch",
        )


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


def _resolve_kernel(cfg: Mapping[str, Any]) -> str:
    kernel = str(cfg_get(cfg, "kernel", "rbf")).strip().lower()
    if not kernel:
        kernel = "rbf"
    if kernel not in _VALID_KERNELS:
        raise ValueError(f"kernel must be one of {_VALID_KERNELS}, got {kernel}")
    return kernel


def _resolve_rank(cfg: Mapping[str, Any]) -> int:
    rank = int(cfg_get(cfg, "rank", 1))
    if rank <= 0:
        raise ValueError("rank must be > 0")
    return rank


def _resolve_training_iter(cfg: Mapping[str, Any]) -> int:
    training_iter = int(cfg_get(cfg, "training_iter", 50))
    if training_iter <= 0:
        raise ValueError("training_iter must be > 0")
    return training_iter


def _resolve_learning_rate(cfg: Mapping[str, Any]) -> float:
    lr = float(cfg_get(cfg, "learning_rate", 0.1))
    if lr <= 0:
        raise ValueError("learning_rate must be > 0")
    return lr


if gpytorch is not None and torch is not None:
    class _MTGPModel(gpytorch.models.ExactGP):
        def __init__(
            self,
            train_x: torch.Tensor,
            train_y: torch.Tensor,
            likelihood: gpytorch.likelihoods.MultitaskGaussianLikelihood,
            *,
            num_tasks: int,
            rank: int,
            kernel_name: str,
        ) -> None:
            super().__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.MultitaskMean(
                gpytorch.means.ConstantMean(), num_tasks=num_tasks
            )
            if kernel_name == "rbf":
                base_kernel = gpytorch.kernels.RBFKernel()
            else:
                raise ValueError(f"Unsupported kernel: {kernel_name}")
            self.covar_module = gpytorch.kernels.MultitaskKernel(
                base_kernel,
                num_tasks=num_tasks,
                rank=rank,
            )

        def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultitaskMultivariateNormal:
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)
else:
    _MTGPModel = None


@register_regressor("mtgp")
class MTGPRegressor(BaseRegressor):
    """Multi-task GP regressor with correlated outputs (optional gpytorch)."""

    def __init__(self, *, cfg: Mapping[str, Any]) -> None:
        super().__init__(cfg=cfg)
        _require_dependency()
        self.name = "mtgp"
        self.target_space = _resolve_target_space(cfg)
        self._cond_scaler_name = _resolve_scaler(cfg)
        self._scaler = StandardScaler() if self._cond_scaler_name == "standardize" else None
        self._kernel_name = _resolve_kernel(cfg)
        self._rank = _resolve_rank(cfg)
        self._training_iter = _resolve_training_iter(cfg)
        self._learning_rate = _resolve_learning_rate(cfg)
        self._random_state = _resolve_random_state(cfg)
        self._model: Any | None = None
        self._likelihood: Any | None = None
        self._target_dim: int | None = None

    def fit(self, X_cond: np.ndarray, Y: np.ndarray) -> "MTGPRegressor":
        _require_dependency()
        X_cond = _ensure_2d_features(X_cond, "X_cond")
        Y = _ensure_2d_targets(Y, X_cond.shape[0])

        if self._scaler is not None:
            X_cond = self._scaler.fit_transform(X_cond)

        if self._random_state is not None:
            np.random.seed(self._random_state)
            torch.manual_seed(self._random_state)

        train_x = torch.as_tensor(X_cond, dtype=torch.float32)
        train_y = torch.as_tensor(Y, dtype=torch.float32)
        num_tasks = int(train_y.shape[1])
        if self._rank > num_tasks:
            raise ValueError("rank must be <= number of tasks")

        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks)
        model = _MTGPModel(
            train_x,
            train_y,
            likelihood,
            num_tasks=num_tasks,
            rank=self._rank,
            kernel_name=self._kernel_name,
        )
        model.train()
        likelihood.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=self._learning_rate)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        final_loss = None
        for _ in range(self._training_iter):
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()
            final_loss = float(loss.item())

        self._model = model
        self._likelihood = likelihood
        self._target_dim = num_tasks
        self._state = {
            "method": self.name,
            "target_space": self.target_space,
            "cond_scaler": self._cond_scaler_name,
            "kernel": self._kernel_name,
            "rank": self._rank,
            "training_iter": self._training_iter,
            "learning_rate": self._learning_rate,
            "random_state": self._random_state,
            "final_loss": final_loss,
        }
        self._mark_fitted()
        return self

    def _predict_with_std(self, X_cond: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self._require_fitted()
        if self._model is None or self._likelihood is None:
            raise ValueError("model state is missing")
        X_cond = _ensure_2d_features(X_cond, "X_cond")
        if self._scaler is not None:
            X_cond = self._scaler.transform(X_cond)
        train_x = self._model.train_inputs[0]
        test_x = torch.as_tensor(X_cond, dtype=train_x.dtype, device=train_x.device)

        self._model.eval()
        self._likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred_dist = self._likelihood(self._model(test_x))
            mean = pred_dist.mean
            std = pred_dist.stddev

        mean_np = mean.detach().cpu().numpy()
        std_np = std.detach().cpu().numpy()
        mean_np = _ensure_2d_predictions(mean_np, n_samples=X_cond.shape[0], target_dim=self._target_dim)
        std_np = _ensure_2d_predictions(std_np, n_samples=X_cond.shape[0], target_dim=self._target_dim)
        if mean_np.shape != std_np.shape:
            raise ValueError("predict_std shape does not match predicted mean")
        return mean_np, std_np

    def predict(self, X_cond: np.ndarray) -> np.ndarray:
        mean, _ = self._predict_with_std(X_cond)
        return mean

    def predict_std(self, X_cond: np.ndarray) -> np.ndarray | None:
        _, std = self._predict_with_std(X_cond)
        return std

    def predict_with_std(self, X_cond: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        mean, std = self._predict_with_std(X_cond)
        return mean, std


__all__ = ["MTGPRegressor"]
