"""POD decomposer that can fit with varying masks via iterative missing-value imputation (EM/ALS style)."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from mode_decomp_ml.domain import DomainSpec, validate_decomposer_compatibility
from mode_decomp_ml.plugins.registry import register_decomposer

from .base import BaseDecomposer, ChannelwiseAdapter, _combine_masks, _normalize_field, _normalize_mask, require_cfg
from .pod_utils import auto_n_modes, parse_n_modes, solve_ridge, svd_snapshots_basis

_MASK_POLICIES = {"allow"}
_INNER_PRODUCTS = {"euclidean", "domain_weights"}
_INITS = {"mean_fill", "zero_fill"}

class _PODScalarEMDecomposer(BaseDecomposer):
    """Scalar POD-EM decomposer (no channel branching)."""

    def __init__(self, *, cfg: Mapping[str, Any]) -> None:
        super().__init__(cfg=cfg)
        self.name = "pod_em"
        n_modes = require_cfg(cfg, "n_modes", label="decompose")
        self._n_modes = parse_n_modes(n_modes, method="pod_em")
        self._n_modes_config = n_modes
        self._n_iter = int(require_cfg(cfg, "n_iter", label="decompose"))
        if self._n_iter < 1:
            raise ValueError("decompose.n_iter must be >= 1 for pod_em")
        self._ridge_alpha = float(require_cfg(cfg, "ridge_alpha", label="decompose"))
        if self._ridge_alpha < 0:
            raise ValueError("decompose.ridge_alpha must be >= 0 for pod_em")
        self._init = str(cfg.get("init", "mean_fill")).strip().lower() or "mean_fill"
        if self._init not in _INITS:
            raise ValueError(f"decompose.init must be one of {_INITS}, got {self._init}")
        self._inner_product = str(require_cfg(cfg, "inner_product", label="decompose")).strip().lower()
        if self._inner_product not in _INNER_PRODUCTS:
            raise ValueError(f"decompose.inner_product must be one of {_INNER_PRODUCTS}, got {self._inner_product}")
        self._mask_policy = str(require_cfg(cfg, "mask_policy", label="decompose")).strip().lower()
        if self._mask_policy not in _MASK_POLICIES:
            raise ValueError(f"decompose.mask_policy must be one of {_MASK_POLICIES}, got {self._mask_policy}")

        self._mu: np.ndarray | None = None  # (F,)
        self._basis: np.ndarray | None = None  # (F,K) in weighted feature space if domain_weights
        self._sqrt_w: np.ndarray | None = None  # (F,) or None
        self._grid_shape: tuple[int, int] | None = None
        self._n_samples: int | None = None
        self._n_modes_effective: int | None = None

        self._coeff_shape: tuple[int, ...] | None = None
        self._field_ndim: int | None = None

    def _require_weights(self, domain_spec: DomainSpec) -> np.ndarray:
        weights = domain_spec.integration_weights()
        if weights is None:
            raise ValueError("pod_em inner_product=domain_weights requires domain integration weights")
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != domain_spec.grid_shape:
            raise ValueError(f"weights shape {weights.shape} does not match {domain_spec.grid_shape}")
        if not np.isfinite(weights).all():
            raise ValueError("pod_em weights must be finite")
        return weights

    def _prepare_scalar(
        self,
        field: np.ndarray,
        mask: np.ndarray | None,
        domain_spec: DomainSpec,
    ) -> tuple[np.ndarray, np.ndarray]:
        arr, was_2d = _normalize_field(field)
        _ = was_2d
        if arr.shape[:2] != domain_spec.grid_shape:
            raise ValueError(f"field shape {arr.shape[:2]} does not match domain {domain_spec.grid_shape}")
        if arr.shape[-1] != 1:
            raise ValueError("pod_em scalar decomposer expects 1 channel")
        field_mask = _normalize_mask(mask, arr.shape[:2])
        combined = _combine_masks(field_mask, domain_spec.mask)
        if combined is None:
            combined = np.ones(domain_spec.grid_shape, dtype=bool)
        vec = np.asarray(arr[..., 0], dtype=np.float64).reshape(-1, order="C")
        m = np.asarray(combined, dtype=bool).reshape(-1, order="C")
        return vec, m

    def fit(self, dataset: Any | None = None, domain_spec: DomainSpec | None = None) -> "_PODScalarEMDecomposer":
        if dataset is None:
            raise ValueError("pod_em requires dataset for fit")
        if domain_spec is None:
            raise ValueError("pod_em requires domain_spec for fit")
        validate_decomposer_compatibility(domain_spec, self._cfg)
        self._grid_shape = domain_spec.grid_shape
        n_samples = int(len(dataset))
        if n_samples <= 0:
            raise ValueError("pod_em requires at least one sample")
        self._n_samples = n_samples

        sqrt_w = None
        if self._inner_product == "domain_weights":
            w = self._require_weights(domain_spec).reshape(-1, order="C")
            sqrt_w = np.sqrt(np.clip(w, 0.0, None))
            if not np.any(sqrt_w > 0):
                raise ValueError("pod_em weights are empty")
        self._sqrt_w = sqrt_w

        X_list: list[np.ndarray] = []
        M_list: list[np.ndarray] = []
        for idx in range(n_samples):
            sample = dataset[idx]
            vec, m = self._prepare_scalar(sample.field, sample.mask, domain_spec)
            if not np.isfinite(vec[m]).all():
                raise ValueError("pod_em training data contains non-finite values within observed mask")
            if int(np.count_nonzero(m)) == 0:
                raise ValueError("pod_em requires at least one observed entry per sample")
            X_list.append(vec)
            M_list.append(m)
        X = np.stack(X_list, axis=0)  # (N,F)
        M = np.stack(M_list, axis=0)  # (N,F)

        # Initial mean from observed values only.
        counts = np.sum(M, axis=0).astype(np.int64)
        sum_x = np.sum(X * M, axis=0)
        mu = np.zeros(X.shape[1], dtype=np.float64)
        valid = counts > 0
        mu[valid] = sum_x[valid] / counts[valid]

        if self._init == "zero_fill":
            X_filled = np.where(M, X, 0.0)
        else:
            X_filled = np.where(M, X, mu[None, :])

        n_features = int(X.shape[1])
        n_modes_req = self._n_modes
        if n_modes_req is None:
            n_modes_req = auto_n_modes(n_samples, n_features)
        if n_modes_req > min(n_samples, n_features):
            raise ValueError("pod_em n_modes exceeds available rank")

        # Initialize basis from current filled data.
        X_center = X_filled - mu[None, :]
        if sqrt_w is not None:
            X_center = X_center * sqrt_w[None, :]
        basis = svd_snapshots_basis(X_center, n_modes=int(n_modes_req), method="pod_em")
        n_modes_eff = int(n_modes_req)

        # EM/ALS iterations.
        for _ in range(self._n_iter):
            coeffs = np.zeros((n_samples, n_modes_eff), dtype=np.float64)
            for i in range(n_samples):
                obs = M[i]
                obs_idx = np.flatnonzero(obs)
                if obs_idx.size < n_modes_eff:
                    raise ValueError("pod_em requires at least n_modes observed entries per sample")
                y = X[i, obs_idx] - mu[obs_idx]
                if sqrt_w is not None:
                    y = y * sqrt_w[obs_idx]
                A = basis[obs_idx, :]
                coeffs[i] = solve_ridge(A, y, self._ridge_alpha, method="pod_em")

            # Update filled values using current reconstruction, but keep observed values fixed.
            for i in range(n_samples):
                if sqrt_w is not None:
                    centered_w = basis @ coeffs[i]
                    centered = np.zeros_like(centered_w)
                    nz = sqrt_w > 0
                    centered[nz] = centered_w[nz] / sqrt_w[nz]
                    x_hat = mu + centered
                else:
                    x_hat = mu + basis @ coeffs[i]
                miss = ~M[i]
                if np.any(miss):
                    X_filled[i, miss] = x_hat[miss]
                X_filled[i, M[i]] = X[i, M[i]]

            mu = np.mean(X_filled, axis=0)
            X_center = X_filled - mu[None, :]
            if sqrt_w is not None:
                X_center = X_center * sqrt_w[None, :]
            basis = svd_snapshots_basis(X_center, n_modes=int(n_modes_eff), method="pod_em")

        self._mu = mu
        self._basis = basis
        self._n_modes_effective = int(n_modes_eff)
        return self

    def transform(self, field: np.ndarray, mask: np.ndarray | None, domain_spec: DomainSpec) -> np.ndarray:
        validate_decomposer_compatibility(domain_spec, self._cfg)
        if self._mu is None or self._basis is None or self._grid_shape is None or self._n_modes_effective is None:
            raise ValueError("pod_em fit must be called before transform")
        if domain_spec.grid_shape != self._grid_shape:
            raise ValueError("pod_em domain grid does not match fit")

        arr, was_2d = _normalize_field(field)
        if arr.shape[-1] != 1:
            raise ValueError("pod_em scalar decomposer expects 1 channel")
        vec, m = self._prepare_scalar(arr[..., 0], mask, domain_spec)
        mu = self._mu
        basis = self._basis
        sqrt_w = self._sqrt_w
        if m is None or bool(np.all(m)):
            z = vec - mu
            if sqrt_w is not None:
                z = z * sqrt_w
            coeff_vec = z @ basis
        else:
            obs_idx = np.flatnonzero(m)
            if obs_idx.size < self._n_modes_effective:
                raise ValueError("pod_em requires at least n_modes observed entries per sample")
            y = vec[obs_idx] - mu[obs_idx]
            if sqrt_w is not None:
                y = y * sqrt_w[obs_idx]
            A = basis[obs_idx, :]
            coeff_vec = solve_ridge(A, y, self._ridge_alpha, method="pod_em")
        coeff_tensor = np.asarray(coeff_vec, dtype=np.float64)[None, :]

        valid_count = int(np.count_nonzero(m))
        valid_fraction = float(valid_count / float(m.size)) if m.size else 0.0
        self._coeff_shape = coeff_tensor.shape
        self._field_ndim = 2 if was_2d else 3
        meta = self._coeff_meta_base(
            field_shape=arr.shape[:2] if was_2d else arr.shape,
            field_ndim=self._field_ndim,
            field_layout="HW" if was_2d else "HWC",
            channels=1,
            coeff_shape=coeff_tensor.shape,
            coeff_layout="CK",
            complex_format="real",
        )
        meta.update(
            {
                "n_samples": int(self._n_samples) if self._n_samples is not None else None,
                "n_modes": int(self._n_modes_effective),
                "n_modes_config": self._n_modes_config,
                "n_iter": int(self._n_iter),
                "ridge_alpha": float(self._ridge_alpha),
                "init": str(self._init),
                "inner_product": self._inner_product,
                "mask_policy": self._mask_policy,
                "mask_valid_count": valid_count,
                "mask_valid_fraction": valid_fraction,
                "projection": "pod_em_svd_snapshots",
            }
        )
        self._coeff_meta = meta
        return coeff_tensor

    def inverse_transform(self, coeff: np.ndarray, domain_spec: DomainSpec) -> np.ndarray:
        validate_decomposer_compatibility(domain_spec, self._cfg)
        if (
            self._mu is None
            or self._basis is None
            or self._grid_shape is None
            or self._coeff_shape is None
            or self._field_ndim is None
        ):
            raise ValueError("pod_em transform must be called before inverse_transform")
        if domain_spec.grid_shape != self._grid_shape:
            raise ValueError("pod_em domain grid does not match fit")
        coeff_tensor = self._reshape_coeff(coeff, self._coeff_shape, name=self.name)
        coeff_vec = coeff_tensor.reshape(-1)
        centered_w = self._basis @ coeff_vec
        if self._sqrt_w is not None:
            sqrt_w = self._sqrt_w
            centered = np.zeros_like(centered_w)
            nz = sqrt_w > 0
            centered[nz] = centered_w[nz] / sqrt_w[nz]
        else:
            centered = centered_w
        vec = self._mu + centered
        height, width = self._grid_shape
        field_hat = vec.reshape((height, width), order="C")
        if domain_spec.mask is not None:
            field_hat = field_hat.copy()
            field_hat[~domain_spec.mask] = 0.0
        if self._field_ndim == 2:
            return field_hat
        return field_hat[..., None]


@register_decomposer("pod_em")
class PODEMDecomposer(BaseDecomposer):
    """POD-EM decomposer with optional channel-wise adapter for vector fields."""

    def __init__(self, *, cfg: Mapping[str, Any]) -> None:
        super().__init__(cfg=cfg)
        self.name = "pod_em"
        self._impl: BaseDecomposer | None = None

    def _make_scalar(self) -> _PODScalarEMDecomposer:
        return _PODScalarEMDecomposer(cfg=dict(self._cfg))

    def _ensure_impl(self, *, dataset: Any | None = None, field: np.ndarray | None = None) -> BaseDecomposer:
        if self._impl is not None:
            return self._impl
        if dataset is not None:
            if len(dataset) == 0:
                raise ValueError("pod_em requires non-empty dataset for fit")
            sample = dataset[0]
            field_arr = np.asarray(sample.field)
        elif field is not None:
            field_arr = np.asarray(field)
        else:
            raise ValueError("pod_em requires dataset or field to infer channel layout")
        if field_arr.ndim == 3 and int(field_arr.shape[-1]) > 1:
            self._impl = ChannelwiseAdapter(
                cfg=dict(self._cfg),
                decomposer_factory=lambda cfg: _PODScalarEMDecomposer(cfg=cfg),
                label=self.name,
            )
        else:
            self._impl = self._make_scalar()
        return self._impl

    def fit(self, dataset: Any | None = None, domain_spec: DomainSpec | None = None) -> "PODEMDecomposer":
        if dataset is None:
            raise ValueError("pod_em requires dataset for fit")
        impl = self._ensure_impl(dataset=dataset)
        impl.fit(dataset=dataset, domain_spec=domain_spec)
        return self

    def transform(self, field: np.ndarray, mask: np.ndarray | None, domain_spec: DomainSpec) -> np.ndarray:
        if self._impl is None:
            raise ValueError("pod_em fit must be called before transform")
        coeff = self._impl.transform(field, mask=mask, domain_spec=domain_spec)
        self._coeff_meta = dict(self._impl.coeff_meta())
        return coeff

    def inverse_transform(self, coeff: np.ndarray, domain_spec: DomainSpec) -> np.ndarray:
        if self._impl is None:
            raise ValueError("pod_em transform must be called before inverse_transform")
        return self._impl.inverse_transform(coeff, domain_spec=domain_spec)


__all__ = ["PODEMDecomposer"]
