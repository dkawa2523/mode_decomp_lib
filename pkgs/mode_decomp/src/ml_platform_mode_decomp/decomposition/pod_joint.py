"""Joint POD/SVD decomposer across channels (vector fields) for grid domains."""

from __future__ import annotations

from ml_platform_mode_decomp.config import constructor_config

from typing import Any, Mapping

import numpy as np

from ml_platform_mode_decomp.geometry import DomainSpec, validate_decomposer_compatibility

from .base import BaseDecomposer, _combine_masks, _normalize_mask, require_cfg
from .pod_utils import parse_n_modes, resolve_centering, svd_snapshots_basis

_MASK_POLICIES = {"error", "zero_fill"}
_INNER_PRODUCTS = {"euclidean", "domain_weights"}


class JointPODDecomposer(BaseDecomposer[np.ndarray]):
    """POD/SVD over flattened (H*W*C) state vectors (captures cross-channel correlations)."""

    synthesis_kind = "affine"

    def __init__(
        self,
        *,
        cfg: Mapping[str, Any] | None = None,
        n_modes: Any = 16,
        center: Any = None,
        mean_centered: Any = None,
        mask_policy: Any = "zero_fill",
        inner_product: Any = "euclidean",
    ) -> None:
        cfg = constructor_config(cfg, locals(), name="pod_joint")
        super().__init__(cfg=cfg)
        self.name = "pod_joint"
        n_modes = require_cfg(cfg, "n_modes", label="decompose")
        self._n_modes = parse_n_modes(n_modes, method="pod_joint")
        self._n_modes_config = n_modes
        self._mask_policy = str(require_cfg(cfg, "mask_policy", label="decompose")).strip().lower()
        if self._mask_policy not in _MASK_POLICIES:
            raise ValueError(f"decompose.mask_policy must be one of {_MASK_POLICIES}, got {self._mask_policy}")
        self._inner_product = str(require_cfg(cfg, "inner_product", label="decompose")).strip().lower()
        if self._inner_product not in _INNER_PRODUCTS:
            raise ValueError(f"decompose.inner_product must be one of {_INNER_PRODUCTS}, got {self._inner_product}")
        self._mean_centered = resolve_centering(cfg, default=False)
        self._mu: np.ndarray | None = None

        self._basis: np.ndarray | None = None  # (F, K) in weighted feature space if domain_weights
        self._sqrt_w: np.ndarray | None = None  # (H*W,) or None
        self._grid_shape: tuple[int, int] | None = None
        self._channels: int | None = None
        self._n_samples: int | None = None
        self._n_modes_effective: int | None = None

        self._coeff_shape: tuple[int, ...] | None = None
        self._field_ndim: int | None = None

    def _require_weights(self, domain_spec: DomainSpec) -> np.ndarray:
        weights = domain_spec.integration_weights()
        if weights is None:
            raise ValueError("pod_joint inner_product=domain_weights requires domain integration weights")
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != domain_spec.grid_shape:
            raise ValueError(f"weights shape {weights.shape} does not match {domain_spec.grid_shape}")
        if not np.isfinite(weights).all():
            raise ValueError("pod_joint weights must be finite")
        return weights

    def _prepare_vector(
        self,
        field: np.ndarray,
        mask: np.ndarray | None,
        domain_spec: DomainSpec,
        *,
        channels_expected: int | None,
    ) -> tuple[np.ndarray, int]:
        arr = np.asarray(field)
        if arr.ndim != 3:
            raise ValueError(f"pod_joint expects 3D fields (H,W,C), got {arr.shape}")
        if arr.shape[:2] != domain_spec.grid_shape:
            raise ValueError(f"field shape {arr.shape[:2]} does not match domain {domain_spec.grid_shape}")
        channels = int(arr.shape[-1])
        if channels_expected is not None and channels != channels_expected:
            raise ValueError("pod_joint requires consistent channel count across samples")

        field_mask = _normalize_mask(mask, arr.shape[:2])
        combined = _combine_masks(field_mask, domain_spec.mask)
        if self._mask_policy == "error":
            if combined is not None and not combined.all():
                raise ValueError("pod_joint mask_policy=error does not allow masks")
        elif self._mask_policy == "zero_fill":
            if combined is not None and not combined.all():
                arr = arr.copy()
                arr[~combined] = 0.0
            elif domain_spec.mask is not None and not domain_spec.mask.all():
                arr = arr.copy()
                arr[~domain_spec.mask] = 0.0
        else:
            raise ValueError(f"Unsupported mask_policy for pod_joint: {self._mask_policy}")

        vec = arr.reshape(-1, order="C").astype(np.float64, copy=False)
        return vec, channels

    def fit(self, dataset: Any | None = None, domain_spec: DomainSpec | None = None) -> "JointPODDecomposer":
        if domain_spec is None:
            raise ValueError("pod_joint requires domain_spec for fit")
        if dataset is None:
            raise ValueError("pod_joint requires dataset for fit")
        validate_decomposer_compatibility(domain_spec, self._cfg)

        self._grid_shape = domain_spec.grid_shape
        n_samples = int(len(dataset))
        if n_samples <= 0:
            raise ValueError("pod_joint requires at least one sample")

        sqrt_w = None
        if self._inner_product == "domain_weights":
            w = self._require_weights(domain_spec)
            sqrt_w = np.sqrt(np.clip(w.reshape(-1, order="C"), 0.0, None))
            if not np.any(sqrt_w > 0):
                raise ValueError("pod_joint weights are empty")
        self._sqrt_w = sqrt_w

        first = dataset[0]
        vec, channels = self._prepare_vector(first.field, first.mask, domain_spec, channels_expected=None)
        X = np.empty((n_samples, len(vec)), dtype=np.float64)
        for idx in range(n_samples):
            if idx:
                sample = dataset[idx]
                vec, _ = self._prepare_vector(sample.field, sample.mask, domain_spec, channels_expected=channels)
            X[idx] = vec
        self._channels = int(channels)
        self._n_samples = n_samples

        self._mu = np.mean(X, axis=0) if self._mean_centered else np.zeros(X.shape[1])
        X -= self._mu
        if sqrt_w is not None:
            # Apply spatial weights per channel.
            sqrt_rep = np.repeat(sqrt_w, self._channels)
            X *= sqrt_rep[None, :]
        if not np.isfinite(X).all():
            raise ValueError("pod_joint training data contains non-finite values")

        self._basis = svd_snapshots_basis(X, n_modes=self._n_modes, method="pod_joint")
        self._n_modes_effective = int(self._basis.shape[1])
        return self

    def transform(self, field: np.ndarray, mask: np.ndarray | None, domain_spec: DomainSpec) -> np.ndarray:
        validate_decomposer_compatibility(domain_spec, self._cfg)
        if self._basis is None or self._grid_shape is None or self._channels is None:
            raise ValueError("pod_joint fit must be called before transform")
        if domain_spec.grid_shape != self._grid_shape:
            raise ValueError("pod_joint domain grid does not match fit")

        vec, channels = self._prepare_vector(field, mask, domain_spec, channels_expected=self._channels)
        if channels != self._channels:
            raise ValueError("pod_joint field channels do not match fit")
        vec = vec - self._mu

        if self._sqrt_w is not None:
            sqrt_rep = np.repeat(self._sqrt_w, self._channels)
            vec = vec * sqrt_rep
        coeff_vec = vec @ self._basis
        coeff_tensor = np.asarray(coeff_vec, dtype=np.float64).reshape(1, -1)

        self._coeff_shape = coeff_tensor.shape
        self._field_ndim = 3
        self._coeff_meta = {
            "method": self.name,
            "field_shape": [int(self._grid_shape[0]), int(self._grid_shape[1]), int(self._channels)],
            "field_ndim": int(self._field_ndim),
            "field_layout": "HWC",
            "channels": int(self._channels),
            "coeff_shape": [int(x) for x in coeff_tensor.shape],
            "coeff_layout": "CK",
            "flatten_order": "C",
            "complex_format": "real",
            "keep": "all",
            "n_samples": int(self._n_samples) if self._n_samples is not None else None,
            "n_modes": int(self._n_modes_effective) if self._n_modes_effective is not None else None,
            "n_modes_config": self._n_modes_config,
            "projection": "svd_snapshots",
            "inner_product": self._inner_product,
            "mask_policy": self._mask_policy,
            "mean_centered": bool(self._mean_centered),
        }
        return coeff_tensor

    def inverse_transform(self, coeff: np.ndarray, domain_spec: DomainSpec) -> np.ndarray:
        validate_decomposer_compatibility(domain_spec, self._cfg)
        if self._basis is None or self._coeff_shape is None or self._grid_shape is None or self._channels is None:
            raise ValueError("pod_joint transform must be called before inverse_transform")
        if domain_spec.grid_shape != self._grid_shape:
            raise ValueError("pod_joint domain grid does not match fit")

        coeff_tensor = self._reshape_coeff(coeff, self._coeff_shape, name=self.name)
        vec_w = coeff_tensor.reshape(-1) @ self._basis.T  # (F,)
        if self._sqrt_w is not None:
            sqrt_rep = np.repeat(self._sqrt_w, self._channels)
            out = np.zeros_like(vec_w)
            valid = sqrt_rep > 0
            out[valid] = vec_w[valid] / sqrt_rep[valid]
            vec = out
        else:
            vec = vec_w
        vec = vec + self._mu
        field_hat = vec.reshape((self._grid_shape[0], self._grid_shape[1], self._channels), order="C")
        if domain_spec.mask is not None:
            field_hat = field_hat.copy()
            field_hat[~domain_spec.mask] = 0.0
        return field_hat


__all__ = ["JointPODDecomposer"]
