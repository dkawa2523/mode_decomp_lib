"""Fourier-Jacobi decomposer for disk domain."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from scipy.special import eval_jacobi

from mode_decomp_ml.domain import DomainSpec, validate_decomposer_compatibility
from mode_decomp_ml.plugins.registry import register_decomposer

from .base import BaseDecomposer, _combine_masks, _normalize_field, _normalize_mask, require_cfg

_NORMALIZATION_OPTIONS = {"orthonormal", "none"}
_ORDERING_OPTIONS = {"m_then_k"}
_MASK_POLICIES = {"ignore_masked_points"}


def _basis_cache_key(domain_spec: DomainSpec, *, m_max: int, k_max: int, normalization: str) -> tuple[Any, ...]:
    meta = domain_spec.meta or {}
    return (
        domain_spec.name,
        domain_spec.grid_shape,
        tuple(meta.get("center", ())),
        float(meta.get("radius", 0.0)),
        tuple(meta.get("x_range", ())),
        tuple(meta.get("y_range", ())),
        int(m_max),
        int(k_max),
        str(normalization),
    )


@register_decomposer("fourier_jacobi")
class FourierJacobiDecomposer(BaseDecomposer):
    """Fourier-Jacobi decomposer for disk domain (real coefficients)."""

    def __init__(self, *, cfg: Mapping[str, Any]) -> None:
        super().__init__(cfg=cfg)
        self.name = "fourier_jacobi"
        m_max = require_cfg(cfg, "m_max", label="decompose")
        k_max = require_cfg(cfg, "k_max", label="decompose")
        self._m_max = int(m_max)
        self._k_max = int(k_max)
        if self._m_max < 0:
            raise ValueError("decompose.m_max must be >= 0 for fourier_jacobi")
        if self._k_max < 0:
            raise ValueError("decompose.k_max must be >= 0 for fourier_jacobi")
        self._ordering = str(require_cfg(cfg, "ordering", label="decompose"))
        if self._ordering not in _ORDERING_OPTIONS:
            raise ValueError(f"decompose.ordering must be one of {_ORDERING_OPTIONS}, got {self._ordering}")
        self._normalization = str(require_cfg(cfg, "normalization", label="decompose"))
        if self._normalization not in _NORMALIZATION_OPTIONS:
            raise ValueError(
                f"decompose.normalization must be one of {_NORMALIZATION_OPTIONS}, got {self._normalization}"
            )
        # CONTRACT: boundary_condition is required for comparability across disk bases.
        self._boundary_condition = str(require_cfg(cfg, "boundary_condition", label="decompose"))
        self._mask_policy = str(require_cfg(cfg, "mask_policy", label="decompose"))
        if self._mask_policy not in _MASK_POLICIES:
            raise ValueError(f"decompose.mask_policy must be one of {_MASK_POLICIES}, got {self._mask_policy}")

        # Mode list defines coefficient index semantics.
        self._mode_list: list[tuple[int, int, str]] = []
        for m in range(self._m_max + 1):
            for k in range(self._k_max + 1):
                if m == 0:
                    self._mode_list.append((m, k, "cos"))
                else:
                    self._mode_list.append((m, k, "cos"))
                    self._mode_list.append((m, k, "sin"))

        self._coeff_shape: tuple[int, ...] | None = None
        self._field_ndim: int | None = None
        self._basis_cache: np.ndarray | None = None
        self._basis_cache_key: tuple[Any, ...] | None = None

    def _get_basis(self, domain_spec: DomainSpec) -> np.ndarray:
        key = _basis_cache_key(domain_spec, m_max=self._m_max, k_max=self._k_max, normalization=self._normalization)
        if self._basis_cache is not None and self._basis_cache_key == key:
            return self._basis_cache
        r = domain_spec.coords.get("r")
        theta = domain_spec.coords.get("theta")
        if r is None or theta is None:
            raise ValueError("disk domain must provide r/theta coords for fourier_jacobi")
        r = np.asarray(r, dtype=np.float64)
        theta = np.asarray(theta, dtype=np.float64)
        t = 2.0 * (r * r) - 1.0

        basis = np.empty((len(self._mode_list),) + r.shape, dtype=np.float64)
        for idx, (m, k, kind) in enumerate(self._mode_list):
            radial = np.power(r, m) * eval_jacobi(k, 0.0, float(m), t)
            if m == 0:
                angular = 1.0
            elif kind == "cos":
                angular = np.cos(m * theta)
            else:
                angular = np.sin(m * theta)
            basis[idx] = radial * angular
        if domain_spec.mask is not None:
            basis[:, ~domain_spec.mask] = 0.0
        if self._normalization == "orthonormal":
            weights = domain_spec.weights
            if weights is None:
                weights = np.ones(r.shape, dtype=np.float64)
            else:
                weights = np.asarray(weights, dtype=np.float64)
            if weights.shape != r.shape:
                raise ValueError(f"weights shape {weights.shape} does not match {r.shape}")
            if domain_spec.mask is not None:
                weights = np.where(domain_spec.mask, weights, 0.0)
            norms = np.sqrt(np.sum(basis * basis * weights, axis=(1, 2)))
            if np.any(norms <= 0):
                raise ValueError("fourier_jacobi basis has zero norm under weights")
            basis = basis / norms[:, None, None]

        self._basis_cache = basis
        self._basis_cache_key = key
        return basis

    def transform(
        self,
        field: np.ndarray,
        mask: np.ndarray | None,
        domain_spec: DomainSpec,
    ) -> np.ndarray:
        validate_decomposer_compatibility(domain_spec, self._cfg)
        field_3d, was_2d = _normalize_field(field)
        if field_3d.shape[:2] != domain_spec.grid_shape:
            raise ValueError(f"field shape {field_3d.shape[:2]} does not match domain {domain_spec.grid_shape}")

        field_mask = _normalize_mask(mask, field_3d.shape[:2])
        combined_mask = _combine_masks(field_mask, domain_spec.mask)

        weights = domain_spec.weights
        if weights is None:
            weights = np.ones(field_3d.shape[:2], dtype=np.float64)
        else:
            weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != field_3d.shape[:2]:
            raise ValueError(f"weights shape {weights.shape} does not match {field_3d.shape[:2]}")
        if combined_mask is not None:
            if self._mask_policy != "ignore_masked_points":
                raise ValueError("fourier_jacobi only supports mask_policy=ignore_masked_points")
            # REVIEW: masked points are ignored via zeroed weights to avoid silent fill.
            weights = np.where(combined_mask, weights, 0.0)

        basis = self._get_basis(domain_spec)
        weights_flat = weights.reshape(-1)
        valid = weights_flat > 0
        if not np.any(valid):
            raise ValueError("fourier_jacobi weights are empty after masking")
        basis_flat = basis.reshape(len(self._mode_list), -1).T
        design = basis_flat[valid]
        if design.shape[0] < design.shape[1]:
            raise ValueError("fourier_jacobi basis has more modes than valid samples")

        field_flat = field_3d.reshape(-1, field_3d.shape[-1])
        if not np.isfinite(field_flat[valid]).all():
            raise ValueError("field has non-finite values within valid mask")
        sqrt_w = np.sqrt(weights_flat[valid])
        design_w = design * sqrt_w[:, None]
        field_w = field_flat[valid] * sqrt_w[:, None]
        coeffs, _, rank, _ = np.linalg.lstsq(design_w, field_w, rcond=None)
        if rank < coeffs.shape[0]:
            raise ValueError("fourier_jacobi basis is rank-deficient; reduce m_max/k_max or check mask")

        coeff_tensor = coeffs.T
        channels = field_3d.shape[-1]
        self._coeff_shape = coeff_tensor.shape
        self._field_ndim = 2 if was_2d else 3
        self._coeff_meta = {
            "method": self.name,
            "field_shape": [int(field_3d.shape[0]), int(field_3d.shape[1])]
            if was_2d
            else [int(field_3d.shape[0]), int(field_3d.shape[1]), int(channels)],
            "field_ndim": self._field_ndim,
            "field_layout": "HW" if was_2d else "HWC",
            "channels": int(channels),
            "coeff_shape": [int(x) for x in coeff_tensor.shape],
            "coeff_layout": "CK",
            "flatten_order": "C",
            "complex_format": "real",
            "keep": "all",
            "m_max": int(self._m_max),
            "k_max": int(self._k_max),
            "ordering": self._ordering,
            "boundary_condition": self._boundary_condition,
            "normalization": self._normalization,
            "normalization_reference": "discrete_weighted_l2" if self._normalization == "orthonormal" else "none",
            "mode_list": [[int(m), int(k), kind] for (m, k, kind) in self._mode_list],
            "projection": "weighted_least_squares",
            "mask_policy": self._mask_policy,
        }
        return coeff_tensor

    def inverse_transform(self, coeff: np.ndarray, domain_spec: DomainSpec) -> np.ndarray:
        validate_decomposer_compatibility(domain_spec, self._cfg)
        if self._coeff_shape is None or self._field_ndim is None:
            raise ValueError("transform must be called before inverse_transform")
        coeff_tensor = self._reshape_coeff(coeff, self._coeff_shape, name=self.name)
        basis = self._get_basis(domain_spec)

        field_channels = []
        for idx in range(coeff_tensor.shape[0]):
            field_c = np.tensordot(coeff_tensor[idx], basis, axes=(0, 0))
            field_channels.append(field_c)
        field_hat = np.stack(field_channels, axis=-1)
        if domain_spec.mask is not None:
            field_hat = field_hat.copy()
            field_hat[~domain_spec.mask] = 0.0
        if self._field_ndim == 2 and field_hat.shape[-1] == 1:
            return field_hat[..., 0]
        return field_hat


__all__ = ["FourierJacobiDecomposer"]

