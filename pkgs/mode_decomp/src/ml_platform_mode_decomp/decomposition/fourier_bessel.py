"""Fourier-Bessel decomposer for disk domain."""

from __future__ import annotations

from ml_platform_mode_decomp.config import constructor_config

from typing import Any, Mapping

import numpy as np
from scipy.special import jn_zeros, jnp_zeros, jv

from ml_platform_mode_decomp.geometry import DomainSpec, validate_decomposer_compatibility
from ml_platform_mode_decomp.geometry.identity import geometry_fingerprint

from .base import WeightedBasisDecomposerBase, require_cfg

_NORMALIZATION_OPTIONS = {"orthonormal", "none"}
_ORDERING_OPTIONS = {"m_then_n"}
_BOUNDARY_CONDITIONS = {"dirichlet", "neumann"}
_MASK_POLICIES = {"ignore_masked_points"}


def _basis_cache_key(domain_spec: DomainSpec) -> tuple[Any, ...]:
    return (geometry_fingerprint(domain_spec),)


class FourierBesselDecomposer(WeightedBasisDecomposerBase):
    """Fourier-Bessel decomposer for disk domain (real coefficients)."""

    synthesis_kind = "affine"

    def __init__(
        self,
        *,
        cfg: Mapping[str, Any] | None = None,
        m_max: Any = 4,
        n_max: Any = 4,
        ordering: Any = "m_then_n",
        normalization: Any = "orthonormal",
        boundary_condition: Any = "dirichlet",
        mask_policy: Any = "ignore_masked_points",
    ) -> None:
        cfg = constructor_config(cfg, locals(), name="fourier_bessel")
        super().__init__(cfg=cfg)
        self.name = "fourier_bessel"
        m_max = require_cfg(cfg, "m_max", label="decompose")
        n_max = require_cfg(cfg, "n_max", label="decompose")
        self._m_max = int(m_max)
        self._n_max = int(n_max)
        if self._m_max < 0:
            raise ValueError("decompose.m_max must be >= 0 for fourier_bessel")
        if self._n_max <= 0:
            raise ValueError("decompose.n_max must be > 0 for fourier_bessel")
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
        if self._boundary_condition not in _BOUNDARY_CONDITIONS:
            raise ValueError(
                f"decompose.boundary_condition must be one of {_BOUNDARY_CONDITIONS}, got {self._boundary_condition}"
            )
        self._mask_policy = str(require_cfg(cfg, "mask_policy", label="decompose"))
        if self._mask_policy not in _MASK_POLICIES:
            raise ValueError(f"decompose.mask_policy must be one of {_MASK_POLICIES}, got {self._mask_policy}")
        if self._boundary_condition == "dirichlet":
            self._radial_roots = [jn_zeros(m, self._n_max).astype(np.float64) for m in range(self._m_max + 1)]
        else:
            # Neumann BC includes a DC mode at root=0 for m=0 (constant radial component).
            root_arrays: list[np.ndarray] = []
            for m in range(self._m_max + 1):
                if m == 0:
                    if self._n_max == 1:
                        arr = np.array([0.0], dtype=np.float64)
                    else:
                        arr = np.concatenate(
                            [np.array([0.0], dtype=np.float64), jnp_zeros(0, self._n_max - 1).astype(np.float64)]
                        )
                    root_arrays.append(arr)
                else:
                    root_arrays.append(jnp_zeros(m, self._n_max).astype(np.float64))
            self._radial_roots = root_arrays
        self._mode_list: list[tuple[int, int, str, float]] = []
        for m in range(self._m_max + 1):
            roots = self._radial_roots[m]
            start = 0 if (self._boundary_condition == "neumann" and m == 0) else 1
            for n_idx, root in enumerate(roots, start=start):
                if m == 0:
                    self._mode_list.append((m, n_idx, "cos", float(root)))
                else:
                    self._mode_list.append((m, n_idx, "cos", float(root)))
                    self._mode_list.append((m, n_idx, "sin", float(root)))
        self._coeff_shape: tuple[int, ...] | None = None
        self._field_ndim: int | None = None
        self._basis_cache: np.ndarray | None = None
        self._basis_cache_key: tuple[Any, ...] | None = None

    def _get_basis(self, domain_spec: DomainSpec) -> np.ndarray:
        key = _basis_cache_key(domain_spec)
        if self._basis_cache is not None and self._basis_cache_key == key:
            return self._basis_cache
        r = domain_spec.coords.get("r")
        theta = domain_spec.coords.get("theta")
        if r is None or theta is None:
            raise ValueError("disk domain must provide r/theta coords for fourier_bessel")
        basis = np.empty((len(self._mode_list),) + r.shape, dtype=np.float64)
        for idx, (m, _, kind, root) in enumerate(self._mode_list):
            radial = jv(m, root * r)
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
                raise ValueError("fourier_bessel basis has zero norm under weights")
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
        field_3d, was_2d, weights, _ = self._prepare_zernike_inputs(field, mask, domain_spec)
        basis = self._get_basis(domain_spec)
        coeff_tensor = self._solve_weighted_least_squares(basis, field_3d, weights)
        return self._projection_layout(field_3d, was_2d, coeff_tensor, domain_spec)

    def _projection_layout(self, field_3d, was_2d, coeff_tensor, domain_spec):
        channels = field_3d.shape[-1]
        self._coeff_shape = coeff_tensor.shape
        self._field_ndim = 2 if was_2d else 3
        # REVIEW: mode ordering + normalization define coeff index semantics.
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
            "n_max": int(self._n_max),
            "ordering": self._ordering,
            "boundary_condition": self._boundary_condition,
            "normalization": self._normalization,
            "normalization_reference": "discrete_weighted_l2" if self._normalization == "orthonormal" else "none",
            "radial_root_type": "Jm_zero" if self._boundary_condition == "dirichlet" else "Jm_prime_zero",
            "mode_list": [[int(m), int(n), kind, float(root)] for (m, n, kind, root) in self._mode_list],
            "projection": "weighted_least_squares",
            "mask_policy": self._mask_policy,
        }
        return coeff_tensor

    def inverse_transform(self, coeff: np.ndarray, domain_spec: DomainSpec) -> np.ndarray:
        validate_decomposer_compatibility(domain_spec, self._cfg)
        if self._coeff_shape is None or self._field_ndim is None:
            raise ValueError("transform must be called before inverse_transform")
        coeff = np.asarray(coeff).reshape(-1)
        expected = int(np.prod(self._coeff_shape))
        if coeff.size > expected:
            raise ValueError(f"coeff size {coeff.size} exceeds expected {expected}")
        if coeff.size < expected:
            raise ValueError("Coefficient size is smaller than the fitted layout")
        coeff_tensor = coeff.reshape(self._coeff_shape, order="C")
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
