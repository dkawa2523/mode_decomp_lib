"""Spherical harmonics decomposer for sphere_grid domain."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from mode_decomp_ml.config import cfg_get
try:  # SciPy 1.15+ deprecates sph_harm in favor of sph_harm_y
    from scipy.special import sph_harm_y as _sph_harm_y

    def _sph_harm(m: int, l: int, theta: np.ndarray, phi: np.ndarray):
        # `sph_harm_y(n, m, theta, phi)` uses (theta=polar, phi=azimuth) while the deprecated
        # `sph_harm(m, n, theta, phi)` uses (theta=azimuth, phi=polar). Swap angles to preserve
        # `sph_harm` semantics for downstream code.
        return _sph_harm_y(l, m, phi, theta)
except Exception:  # pragma: no cover - fallback for older SciPy
    from scipy.special import sph_harm as _sph_harm

from mode_decomp_ml.domain import DomainSpec, validate_decomposer_compatibility
from mode_decomp_ml.optional import require_dependency
from mode_decomp_ml.plugins.registry import register_decomposer

from .base import BaseDecomposer, _combine_masks, _normalize_field, _normalize_mask, require_cfg, parse_bool

try:  # optional dependency
    import pyshtools  # noqa: F401
except Exception:  # pragma: no cover - optional dependency
    pyshtools = None

_ALLOWED_BACKENDS = {"pyshtools", "scipy"}
_ALLOWED_NORMS = {"ortho", "schmidt", "4pi"}


def _basis_cache_key(domain_spec: DomainSpec) -> tuple[Any, ...]:
    meta = domain_spec.meta or {}
    return (
        domain_spec.name,
        domain_spec.grid_shape,
        float(meta.get("radius", 1.0)),
        tuple(meta.get("lat_range", ())),
        tuple(meta.get("lon_range", ())),
        str(meta.get("angle_unit", "")),
    )


def _real_mode_sign(m: int) -> float:
    return -1.0 if (m % 2) else 1.0


@register_decomposer("spherical_harmonics")
class SphericalHarmonicsDecomposer(BaseDecomposer):
    """Spherical harmonics decomposer for sphere_grid domain."""

    def __init__(self, *, cfg: Mapping[str, Any]) -> None:
        super().__init__(cfg=cfg)
        self.name = "spherical_harmonics"
        l_max = require_cfg(cfg, "l_max", label="decompose")
        self._l_max = int(l_max)
        if self._l_max < 0:
            raise ValueError("decompose.l_max must be >= 0 for spherical_harmonics")
        self._real_form = parse_bool(cfg_get(cfg, "real_form", True), default=True)
        self._norm = str(cfg_get(cfg, "norm", "ortho")).strip().lower() or "ortho"
        if self._norm not in _ALLOWED_NORMS:
            raise ValueError(
                f"decompose.norm must be one of {_ALLOWED_NORMS}, got {self._norm}"
            )
        self._backend = str(cfg_get(cfg, "backend", "pyshtools")).strip().lower() or "pyshtools"
        if self._backend not in _ALLOWED_BACKENDS:
            raise ValueError(
                f"decompose.backend must be one of {_ALLOWED_BACKENDS}, got {self._backend}"
            )

        self._lm_kind_list: list[list[Any]] = []
        if self._real_form:
            for l in range(self._l_max + 1):
                self._lm_kind_list.append([int(l), 0, "cos"])
                for m in range(1, l + 1):
                    self._lm_kind_list.append([int(l), int(m), "cos"])
                    self._lm_kind_list.append([int(l), int(m), "sin"])
        else:
            for l in range(self._l_max + 1):
                for m in range(-l, l + 1):
                    self._lm_kind_list.append([int(l), int(m), "complex"])

        self._coeff_shape: tuple[int, ...] | None = None
        self._field_ndim: int | None = None
        self._grid_shape: tuple[int, int] | None = None
        self._basis_cache: np.ndarray | None = None
        self._basis_cache_key: tuple[Any, ...] | None = None

    def _require_backend(self) -> None:
        if self._backend == "pyshtools":
            require_dependency(
                pyshtools,
                name="spherical_harmonics decomposer",
                pip_name="pyshtools",
                extra_hint="Set decompose.backend=scipy to use the SciPy backend.",
            )

    def _get_basis(self, domain_spec: DomainSpec) -> np.ndarray:
        key = (_basis_cache_key(domain_spec), self._l_max, self._real_form, self._norm, self._backend)
        if self._basis_cache is not None and self._basis_cache_key == key:
            return self._basis_cache

        theta = domain_spec.coords.get("theta")
        phi = domain_spec.coords.get("phi")
        if theta is None or phi is None:
            raise ValueError("sphere_grid domain must provide theta/phi coords for spherical_harmonics")
        theta = np.asarray(theta, dtype=np.float64)
        phi = np.asarray(phi, dtype=np.float64)

        self._require_backend()

        n_modes = len(self._lm_kind_list)
        basis_dtype = np.float64 if self._real_form else np.complex128
        basis = np.empty((n_modes,) + theta.shape, dtype=basis_dtype)
        for idx, (l_val, m_val, kind) in enumerate(self._lm_kind_list):
            l = int(l_val)
            m = int(m_val)
            if self._real_form:
                if m == 0:
                    mode = _sph_harm(0, l, theta, phi).real
                else:
                    ylm = _sph_harm(m, l, theta, phi)
                    factor = np.sqrt(2.0) * _real_mode_sign(m)
                    if kind == "cos":
                        mode = factor * ylm.real
                    else:
                        mode = factor * ylm.imag
                basis[idx] = mode
            else:
                basis[idx] = _sph_harm(m, l, theta, phi)

        if domain_spec.mask is not None:
            basis[:, ~domain_spec.mask] = 0.0

        self._basis_cache = basis
        self._basis_cache_key = key
        return basis

    @staticmethod
    def _solve_weighted_least_squares(
        basis: np.ndarray,
        field_3d: np.ndarray,
        weights: np.ndarray,
    ) -> np.ndarray:
        weights_flat = weights.reshape(-1)
        valid = weights_flat > 0
        if not np.any(valid):
            raise ValueError("spherical_harmonics weights are empty after masking")
        basis_flat = basis.reshape(basis.shape[0], -1).T
        design = basis_flat[valid]
        if design.shape[0] < design.shape[1]:
            raise ValueError("spherical_harmonics basis has more modes than valid samples")
        field_flat = field_3d.reshape(-1, field_3d.shape[-1])
        if not np.isfinite(field_flat[valid]).all():
            raise ValueError("field has non-finite values within valid mask")
        sqrt_w = np.sqrt(weights_flat[valid])
        design_w = design * sqrt_w[:, None]
        field_w = field_flat[valid] * sqrt_w[:, None]
        coeffs, _, rank, _ = np.linalg.lstsq(design_w, field_w, rcond=None)
        if rank < coeffs.shape[0]:
            raise ValueError("spherical_harmonics basis is rank-deficient; reduce l_max or check mask")
        return coeffs.T

    def transform(
        self,
        field: np.ndarray,
        mask: np.ndarray | None,
        domain_spec: DomainSpec,
    ) -> np.ndarray:
        validate_decomposer_compatibility(domain_spec, self._cfg)
        field_3d, was_2d = _normalize_field(field)
        if field_3d.shape[:2] != domain_spec.grid_shape:
            raise ValueError(
                f"field shape {field_3d.shape[:2]} does not match domain {domain_spec.grid_shape}"
            )
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
            weights = np.where(combined_mask, weights, 0.0)

        basis = self._get_basis(domain_spec)
        coeff_tensor = self._solve_weighted_least_squares(basis, field_3d, weights)

        channels = field_3d.shape[-1]
        self._coeff_shape = coeff_tensor.shape
        self._field_ndim = 2 if was_2d else 3
        self._grid_shape = domain_spec.grid_shape

        meta = self._coeff_meta_base(
            field_shape=field_3d.shape[:2] if was_2d else field_3d.shape,
            field_ndim=self._field_ndim,
            field_layout="HW" if was_2d else "HWC",
            channels=int(channels),
            coeff_shape=coeff_tensor.shape,
            coeff_layout="CK",
            complex_format="real" if self._real_form else "complex",
        )
        meta.update(
            {
                "l_max": int(self._l_max),
                "real_form": bool(self._real_form),
                "norm": self._norm,
                "backend": self._backend,
                "lm_kind_list": self._lm_kind_list,
                "projection": "weighted_least_squares",
                "mask_policy": "ignore_masked_points",
            }
        )
        self._coeff_meta = meta
        return coeff_tensor

    def inverse_transform(self, coeff: np.ndarray, domain_spec: DomainSpec) -> np.ndarray:
        validate_decomposer_compatibility(domain_spec, self._cfg)
        if self._coeff_shape is None or self._field_ndim is None:
            raise ValueError("transform must be called before inverse_transform")
        if self._grid_shape is not None and domain_spec.grid_shape != self._grid_shape:
            raise ValueError("spherical_harmonics domain grid does not match cached shape")
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


__all__ = ["SphericalHarmonicsDecomposer"]
