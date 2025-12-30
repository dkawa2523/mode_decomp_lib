"""Zernike decomposer for disk domain."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from scipy.special import comb

from mode_decomp_ml.domain import DomainSpec, validate_decomposer_compatibility

from . import BaseDecomposer, register_decomposer

_NORMALIZATION_OPTIONS = {"orthonormal", "none"}
_ORDERING_OPTIONS = {"n_then_m"}


def _cfg_get(cfg: Mapping[str, Any] | None, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        try:
            return cfg.get(key, default)
        except TypeError:
            pass
    return getattr(cfg, key, default)


def _normalize_field(field: np.ndarray) -> tuple[np.ndarray, bool]:
    field = np.asarray(field)
    if field.ndim == 2:
        return field[..., None], True
    if field.ndim == 3:
        return field, False
    raise ValueError(f"field must be 2D or 3D, got shape {field.shape}")


def _normalize_mask(mask: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray | None:
    if mask is None:
        return None
    mask = np.asarray(mask)
    if mask.shape != shape:
        raise ValueError(f"mask shape {mask.shape} does not match {shape}")
    return mask.astype(bool)


def _combine_masks(
    field_mask: np.ndarray | None,
    domain_mask: np.ndarray | None,
) -> np.ndarray | None:
    if field_mask is None and domain_mask is None:
        return None
    if field_mask is None:
        return domain_mask
    if domain_mask is None:
        return field_mask
    return field_mask & domain_mask


def _require_cfg(cfg: Mapping[str, Any], key: str) -> Any:
    value = _cfg_get(cfg, key, None)
    if value is None:
        raise ValueError(f"decompose.{key} is required for zernike")
    if isinstance(value, str) and not value.strip():
        raise ValueError(f"decompose.{key} must be non-empty for zernike")
    return value


def _build_nm_list(n_max: int, ordering: str) -> list[tuple[int, int]]:
    if ordering not in _ORDERING_OPTIONS:
        raise ValueError(f"zernike ordering must be one of {_ORDERING_OPTIONS}, got {ordering}")
    nm_list: list[tuple[int, int]] = []
    for n in range(n_max + 1):
        for m in range(-n, n + 1, 2):
            nm_list.append((n, m))
    return nm_list


def _radial_poly(n: int, m: int, r: np.ndarray) -> np.ndarray:
    m_abs = abs(m)
    if (n - m_abs) % 2 != 0:
        return np.zeros_like(r, dtype=np.float64)
    radial = np.zeros_like(r, dtype=np.float64)
    k_max = (n - m_abs) // 2
    for k in range(k_max + 1):
        coeff = (-1.0) ** k * comb(n - k, k) * comb(n - 2 * k, k_max - k)
        radial += coeff * np.power(r, n - 2 * k)
    return radial


def _zernike_mode(
    n: int,
    m: int,
    r: np.ndarray,
    theta: np.ndarray,
    normalization: str,
) -> np.ndarray:
    radial = _radial_poly(n, m, r)
    if m == 0:
        angular = 1.0
    elif m > 0:
        angular = np.cos(m * theta)
    else:
        angular = np.sin(abs(m) * theta)
    mode = radial * angular
    if normalization == "none":
        return mode
    if normalization == "orthonormal":
        factor = np.sqrt(n + 1.0) if m == 0 else np.sqrt(2.0 * (n + 1.0))
        return mode * factor
    raise ValueError(f"zernike normalization must be one of {_NORMALIZATION_OPTIONS}, got {normalization}")


def _basis_cache_key(domain_spec: DomainSpec) -> tuple[Any, ...]:
    meta = domain_spec.meta or {}
    return (
        domain_spec.name,
        domain_spec.grid_shape,
        tuple(meta.get("center", ())),
        float(meta.get("radius", 0.0)),
        tuple(meta.get("x_range", ())),
        tuple(meta.get("y_range", ())),
    )


@register_decomposer("zernike")
class ZernikeDecomposer(BaseDecomposer):
    """Zernike decomposer for disk domain (real coefficients)."""

    def __init__(self, *, cfg: Mapping[str, Any]) -> None:
        super().__init__(cfg=cfg)
        self.name = "zernike"
        n_max = _require_cfg(cfg, "n_max")
        self._n_max = int(n_max)
        if self._n_max < 0:
            raise ValueError("decompose.n_max must be >= 0 for zernike")
        self._ordering = str(_require_cfg(cfg, "ordering"))
        self._normalization = str(_require_cfg(cfg, "normalization"))
        if self._normalization not in _NORMALIZATION_OPTIONS:
            raise ValueError(
                f"decompose.normalization must be one of {_NORMALIZATION_OPTIONS}, got {self._normalization}"
            )
        # CONTRACT: boundary_condition is required for comparability across disk bases.
        self._boundary_condition = str(_require_cfg(cfg, "boundary_condition"))
        self._nm_list = _build_nm_list(self._n_max, self._ordering)
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
            raise ValueError("disk domain must provide r/theta coords for zernike")
        basis = np.empty((len(self._nm_list),) + r.shape, dtype=np.float64)
        for idx, (n, m) in enumerate(self._nm_list):
            basis[idx] = _zernike_mode(n, m, r, theta, self._normalization)
        if domain_spec.mask is not None:
            basis[:, ~domain_spec.mask] = 0.0
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
            # REVIEW: masked points are ignored via zeroed weights to avoid silent fill.
            weights = np.where(combined_mask, weights, 0.0)
        basis = self._get_basis(domain_spec)
        weights_flat = weights.reshape(-1)
        valid = weights_flat > 0
        if not np.any(valid):
            raise ValueError("zernike weights are empty after masking")
        basis_flat = basis.reshape(len(self._nm_list), -1).T
        design = basis_flat[valid]
        if design.shape[0] < design.shape[1]:
            raise ValueError("zernike basis has more modes than valid samples")
        field_flat = field_3d.reshape(-1, field_3d.shape[-1])
        if not np.isfinite(field_flat[valid]).all():
            raise ValueError("field has non-finite values within valid mask")
        sqrt_w = np.sqrt(weights_flat[valid])
        design_w = design * sqrt_w[:, None]
        field_w = field_flat[valid] * sqrt_w[:, None]
        coeffs, _, rank, _ = np.linalg.lstsq(design_w, field_w, rcond=None)
        if rank < coeffs.shape[0]:
            raise ValueError("zernike basis is rank-deficient; reduce n_max or check mask coverage")
        coeff_tensor = coeffs.T
        channels = field_3d.shape[-1]

        flat = coeff_tensor.reshape(-1, order="C")
        self._coeff_shape = coeff_tensor.shape
        self._field_ndim = 2 if was_2d else 3
        # REVIEW: coeff ordering + normalization + boundary_condition are required for comparability.
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
            "n_max": int(self._n_max),
            "ordering": self._ordering,
            "nm_list": [[int(n), int(m)] for (n, m) in self._nm_list],
            "normalization": self._normalization,
            "normalization_reference": "integral_over_unit_disk_equals_pi"
            if self._normalization == "orthonormal"
            else "none",
            "boundary_condition": self._boundary_condition,
            "projection": "weighted_least_squares",
            "mask_policy": "ignore_masked_points",
        }
        return flat

    def inverse_transform(self, coeff: np.ndarray, domain_spec: DomainSpec) -> np.ndarray:
        validate_decomposer_compatibility(domain_spec, self._cfg)
        if self._coeff_shape is None or self._field_ndim is None:
            raise ValueError("transform must be called before inverse_transform")
        coeff = np.asarray(coeff).reshape(-1)
        expected = int(np.prod(self._coeff_shape))
        if coeff.size > expected:
            raise ValueError(f"coeff size {coeff.size} exceeds expected {expected}")
        if coeff.size < expected:
            padded = np.zeros(expected, dtype=coeff.dtype)
            padded[: coeff.size] = coeff
            coeff = padded
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
