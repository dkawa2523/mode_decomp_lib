"""Helmholtz decomposer for 2D vector fields via FFT projection."""

from __future__ import annotations

from ml_platform_mode_decomp.config import constructor_config

from typing import Any, Mapping

import numpy as np

from ml_platform_mode_decomp.geometry import DomainSpec, validate_decomposer_compatibility

from .base import BaseDecomposer, require_cfg_stripped

_BOUNDARY_CONDITIONS = {"periodic"}
_MASK_POLICIES = {"error", "zero_fill"}
_FFT_NORM = "ortho"
_DERIVATIVE_CONVENTION = "spectral_first_derivative_nyquist_zero"


def _periodic_wavenumbers(count: int, spacing: float) -> np.ndarray:
    """Symbol of a real periodic first derivative on the sampled grid.

    An even grid's Nyquist cosine has no sampled sine partner. Its first
    derivative is therefore zero, as for real trigonometric interpolation.
    This odd symbol preserves conjugacy and makes the vector projection a
    real orthogonal projector, including all Nyquist sectors.
    """
    values = 2.0 * np.pi * np.fft.fftfreq(count, d=spacing)
    if count % 2 == 0:
        values[count // 2] = 0.0
    return values


def _grid_spacing(domain_spec: DomainSpec) -> tuple[float, float]:
    units = domain_spec.meta.get("coordinate_units", domain_spec.meta.get("units"))
    if isinstance(units, Mapping) and units.get("x") != units.get("y"):
        raise ValueError("helmholtz x/y must use the same coordinate unit; convert axes before decomposition")
    x = domain_spec.coords.get("x")
    y = domain_spec.coords.get("y")
    if x is None or y is None:
        raise ValueError("helmholtz requires grid coordinates x/y in domain_spec")
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("helmholtz requires 2D grid coordinates for x/y")
    dx = float(abs(x[0, 1] - x[0, 0])) if x.shape[1] > 1 else 1.0
    dy = float(abs(y[1, 0] - y[0, 0])) if y.shape[0] > 1 else 1.0
    if dx <= 0 or dy <= 0:
        raise ValueError("helmholtz grid spacing must be positive")
    return dx, dy


def _fft_helmholtz(
    field: np.ndarray,
    *,
    dx: float,
    dy: float,
) -> tuple[np.ndarray, np.ndarray]:
    height, width, channels = field.shape
    if channels != 2:
        raise ValueError("helmholtz expects vector field with 2 channels")
    u = field[..., 0]
    v = field[..., 1]
    u_hat = np.fft.fft2(u, norm=_FFT_NORM)
    v_hat = np.fft.fft2(v, norm=_FFT_NORM)

    kx = _periodic_wavenumbers(width, dx)
    ky = _periodic_wavenumbers(height, dy)
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing="xy")
    k2 = kx_grid**2 + ky_grid**2

    k_dot = kx_grid * u_hat + ky_grid * v_hat
    scale = np.zeros_like(k_dot, dtype=np.complex128)
    mask = k2 > 0
    scale[mask] = k_dot[mask] / k2[mask]

    u_curl_free_hat = kx_grid * scale
    v_curl_free_hat = ky_grid * scale
    u_div_free_hat = u_hat - u_curl_free_hat
    v_div_free_hat = v_hat - v_curl_free_hat

    # The common derivative nullspace (DC and even-grid Nyquist corners) is
    # assigned to div_free; it is not duplicated between the two parts.
    u_curl_free = np.fft.ifft2(u_curl_free_hat, norm=_FFT_NORM).real
    v_curl_free = np.fft.ifft2(v_curl_free_hat, norm=_FFT_NORM).real
    u_div_free = np.fft.ifft2(u_div_free_hat, norm=_FFT_NORM).real
    v_div_free = np.fft.ifft2(v_div_free_hat, norm=_FFT_NORM).real

    curl_free = np.stack([u_curl_free, v_curl_free], axis=-1)
    div_free = np.stack([u_div_free, v_div_free], axis=-1)
    return curl_free, div_free


class HelmholtzDecomposer(BaseDecomposer[np.ndarray]):
    """Helmholtz decomposer (curl-free + div-free parts) for 2D vector fields."""

    synthesis_kind = "component"

    def __init__(
        self, *, cfg: Mapping[str, Any] | None = None, boundary_condition: Any = "periodic", mask_policy: Any = "error"
    ) -> None:
        cfg = constructor_config(cfg, locals(), name="helmholtz")
        super().__init__(cfg=cfg)
        self.name = "helmholtz"
        self._boundary_condition = require_cfg_stripped(cfg, "boundary_condition", label="decompose")
        if self._boundary_condition not in _BOUNDARY_CONDITIONS:
            raise ValueError(
                f"decompose.boundary_condition must be one of {_BOUNDARY_CONDITIONS}, got {self._boundary_condition}"
            )
        self._mask_policy = require_cfg_stripped(cfg, "mask_policy", label="decompose")
        if self._mask_policy not in _MASK_POLICIES:
            raise ValueError(f"decompose.mask_policy must be one of {_MASK_POLICIES}, got {self._mask_policy}")
        self._coeff_shape: tuple[int, ...] | None = None
        self._field_ndim: int | None = None
        self._grid_shape: tuple[int, int] | None = None
        self._grid_spacing: tuple[float, float] | None = None

    def transform(
        self,
        field: np.ndarray,
        mask: np.ndarray | None,
        domain_spec: DomainSpec,
    ) -> np.ndarray:
        validate_decomposer_compatibility(domain_spec, self._cfg)
        allow_zero_fill = self._mask_policy == "zero_fill"
        field_3d, was_2d = self._prepare_field(field, mask, domain_spec, allow_zero_fill=allow_zero_fill)
        if was_2d or field_3d.shape[-1] != 2:
            raise ValueError("helmholtz requires vector field with shape (H, W, 2)")

        dx, dy = _grid_spacing(domain_spec)
        curl_free, div_free = _fft_helmholtz(field_3d, dx=dx, dy=dy)

        coeff_tensor = np.stack([curl_free, div_free], axis=0)
        self._coeff_shape = coeff_tensor.shape
        self._field_ndim = 3
        self._grid_shape = domain_spec.grid_shape
        self._grid_spacing = (dx, dy)
        # CONTRACT: part ordering and projection define coeff semantics.
        self._coeff_meta = {
            "method": self.name,
            "field_shape": [int(field_3d.shape[0]), int(field_3d.shape[1]), 2],
            "field_ndim": self._field_ndim,
            "field_layout": "HWC",
            "channels": 2,
            "coeff_shape": [int(x) for x in coeff_tensor.shape],
            "coeff_layout": "PHWC",
            "flatten_order": "C",
            "complex_format": "real",
            "keep": "all",
            "parts": ["curl_free", "div_free"],
            "part_axis": 0,
            "projection": "fft_helmholtz",
            "boundary_condition": self._boundary_condition,
            "mask_policy": self._mask_policy,
            "grid_spacing": [float(dx), float(dy)],
            "fft_norm": _FFT_NORM,
            "derivative_convention": _DERIVATIVE_CONVENTION,
            "projection_inner_product": "euclidean_grid",
            "derivative_nullspace_assignment": "div_free",
        }
        return coeff_tensor

    def validate_field_spec(self, spec, *, preprocess_name="none") -> None:
        spec.require_vector(self.name)
        if preprocess_name != "none":
            raise ValueError("Physical helmholtz decomposition requires field_preprocess.name: none")

    def inverse_transform(self, coeff: np.ndarray, domain_spec: DomainSpec) -> np.ndarray:
        validate_decomposer_compatibility(domain_spec, self._cfg)
        if self._coeff_shape is None or self._field_ndim is None:
            raise ValueError("helmholtz transform must be called before inverse_transform")
        if self._grid_shape is None:
            raise ValueError("helmholtz grid shape is not available")
        if domain_spec.grid_shape != self._grid_shape:
            raise ValueError("helmholtz domain grid does not match cached shape")

        coeff = np.asarray(coeff).reshape(-1)
        expected = int(np.prod(self._coeff_shape))
        if coeff.size > expected:
            raise ValueError(f"coeff size {coeff.size} exceeds expected {expected}")
        if coeff.size < expected:
            raise ValueError("Coefficient size is smaller than the fitted layout")
        coeff_tensor = coeff.reshape(self._coeff_shape, order="C")
        field_hat = coeff_tensor.sum(axis=0)
        return field_hat
