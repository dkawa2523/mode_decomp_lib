"""Helmholtz decomposer for 2D vector fields via Poisson solves (rectangle domain)."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from scipy.fft import dctn, idctn, dstn, idstn

from mode_decomp_ml.domain import DomainSpec, validate_decomposer_compatibility
from mode_decomp_ml.plugins.registry import register_decomposer

from .base import BaseDecomposer, require_cfg_stripped
from .helmholtz import _fft_helmholtz, _grid_spacing

_BOUNDARY_CONDITIONS = {"periodic", "dirichlet", "neumann"}
_MASK_POLICIES = {"error"}


def _laplacian_eigs_dirichlet(n: int) -> np.ndarray:
    # Dirichlet on boundaries, interior points count n (= H-2 or W-2).
    k = np.arange(1, n + 1, dtype=np.float64)
    return 2.0 - 2.0 * np.cos(np.pi * k / float(n + 1))


def _laplacian_eigs_neumann(n: int) -> np.ndarray:
    # Neumann on boundaries, points count n (= H or W), k=0..n-1.
    # This matches the DCT-II diagonalization used below.
    k = np.arange(0, n, dtype=np.float64)
    return 2.0 - 2.0 * np.cos(np.pi * k / float(n))


def _poisson_solve(
    rhs: np.ndarray,
    *,
    dx: float,
    dy: float,
    bc: str,
) -> np.ndarray:
    f = np.asarray(rhs, dtype=np.float64)
    if f.ndim != 2:
        raise ValueError("poisson rhs must be 2D")
    height, width = f.shape
    if height < 3 or width < 3:
        raise ValueError("poisson solver requires grid at least 3x3")
    if bc == "dirichlet":
        # Solve on interior with homogeneous Dirichlet boundary.
        fin = f[1:-1, 1:-1]
        ny, nx = fin.shape
        fhat = dstn(fin, type=1, norm="ortho")
        lam_y = _laplacian_eigs_dirichlet(ny) / (dy * dy)
        lam_x = _laplacian_eigs_dirichlet(nx) / (dx * dx)
        denom = -(lam_y[:, None] + lam_x[None, :])
        uhat = fhat / denom
        uin = idstn(uhat, type=1, norm="ortho")
        out = np.zeros_like(f, dtype=np.float64)
        out[1:-1, 1:-1] = uin
        return out
    if bc == "neumann":
        # Homogeneous Neumann boundary via DCT-II on full grid.
        # NOTE: A solution exists only if mean(rhs)=0. Finite differences and masks can introduce
        # a small mean component, so we explicitly remove it to stabilize the solve.
        f = f - float(np.mean(f))
        fhat = dctn(f, type=2, norm="ortho")
        lam_y = _laplacian_eigs_neumann(height) / (dy * dy)
        lam_x = _laplacian_eigs_neumann(width) / (dx * dx)
        denom = -(lam_y[:, None] + lam_x[None, :])
        denom[0, 0] = 1.0
        uhat = fhat / denom
        uhat[0, 0] = 0.0
        return idctn(uhat, type=2, norm="ortho")
    raise ValueError(f"Unsupported poisson boundary_condition: {bc}")


def _div_curl(field: np.ndarray, *, dx: float, dy: float) -> tuple[np.ndarray, np.ndarray]:
    if field.ndim != 3 or field.shape[-1] != 2:
        raise ValueError("field must be (H,W,2)")
    u = np.asarray(field[..., 0], dtype=np.float64)
    v = np.asarray(field[..., 1], dtype=np.float64)
    du_dx = np.gradient(u, dx, axis=1, edge_order=2)
    dv_dy = np.gradient(v, dy, axis=0, edge_order=2)
    div = du_dx + dv_dy
    dv_dx = np.gradient(v, dx, axis=1, edge_order=2)
    du_dy = np.gradient(u, dy, axis=0, edge_order=2)
    curl = dv_dx - du_dy
    return div, curl


def _grad(phi: np.ndarray, *, dx: float, dy: float) -> np.ndarray:
    dphi_dx = np.gradient(phi, dx, axis=1, edge_order=2)
    dphi_dy = np.gradient(phi, dy, axis=0, edge_order=2)
    return np.stack([dphi_dx, dphi_dy], axis=-1)


def _rot_grad(psi: np.ndarray, *, dx: float, dy: float) -> np.ndarray:
    dpsi_dx = np.gradient(psi, dx, axis=1, edge_order=2)
    dpsi_dy = np.gradient(psi, dy, axis=0, edge_order=2)
    return np.stack([dpsi_dy, -dpsi_dx], axis=-1)


@register_decomposer("helmholtz_poisson")
class HelmholtzPoissonDecomposer(BaseDecomposer):
    """Helmholtz decomposer for rectangle domains using Poisson solves for potentials."""

    def __init__(self, *, cfg: Mapping[str, Any]) -> None:
        super().__init__(cfg=cfg)
        self.name = "helmholtz_poisson"
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

    def transform(self, field: np.ndarray, mask: np.ndarray | None, domain_spec: DomainSpec) -> np.ndarray:
        validate_decomposer_compatibility(domain_spec, self._cfg)
        field_3d, was_2d = self._prepare_field(field, mask, domain_spec, allow_zero_fill=False)
        if was_2d or field_3d.shape[-1] != 2:
            raise ValueError("helmholtz_poisson requires vector field with shape (H, W, 2)")

        dx, dy = _grid_spacing(domain_spec)
        if self._boundary_condition == "periodic":
            curl_free, div_free = _fft_helmholtz(field_3d, dx=dx, dy=dy)
            projection = "fft_helmholtz"
        else:
            div, curl = _div_curl(field_3d, dx=dx, dy=dy)
            phi = _poisson_solve(div, dx=dx, dy=dy, bc=self._boundary_condition)
            psi = _poisson_solve(-curl, dx=dx, dy=dy, bc=self._boundary_condition)
            curl_free = _grad(phi, dx=dx, dy=dy)
            div_free = _rot_grad(psi, dx=dx, dy=dy)
            projection = f"poisson_{self._boundary_condition}"

        coeff_tensor = np.stack([curl_free, div_free], axis=0)
        self._coeff_shape = coeff_tensor.shape
        self._field_ndim = 3
        self._grid_shape = domain_spec.grid_shape
        self._grid_spacing = (dx, dy)
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
            "projection": projection,
            "boundary_condition": self._boundary_condition,
            "mask_policy": self._mask_policy,
            "grid_spacing": [float(dx), float(dy)],
        }
        return coeff_tensor

    def inverse_transform(self, coeff: np.ndarray, domain_spec: DomainSpec) -> np.ndarray:
        validate_decomposer_compatibility(domain_spec, self._cfg)
        if self._coeff_shape is None or self._field_ndim is None:
            raise ValueError("helmholtz_poisson transform must be called before inverse_transform")
        if self._grid_shape is None:
            raise ValueError("helmholtz_poisson grid shape is not available")
        if domain_spec.grid_shape != self._grid_shape:
            raise ValueError("helmholtz_poisson domain grid does not match cached shape")

        coeff = np.asarray(coeff).reshape(-1)
        expected = int(np.prod(self._coeff_shape))
        if coeff.size > expected:
            raise ValueError(f"coeff size {coeff.size} exceeds expected {expected}")
        if coeff.size < expected:
            padded = np.zeros(expected, dtype=coeff.dtype)
            padded[: coeff.size] = coeff
            coeff = padded
        coeff_tensor = coeff.reshape(self._coeff_shape, order="C")
        return coeff_tensor.sum(axis=0)


__all__ = ["HelmholtzPoissonDecomposer"]
