"""PSWF2D tensor decomposer using discrete prolate spheroidal sequences."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from mode_decomp_ml.config import cfg_get

from mode_decomp_ml.domain import DomainSpec
from mode_decomp_ml.optional import require_dependency
from mode_decomp_ml.plugins.decomposers.base import GridDecomposerBase, require_cfg
from mode_decomp_ml.plugins.registry import register_decomposer

try:  # optional dependency
    from scipy.signal.windows import dpss as _dpss
except Exception:  # pragma: no cover - SciPy optional for this decomposer
    _dpss = None

_MASK_POLICIES = {"error", "zero_fill"}
_DPSS_NORM = 2



def _parse_positive_int(value: Any, *, key: str) -> int:
    if isinstance(value, (int, np.integer)):
        num = int(value)
    elif isinstance(value, str) and value.strip():
        num = int(value)
    else:
        raise ValueError(f"decompose.{key} must be a positive int for pswf2d_tensor")
    if num <= 0:
        raise ValueError(f"decompose.{key} must be positive for pswf2d_tensor")
    return num


def _parse_positive_float(value: Any, *, key: str) -> float:
    try:
        num = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"decompose.{key} must be a positive float for pswf2d_tensor") from exc
    if not np.isfinite(num) or num <= 0:
        raise ValueError(f"decompose.{key} must be a positive float for pswf2d_tensor")
    return float(num)


def _require_dpss() -> None:
    require_dependency(
        _dpss,
        name="pswf2d_tensor decomposer",
        pip_name="scipy",
        extra_hint="Install scipy or choose another decomposer.",
    )


def _ensure_basis_matrix(arr: np.ndarray, expected_len: int, label: str) -> np.ndarray:
    mat = np.asarray(arr, dtype=np.float64)
    if mat.ndim == 1:
        mat = mat[None, :]
    if mat.ndim != 2 or mat.shape[1] != expected_len:
        raise ValueError(f"pswf2d_tensor {label} basis has unexpected shape {mat.shape}")
    return mat


@register_decomposer("pswf2d_tensor")
class PSWF2DTensorDecomposer(GridDecomposerBase):
    """PSWF2D decomposer using a tensor product of 1D DPSS bases."""

    def __init__(self, *, cfg: Mapping[str, Any]) -> None:
        super().__init__(cfg=cfg)
        self.name = "pswf2d_tensor"
        self._c_x = _parse_positive_float(require_cfg(cfg, "c_x", label="decompose"), key="c_x")
        self._c_y = _parse_positive_float(require_cfg(cfg, "c_y", label="decompose"), key="c_y")
        self._n_x = _parse_positive_int(require_cfg(cfg, "n_x", label="decompose"), key="n_x")
        self._n_y = _parse_positive_int(require_cfg(cfg, "n_y", label="decompose"), key="n_y")
        self._mask_policy = str(cfg_get(cfg, "mask_policy", "error")).strip() or "error"
        if self._mask_policy not in _MASK_POLICIES:
            raise ValueError(
                f"decompose.mask_policy must be one of {_MASK_POLICIES}, got {self._mask_policy}"
            )
        self._basis_cache_key: tuple[Any, ...] | None = None
        self._basis_x: np.ndarray | None = None
        self._basis_y: np.ndarray | None = None
        self._eig_x: np.ndarray | None = None
        self._eig_y: np.ndarray | None = None
        self._grid_shape: tuple[int, int] | None = None

    def _validate_params(self, grid_shape: tuple[int, int]) -> None:
        height, width = grid_shape
        if self._n_x > width:
            raise ValueError("decompose.n_x must be <= grid width for pswf2d_tensor")
        if self._n_y > height:
            raise ValueError("decompose.n_y must be <= grid height for pswf2d_tensor")
        if self._c_x >= 0.5 * width:
            raise ValueError("decompose.c_x must be < grid_width/2 for pswf2d_tensor")
        if self._c_y >= 0.5 * height:
            raise ValueError("decompose.c_y must be < grid_height/2 for pswf2d_tensor")

    def _get_basis(self, domain_spec: DomainSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        _require_dpss()
        grid_shape = tuple(int(x) for x in domain_spec.grid_shape)
        key = (grid_shape, float(self._c_x), float(self._c_y), int(self._n_x), int(self._n_y))
        if self._basis_cache_key == key and self._basis_x is not None and self._basis_y is not None:
            if self._eig_x is None or self._eig_y is None:
                raise ValueError("pswf2d_tensor eigenvalues are missing from cache")
            return self._basis_x, self._basis_y, self._eig_x, self._eig_y

        self._validate_params(grid_shape)

        height, width = grid_shape
        basis_x, eig_x = _dpss(
            width,
            self._c_x,
            Kmax=self._n_x,
            return_ratios=True,
            norm=_DPSS_NORM,
        )
        basis_y, eig_y = _dpss(
            height,
            self._c_y,
            Kmax=self._n_y,
            return_ratios=True,
            norm=_DPSS_NORM,
        )
        basis_x = _ensure_basis_matrix(basis_x, width, "x")
        basis_y = _ensure_basis_matrix(basis_y, height, "y")
        eig_x = np.atleast_1d(np.asarray(eig_x, dtype=np.float64))
        eig_y = np.atleast_1d(np.asarray(eig_y, dtype=np.float64))
        if eig_x.size != basis_x.shape[0] or eig_y.size != basis_y.shape[0]:
            raise ValueError("pswf2d_tensor eigenvalue count mismatch with basis")

        self._basis_cache_key = key
        self._basis_x = basis_x
        self._basis_y = basis_y
        self._eig_x = eig_x
        self._eig_y = eig_y
        return basis_x, basis_y, eig_x, eig_y

    def _project(self, field_2d: np.ndarray, basis_x: np.ndarray, basis_y: np.ndarray) -> np.ndarray:
        field = np.asarray(field_2d, dtype=np.float64)
        return basis_y @ field @ basis_x.T

    def _reconstruct(self, coeff_2d: np.ndarray, basis_x: np.ndarray, basis_y: np.ndarray) -> np.ndarray:
        coeff = np.asarray(coeff_2d, dtype=np.float64)
        return basis_y.T @ coeff @ basis_x

    def transform(
        self,
        field: np.ndarray,
        mask: np.ndarray | None,
        domain_spec: DomainSpec,
    ) -> np.ndarray:
        basis_x, basis_y, eig_x, eig_y = self._get_basis(domain_spec)
        allow_zero_fill = self._mask_policy == "zero_fill"

        coeff_tensor = self._grid_transform(
            field,
            mask,
            domain_spec,
            allow_zero_fill=allow_zero_fill,
            forward_fn=lambda arr: self._project(arr, basis_x, basis_y),
            coeff_layout="CYX",
            complex_format="real",
            extra_meta={
                "c_x": float(self._c_x),
                "c_y": float(self._c_y),
                "n_x": int(self._n_x),
                "n_y": int(self._n_y),
                "basis_type": "dpss",
                "dpss_norm": _DPSS_NORM,
                "mask_policy": self._mask_policy,
                "mode_axes": ["y", "x"],
                "mode_order": "row_major",
                "mode_shape": [int(self._n_y), int(self._n_x)],
                "eigvals_x": [float(val) for val in eig_x],
                "eigvals_y": [float(val) for val in eig_y],
                "projection": "tensor_product",
                "approximation": "discrete_pswf_dpss",
            },
        )
        self._grid_shape = domain_spec.grid_shape
        return coeff_tensor

    def inverse_transform(self, coeff: np.ndarray, domain_spec: DomainSpec) -> np.ndarray:
        if self._grid_shape is not None and domain_spec.grid_shape != self._grid_shape:
            raise ValueError("pswf2d_tensor domain grid does not match cached shape")
        basis_x, basis_y, _, _ = self._get_basis(domain_spec)
        return self._grid_inverse(
            coeff,
            domain_spec,
            inverse_fn=lambda arr: self._reconstruct(arr, basis_x, basis_y),
        )


__all__ = ["PSWF2DTensorDecomposer"]
