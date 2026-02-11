"""Shared helpers for Zernike-family decomposers."""
from __future__ import annotations

from typing import Any, Callable, Mapping

import numpy as np
from scipy.special import comb, gammaln

from mode_decomp_ml.config import cfg_get
from .base import ZernikeFamilyBase
from mode_decomp_ml.domain import DomainSpec

_NORMALIZATION_OPTIONS = {"orthonormal", "none"}
_ORDERING_OPTIONS = {"n_then_m"}


def build_nm_list(n_max: int, ordering: str, *, m_max: int | None = None) -> list[tuple[int, int]]:
    if ordering not in _ORDERING_OPTIONS:
        raise ValueError(f"zernike ordering must be one of {_ORDERING_OPTIONS}, got {ordering}")
    if m_max is None:
        m_max = n_max
    nm_list: list[tuple[int, int]] = []
    for n in range(n_max + 1):
        m_cap = min(n, m_max)
        for m in range(-m_cap, m_cap + 1, 2):
            nm_list.append((n, m))
    return nm_list


def radial_poly_standard(n: int, m: int, r: np.ndarray) -> np.ndarray:
    m_abs = abs(m)
    if (n - m_abs) % 2 != 0:
        return np.zeros_like(r, dtype=np.float64)
    radial = np.zeros_like(r, dtype=np.float64)
    k_max = (n - m_abs) // 2
    for k in range(k_max + 1):
        coeff = (-1.0) ** k * comb(n - k, k) * comb(n - 2 * k, k_max - k)
        radial += coeff * np.power(r, n - 2 * k)
    return radial


def build_nm_list_pseudo(
    n_max: int,
    ordering: str,
    *,
    m_max: int | None = None,
) -> list[tuple[int, int]]:
    """(n,m) list for pseudo-zernike family (no parity constraint on m)."""
    if ordering not in _ORDERING_OPTIONS:
        raise ValueError(f"pseudo_zernike ordering must be one of {_ORDERING_OPTIONS}, got {ordering}")
    if m_max is None:
        m_max = n_max
    nm_list: list[tuple[int, int]] = []
    for n in range(n_max + 1):
        m_cap = min(n, m_max)
        for m in range(-m_cap, m_cap + 1):
            nm_list.append((n, m))
    return nm_list


def radial_poly_pseudo(n: int, m: int, r: np.ndarray) -> np.ndarray:
    """Pseudo-Zernike radial polynomial R_nm(r).

    Definition (Dai et al., also matches common references):
      R_nm(r) = sum_{s=0}^{n-|m|} D_{n,|m|,s} * r^{n-s}
      D_{n,|m|,s} = (-1)^s * (2n+1-s)! / ( s! * (n-|m|-s)! * (n+|m|-s+1)! )

    We compute factorial ratios via log-gamma for numerical stability.
    """
    m_abs = abs(int(m))
    if n < 0:
        raise ValueError("n must be >= 0")
    if m_abs > n:
        return np.zeros_like(r, dtype=np.float64)
    out = np.zeros_like(r, dtype=np.float64)
    # s = 0..(n-|m|)
    s_max = n - m_abs
    n_f = float(n)
    for s in range(s_max + 1):
        # log((2n+1-s)!) = gammaln(2n+2-s)
        # denom: s! (n-|m|-s)! (n+|m|-s+1)!
        log_num = gammaln(2.0 * n_f + 2.0 - float(s))
        log_den = (
            gammaln(float(s) + 1.0)
            + gammaln(float(n - m_abs - s) + 1.0)
            + gammaln(float(n + m_abs - s) + 2.0)
        )
        coeff = np.exp(log_num - log_den)
        if s % 2 == 1:
            coeff = -coeff
        out += coeff * np.power(r, n - s)
    return out


def zernike_mode(
    n: int,
    m: int,
    r: np.ndarray,
    theta: np.ndarray,
    normalization: str,
    radial_fn: Callable[[int, int, np.ndarray], np.ndarray],
) -> np.ndarray:
    radial = radial_fn(n, m, r)
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


def build_basis(
    nm_list: list[tuple[int, int]],
    r: np.ndarray,
    theta: np.ndarray,
    normalization: str,
    radial_fn: Callable[[int, int, np.ndarray], np.ndarray],
    mask: np.ndarray | None,
) -> np.ndarray:
    basis = np.empty((len(nm_list),) + r.shape, dtype=np.float64)
    for idx, (n, m) in enumerate(nm_list):
        basis[idx] = zernike_mode(n, m, r, theta, normalization, radial_fn)
    if mask is not None:
        basis[:, ~mask] = 0.0
    return basis


def _mode_kind(m: int) -> str:
    return "sin" if m < 0 else "cos"


def build_nm_kind_list(nm_list: list[tuple[int, int]]) -> list[list[Any]]:
    return [[int(n), int(m), _mode_kind(int(m))] for (n, m) in nm_list]


__all__ = [
    "_NORMALIZATION_OPTIONS",
    "_ORDERING_OPTIONS",
    "cfg_get",
    "build_nm_list",
    "build_nm_list_pseudo",
    "radial_poly_standard",
    "radial_poly_pseudo",
    "zernike_mode",
    "build_basis",
    "build_nm_kind_list",
    "ZernikeBasisDecomposer",
]


class ZernikeBasisDecomposer(ZernikeFamilyBase):
    """Shared Zernike-family decomposer core with radial hook."""

    def __init__(
        self,
        *,
        cfg: Mapping[str, Any],
        name: str,
        n_max: int,
        m_max: int,
        ordering: str,
        normalization: str,
        boundary_condition: str,
        normalization_reference: str | None = None,
    ) -> None:
        super().__init__(cfg=cfg)
        self.name = name
        self._n_max = int(n_max)
        self._m_max = int(m_max)
        self._ordering = str(ordering)
        self._normalization = str(normalization)
        self._boundary_condition = str(boundary_condition)
        self._nm_list = build_nm_list(self._n_max, self._ordering, m_max=self._m_max)
        self._nm_kind_list = build_nm_kind_list(self._nm_list)
        if normalization_reference is None:
            normalization_reference = (
                "integral_over_unit_disk_equals_pi" if self._normalization == "orthonormal" else "none"
            )
        self._normalization_reference = normalization_reference
        self._basis_cache: np.ndarray | None = None
        self._basis_cache_key_value: tuple[Any, ...] | None = None
        self._coeff_shape: tuple[int, ...] | None = None
        self._field_ndim: int | None = None

    def _basis_cache_key(self, domain_spec: DomainSpec) -> tuple[Any, ...]:
        raise NotImplementedError

    def _radial_fn(self, domain_spec: DomainSpec) -> Callable[[int, int, np.ndarray], np.ndarray]:
        raise NotImplementedError

    def _meta_extras(self, domain_spec: DomainSpec) -> Mapping[str, Any]:
        return {}

    def _get_basis(self, domain_spec: DomainSpec) -> np.ndarray:
        key = self._basis_cache_key(domain_spec)
        if self._basis_cache is not None and self._basis_cache_key_value == key:
            return self._basis_cache
        r = domain_spec.coords.get("r")
        theta = domain_spec.coords.get("theta")
        if r is None or theta is None:
            raise ValueError(f"{self.name} requires r/theta coords from the domain")
        basis = build_basis(
            self._nm_list,
            r,
            theta,
            self._normalization,
            self._radial_fn(domain_spec),
            domain_spec.mask,
        )
        self._basis_cache = basis
        self._basis_cache_key_value = key
        return basis

    def transform(
        self,
        field: np.ndarray,
        mask: np.ndarray | None,
        domain_spec: DomainSpec,
    ) -> np.ndarray:
        field_3d, was_2d, weights, _ = self._prepare_zernike_inputs(
            field, mask, domain_spec
        )
        basis = self._get_basis(domain_spec)
        coeff_tensor = self._solve_weighted_least_squares(basis, field_3d, weights)
        channels = field_3d.shape[-1]
        self._coeff_shape = coeff_tensor.shape
        self._field_ndim = 2 if was_2d else 3
        meta = self._coeff_meta_base(
            field_shape=field_3d.shape[:2] if was_2d else field_3d.shape,
            field_ndim=self._field_ndim,
            field_layout="HW" if was_2d else "HWC",
            channels=int(channels),
            coeff_shape=coeff_tensor.shape,
            coeff_layout="CK",
            complex_format="real",
        )
        meta.update(
            {
                "n_max": int(self._n_max),
                "m_max": int(self._m_max),
                "ordering": self._ordering,
                "nm_list": [[int(n), int(m)] for (n, m) in self._nm_list],
                "nm_kind_list": self._nm_kind_list,
                "normalization": self._normalization,
                "normalization_reference": self._normalization_reference,
                "boundary_condition": self._boundary_condition,
                "projection": "weighted_least_squares",
                "mask_policy": "ignore_masked_points",
            }
        )
        meta.update(dict(self._meta_extras(domain_spec)))
        self._coeff_meta = meta
        return coeff_tensor

    def inverse_transform(self, coeff: np.ndarray, domain_spec: DomainSpec) -> np.ndarray:
        if self._coeff_shape is None or self._field_ndim is None:
            raise ValueError("transform must be called before inverse_transform")
        coeff_tensor = self._reshape_coeff(coeff, self._coeff_shape, name=self.name)
        basis = self._get_basis(domain_spec)

        field_hat = self._reconstruct_from_basis(coeff_tensor, basis)
        if domain_spec.mask is not None:
            field_hat = field_hat.copy()
            field_hat[~domain_spec.mask] = 0.0
        if self._field_ndim == 2 and field_hat.shape[-1] == 1:
            return field_hat[..., 0]
        return field_hat
