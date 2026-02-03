"""Annular Zernike decomposer for annulus domain."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from mode_decomp_ml.config import cfg_get
from mode_decomp_ml.domain import DomainSpec

from mode_decomp_ml.plugins.registry import register_decomposer
from .base import require_cfg
from .zernike_shared import (
    _NORMALIZATION_OPTIONS,
    ZernikeBasisDecomposer,
    radial_poly_standard,
)


def _annular_radial_factory(r_inner: float, r_outer: float):
    if r_outer <= 0.0:
        raise ValueError("annular_zernike requires r_outer > 0")
    eps = float(r_inner) / float(r_outer)
    if eps <= 0.0:
        return radial_poly_standard

    denom = 1.0 - eps
    if denom <= 0.0:
        raise ValueError("annular_zernike requires r_inner < r_outer")

    def _radial(n: int, m: int, r: np.ndarray) -> np.ndarray:
        t = (r - eps) / denom
        t = np.clip(t, 0.0, 1.0)
        return radial_poly_standard(n, m, t)

    return _radial


@register_decomposer("annular_zernike")
class AnnularZernikeDecomposer(ZernikeBasisDecomposer):
    """Annular Zernike decomposer for annulus domain (real coefficients)."""

    def __init__(self, *, cfg: Mapping[str, Any]) -> None:
        n_max = require_cfg(cfg, "n_max", label="decompose")
        n_max = int(n_max)
        if n_max < 0:
            raise ValueError("decompose.n_max must be >= 0 for annular_zernike")
        m_max = cfg_get(cfg, "m_max", n_max)
        m_max = int(m_max)
        if m_max < 0:
            raise ValueError("decompose.m_max must be >= 0 for annular_zernike")
        if m_max > n_max:
            raise ValueError("decompose.m_max must be <= n_max for annular_zernike")
        ordering = str(require_cfg(cfg, "ordering", label="decompose"))
        normalization = str(require_cfg(cfg, "normalization", label="decompose"))
        if normalization not in _NORMALIZATION_OPTIONS:
            raise ValueError(
                f"decompose.normalization must be one of {_NORMALIZATION_OPTIONS}, got {normalization}"
            )
        boundary_condition = str(require_cfg(cfg, "boundary_condition", label="decompose"))
        normalization_ref = "mapped_unit_disk" if normalization == "orthonormal" else "none"
        super().__init__(
            cfg=cfg,
            name="annular_zernike",
            n_max=n_max,
            m_max=m_max,
            ordering=ordering,
            normalization=normalization,
            boundary_condition=boundary_condition,
            normalization_reference=normalization_ref,
        )

    def _basis_cache_key(self, domain_spec: DomainSpec) -> tuple[Any, ...]:
        meta = domain_spec.meta or {}
        return (
            domain_spec.name,
            domain_spec.grid_shape,
            tuple(meta.get("center", ())),
            float(meta.get("r_inner", 0.0)),
            float(meta.get("r_outer", 0.0)),
            tuple(meta.get("x_range", ())),
            tuple(meta.get("y_range", ())),
        )

    def _radial_fn(self, domain_spec: DomainSpec):  # type: ignore[override]
        meta = domain_spec.meta or {}
        r_inner = float(meta.get("r_inner", 0.0))
        r_outer = float(meta.get("r_outer", 0.0))
        return _annular_radial_factory(r_inner, r_outer)

    def _meta_extras(self, domain_spec: DomainSpec) -> Mapping[str, Any]:
        meta = domain_spec.meta or {}
        r_inner = float(meta.get("r_inner", 0.0))
        r_outer = float(meta.get("r_outer", 0.0))
        return {
            "domain": "annulus",
            "r_inner": r_inner,
            "r_outer": r_outer,
            "r_inner_norm": r_inner / r_outer if r_outer else 0.0,
        }
