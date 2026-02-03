"""Zernike decomposer for disk domain."""
from __future__ import annotations

from typing import Any, Mapping

from mode_decomp_ml.domain import DomainSpec

from .base import require_cfg
from .zernike_shared import ZernikeBasisDecomposer
from mode_decomp_ml.plugins.registry import register_decomposer
from .zernike_shared import (
    _NORMALIZATION_OPTIONS,
    radial_poly_standard,
)

@register_decomposer("zernike")
class ZernikeDecomposer(ZernikeBasisDecomposer):
    """Zernike decomposer for disk domain (real coefficients)."""

    def __init__(self, *, cfg: Mapping[str, Any]) -> None:
        n_max = require_cfg(cfg, "n_max", label="decompose")
        n_max = int(n_max)
        if n_max < 0:
            raise ValueError("decompose.n_max must be >= 0 for zernike")
        ordering = str(require_cfg(cfg, "ordering", label="decompose"))
        normalization = str(require_cfg(cfg, "normalization", label="decompose"))
        if normalization not in _NORMALIZATION_OPTIONS:
            raise ValueError(
                f"decompose.normalization must be one of {_NORMALIZATION_OPTIONS}, got {normalization}"
            )
        # CONTRACT: boundary_condition is required for comparability across disk bases.
        boundary_condition = str(require_cfg(cfg, "boundary_condition", label="decompose"))
        normalization_ref = (
            "integral_over_unit_disk_equals_pi" if normalization == "orthonormal" else "none"
        )
        super().__init__(
            cfg=cfg,
            name="zernike",
            n_max=n_max,
            m_max=n_max,
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
            float(meta.get("radius", 0.0)),
            tuple(meta.get("x_range", ())),
            tuple(meta.get("y_range", ())),
        )

    def _radial_fn(self, domain_spec: DomainSpec):  # type: ignore[override]
        return radial_poly_standard
