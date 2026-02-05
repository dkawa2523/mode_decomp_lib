"""Slepian decomposer for sphere_grid domain."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
import json

import numpy as np
from mode_decomp_ml.config import cfg_get
try:  # SciPy 1.15+ deprecates sph_harm in favor of sph_harm_y
    from scipy.special import sph_harm_y as _sph_harm
except Exception:  # pragma: no cover - fallback for older SciPy
    from scipy.special import sph_harm as _sph_harm

from mode_decomp_ml.domain import DomainSpec, validate_decomposer_compatibility
from mode_decomp_ml.optional import require_dependency
from mode_decomp_ml.plugins.registry import register_decomposer

from .base import EigenBasisDecomposerBase, _combine_masks, _normalize_field, _normalize_mask, require_cfg

try:  # optional dependency
    import pyshtools  # noqa: F401
except Exception:  # pragma: no cover - optional dependency
    pyshtools = None

_ALLOWED_BACKENDS = {"pyshtools", "scipy"}
_ALLOWED_REGION_MASK_SOURCES = {"dataset", "domain"}
_ALLOWED_ANGLE_UNITS = {"deg", "degree", "degrees", "rad", "radian", "radians"}



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


def _ensure_angle_unit(value: Any) -> str:
    unit = str(value).strip().lower() if value is not None else ""
    if not unit:
        return "deg"
    if unit not in _ALLOWED_ANGLE_UNITS:
        raise ValueError(f"angle_unit must be one of {_ALLOWED_ANGLE_UNITS}, got {value}")
    return unit


def _to_radians(value: float, angle_unit: str) -> float:
    if angle_unit in {"deg", "degree", "degrees"}:
        return float(np.deg2rad(value))
    if angle_unit in {"rad", "radian", "radians"}:
        return float(value)
    raise ValueError(f"Unsupported angle_unit: {angle_unit}")


def _build_cap_mask(domain_spec: DomainSpec, cap_spec: Mapping[str, Any]) -> np.ndarray:
    lat = domain_spec.coords.get("lat")
    lon = domain_spec.coords.get("lon")
    if lat is None or lon is None:
        raise ValueError("sphere_grid domain must provide lat/lon coords for spherical_slepian")

    if "lat" not in cap_spec or "lon" not in cap_spec or "radius" not in cap_spec:
        raise ValueError("decompose.region_cap requires lat/lon/radius")

    angle_unit = _ensure_angle_unit(cap_spec.get("angle_unit", domain_spec.meta.get("angle_unit", "deg")))
    lat0 = _to_radians(float(cap_spec["lat"]), angle_unit)
    lon0 = _to_radians(float(cap_spec["lon"]), angle_unit)
    radius = _to_radians(float(cap_spec["radius"]), angle_unit)
    if radius <= 0.0:
        raise ValueError("decompose.region_cap.radius must be > 0")

    lat_arr = np.asarray(lat, dtype=np.float64)
    lon_arr = np.asarray(lon, dtype=np.float64)
    cos_gamma = (
        np.sin(lat_arr) * np.sin(lat0)
        + np.cos(lat_arr) * np.cos(lat0) * np.cos(lon_arr - lon0)
    )
    cos_gamma = np.clip(cos_gamma, -1.0, 1.0)
    gamma = np.arccos(cos_gamma)
    return gamma <= radius


def _normalize_region_mask(mask: np.ndarray, grid_shape: tuple[int, int]) -> np.ndarray:
    mask_arr = np.asarray(mask).astype(bool)
    if mask_arr.shape != grid_shape:
        raise ValueError(f"region_mask shape {mask_arr.shape} does not match {grid_shape}")
    return mask_arr


def _build_real_sh_basis(
    *,
    l_max: int,
    theta: np.ndarray,
    phi: np.ndarray,
) -> tuple[np.ndarray, list[list[Any]]]:
    lm_kind_list: list[list[Any]] = []
    basis_list: list[np.ndarray] = []
    for l in range(l_max + 1):
        lm_kind_list.append([int(l), 0, "cos"])
        basis_list.append(_sph_harm(0, l, theta, phi).real)
        for m in range(1, l + 1):
            ylm = _sph_harm(m, l, theta, phi)
            factor = np.sqrt(2.0) * _real_mode_sign(m)
            lm_kind_list.append([int(l), int(m), "cos"])
            basis_list.append(factor * ylm.real)
            lm_kind_list.append([int(l), int(m), "sin"])
            basis_list.append(factor * ylm.imag)
    basis = np.stack(basis_list, axis=0)
    return basis, lm_kind_list


@register_decomposer("spherical_slepian")
class SphericalSlepianDecomposer(EigenBasisDecomposerBase):
    """Slepian (region-concentrated) eigenbasis for sphere_grid domain."""

    def __init__(self, *, cfg: Mapping[str, Any]) -> None:
        super().__init__(cfg=cfg)
        self.name = "spherical_slepian"
        l_max = require_cfg(cfg, "l_max", label="decompose")
        self._l_max = int(l_max)
        if self._l_max < 0:
            raise ValueError("decompose.l_max must be >= 0 for spherical_slepian")
        k_val = require_cfg(cfg, "k", label="decompose")
        self._k = int(k_val)
        if self._k <= 0:
            raise ValueError("decompose.k must be > 0 for spherical_slepian")

        self._backend = str(cfg_get(cfg, "backend", "pyshtools")).strip().lower() or "pyshtools"
        if self._backend not in _ALLOWED_BACKENDS:
            raise ValueError(f"decompose.backend must be one of {_ALLOWED_BACKENDS}, got {self._backend}")

        self._region_mask_cfg = cfg_get(cfg, "region_mask", None)
        self._region_cap_cfg = cfg_get(cfg, "region_cap", None)

        self._coeff_shape: tuple[int, ...] | None = None
        self._field_ndim: int | None = None
        self._grid_shape: tuple[int, int] | None = None
        self._channels: int | None = None
        self._domain_key: tuple[Any, ...] | None = None
        self._region_mask: np.ndarray | None = None
        self._region_spec: dict[str, Any] | None = None
        self._lm_kind_list: list[list[Any]] | None = None

    def _require_backend(self) -> None:
        if self._backend == "pyshtools":
            require_dependency(
                pyshtools,
                name="spherical_slepian decomposer",
                pip_name="pyshtools",
                extra_hint="Set decompose.backend=scipy to use the SciPy backend.",
            )

    def _resolve_region_mask(
        self,
        *,
        dataset: Any | None,
        domain_spec: DomainSpec,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if self._region_cap_cfg is not None:
            if not isinstance(self._region_cap_cfg, Mapping):
                raise ValueError("decompose.region_cap must be a mapping")
            region_mask = _build_cap_mask(domain_spec, self._region_cap_cfg)
            region_spec = {
                "source": "cap",
                "cap": {
                    "lat": float(self._region_cap_cfg["lat"]),
                    "lon": float(self._region_cap_cfg["lon"]),
                    "radius": float(self._region_cap_cfg["radius"]),
                    "angle_unit": _ensure_angle_unit(
                        self._region_cap_cfg.get("angle_unit", domain_spec.meta.get("angle_unit", "deg"))
                    ),
                },
            }
        elif self._region_mask_cfg is not None:
            if isinstance(self._region_mask_cfg, str):
                source = self._region_mask_cfg.strip().lower()
                if source not in _ALLOWED_REGION_MASK_SOURCES:
                    raise ValueError(
                        f"decompose.region_mask must be one of {_ALLOWED_REGION_MASK_SOURCES}, got {source}"
                    )
                if source == "dataset":
                    if dataset is None:
                        raise ValueError("spherical_slepian requires dataset to resolve region_mask=dataset")
                    masks: list[np.ndarray] = []
                    for idx in range(len(dataset)):
                        sample = dataset[idx]
                        if sample.mask is None:
                            raise ValueError("spherical_slepian region_mask=dataset requires masks for all samples")
                        mask_arr = _normalize_region_mask(sample.mask, domain_spec.grid_shape)
                        masks.append(mask_arr)
                    if not masks:
                        raise ValueError("spherical_slepian requires at least one mask in dataset")
                    region_mask = masks[0]
                    for mask_item in masks[1:]:
                        if not np.array_equal(mask_item, region_mask):
                            raise ValueError("spherical_slepian requires a fixed region mask across samples")
                    region_spec = {"source": "dataset"}
                else:
                    if domain_spec.mask is None:
                        raise ValueError("spherical_slepian region_mask=domain requires domain mask")
                    region_mask = _normalize_region_mask(domain_spec.mask, domain_spec.grid_shape)
                    region_spec = {"source": "domain"}
            else:
                region_mask = _normalize_region_mask(self._region_mask_cfg, domain_spec.grid_shape)
                region_spec = {"source": "inline"}
        else:
            raise ValueError("spherical_slepian requires decompose.region_cap or decompose.region_mask")

        if domain_spec.mask is not None:
            region_mask = _combine_masks(region_mask, domain_spec.mask)

        if region_mask is None:
            raise ValueError("spherical_slepian region mask could not be resolved")
        if not np.any(region_mask):
            raise ValueError("spherical_slepian region mask has no valid entries")

        region_spec = dict(region_spec)
        region_spec.update(
            {
                "mask_shape": [int(region_mask.shape[0]), int(region_mask.shape[1])],
                "mask_valid_count": int(np.sum(region_mask)),
            }
        )
        return region_mask, region_spec

    def _build_basis(self, domain_spec: DomainSpec, region_mask: np.ndarray) -> None:
        if domain_spec.name != "sphere_grid":
            raise ValueError("spherical_slepian requires sphere_grid domain")
        theta = domain_spec.coords.get("theta")
        phi = domain_spec.coords.get("phi")
        if theta is None or phi is None:
            raise ValueError("sphere_grid domain must provide theta/phi coords for spherical_slepian")

        self._require_backend()

        theta = np.asarray(theta, dtype=np.float64)
        phi = np.asarray(phi, dtype=np.float64)
        basis_sh, lm_kind_list = _build_real_sh_basis(l_max=self._l_max, theta=theta, phi=phi)
        if domain_spec.mask is not None:
            basis_sh[:, ~domain_spec.mask] = 0.0
        self._lm_kind_list = lm_kind_list

        weights = domain_spec.weights
        if weights is None:
            weights = np.ones(domain_spec.grid_shape, dtype=np.float64)
        else:
            weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != domain_spec.grid_shape:
            raise ValueError(f"weights shape {weights.shape} does not match {domain_spec.grid_shape}")
        if not np.isfinite(weights).all():
            raise ValueError("spherical_slepian weights must be finite")
        if np.any(weights < 0):
            raise ValueError("spherical_slepian weights must be non-negative")

        basis_flat = basis_sh.reshape(basis_sh.shape[0], -1)
        weights_flat = weights.reshape(-1)
        region_flat = region_mask.reshape(-1).astype(np.float64)
        if not np.any(region_flat > 0):
            raise ValueError("spherical_slepian region mask has no valid weights")

        sqrt_w = np.sqrt(weights_flat)
        yw = basis_flat * sqrt_w[None, :]
        gram = yw @ yw.T
        gram = 0.5 * (gram + gram.T)
        evals, evecs = np.linalg.eigh(gram)
        max_eval = float(np.max(evals))
        tol = max(1e-12, 1e-12 * max_eval)
        if np.min(evals) < -tol:
            raise ValueError("spherical_slepian gram matrix is not positive definite")
        evals = np.clip(evals, tol, None)
        inv_sqrt = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T
        y_ortho = inv_sqrt @ basis_flat

        sqrt_wr = np.sqrt(weights_flat * region_flat)
        ywr = y_ortho * sqrt_wr[None, :]
        conc = ywr @ ywr.T
        conc = 0.5 * (conc + conc.T)
        eigvals, eigvecs = np.linalg.eigh(conc)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        n_modes = y_ortho.shape[0]
        if self._k > n_modes:
            raise ValueError("spherical_slepian k exceeds available spherical harmonic modes")

        slepian_basis = eigvecs.T @ y_ortho
        slepian_basis = slepian_basis[: self._k]
        eigvals = eigvals[: self._k]

        basis_nodes = slepian_basis.reshape(self._k, -1).T
        self._set_basis(basis_nodes, eigvals)

    def _ensure_basis(self, domain_spec: DomainSpec, region_mask: np.ndarray) -> None:
        if self._basis is None:
            self._build_basis(domain_spec, region_mask)
            self._domain_key = _basis_cache_key(domain_spec)
            return
        if self._domain_key is None:
            raise ValueError("spherical_slepian basis is not initialized correctly")
        if self._domain_key != _basis_cache_key(domain_spec):
            raise ValueError("spherical_slepian domain spec does not match cached basis")
        if self._region_mask is None:
            raise ValueError("spherical_slepian region mask is not cached")
        if not np.array_equal(region_mask, self._region_mask):
            raise ValueError("spherical_slepian requires the same region mask used during fit")

    @staticmethod
    def _solve_weighted_least_squares(
        basis: np.ndarray,
        field_3d: np.ndarray,
        weights: np.ndarray,
    ) -> np.ndarray:
        if basis.ndim != 2:
            raise ValueError("spherical_slepian basis must be 2D (nodes x modes)")
        weights_flat = weights.reshape(-1)
        valid = weights_flat > 0
        if not np.any(valid):
            raise ValueError("spherical_slepian weights are empty after masking")
        design = basis[valid]
        if design.shape[0] < design.shape[1]:
            raise ValueError("spherical_slepian basis has more modes than valid samples")
        field_flat = field_3d.reshape(-1, field_3d.shape[-1])
        if not np.isfinite(field_flat[valid]).all():
            raise ValueError("spherical_slepian field has non-finite values within valid mask")
        sqrt_w = np.sqrt(weights_flat[valid])
        design_w = design * sqrt_w[:, None]
        field_w = field_flat[valid] * sqrt_w[:, None]
        coeffs, _, rank, _ = np.linalg.lstsq(design_w, field_w, rcond=None)
        if rank < coeffs.shape[0]:
            raise ValueError("spherical_slepian basis is rank-deficient; reduce k or check region")
        return coeffs.T

    @staticmethod
    def _project_coefficients(
        basis: np.ndarray,
        field_3d: np.ndarray,
        weights: np.ndarray,
    ) -> np.ndarray:
        if basis.ndim != 2:
            raise ValueError("spherical_slepian basis must be 2D (nodes x modes)")
        weights_flat = weights.reshape(-1)
        valid = weights_flat > 0
        if not np.any(valid):
            raise ValueError("spherical_slepian weights are empty after masking")
        field_flat = field_3d.reshape(-1, field_3d.shape[-1])
        if not np.isfinite(field_flat[valid]).all():
            raise ValueError("spherical_slepian field has non-finite values within valid mask")
        coeffs = basis.T @ (field_flat * weights_flat[:, None])
        return coeffs.T

    def fit(
        self,
        dataset: Any | None = None,
        domain_spec: DomainSpec | None = None,
    ) -> "SphericalSlepianDecomposer":
        if domain_spec is None:
            raise ValueError("spherical_slepian requires domain_spec for fit")
        channels: int | None = None
        if dataset is not None:
            for idx in range(len(dataset)):
                sample = dataset[idx]
                field = np.asarray(sample.field)
                if field.ndim != 3:
                    raise ValueError(f"field must be 3D per sample, got {field.shape}")
                if field.shape[:2] != domain_spec.grid_shape:
                    raise ValueError(
                        f"field shape {field.shape[:2]} does not match domain {domain_spec.grid_shape}"
                    )
                if channels is None:
                    channels = int(field.shape[-1])
                elif channels != field.shape[-1]:
                    raise ValueError("spherical_slepian requires consistent channel count across samples")

        region_mask, region_spec = self._resolve_region_mask(dataset=dataset, domain_spec=domain_spec)
        self._ensure_basis(domain_spec, region_mask)
        self._region_mask = region_mask.astype(bool)
        self._region_spec = region_spec
        self._grid_shape = domain_spec.grid_shape
        if channels is not None:
            self._channels = channels
        return self

    def transform(
        self,
        field: np.ndarray,
        mask: np.ndarray | None,
        domain_spec: DomainSpec,
    ) -> np.ndarray:
        validate_decomposer_compatibility(domain_spec, self._cfg)
        if self._basis is None or self._region_mask is None:
            raise ValueError("spherical_slepian fit must be called before transform")

        field_3d, was_2d = _normalize_field(field)
        if field_3d.shape[:2] != domain_spec.grid_shape:
            raise ValueError(
                f"field shape {field_3d.shape[:2]} does not match domain {domain_spec.grid_shape}"
            )
        if self._channels is not None and field_3d.shape[-1] != self._channels:
            raise ValueError("spherical_slepian field channels do not match fit")

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

        if combined_mask is None or np.all(combined_mask):
            coeff_tensor = self._project_coefficients(self._basis, field_3d, weights)
        else:
            coeff_tensor = self._solve_weighted_least_squares(self._basis, field_3d, weights)

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
                "l_max": int(self._l_max),
                "k": int(self._k),
                "backend": self._backend,
                "projection": "weighted_least_squares",
                "mask_policy": "ignore_masked_points",
                "region_spec": dict(self._region_spec or {}),
                "lm_kind_list": self._lm_kind_list,
            }
        )
        meta.update(self._eigen_meta(projection="slepian_concentration", mode_order="descending_concentration"))
        meta["concentration"] = meta.get("eigenvalues")
        self._coeff_meta = meta
        return coeff_tensor

    def inverse_transform(self, coeff: np.ndarray, domain_spec: DomainSpec) -> np.ndarray:
        validate_decomposer_compatibility(domain_spec, self._cfg)
        if self._basis is None or self._coeff_shape is None or self._field_ndim is None:
            raise ValueError("spherical_slepian transform must be called before inverse_transform")
        if self._grid_shape is None:
            raise ValueError("spherical_slepian grid shape is not available")
        if domain_spec.grid_shape != self._grid_shape:
            raise ValueError("spherical_slepian domain grid does not match cached basis")

        coeff_tensor = self._reshape_coeff(coeff, self._coeff_shape, name=self.name)
        height, width = self._grid_shape
        field_channels = []
        for ch in range(coeff_tensor.shape[0]):
            vec = self._basis @ coeff_tensor[ch]
            field_c = vec.reshape(height, width, order="C")
            field_channels.append(field_c)
        field_hat = np.stack(field_channels, axis=-1)
        if domain_spec.mask is not None:
            field_hat = field_hat.copy()
            field_hat[~domain_spec.mask] = 0.0
        if self._field_ndim == 2 and field_hat.shape[-1] == 1:
            return field_hat[..., 0]
        return field_hat

    def save_state(self, run_dir: str | Path) -> Path:
        path = super().save_state(run_dir)
        if self._basis is None or self._eigenvalues is None or self._region_spec is None:
            raise ValueError("spherical_slepian basis is not initialized")

        out_dir = Path(run_dir) / "outputs" / "states" / "decomposer"
        out_dir.mkdir(parents=True, exist_ok=True)
        basis_path = out_dir / "slepian_basis.npz"
        payload = {
            "eigenvalues": np.asarray(self._eigenvalues, dtype=np.float64),
            "eigenvectors": np.asarray(self._basis, dtype=np.float64),
            "region_spec": np.asarray(
                json.dumps(self._region_spec, ensure_ascii=True), dtype="U"
            ),
            "grid_shape": np.asarray(self._grid_shape, dtype=np.int64),
            "l_max": np.asarray([self._l_max], dtype=np.int64),
            "k": np.asarray([self._k], dtype=np.int64),
            "backend": np.asarray([self._backend], dtype="U"),
        }
        np.savez_compressed(basis_path, **payload)
        return path


__all__ = ["SphericalSlepianDecomposer"]
