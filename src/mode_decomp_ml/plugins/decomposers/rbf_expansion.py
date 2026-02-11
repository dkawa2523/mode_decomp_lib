"""RBF expansion decomposer for grid domains with (possibly varying) masks."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from mode_decomp_ml.config import cfg_get

from mode_decomp_ml.domain import DomainSpec, validate_decomposer_compatibility
from mode_decomp_ml.plugins.registry import register_decomposer

from .base import BaseDecomposer, _combine_masks, _normalize_field, _normalize_mask, require_cfg

_KERNELS = {"gaussian", "thin_plate"}
_CENTERS = {"stride", "farthest"}
_MASK_POLICIES = {"ignore_masked_points"}


def _basis_cache_key(
    domain_spec: DomainSpec,
    *,
    kernel: str,
    centers: str,
    stride: int,
    n_centers: int | None,
    length_scale: float,
    include_bias: bool,
) -> tuple[Any, ...]:
    meta = domain_spec.meta or {}
    return (
        domain_spec.name,
        domain_spec.grid_shape,
        tuple(meta.get("x_range", ())),
        tuple(meta.get("y_range", ())),
        str(kernel),
        str(centers),
        int(stride),
        None if n_centers is None else int(n_centers),
        float(length_scale),
        bool(include_bias),
    )


def _stride_centers(grid_shape: tuple[int, int], stride: int) -> list[tuple[int, int]]:
    height, width = grid_shape
    return [(int(i), int(j)) for i in range(0, height, stride) for j in range(0, width, stride)]


def _rbf_kernel(
    *,
    kernel: str,
    dx: np.ndarray,
    dy: np.ndarray,
    length_scale: float,
) -> np.ndarray:
    d2 = dx * dx + dy * dy
    if kernel == "gaussian":
        # exp(-0.5 * (r/ls)^2)
        inv = 1.0 / max(float(length_scale), 1e-12)
        return np.exp(-0.5 * d2 * (inv * inv))
    if kernel == "thin_plate":
        # r^2 log(r), with r scaled by length_scale to keep units consistent.
        inv = 1.0 / max(float(length_scale), 1e-12)
        r = np.sqrt(d2) * inv
        # Define phi(0)=0.
        eps = 1e-12
        out = (r * r) * np.log(r + eps)
        out = np.where(r > 0, out, 0.0)
        return out
    raise ValueError(f"Unsupported kernel: {kernel}")


@register_decomposer("rbf_expansion")
class RBFExpansionDecomposer(BaseDecomposer):
    """RBF basis expansion for arbitrary masks (per-sample masks supported)."""

    def __init__(self, *, cfg: Mapping[str, Any]) -> None:
        super().__init__(cfg=cfg)
        self.name = "rbf_expansion"
        self._kernel = str(require_cfg(cfg, "kernel", label="decompose")).strip().lower()
        if self._kernel not in _KERNELS:
            raise ValueError(f"decompose.kernel must be one of {_KERNELS}, got {self._kernel}")
        self._centers = str(require_cfg(cfg, "centers", label="decompose")).strip().lower()
        if self._centers not in _CENTERS:
            raise ValueError(f"decompose.centers must be one of {_CENTERS}, got {self._centers}")
        n_centers = cfg_get(cfg, "n_centers", None)
        self._n_centers: int | None
        if n_centers is None or n_centers == "" or n_centers == "null":
            self._n_centers = None
        else:
            self._n_centers = int(n_centers)
            if self._n_centers <= 0:
                raise ValueError("decompose.n_centers must be > 0 for rbf_expansion")

        stride = cfg_get(cfg, "stride", None)
        if stride is None:
            if self._n_centers is None:
                raise ValueError("decompose.stride is required for rbf_expansion when n_centers is not set")
            stride = 1
        self._stride = int(stride)
        if self._stride < 1:
            raise ValueError("decompose.stride must be >= 1 for rbf_expansion")
        length_scale = require_cfg(cfg, "length_scale", label="decompose")
        self._length_scale = float(length_scale)
        if self._length_scale <= 0:
            raise ValueError("decompose.length_scale must be > 0 for rbf_expansion")
        ridge_alpha = require_cfg(cfg, "ridge_alpha", label="decompose")
        self._ridge_alpha = float(ridge_alpha)
        if self._ridge_alpha < 0:
            raise ValueError("decompose.ridge_alpha must be >= 0 for rbf_expansion")
        self._mask_policy = str(require_cfg(cfg, "mask_policy", label="decompose")).strip().lower()
        if self._mask_policy not in _MASK_POLICIES:
            raise ValueError(f"decompose.mask_policy must be one of {_MASK_POLICIES}, got {self._mask_policy}")

        self._include_bias = bool(cfg_get(cfg, "include_bias", False))

        self._basis_cache: np.ndarray | None = None
        self._basis_cache_key: tuple[Any, ...] | None = None
        self._center_indices: list[tuple[int, int]] | None = None
        self._effective_stride: int | None = None
        self._coeff_shape: tuple[int, ...] | None = None
        self._field_ndim: int | None = None

    def _select_centers(self, domain_spec: DomainSpec) -> tuple[list[tuple[int, int]], int]:
        height, width = domain_spec.grid_shape
        domain_mask = None if domain_spec.mask is None else np.asarray(domain_spec.mask).astype(bool)
        n_valid = int(domain_mask.sum()) if domain_mask is not None else int(height * width)

        if self._n_centers is None:
            if self._centers == "farthest":
                raise ValueError("decompose.n_centers is required for centers='farthest'")
            stride = int(self._stride)
            centers = _stride_centers(domain_spec.grid_shape, stride)
            if domain_mask is not None:
                centers = [(i, j) for (i, j) in centers if bool(domain_mask[i, j])]
            return centers, stride

        desired = min(int(self._n_centers), max(1, n_valid))
        # Choose a stride that yields at least ~desired candidates, then downsample if needed.
        stride = int(np.floor(np.sqrt(float(n_valid) / float(desired)))) if desired > 0 else int(self._stride)
        stride = max(1, stride)
        centers = _stride_centers(domain_spec.grid_shape, stride)
        if domain_mask is not None:
            centers = [(i, j) for (i, j) in centers if bool(domain_mask[i, j])]
        if not centers:
            raise ValueError("rbf_expansion produced no centers after applying domain mask")
        if len(centers) > desired:
            if self._centers == "stride":
                step = float(len(centers)) / float(desired)
                idxs = [min(int(i * step), len(centers) - 1) for i in range(desired)]
                centers = [centers[i] for i in idxs]
            elif self._centers == "farthest":
                # Farthest-point sampling in physical (x,y) coordinates for better coverage at low K.
                x = domain_spec.coords.get("x")
                y = domain_spec.coords.get("y")
                if x is None or y is None:
                    raise ValueError("rbf_expansion centers='farthest' requires x/y coords in domain_spec")
                x = np.asarray(x, dtype=np.float64)
                y = np.asarray(y, dtype=np.float64)
                pts = np.array([[x[i, j], y[i, j]] for (i, j) in centers], dtype=np.float64)
                if pts.shape[0] != len(centers):
                    raise ValueError("rbf_expansion farthest sampling internal error")
                centroid = np.mean(pts, axis=0, keepdims=True)
                dist2 = np.sum((pts - centroid) ** 2, axis=1)
                first = int(np.argmax(dist2))
                chosen = [first]
                min_dist2 = np.sum((pts - pts[first : first + 1]) ** 2, axis=1)
                for _ in range(int(desired) - 1):
                    nxt = int(np.argmax(min_dist2))
                    chosen.append(nxt)
                    d2 = np.sum((pts - pts[nxt : nxt + 1]) ** 2, axis=1)
                    min_dist2 = np.minimum(min_dist2, d2)
                centers = [centers[i] for i in chosen]
            else:  # pragma: no cover
                raise ValueError(f"Unknown centers method: {self._centers}")
        return centers, stride

    def _get_basis(self, domain_spec: DomainSpec) -> np.ndarray:
        x = domain_spec.coords.get("x")
        y = domain_spec.coords.get("y")
        if x is None or y is None:
            raise ValueError("rbf_expansion requires x/y coords in domain_spec")
        centers, effective_stride = self._select_centers(domain_spec)
        key = _basis_cache_key(
            domain_spec,
            kernel=self._kernel,
            centers=self._centers,
            stride=effective_stride,
            n_centers=self._n_centers,
            length_scale=self._length_scale,
            include_bias=self._include_bias,
        )
        if self._basis_cache is not None and self._basis_cache_key == key:
            return self._basis_cache
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if x.shape != domain_spec.grid_shape or y.shape != domain_spec.grid_shape:
            raise ValueError("rbf_expansion x/y shape mismatch with grid_shape")
        if not centers:
            raise ValueError("rbf_expansion produced no centers")
        basis_list: list[np.ndarray] = []
        if self._include_bias:
            bias = np.ones(domain_spec.grid_shape, dtype=np.float64)
            if domain_spec.mask is not None:
                bias = np.where(domain_spec.mask, bias, 0.0)
            basis_list.append(bias)
        basis = np.empty((len(centers),) + domain_spec.grid_shape, dtype=np.float64)
        for idx, (i, j) in enumerate(centers):
            xc = float(x[i, j])
            yc = float(y[i, j])
            basis[idx] = _rbf_kernel(kernel=self._kernel, dx=(x - xc), dy=(y - yc), length_scale=self._length_scale)
        if domain_spec.mask is not None:
            basis[:, ~domain_spec.mask] = 0.0
        if basis_list:
            basis = np.concatenate([np.stack(basis_list, axis=0), basis], axis=0)
        self._basis_cache = basis
        self._basis_cache_key = key
        self._center_indices = centers
        self._effective_stride = effective_stride
        return basis

    def transform(self, field: np.ndarray, mask: np.ndarray | None, domain_spec: DomainSpec) -> np.ndarray:
        validate_decomposer_compatibility(domain_spec, self._cfg)
        field_3d, was_2d = _normalize_field(field)
        if field_3d.shape[:2] != domain_spec.grid_shape:
            raise ValueError(f"field shape {field_3d.shape[:2]} does not match domain {domain_spec.grid_shape}")
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
        weights_flat = weights.reshape(-1)
        valid = weights_flat > 0
        if not np.any(valid):
            raise ValueError("rbf_expansion weights are empty after masking")
        basis_flat = basis.reshape(basis.shape[0], -1).T  # (HW, K)
        design = basis_flat[valid]  # (Nvalid, K)
        if design.shape[0] < design.shape[1]:
            raise ValueError("rbf_expansion basis has more centers than valid samples")

        sqrt_w = np.sqrt(weights_flat[valid])
        design_w = design * sqrt_w[:, None]

        # Precompute normal equations for stability and speed.
        k = design_w.shape[1]
        if self._ridge_alpha > 0:
            gram = design_w.T @ design_w
            gram = gram + (self._ridge_alpha * np.eye(k, dtype=np.float64))

        field_flat = field_3d.reshape(-1, field_3d.shape[-1])
        coeffs = []
        for ch in range(field_3d.shape[-1]):
            y = field_flat[valid, ch]
            if not np.isfinite(y).all():
                raise ValueError("rbf_expansion field has non-finite values within valid mask")
            y_w = y * sqrt_w
            if self._ridge_alpha > 0:
                rhs = design_w.T @ y_w
                w = np.linalg.solve(gram, rhs)
            else:
                w, _, rank, _ = np.linalg.lstsq(design_w, y_w, rcond=None)
                if rank < w.shape[0]:
                    raise ValueError("rbf_expansion basis is rank-deficient; increase stride or add ridge_alpha")
            coeffs.append(w)
        coeff_tensor = np.stack(coeffs, axis=0)

        self._coeff_shape = coeff_tensor.shape
        self._field_ndim = 2 if was_2d else 3
        centers = self._center_indices or self._select_centers(domain_spec)[0]
        self._coeff_meta = {
            "method": self.name,
            "field_shape": [int(field_3d.shape[0]), int(field_3d.shape[1])]
            if was_2d
            else [int(field_3d.shape[0]), int(field_3d.shape[1]), int(field_3d.shape[-1])],
            "field_ndim": self._field_ndim,
            "field_layout": "HW" if was_2d else "HWC",
            "channels": int(field_3d.shape[-1]),
            "coeff_shape": [int(x) for x in coeff_tensor.shape],
            "coeff_layout": "CK",
            "flatten_order": "C",
            "complex_format": "real",
            "keep": "all",
            "kernel": self._kernel,
            "centers": self._centers,
            "include_bias": bool(self._include_bias),
            "n_centers": int(self._n_centers) if self._n_centers is not None else None,
            "stride": int(self._stride),
            "effective_stride": int(self._effective_stride) if self._effective_stride is not None else int(self._stride),
            "length_scale": float(self._length_scale),
            "ridge_alpha": float(self._ridge_alpha),
            "center_indices": [[int(i), int(j)] for (i, j) in centers],
            "projection": "weighted_ridge",
            "mask_policy": self._mask_policy,
        }
        return coeff_tensor

    def inverse_transform(self, coeff: np.ndarray, domain_spec: DomainSpec) -> np.ndarray:
        validate_decomposer_compatibility(domain_spec, self._cfg)
        if self._coeff_shape is None or self._field_ndim is None:
            raise ValueError("transform must be called before inverse_transform")
        coeff_tensor = self._reshape_coeff(coeff, self._coeff_shape, name=self.name)
        basis = self._get_basis(domain_spec)

        field_channels = []
        for ch in range(coeff_tensor.shape[0]):
            field_c = np.tensordot(coeff_tensor[ch], basis, axes=(0, 0))
            field_channels.append(field_c)
        field_hat = np.stack(field_channels, axis=-1)
        if domain_spec.mask is not None:
            field_hat = field_hat.copy()
            field_hat[~domain_spec.mask] = 0.0
        if self._field_ndim == 2 and field_hat.shape[-1] == 1:
            return field_hat[..., 0]
        return field_hat


__all__ = ["RBFExpansionDecomposer"]
