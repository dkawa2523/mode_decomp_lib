"""POD/SVD decomposer for data-driven bases."""
from __future__ import annotations

import warnings
from typing import Any, Mapping

import numpy as np
from mode_decomp_ml.config import cfg_get

from mode_decomp_ml.domain import DomainSpec, validate_decomposer_compatibility

from .base import BaseDecomposer, _combine_masks, _normalize_field, _normalize_mask, require_cfg, parse_bool
from mode_decomp_ml.plugins.registry import register_decomposer

_MASK_POLICIES = {"error", "ignore_masked_points"}
_INNER_PRODUCTS = {"euclidean", "domain_weights"}
_RANK_SELECT_METHODS = {"energy"}
_MODE_WEIGHT_METHODS = {"eigval_scale"}


def _resolve_rank_select(cfg: Mapping[str, Any]) -> dict[str, Any]:
    options = cfg_get(cfg, "options", None)
    rank_cfg = {}
    if isinstance(options, Mapping):
        rank_cfg = options.get("rank_select") or {}
    if not isinstance(rank_cfg, Mapping):
        rank_cfg = {}
    enable = parse_bool(rank_cfg.get("enable", False), default=False)
    method = str(rank_cfg.get("method", "energy")).strip().lower() or "energy"
    if method not in _RANK_SELECT_METHODS:
        raise ValueError(f"options.rank_select.method must be one of {_RANK_SELECT_METHODS}, got {method}")
    energy = float(rank_cfg.get("energy", 0.99))
    if not (0.0 < energy <= 1.0):
        raise ValueError("options.rank_select.energy must be in (0, 1]")
    max_modes = rank_cfg.get("max_modes", None)
    if max_modes is not None:
        max_modes = int(max_modes)
        if max_modes <= 0:
            raise ValueError("options.rank_select.max_modes must be > 0")
    return {"enable": enable, "method": method, "energy": energy, "max_modes": max_modes}


def _resolve_mode_weight(cfg: Mapping[str, Any]) -> dict[str, Any]:
    options = cfg_get(cfg, "options", None)
    mode_cfg = {}
    if isinstance(options, Mapping):
        mode_cfg = options.get("mode_weight") or {}
    if not isinstance(mode_cfg, Mapping):
        mode_cfg = {}
    enable = parse_bool(mode_cfg.get("enable", False), default=False)
    method = str(mode_cfg.get("method", "eigval_scale")).strip().lower() or "eigval_scale"
    if method not in _MODE_WEIGHT_METHODS:
        raise ValueError(f"options.mode_weight.method must be one of {_MODE_WEIGHT_METHODS}, got {method}")
    return {"enable": enable, "method": method}


def _select_rank_by_energy(eigvals: np.ndarray, energy: float, max_modes: int | None) -> int:
    if eigvals.size == 0:
        return 0
    vals = np.maximum(np.asarray(eigvals, dtype=np.float64), 0.0)
    total = float(np.sum(vals))
    if total <= 0.0:
        return min(max_modes or eigvals.size, eigvals.size)
    cumsum = np.cumsum(vals)
    ratio = cumsum / total
    idx = int(np.searchsorted(ratio, energy, side="left")) + 1
    if max_modes is not None:
        idx = min(idx, int(max_modes))
    return min(idx, eigvals.size)



@register_decomposer("pod_svd")
class PODSVDDecomposer(BaseDecomposer):
    """POD/SVD decomposer fitted on training snapshots."""

    def __init__(self, *, cfg: Mapping[str, Any]) -> None:
        super().__init__(cfg=cfg)
        self.name = "pod_svd"
        self._mask_policy = str(require_cfg(cfg, "mask_policy", label="decompose"))
        if self._mask_policy not in _MASK_POLICIES:
            raise ValueError(
                f"decompose.mask_policy must be one of {_MASK_POLICIES}, got {self._mask_policy}"
            )
        n_modes = cfg_get(cfg, "n_modes", None)
        if n_modes is None:
            self._n_modes = None
        else:
            self._n_modes = int(n_modes)
            if self._n_modes <= 0:
                raise ValueError("decompose.n_modes must be > 0 for pod_svd")
        self._inner_product = str(cfg_get(cfg, "inner_product", "euclidean")).strip() or "euclidean"
        if self._inner_product not in _INNER_PRODUCTS:
            raise ValueError(
                f"decompose.inner_product must be one of {_INNER_PRODUCTS}, got {self._inner_product}"
            )
        self._inner_product_effective = self._inner_product
        self._rank_select = _resolve_rank_select(cfg)
        self._mode_weight = _resolve_mode_weight(cfg)
        self._basis: list[np.ndarray] | None = None
        self._singular_values: list[np.ndarray] | None = None
        self._eigvals: list[np.ndarray] | None = None
        self._mode_weight_scale: list[np.ndarray] | None = None
        self._mask: np.ndarray | None = None
        self._mask_indices: np.ndarray | None = None
        self._sqrt_weights: np.ndarray | None = None
        self._grid_shape: tuple[int, int] | None = None
        self._channels: int | None = None
        self._n_samples: int | None = None
        self._n_modes_effective: int | None = None
        self._coeff_shape: tuple[int, ...] | None = None
        self._field_ndim: int | None = None

    def _resolve_mask(
        self,
        domain_spec: DomainSpec,
        mask_arr: np.ndarray | None,
    ) -> np.ndarray | None:
        domain_mask = domain_spec.mask
        if self._mask_policy == "error":
            if domain_mask is not None or mask_arr is not None:
                raise ValueError("pod_svd mask_policy=error does not allow masks")
            return None
        if self._mask_policy != "ignore_masked_points":
            raise ValueError(f"Unsupported mask_policy for pod_svd: {self._mask_policy}")
        # CONTRACT: ignore_masked_points requires a fixed mask across samples.
        mask = domain_mask
        if mask_arr is not None:
            sample_mask = np.asarray(mask_arr[0]).astype(bool)
            if not np.all(mask_arr == sample_mask):
                raise ValueError("pod_svd requires a fixed mask across samples")
            mask = sample_mask if mask is None else (mask & sample_mask)
        if mask is not None and mask.shape != domain_spec.grid_shape:
            raise ValueError(f"mask shape {mask.shape} does not match {domain_spec.grid_shape}")
        return mask

    def fit(self, dataset: Any | None = None, domain_spec: DomainSpec | None = None) -> "PODSVDDecomposer":
        if dataset is None:
            raise ValueError("pod_svd requires dataset for fit")
        if domain_spec is None:
            raise ValueError("pod_svd requires domain_spec for fit")
        fields: list[np.ndarray] = []
        masks: list[np.ndarray | None] = []
        for idx in range(len(dataset)):
            sample = dataset[idx]
            field = np.asarray(sample.field)
            if field.ndim != 3:
                raise ValueError(f"field must be 3D per sample, got {field.shape}")
            if field.shape[:2] != domain_spec.grid_shape:
                raise ValueError(
                    f"field shape {field.shape[:2]} does not match domain {domain_spec.grid_shape}"
                )
            fields.append(field)
            masks.append(None if sample.mask is None else np.asarray(sample.mask))
        if not fields:
            raise ValueError("pod_svd requires at least one training sample")
        field_arr = np.stack(fields, axis=0)
        if all(mask is None for mask in masks):
            mask_arr = None
        elif any(mask is None for mask in masks):
            raise ValueError("mask must be present for all samples or none")
        else:
            mask_arr = np.stack([mask for mask in masks if mask is not None], axis=0)

        mask = self._resolve_mask(domain_spec, mask_arr)
        if mask is not None:
            mask_flat = mask.reshape(-1, order="C")
            mask_indices = np.flatnonzero(mask_flat)
            if mask_indices.size == 0:
                raise ValueError("pod_svd mask has no valid entries")
        else:
            mask_indices = None

        self._sqrt_weights = None
        self._inner_product_effective = self._inner_product
        if self._inner_product == "domain_weights":
            weights = domain_spec.integration_weights()
            if weights is None:
                warnings.warn(
                    "pod_svd inner_product=domain_weights but domain has no integration_weights; "
                    "falling back to euclidean",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._inner_product_effective = "euclidean"
            else:
                weights_arr = np.asarray(weights, dtype=np.float64)
                if weights_arr.shape != domain_spec.grid_shape:
                    raise ValueError(
                        f"pod_svd weights shape {weights_arr.shape} does not match domain {domain_spec.grid_shape}"
                    )
                weights_flat = weights_arr.reshape(-1, order="C")
                if mask_indices is not None:
                    weights_flat = weights_flat[mask_indices]
                if not np.isfinite(weights_flat).all():
                    raise ValueError("pod_svd requires finite integration weights")
                if np.any(weights_flat < 0.0):
                    raise ValueError("pod_svd integration weights must be non-negative")
                if np.all(weights_flat == 0.0):
                    raise ValueError("pod_svd integration weights are all zero")
                self._sqrt_weights = np.sqrt(weights_flat)

        n_samples = field_arr.shape[0]
        channels = field_arr.shape[-1]
        basis_list: list[np.ndarray] = []
        sv_list: list[np.ndarray] = []
        resolved_modes: int | None = None
        for ch in range(channels):
            snapshots = field_arr[..., ch].reshape(n_samples, -1, order="C")
            if mask_indices is not None:
                snapshots = snapshots[:, mask_indices]
            if self._inner_product_effective == "domain_weights":
                if self._sqrt_weights is None:
                    raise ValueError("pod_svd weights are not initialized")
                snapshots = snapshots * self._sqrt_weights[None, :]
            if not np.isfinite(snapshots).all():
                raise ValueError("pod_svd requires finite snapshots")
            _, singular, vh = np.linalg.svd(snapshots, full_matrices=False)
            rank = vh.shape[0]
            target_modes = rank if self._n_modes is None else self._n_modes
            if target_modes > rank:
                raise ValueError("pod_svd n_modes exceeds available rank")
            if resolved_modes is None:
                resolved_modes = target_modes
            elif resolved_modes != target_modes:
                raise ValueError("pod_svd mode count mismatch across channels")
            basis_list.append(vh[:resolved_modes].T)
            sv_list.append(singular[:resolved_modes])

        if resolved_modes is None:
            raise ValueError("pod_svd failed to resolve modes")

        eigvals_list = []
        for singular in sv_list:
            denom = max(n_samples - 1, 1)
            eigvals_list.append((singular ** 2) / float(denom))

        if self._n_modes is None and self._rank_select["enable"]:
            mean_eigvals = np.mean(np.stack(eigvals_list, axis=0), axis=0)
            selected = _select_rank_by_energy(
                mean_eigvals,
                float(self._rank_select["energy"]),
                self._rank_select["max_modes"],
            )
            if selected <= 0:
                raise ValueError("pod_svd rank_select resulted in zero modes")
            basis_list = [basis[:, :selected] for basis in basis_list]
            sv_list = [sv[:selected] for sv in sv_list]
            eigvals_list = [eig[:selected] for eig in eigvals_list]
            resolved_modes = selected

        self._basis = basis_list
        self._singular_values = sv_list
        self._eigvals = eigvals_list
        self._mask = mask
        self._mask_indices = mask_indices
        self._grid_shape = domain_spec.grid_shape
        self._channels = channels
        self._n_samples = n_samples
        self._n_modes_effective = resolved_modes
        return self

    def _check_mask_consistency(
        self,
        field_mask: np.ndarray | None,
        domain_spec: DomainSpec,
    ) -> np.ndarray | None:
        combined = _combine_masks(field_mask, domain_spec.mask)
        if self._mask_policy == "error":
            if combined is not None:
                raise ValueError("pod_svd mask_policy=error does not allow masks")
            if self._mask is not None:
                raise ValueError("pod_svd was fit without mask but mask is present")
            return None
        if combined is None:
            if self._mask is not None:
                raise ValueError("pod_svd requires mask but none was provided")
            return None
        if self._mask is None:
            raise ValueError("pod_svd was fit without mask but mask is present")
        if combined.shape != self._mask.shape or not np.array_equal(combined, self._mask):
            raise ValueError("pod_svd requires the same mask used during fit")
        return combined

    def transform(
        self,
        field: np.ndarray,
        mask: np.ndarray | None,
        domain_spec: DomainSpec,
    ) -> np.ndarray:
        validate_decomposer_compatibility(domain_spec, self._cfg)
        if self._basis is None or self._grid_shape is None or self._channels is None:
            raise ValueError("pod_svd fit must be called before transform")
        inner_product = getattr(self, "_inner_product_effective", "euclidean")
        sqrt_weights = getattr(self, "_sqrt_weights", None)
        field_3d, was_2d = _normalize_field(field)
        if field_3d.shape[:2] != domain_spec.grid_shape:
            raise ValueError(
                f"field shape {field_3d.shape[:2]} does not match domain {domain_spec.grid_shape}"
            )
        if field_3d.shape[:2] != self._grid_shape:
            raise ValueError("pod_svd domain grid does not match fit")
        if field_3d.shape[-1] != self._channels:
            raise ValueError("pod_svd field channels do not match fit")
        field_mask = _normalize_mask(mask, field_3d.shape[:2])
        self._check_mask_consistency(field_mask, domain_spec)
        mask_indices = self._mask_indices
        coeffs = []
        mode_scales: list[np.ndarray] = []
        for ch in range(field_3d.shape[-1]):
            vec = field_3d[..., ch].reshape(-1, order="C")
            if mask_indices is not None:
                vec = vec[mask_indices]
            if inner_product == "domain_weights":
                if sqrt_weights is None:
                    raise ValueError("pod_svd weights are not initialized")
                vec = vec * sqrt_weights
            coeff_vec = vec @ self._basis[ch]
            if self._mode_weight["enable"]:
                if self._eigvals is None:
                    raise ValueError("pod_svd mode_weight requires eigvals")
                scale = np.sqrt(np.asarray(self._eigvals[ch], dtype=np.float64) + 1e-12)
                coeff_vec = coeff_vec * scale
                mode_scales.append(scale)
            coeffs.append(coeff_vec)
        coeff_tensor = np.stack(coeffs, axis=0)
        if self._mode_weight["enable"]:
            self._mode_weight_scale = mode_scales
        self._coeff_shape = coeff_tensor.shape
        self._field_ndim = 2 if was_2d else 3
        valid_count = (
            int(self._mask_indices.size)
            if self._mask_indices is not None
            else int(field_3d.shape[0] * field_3d.shape[1])
        )
        # REVIEW: channel-first coeff layout keeps POD consistent with other decomposers.
        self._coeff_meta = {
            "method": self.name,
            "field_shape": [int(field_3d.shape[0]), int(field_3d.shape[1])]
            if was_2d
            else [int(field_3d.shape[0]), int(field_3d.shape[1]), int(field_3d.shape[2])],
            "field_ndim": self._field_ndim,
            "field_layout": "HW" if was_2d else "HWC",
            "channels": int(field_3d.shape[-1]),
            "coeff_shape": [int(x) for x in coeff_tensor.shape],
            "coeff_layout": "CK",
            "flatten_order": "C",
            "complex_format": "real",
            "keep": "all",
            "n_samples": int(self._n_samples) if self._n_samples is not None else None,
            "n_modes": int(self._n_modes_effective) if self._n_modes_effective is not None else None,
            "projection": "svd",
            "eigvals": [eig.tolist() for eig in self._eigvals] if self._eigvals is not None else None,
            "mask_policy": self._mask_policy,
            "mask_valid_count": valid_count,
            "inner_product": inner_product,
            "mean_centered": False,
            "rank_select": dict(self._rank_select),
            "mode_weight": dict(self._mode_weight),
        }
        return coeff_tensor

    def inverse_transform(self, coeff: np.ndarray, domain_spec: DomainSpec) -> np.ndarray:
        validate_decomposer_compatibility(domain_spec, self._cfg)
        if self._basis is None or self._grid_shape is None or self._field_ndim is None:
            raise ValueError("pod_svd transform must be called before inverse_transform")
        inner_product = getattr(self, "_inner_product_effective", "euclidean")
        sqrt_weights = getattr(self, "_sqrt_weights", None)
        if domain_spec.grid_shape != self._grid_shape:
            raise ValueError("pod_svd domain grid does not match fit")
        if self._coeff_shape is None:
            raise ValueError("pod_svd coeff_shape is not available")
        coeff = np.asarray(coeff).reshape(-1)
        expected = int(np.prod(self._coeff_shape))
        if coeff.size > expected:
            raise ValueError(f"coeff size {coeff.size} exceeds expected {expected}")
        if coeff.size < expected:
            padded = np.zeros(expected, dtype=coeff.dtype)
            padded[: coeff.size] = coeff
            coeff = padded
        coeff_tensor = coeff.reshape(self._coeff_shape, order="C")
        if self._mode_weight["enable"]:
            if self._mode_weight_scale is None:
                if self._eigvals is None:
                    raise ValueError("pod_svd mode_weight requires eigvals")
                self._mode_weight_scale = [
                    np.sqrt(np.asarray(eig, dtype=np.float64) + 1e-12) for eig in self._eigvals
                ]
            coeff_tensor = np.stack(
                [coeff_tensor[ch] / self._mode_weight_scale[ch] for ch in range(coeff_tensor.shape[0])],
                axis=0,
            )
        height, width = self._grid_shape
        mask_indices = self._mask_indices

        field_channels = []
        for ch in range(coeff_tensor.shape[0]):
            vec = coeff_tensor[ch] @ self._basis[ch].T
            if inner_product == "domain_weights":
                if sqrt_weights is None:
                    raise ValueError("pod_svd weights are not initialized")
                restored = np.zeros_like(vec)
                valid = sqrt_weights > 0.0
                restored[valid] = vec[valid] / sqrt_weights[valid]
                vec = restored
            if mask_indices is not None:
                full = np.zeros(height * width, dtype=vec.dtype)
                full[mask_indices] = vec
                field_c = full.reshape(height, width, order="C")
            else:
                field_c = vec.reshape(height, width, order="C")
            field_channels.append(field_c)
        field_hat = np.stack(field_channels, axis=-1)
        if self._field_ndim == 2 and field_hat.shape[-1] == 1:
            return field_hat[..., 0]
        return field_hat
