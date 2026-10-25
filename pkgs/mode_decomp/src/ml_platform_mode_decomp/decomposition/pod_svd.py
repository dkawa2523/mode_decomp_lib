"""POD/SVD decomposer for data-driven bases."""

from __future__ import annotations

from .base import field_arrays

from ml_platform_mode_decomp.config import constructor_config, integer_parameter

from typing import Any, Mapping

import numpy as np

from ml_platform_mode_decomp.geometry import DomainSpec, validate_decomposer_compatibility

from .base import BaseDecomposer, _combine_masks, _normalize_field, _normalize_mask, require_cfg
from .pod_utils import (
    project_pod,
    reconstruct_pod,
    resolve_rank_select as _resolve_rank_select,
    resolve_mode_weight as _resolve_mode_weight,
    select_pod_modes,
    singular_value_rank,
    snapshot_matrix,
    component_blocks_meta,
)

_MASK_POLICIES = {"error", "ignore_masked_points"}
_INNER_PRODUCTS = {"euclidean", "domain_weights"}


class PODSVDDecomposer(BaseDecomposer[np.ndarray]):
    """POD/SVD decomposer fitted on training snapshots."""

    synthesis_kind = "affine"

    def __init__(
        self,
        *,
        cfg: Mapping[str, Any] | None = None,
        options: Any = None,
        n_modes: Any = None,
        mask_policy: Any = "ignore_masked_points",
        inner_product: Any = "euclidean",
    ) -> None:
        cfg = constructor_config(cfg, locals(), name="pod_svd")
        super().__init__(cfg=cfg)
        self.name = "pod_svd"
        self._mask_policy = str(require_cfg(cfg, "mask_policy", label="decompose"))
        if self._mask_policy not in _MASK_POLICIES:
            raise ValueError(f"decompose.mask_policy must be one of {_MASK_POLICIES}, got {self._mask_policy}")
        n_modes = cfg["n_modes"]
        if n_modes is None:
            self._n_modes = None
        else:
            self._n_modes = integer_parameter(n_modes, "n_modes")
            if self._n_modes <= 0:
                raise ValueError("decompose.n_modes must be > 0 for pod_svd")
        self._inner_product = str(cfg["inner_product"]).strip() or "euclidean"
        if self._inner_product not in _INNER_PRODUCTS:
            raise ValueError(f"decompose.inner_product must be one of {_INNER_PRODUCTS}, got {self._inner_product}")
        self._inner_product_effective = self._inner_product
        self._rank_select = _resolve_rank_select(cfg)
        self._mode_weight = _resolve_mode_weight(cfg)
        self._basis: list[np.ndarray] | None = None
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
        self._diagnostics: dict[str, Any] = {}

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
            if any(not np.array_equal(sample_mask, sample) for sample in mask_arr[1:]):
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
        field_arr, mask_arr = field_arrays(dataset, domain_spec, scalar=False)

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
                raise ValueError("pod_svd inner_product=domain_weights requires domain integration weights")
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
        ranks: list[int] = []
        total_energy = 0.0
        channel_energy = []
        for ch in range(channels):
            snapshots = snapshot_matrix(field_arr[..., ch], mask_indices, method="pod_svd")
            if self._inner_product_effective == "domain_weights":
                if self._sqrt_weights is None:
                    raise ValueError("pod_svd weights are not initialized")
                snapshots *= self._sqrt_weights
            if any(not np.isfinite(row).all() for row in snapshots):
                raise ValueError("pod_svd requires finite snapshots after weighting")
            energy = float(np.einsum("ij,ij->", snapshots, snapshots))
            channel_energy.append(energy)
            total_energy += energy
            _, singular, vh = np.linalg.svd(snapshots, full_matrices=False)
            rank = singular_value_rank(singular, snapshots.shape)
            ranks.append(rank)
            basis_list.append(vh[:rank].T)
            sv_list.append(singular[:rank])

        threshold = float(self._rank_select["energy"]) if self._rank_select["enable"] else None
        selected, diagnostics = [], []
        for ch, (singular, energy) in enumerate(zip(sv_list, channel_energy, strict=True)):
            count, detail = select_pod_modes(
                singular**2,
                energy,
                n_modes=self._n_modes,
                method=f"pod_svd component {ch}",
                energy_threshold=threshold,
                max_modes=self._rank_select["max_modes"],
            )
            selected.append(count)
            diagnostics.append(detail)
        basis_list = [basis[:, :count] for basis, count in zip(basis_list, selected, strict=True)]
        sv_list = [singular[:count] for singular, count in zip(sv_list, selected, strict=True)]
        eigvals_list = [singular**2 / float(max(n_samples - 1, 1)) for singular in sv_list]
        retained = sum(item["retained_training_energy"] for item in diagnostics)
        fraction = float(np.clip(retained / total_energy, 0, 1)) if total_energy else 1.0
        self._diagnostics = {
            "selected_modes": sum(selected),
            "channel_selected_modes": selected,
            "total_training_energy": total_energy,
            "retained_training_energy": retained,
            "retained_energy_fraction": fraction,
            "energy_threshold": threshold,
            "energy_threshold_met": (
                all(item["energy_threshold_met"] for item in diagnostics) if threshold is not None else None
            ),
            "effective_rank": sum(ranks),
            "channel_effective_ranks": ranks,
            "spectrum_complete": True,
            "energy_metric": self._inner_product_effective,
            "centered": False,
        }

        self._basis = basis_list
        self._eigvals = eigvals_list
        self._mask = mask
        self._mask_indices = mask_indices
        self._grid_shape = domain_spec.grid_shape
        self._channels = channels
        self._n_samples = n_samples
        self._n_modes_effective = sum(selected)
        self._component_blocks = True
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
            raise ValueError(f"field shape {field_3d.shape[:2]} does not match domain {domain_spec.grid_shape}")
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
            coeff_vec = project_pod(
                vec, self._basis[ch], projection_weights=sqrt_weights if inner_product == "domain_weights" else None
            )
            if self._mode_weight["enable"]:
                if self._eigvals is None:
                    raise ValueError("pod_svd mode_weight requires eigvals")
                scale = np.sqrt(np.asarray(self._eigvals[ch], dtype=np.float64) + 1e-12)
                coeff_vec = coeff_vec * scale
                mode_scales.append(scale)
            coeffs.append(coeff_vec)
        coeff_tensor = np.concatenate(coeffs) if getattr(self, "_component_blocks", False) else np.stack(coeffs, axis=0)
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
            "diagnostics": dict(self._diagnostics),
            "mode_weight": dict(self._mode_weight),
        }
        if getattr(self, "_component_blocks", False):
            self._coeff_meta = component_blocks_meta(self._coeff_meta, [len(value) for value in coeffs])
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
            raise ValueError("Coefficient size is smaller than the fitted layout")
        if getattr(self, "_component_blocks", False):
            offsets = np.cumsum([0, *(basis.shape[1] for basis in self._basis)])
            coeff_tensor = [coeff[offsets[ch] : offsets[ch + 1]] for ch in range(len(self._basis))]
        else:
            coeff_tensor = list(coeff.reshape(self._coeff_shape, order="C"))
        if self._mode_weight["enable"]:
            if self._mode_weight_scale is None:
                if self._eigvals is None:
                    raise ValueError("pod_svd mode_weight requires eigvals")
                self._mode_weight_scale = [np.sqrt(np.asarray(eig, dtype=np.float64) + 1e-12) for eig in self._eigvals]
            coeff_tensor = [value / self._mode_weight_scale[ch] for ch, value in enumerate(coeff_tensor)]
        height, width = self._grid_shape
        mask_indices = self._mask_indices

        field_channels = []
        for ch in range(len(coeff_tensor)):
            if inner_product == "domain_weights" and sqrt_weights is None:
                raise ValueError("pod_svd weights are not initialized")
            vec = reconstruct_pod(
                coeff_tensor[ch],
                self._basis[ch],
                sqrt_weights=sqrt_weights if inner_product == "domain_weights" else None,
            )
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
