"""POD decomposer with backend/solver/inner-product configuration."""
from __future__ import annotations

import time
import warnings
from typing import Any, Mapping

import numpy as np
from mode_decomp_ml.config import cfg_get
from sklearn.decomposition import IncrementalPCA, PCA

from mode_decomp_ml.domain import DomainSpec, validate_decomposer_compatibility
from mode_decomp_ml.optional import require_dependency

from .base import BaseDecomposer, ChannelwiseAdapter, _combine_masks, _normalize_field, _normalize_mask, parse_bool
from mode_decomp_ml.plugins.registry import register_decomposer

try:  # optional dependency
    import modred as _modred
except Exception:  # pragma: no cover - modred optional
    _modred = None

_MASK_POLICIES = {"error", "ignore_masked_points"}
_BACKENDS = {"sklearn", "modred"}
_SOLVERS = {"direct", "snapshots", "randomized", "incremental"}
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



def _require_modred() -> None:
    require_dependency(
        _modred,
        name="pod backend=modred",
        pip_name="modred",
        extra_hint="Install modred or choose backend=sklearn.",
    )


def _stack_snapshots_for_modred(snapshots: np.ndarray) -> np.ndarray:
    arr = np.asarray(snapshots, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"modred snapshots must be 2D, got shape {arr.shape}")
    return arr.T


class _PODScalarDecomposer(BaseDecomposer):
    """Scalar POD decomposer (no channel branching)."""

    def __init__(self, *, cfg: Mapping[str, Any]) -> None:
        super().__init__(cfg=cfg)
        self.name = "pod"
        self._mask_policy = str(cfg_get(cfg, "mask_policy", "error")).strip() or "error"
        if self._mask_policy not in _MASK_POLICIES:
            raise ValueError(
                f"decompose.mask_policy must be one of {_MASK_POLICIES}, got {self._mask_policy}"
            )
        self._backend = str(cfg_get(cfg, "backend", "sklearn")).strip() or "sklearn"
        if self._backend not in _BACKENDS:
            raise ValueError(f"decompose.backend must be one of {_BACKENDS}, got {self._backend}")
        self._solver = str(cfg_get(cfg, "solver", "direct")).strip() or "direct"
        if self._solver not in _SOLVERS:
            raise ValueError(f"decompose.solver must be one of {_SOLVERS}, got {self._solver}")
        self._inner_product = str(cfg_get(cfg, "inner_product", "euclidean")).strip() or "euclidean"
        if self._inner_product not in _INNER_PRODUCTS:
            raise ValueError(
                f"decompose.inner_product must be one of {_INNER_PRODUCTS}, got {self._inner_product}"
            )
        n_modes = cfg_get(cfg, "n_modes", None)
        if n_modes is None:
            self._n_modes = None
        else:
            self._n_modes = int(n_modes)
            if self._n_modes <= 0:
                raise ValueError("decompose.n_modes must be > 0 for pod")
        seed = cfg_get(cfg, "seed", None)
        self.seed = int(seed) if seed is not None else None
        self._rank_select = _resolve_rank_select(cfg)
        self._mode_weight = _resolve_mode_weight(cfg)

        self.mean: np.ndarray | None = None
        self.modes: np.ndarray | None = None
        self.eigvals: np.ndarray | None = None
        self.var: np.ndarray | None = None
        self.weights_type: str | None = None
        self.inner_product: str | None = None
        self.backend: str | None = None
        self.solver: str | None = None

        self._inner_product_effective = self._inner_product
        self._mask: np.ndarray | None = None
        self._mask_indices: np.ndarray | None = None
        self._sqrt_weights: np.ndarray | None = None
        self._inner_product_weights: np.ndarray | None = None
        self._grid_shape: tuple[int, int] | None = None
        self._n_samples: int | None = None
        self._n_modes_effective: int | None = None
        self._coeff_shape: tuple[int, ...] | None = None
        self._field_ndim: int | None = None
        self._fit_batch_size: int | None = None
        self._fit_n_batches: int | None = None
        self._fit_time_sec: float | None = None
        self._mode_weight_scale: np.ndarray | None = None

    def _resolve_mask(
        self,
        domain_spec: DomainSpec,
        mask_arr: np.ndarray | None,
    ) -> np.ndarray | None:
        domain_mask = domain_spec.mask
        if self._mask_policy == "error":
            if domain_mask is not None or mask_arr is not None:
                raise ValueError("pod mask_policy=error does not allow masks")
            return None
        if self._mask_policy != "ignore_masked_points":
            raise ValueError(f"Unsupported mask_policy for pod: {self._mask_policy}")
        # CONTRACT: ignore_masked_points requires a fixed mask across samples.
        mask = domain_mask
        if mask_arr is not None:
            sample_mask = np.asarray(mask_arr[0]).astype(bool)
            if not np.all(mask_arr == sample_mask):
                raise ValueError("pod requires a fixed mask across samples")
            mask = sample_mask if mask is None else (mask & sample_mask)
        if mask is not None and mask.shape != domain_spec.grid_shape:
            raise ValueError(f"mask shape {mask.shape} does not match {domain_spec.grid_shape}")
        return mask

    def _check_mask_consistency(
        self,
        field_mask: np.ndarray | None,
        domain_spec: DomainSpec,
    ) -> np.ndarray | None:
        combined = _combine_masks(field_mask, domain_spec.mask)
        if self._mask_policy == "error":
            if combined is not None:
                raise ValueError("pod mask_policy=error does not allow masks")
            if self._mask is not None:
                raise ValueError("pod was fit without mask but mask is present")
            return None
        if combined is None:
            if self._mask is not None:
                raise ValueError("pod requires mask but none was provided")
            return None
        if self._mask is None:
            raise ValueError("pod was fit without mask but mask is present")
        if combined.shape != self._mask.shape or not np.array_equal(combined, self._mask):
            raise ValueError("pod requires the same mask used during fit")
        return combined

    def _validate_backend(self) -> None:
        if self._backend == "sklearn":
            return
        if self._backend == "modred":
            if self._solver != "snapshots":
                raise NotImplementedError(
                    "pod backend=modred supports solver=snapshots only in v1"
                )
            _require_modred()
            return
        raise ValueError(f"Unsupported backend for pod: {self._backend}")

    def _fit_incremental(
        self,
        *,
        dataset: Any,
        domain_spec: DomainSpec,
    ) -> "_PODScalarDecomposer":
        if domain_spec is None:
            raise ValueError("pod requires domain_spec for fit")
        n_samples = len(dataset)
        if n_samples <= 0:
            raise ValueError("pod requires at least one training sample")

        domain_mask = domain_spec.mask
        mask: np.ndarray | None = None
        mask_indices: np.ndarray | None = None
        mask_ref: np.ndarray | None = None
        sum_vec: np.ndarray | None = None
        sum_sq: np.ndarray | None = None

        def _sample_field_and_mask(index: int) -> tuple[np.ndarray, np.ndarray | None]:
            sample = dataset[index]
            field = np.asarray(sample.field)
            if field.ndim == 3:
                if field.shape[-1] != 1:
                    raise ValueError(f"pod expects scalar fields, got shape {field.shape}")
                field = field[..., 0]
            elif field.ndim != 2:
                raise ValueError(f"pod expects 2D fields, got shape {field.shape}")
            if field.shape != domain_spec.grid_shape:
                raise ValueError(
                    f"field shape {field.shape} does not match domain {domain_spec.grid_shape}"
                )
            if sample.mask is None:
                mask_arr = None
            else:
                mask_arr = np.asarray(sample.mask).astype(bool)
                if mask_arr.shape != domain_spec.grid_shape:
                    raise ValueError(
                        f"mask shape {mask_arr.shape} does not match {domain_spec.grid_shape}"
                    )
            return field, mask_arr

        for idx in range(n_samples):
            field, sample_mask = _sample_field_and_mask(idx)
            if idx == 0:
                if self._mask_policy == "error":
                    if domain_mask is not None or sample_mask is not None:
                        raise ValueError("pod mask_policy=error does not allow masks")
                    mask = None
                elif self._mask_policy == "ignore_masked_points":
                    if sample_mask is not None:
                        mask_ref = sample_mask
                        mask = sample_mask if domain_mask is None else (domain_mask & sample_mask)
                    else:
                        mask = domain_mask
                else:
                    raise ValueError(f"Unsupported mask_policy for pod: {self._mask_policy}")
                if mask is not None:
                    mask_flat = mask.reshape(-1, order="C")
                    mask_indices = np.flatnonzero(mask_flat)
                    if mask_indices.size == 0:
                        raise ValueError("pod mask has no valid entries")
            else:
                if self._mask_policy == "error":
                    if sample_mask is not None:
                        raise ValueError("pod mask_policy=error does not allow masks")
                else:
                    if mask_ref is None:
                        if sample_mask is not None:
                            raise ValueError("mask must be present for all samples or none")
                    else:
                        if sample_mask is None:
                            raise ValueError("mask must be present for all samples or none")
                        if not np.array_equal(mask_ref, sample_mask):
                            raise ValueError("pod requires a fixed mask across samples")

            vec = field.reshape(-1, order="C")
            if mask_indices is not None:
                vec = vec[mask_indices]
            if not np.isfinite(vec).all():
                raise ValueError("pod requires finite snapshots")
            vec64 = vec.astype(np.float64)
            if sum_vec is None:
                sum_vec = vec64.copy()
                sum_sq = vec64 * vec64
            else:
                sum_vec += vec64
                sum_sq += vec64 * vec64

        if sum_vec is None or sum_sq is None:
            raise ValueError("pod requires at least one training sample")

        mean = sum_vec / float(n_samples)
        var = sum_sq / float(n_samples) - mean * mean
        var = np.maximum(var, 0.0)

        self._inner_product_effective = self._inner_product
        self._sqrt_weights = None
        self._inner_product_weights = None
        weights_type = "euclidean"
        if self._inner_product == "domain_weights":
            weights = domain_spec.integration_weights()
            if weights is None:
                warnings.warn(
                    "pod inner_product=domain_weights but domain has no integration_weights; "
                    "falling back to euclidean",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._inner_product_effective = "euclidean"
            else:
                weights_arr = np.asarray(weights, dtype=np.float64)
                if weights_arr.shape != domain_spec.grid_shape:
                    raise ValueError(
                        f"pod weights shape {weights_arr.shape} does not match domain {domain_spec.grid_shape}"
                    )
                weights_flat = weights_arr.reshape(-1, order="C")
                if mask_indices is not None:
                    weights_flat = weights_flat[mask_indices]
                if not np.isfinite(weights_flat).all():
                    raise ValueError("pod requires finite integration weights")
                if np.any(weights_flat < 0.0):
                    raise ValueError("pod integration weights must be non-negative")
                if np.all(weights_flat == 0.0):
                    raise ValueError("pod integration weights are all zero")
                weights_type = "integration_weights"
                self._sqrt_weights = np.sqrt(weights_flat)

        n_features = int(mean.shape[0])
        rank = min(n_samples, n_features)
        target_modes = rank if self._n_modes is None else self._n_modes
        if self._n_modes is None and self._rank_select["enable"] and self._rank_select["max_modes"]:
            target_modes = min(int(self._rank_select["max_modes"]), rank)
        if target_modes > rank:
            raise ValueError("pod n_modes exceeds available rank")

        batch_cfg = cfg_get(self._cfg, "batch_size", None)
        if batch_cfg is None:
            auto_batch = max(int(target_modes) * 5, 10)
            batch_size = min(n_samples, auto_batch)
        else:
            batch_size = int(batch_cfg)
        if batch_size <= 0:
            raise ValueError("decompose.batch_size must be > 0 for pod incremental")
        if batch_size < int(target_modes):
            batch_size = int(target_modes)
        if batch_size > n_samples:
            batch_size = n_samples

        ipca = IncrementalPCA(n_components=int(target_modes), batch_size=int(batch_size))
        start = time.perf_counter()
        n_batches = 0
        for start_idx in range(0, n_samples, int(batch_size)):
            batch_vecs: list[np.ndarray] = []
            for idx in range(start_idx, min(n_samples, start_idx + int(batch_size))):
                field, _ = _sample_field_and_mask(idx)
                vec = field.reshape(-1, order="C")
                if mask_indices is not None:
                    vec = vec[mask_indices]
                if not np.isfinite(vec).all():
                    raise ValueError("pod requires finite snapshots")
                batch_vecs.append(vec)
            batch_arr = np.stack(batch_vecs, axis=0)
            batch_centered = batch_arr - mean
            if self._inner_product_effective == "domain_weights":
                if self._sqrt_weights is None:
                    raise ValueError("pod weights are not initialized")
                batch_centered = batch_centered * self._sqrt_weights[None, :]
            if not np.isfinite(batch_centered).all():
                raise ValueError("pod requires finite snapshots")
            ipca.partial_fit(batch_centered)
            n_batches += 1
        fit_time = float(time.perf_counter() - start)

        self.mean = mean
        self.modes = ipca.components_.T
        self.eigvals = np.asarray(ipca.explained_variance_, dtype=np.float64)
        if hasattr(ipca, "var_"):
            self.var = np.asarray(ipca.var_, dtype=np.float64)
        else:
            self.var = var
        self.weights_type = weights_type
        self.inner_product = self._inner_product_effective
        self.backend = self._backend
        self.solver = self._solver

        self._mask = mask
        self._mask_indices = mask_indices
        self._grid_shape = domain_spec.grid_shape
        self._n_samples = n_samples
        if self._n_modes is None and self._rank_select["enable"]:
            selected = _select_rank_by_energy(
                self.eigvals,
                float(self._rank_select["energy"]),
                self._rank_select["max_modes"],
            )
            if selected <= 0:
                raise ValueError("pod rank_select resulted in zero modes")
            self.modes = self.modes[:, :selected]
            self.eigvals = self.eigvals[:selected]
            self._n_modes_effective = selected
        else:
            self._n_modes_effective = target_modes
        self._fit_batch_size = int(batch_size)
        self._fit_n_batches = int(n_batches)
        self._fit_time_sec = fit_time
        return self

    def fit(self, dataset: Any | None = None, domain_spec: DomainSpec | None = None) -> "_PODScalarDecomposer":
        if dataset is None:
            raise ValueError("pod requires dataset for fit")
        if domain_spec is None:
            raise ValueError("pod requires domain_spec for fit")
        self._validate_backend()
        self._fit_batch_size = None
        self._fit_n_batches = None
        self._fit_time_sec = None
        if self._backend == "sklearn" and self._solver == "incremental":
            return self._fit_incremental(dataset=dataset, domain_spec=domain_spec)
        fields: list[np.ndarray] = []
        masks: list[np.ndarray | None] = []
        for idx in range(len(dataset)):
            sample = dataset[idx]
            field = np.asarray(sample.field)
            if field.ndim == 3:
                if field.shape[-1] != 1:
                    raise ValueError(f"pod expects scalar fields, got shape {field.shape}")
                field = field[..., 0]
            elif field.ndim != 2:
                raise ValueError(f"pod expects 2D fields, got shape {field.shape}")
            if field.shape != domain_spec.grid_shape:
                raise ValueError(
                    f"field shape {field.shape} does not match domain {domain_spec.grid_shape}"
                )
            fields.append(field)
            masks.append(None if sample.mask is None else np.asarray(sample.mask))
        if not fields:
            raise ValueError("pod requires at least one training sample")
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
                raise ValueError("pod mask has no valid entries")
        else:
            mask_indices = None

        n_samples = field_arr.shape[0]
        snapshots = field_arr.reshape(n_samples, -1, order="C")
        if mask_indices is not None:
            snapshots = snapshots[:, mask_indices]
        if not np.isfinite(snapshots).all():
            raise ValueError("pod requires finite snapshots")

        mean = snapshots.mean(axis=0)
        self.var = snapshots.var(axis=0, dtype=np.float64)
        centered = snapshots - mean

        self._inner_product_effective = self._inner_product
        self._sqrt_weights = None
        self._inner_product_weights = None
        weights_type = "euclidean"
        if self._inner_product == "domain_weights":
            weights = domain_spec.integration_weights()
            if weights is None:
                warnings.warn(
                    "pod inner_product=domain_weights but domain has no integration_weights; "
                    "falling back to euclidean",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._inner_product_effective = "euclidean"
            else:
                weights_arr = np.asarray(weights, dtype=np.float64)
                if weights_arr.shape != domain_spec.grid_shape:
                    raise ValueError(
                        f"pod weights shape {weights_arr.shape} does not match domain {domain_spec.grid_shape}"
                    )
                weights_flat = weights_arr.reshape(-1, order="C")
                if mask_indices is not None:
                    weights_flat = weights_flat[mask_indices]
                if not np.isfinite(weights_flat).all():
                    raise ValueError("pod requires finite integration weights")
                if np.any(weights_flat < 0.0):
                    raise ValueError("pod integration weights must be non-negative")
                if np.all(weights_flat == 0.0):
                    raise ValueError("pod integration weights are all zero")
                weights_type = "integration_weights"
                if self._backend == "sklearn":
                    self._sqrt_weights = np.sqrt(weights_flat)
                else:
                    self._inner_product_weights = weights_flat
        if self._inner_product_effective == "domain_weights" and self._backend == "sklearn":
            if self._sqrt_weights is None:
                raise ValueError("pod weights are not initialized")
            centered = centered * self._sqrt_weights[None, :]

        n_samples, n_features = centered.shape
        rank = min(n_samples, n_features)
        target_modes = rank if self._n_modes is None else self._n_modes
        if self._n_modes is None and self._rank_select["enable"] and self._rank_select["max_modes"]:
            target_modes = min(int(self._rank_select["max_modes"]), rank)
        if target_modes > rank:
            raise ValueError("pod n_modes exceeds available rank")

        if self._backend == "sklearn":
            if self._solver in {"direct", "snapshots"}:
                svd_solver = "full"
            elif self._solver == "randomized":
                svd_solver = "randomized"
            else:
                raise ValueError(f"Unsupported solver for sklearn backend: {self._solver}")
            pca_kwargs: dict[str, Any] = {
                "n_components": int(target_modes),
                "svd_solver": svd_solver,
            }
            if svd_solver == "randomized":
                pca_kwargs["random_state"] = self.seed
            pca = PCA(**pca_kwargs)
            pca.fit(centered)
            self.mean = mean
            self.modes = pca.components_.T
            self.eigvals = np.asarray(pca.explained_variance_, dtype=np.float64)
        else:
            vecs = _stack_snapshots_for_modred(centered)
            weights = self._inner_product_weights
            try:
                eigvals, modes, _ = _modred.pod.compute_POD_arrays_snaps_method(
                    vecs,
                    inner_product_weights=weights,
                )
            except TypeError:
                eigvals, modes, _ = _modred.pod.compute_POD_arrays_snaps_method(vecs)
            eigvals_arr = np.asarray(eigvals, dtype=np.float64).reshape(-1)
            modes_arr = np.asarray(modes, dtype=np.float64)
            if modes_arr.ndim == 1:
                modes_arr = modes_arr[:, None]
            if modes_arr.ndim != 2:
                raise ValueError(f"modred modes must be 2D, got shape {modes_arr.shape}")
            if modes_arr.shape[0] != n_features and modes_arr.shape[1] == n_features:
                modes_arr = modes_arr.T
            if modes_arr.shape[0] != n_features:
                raise ValueError(
                    f"modred modes shape {modes_arr.shape} does not match features {n_features}"
                )
            available_modes = min(modes_arr.shape[1], eigvals_arr.size)
            target_modes = min(target_modes, available_modes)
            self.mean = mean
            self.modes = modes_arr[:, :target_modes]
            self.eigvals = eigvals_arr[:target_modes]
        self.weights_type = weights_type
        self.inner_product = self._inner_product_effective
        self.backend = self._backend
        self.solver = self._solver

        self._mask = mask
        self._mask_indices = mask_indices
        self._grid_shape = domain_spec.grid_shape
        self._n_samples = n_samples
        if self._n_modes is None and self._rank_select["enable"]:
            selected = _select_rank_by_energy(
                self.eigvals if self.eigvals is not None else np.array([]),
                float(self._rank_select["energy"]),
                self._rank_select["max_modes"],
            )
            if selected <= 0:
                raise ValueError("pod rank_select resulted in zero modes")
            if self.modes is not None:
                self.modes = self.modes[:, :selected]
            if self.eigvals is not None:
                self.eigvals = self.eigvals[:selected]
            self._n_modes_effective = selected
        else:
            self._n_modes_effective = target_modes
        return self

    def transform(
        self,
        field: np.ndarray,
        mask: np.ndarray | None,
        domain_spec: DomainSpec,
    ) -> np.ndarray:
        validate_decomposer_compatibility(domain_spec, self._cfg)
        if self.modes is None or self._grid_shape is None:
            raise ValueError("pod fit must be called before transform")
        field_3d, was_2d = _normalize_field(field)
        if field_3d.shape[:2] != domain_spec.grid_shape:
            raise ValueError(
                f"field shape {field_3d.shape[:2]} does not match domain {domain_spec.grid_shape}"
            )
        if field_3d.shape[:2] != self._grid_shape:
            raise ValueError("pod domain grid does not match fit")
        if field_3d.shape[-1] != 1:
            raise ValueError("pod expects scalar fields; use channelwise adapter for vectors")
        field_mask = _normalize_mask(mask, field_3d.shape[:2])
        self._check_mask_consistency(field_mask, domain_spec)

        vec = field_3d[..., 0].reshape(-1, order="C")
        if self._mask_indices is not None:
            vec = vec[self._mask_indices]
        if self.mean is None:
            raise ValueError("pod mean is not available")
        vec = vec - self.mean
        if self._inner_product_effective == "domain_weights":
            if self._backend == "sklearn":
                if self._sqrt_weights is None:
                    raise ValueError("pod weights are not initialized")
                vec = vec * self._sqrt_weights
                coeff_vec = vec @ self.modes
            else:
                if self._inner_product_weights is None:
                    raise ValueError("pod weights are not initialized")
                coeff_vec = (vec * self._inner_product_weights) @ self.modes
        else:
            coeff_vec = vec @ self.modes
        self._mode_weight_scale = None
        if self._mode_weight["enable"]:
            if self.eigvals is None:
                raise ValueError("pod mode_weight requires eigvals")
            scale = np.sqrt(np.asarray(self.eigvals, dtype=np.float64) + 1e-12)
            coeff_vec = coeff_vec * scale
            self._mode_weight_scale = scale
        coeff_tensor = coeff_vec[None, :]
        self._coeff_shape = coeff_tensor.shape
        self._field_ndim = 2 if was_2d else 3
        valid_count = (
            int(self._mask_indices.size)
            if self._mask_indices is not None
            else int(field_3d.shape[0] * field_3d.shape[1])
        )
        meta = self._coeff_meta_base(
            field_shape=field_3d.shape[:2] if was_2d else field_3d.shape,
            field_ndim=self._field_ndim,
            field_layout="HW" if was_2d else "HWC",
            channels=1,
            coeff_shape=coeff_tensor.shape,
            coeff_layout="CK",
            complex_format="real",
        )
        meta.update(
            {
                "n_samples": int(self._n_samples) if self._n_samples is not None else None,
                "n_modes": int(self._n_modes_effective) if self._n_modes_effective is not None else None,
                "projection": "pod",
                "mode_order": "descending_eigval",
                "eigvals": [float(val) for val in self.eigvals] if self.eigvals is not None else None,
                "mask_policy": self._mask_policy,
                "mask_valid_count": valid_count,
                "mean_centered": True,
                "weights_type": self.weights_type,
                "inner_product": self._inner_product_effective,
                "backend": self._backend,
                "solver": self._solver,
                "seed": self.seed,
                "rank_select": dict(self._rank_select),
                "mode_weight": dict(self._mode_weight),
                "metrics": {
                    "batch_size": int(self._fit_batch_size)
                    if self._fit_batch_size is not None
                    else None,
                    "n_batches": int(self._fit_n_batches)
                    if self._fit_n_batches is not None
                    else None,
                    "fit_time": float(self._fit_time_sec)
                    if self._fit_time_sec is not None
                    else None,
                },
            }
        )
        self._coeff_meta = meta
        return coeff_tensor

    def inverse_transform(self, coeff: np.ndarray, domain_spec: DomainSpec) -> np.ndarray:
        validate_decomposer_compatibility(domain_spec, self._cfg)
        if self.modes is None or self._grid_shape is None or self._field_ndim is None:
            raise ValueError("pod transform must be called before inverse_transform")
        if domain_spec.grid_shape != self._grid_shape:
            raise ValueError("pod domain grid does not match fit")
        if self._coeff_shape is None:
            raise ValueError("pod coeff_shape is not available")

        coeff_tensor = self._reshape_coeff(coeff, self._coeff_shape, name=self.name)
        coeff_vec = coeff_tensor[0]
        if self._mode_weight["enable"]:
            if self._mode_weight_scale is None:
                if self.eigvals is None:
                    raise ValueError("pod mode_weight requires eigvals")
                self._mode_weight_scale = np.sqrt(np.asarray(self.eigvals, dtype=np.float64) + 1e-12)
            coeff_vec = coeff_vec / self._mode_weight_scale
        vec = coeff_vec @ self.modes.T
        if self._inner_product_effective == "domain_weights" and self._backend == "sklearn":
            if self._sqrt_weights is None:
                raise ValueError("pod weights are not initialized")
            restored = np.zeros_like(vec)
            valid = self._sqrt_weights > 0.0
            restored[valid] = vec[valid] / self._sqrt_weights[valid]
            vec = restored
        if self.mean is None:
            raise ValueError("pod mean is not available")
        vec = vec + self.mean

        height, width = self._grid_shape
        if self._mask_indices is not None:
            full = np.zeros(height * width, dtype=vec.dtype)
            full[self._mask_indices] = vec
            field_c = full.reshape(height, width, order="C")
        else:
            field_c = vec.reshape(height, width, order="C")
        field_hat = field_c[..., None]
        if self._field_ndim == 2 and field_hat.shape[-1] == 1:
            return field_hat[..., 0]
        return field_hat


@register_decomposer("pod")
class PODDecomposer(BaseDecomposer):
    """POD decomposer with optional channel-wise adapter for vector fields."""

    def __init__(self, *, cfg: Mapping[str, Any]) -> None:
        super().__init__(cfg=cfg)
        self.name = "pod"
        self._impl: BaseDecomposer | None = None
        self._is_channelwise = False
        self.mean: list[np.ndarray] | np.ndarray | None = None
        self.modes: list[np.ndarray] | np.ndarray | None = None
        self.eigvals: list[np.ndarray] | np.ndarray | None = None
        self.var: list[np.ndarray] | np.ndarray | None = None
        self.weights_type: list[str] | str | None = None
        self.inner_product: list[str] | str | None = None
        self.backend: str | None = None
        self.solver: str | None = None
        self.seed: int | None = None
        self.fit_batch_size: list[int | None] | int | None = None
        self.fit_n_batches: list[int | None] | int | None = None
        self.fit_time_sec: list[float | None] | float | None = None

    def _make_scalar(self) -> _PODScalarDecomposer:
        return _PODScalarDecomposer(cfg=dict(self._cfg))

    def _ensure_impl(
        self,
        *,
        dataset: Any | None = None,
        field: np.ndarray | None = None,
    ) -> BaseDecomposer:
        if self._impl is not None:
            return self._impl
        if dataset is not None:
            if len(dataset) == 0:
                raise ValueError("pod requires non-empty dataset for fit")
            sample = dataset[0]
            field_arr = np.asarray(sample.field)
        elif field is not None:
            field_arr = np.asarray(field)
        else:
            raise ValueError("pod requires dataset or field to infer channel layout")
        if field_arr.ndim not in {2, 3}:
            raise ValueError(f"pod expects 2D or 3D field, got shape {field_arr.shape}")
        if field_arr.ndim == 3 and field_arr.shape[-1] > 1:
            self._impl = ChannelwiseAdapter(
                cfg=dict(self._cfg),
                decomposer_factory=lambda cfg: _PODScalarDecomposer(cfg=cfg),
                label=self.name,
            )
            self._is_channelwise = True
        else:
            self._impl = self._make_scalar()
            self._is_channelwise = False
        return self._impl

    def _sync_state(self) -> None:
        if self._impl is None:
            return
        if isinstance(self._impl, ChannelwiseAdapter):
            decomposers = list(self._impl._decomposers)
            if not decomposers:
                return
            self.mean = [dec.mean for dec in decomposers]
            self.modes = [dec.modes for dec in decomposers]
            self.eigvals = [dec.eigvals for dec in decomposers]
            self.var = [dec.var for dec in decomposers]
            self.weights_type = [dec.weights_type for dec in decomposers]
            self.inner_product = [dec.inner_product for dec in decomposers]
            self.backend = decomposers[0].backend if hasattr(decomposers[0], "backend") else None
            self.solver = decomposers[0].solver if hasattr(decomposers[0], "solver") else None
            self.seed = decomposers[0].seed if hasattr(decomposers[0], "seed") else None
            self.fit_batch_size = [getattr(dec, "_fit_batch_size", None) for dec in decomposers]
            self.fit_n_batches = [getattr(dec, "_fit_n_batches", None) for dec in decomposers]
            self.fit_time_sec = [getattr(dec, "_fit_time_sec", None) for dec in decomposers]
            return
        if isinstance(self._impl, _PODScalarDecomposer):
            self.mean = self._impl.mean
            self.modes = self._impl.modes
            self.eigvals = self._impl.eigvals
            self.var = self._impl.var
            self.weights_type = self._impl.weights_type
            self.inner_product = self._impl.inner_product
            self.backend = self._impl.backend
            self.solver = self._impl.solver
            self.seed = self._impl.seed
            self.fit_batch_size = self._impl._fit_batch_size
            self.fit_n_batches = self._impl._fit_n_batches
            self.fit_time_sec = self._impl._fit_time_sec

    def fit(self, dataset: Any | None = None, domain_spec: DomainSpec | None = None) -> "PODDecomposer":
        if dataset is None:
            raise ValueError("pod requires dataset for fit")
        impl = self._ensure_impl(dataset=dataset)
        impl.fit(dataset=dataset, domain_spec=domain_spec)
        self._sync_state()
        return self

    def transform(
        self,
        field: np.ndarray,
        mask: np.ndarray | None,
        domain_spec: DomainSpec,
    ) -> np.ndarray:
        if self._impl is None:
            raise ValueError("pod fit must be called before transform")
        coeff = self._impl.transform(field, mask=mask, domain_spec=domain_spec)
        meta = dict(self._impl.coeff_meta())
        if isinstance(self._impl, ChannelwiseAdapter):
            meta["metrics"] = {
                "batch_size": self.fit_batch_size,
                "n_batches": self.fit_n_batches,
                "fit_time": self.fit_time_sec,
            }
        self._coeff_meta = meta
        return coeff

    def inverse_transform(self, coeff: np.ndarray, domain_spec: DomainSpec) -> np.ndarray:
        if self._impl is None:
            raise ValueError("pod transform must be called before inverse_transform")
        return self._impl.inverse_transform(coeff, domain_spec=domain_spec)


__all__ = ["PODDecomposer"]
