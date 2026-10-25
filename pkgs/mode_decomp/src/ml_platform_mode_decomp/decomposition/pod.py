"""POD decomposer with backend/solver/inner-product configuration."""

from __future__ import annotations

from .base import field_arrays

from ml_platform_mode_decomp.config import constructor_config

from typing import Any, Mapping

import numpy as np
from ml_platform_mode_decomp.config import cfg_get
from sklearn.decomposition import IncrementalPCA
from sklearn.utils.extmath import randomized_svd, svd_flip

from ml_platform_mode_decomp.geometry import DomainSpec, validate_decomposer_compatibility
from ml_platform_mode_decomp.optional import require_dependency

from .base import BaseDecomposer, ChannelwiseAdapter, _combine_masks, _normalize_field, _normalize_mask
from .pod_utils import (
    project_pod,
    reconstruct_pod,
    snapshot_matrix,
    snapshot_svd,
    projection_energy,
    select_pod_modes,
    singular_value_rank,
    component_blocks_meta,
    resolve_rank_select as _resolve_rank_select,
    resolve_mode_weight as _resolve_mode_weight,
)

try:  # optional dependency
    import modred as _modred
except ImportError:  # pragma: no cover - modred optional
    _modred = None

_MASK_POLICIES = {"error", "ignore_masked_points"}
_BACKENDS = {"sklearn", "modred"}
_SOLVERS = {"direct", "snapshots", "randomized", "incremental"}
_INNER_PRODUCTS = {"euclidean", "domain_weights"}


def _require_modred() -> None:
    require_dependency(
        _modred,
        name="pod backend=modred",
        pip_name="modred",
        extra_hint="Install modred or choose backend=sklearn.",
    )


class _PODScalarDecomposer(BaseDecomposer[np.ndarray]):
    """Scalar POD decomposer (no channel branching)."""

    def __init__(self, *, cfg: Mapping[str, Any]) -> None:
        super().__init__(cfg=cfg)
        self.name = "pod"
        self._mask_policy = str(cfg["mask_policy"]).strip() or "error"
        if self._mask_policy not in _MASK_POLICIES:
            raise ValueError(f"decompose.mask_policy must be one of {_MASK_POLICIES}, got {self._mask_policy}")
        self._backend = str(cfg["backend"]).strip() or "sklearn"
        if self._backend not in _BACKENDS:
            raise ValueError(f"decompose.backend must be one of {_BACKENDS}, got {self._backend}")
        self._solver = str(cfg["solver"]).strip() or "direct"
        if self._solver not in _SOLVERS:
            raise ValueError(f"decompose.solver must be one of {_SOLVERS}, got {self._solver}")
        self._inner_product = str(cfg["inner_product"]).strip() or "euclidean"
        if self._inner_product not in _INNER_PRODUCTS:
            raise ValueError(f"decompose.inner_product must be one of {_INNER_PRODUCTS}, got {self._inner_product}")
        n_modes = cfg["n_modes"]
        if n_modes is None:
            self._n_modes = None
        else:
            self._n_modes = int(n_modes)
            if self._n_modes <= 0:
                raise ValueError("decompose.n_modes must be > 0 for pod")
        seed = cfg["seed"]
        self.seed = int(seed) if seed is not None else None
        self._rank_select = _resolve_rank_select(cfg)
        self._mode_weight = _resolve_mode_weight(cfg)

        self.mean: np.ndarray | None = None
        self.modes: np.ndarray | None = None
        self.eigvals: np.ndarray | None = None
        self.weights_type: str | None = None

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
        self._mode_weight_scale: np.ndarray | None = None
        self._diagnostics: dict[str, Any] = {}
        self._candidate_modes: np.ndarray | None = None
        self._candidate_energy: np.ndarray | None = None

    def _retain_basis(self, selected: int) -> None:
        if self._candidate_modes is None or self._candidate_energy is None or self._n_samples is None:
            raise RuntimeError("pod candidate basis is not initialized")
        if not 0 <= selected <= len(self._candidate_energy):
            raise ValueError("pod n_modes exceeds available rank")
        self.modes = self._candidate_modes[:, :selected]
        self.eigvals = self._candidate_energy[:selected] / float(max(self._n_samples - 1, 1))
        self._n_modes_effective = selected
        retained = float(np.sum(self._candidate_energy[:selected]))
        total = self._diagnostics["total_training_energy"]
        fraction = float(np.clip(retained / total, 0.0, 1.0)) if total else 1.0
        threshold = self._diagnostics["energy_threshold"]
        self._diagnostics.update(
            {
                "selected_modes": selected,
                "retained_training_energy": retained,
                "retained_energy_fraction": fraction,
                "energy_threshold_met": (
                    bool(fraction >= threshold - 16 * np.finfo(float).eps) if threshold is not None else None
                ),
            }
        )

    def _select_basis(
        self,
        mode_energy: np.ndarray,
        singular: np.ndarray,
        shape: tuple[int, int],
        total_energy: float,
        *,
        spectrum_complete: bool,
    ) -> None:
        if self.modes is None:
            raise RuntimeError("pod basis is not initialized")
        rank = singular_value_rank(singular, shape)
        energies = np.asarray(mode_energy[:rank], dtype=np.float64)
        # Approximate solvers need not order modes by their actual training
        # projection energy. Measure it in the declared inner product.
        order = np.argsort(energies)[::-1]
        energies = energies[order]
        self.modes = self.modes[:, :rank][:, order]
        threshold = float(self._rank_select["energy"]) if self._rank_select["enable"] else None
        selected, diagnostics = select_pod_modes(
            energies,
            total_energy,
            n_modes=self._n_modes,
            method="pod",
            energy_threshold=threshold,
            max_modes=self._rank_select["max_modes"],
        )
        self._candidate_modes = self.modes
        self._candidate_energy = energies
        self._diagnostics = {
            **diagnostics,
            "effective_rank": rank,
            "spectrum_complete": spectrum_complete,
            "energy_metric": self._inner_product_effective,
            "centered": True,
        }
        self._retain_basis(selected)

    def _discard_candidate_basis(self) -> None:
        # Retained arrays own their storage after channelwise count alignment.
        if self.modes is not None:
            self.modes = self.modes.copy()
        self._candidate_modes = None
        self._candidate_energy = None

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
            if any(not np.array_equal(sample_mask, sample) for sample in mask_arr[1:]):
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
                raise NotImplementedError("pod backend=modred supports solver=snapshots only in v1")
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
        running_mean: np.ndarray | None = None
        centered_sum_sq: np.ndarray | None = None

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
                raise ValueError(f"field shape {field.shape} does not match domain {domain_spec.grid_shape}")
            if sample.mask is None:
                mask_arr = None
            else:
                mask_arr = np.asarray(sample.mask).astype(bool)
                if mask_arr.shape != domain_spec.grid_shape:
                    raise ValueError(f"mask shape {mask_arr.shape} does not match {domain_spec.grid_shape}")
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
            vec64 = np.asarray(vec, dtype=np.float64)
            if running_mean is None:
                running_mean = vec64.copy()
                centered_sum_sq = np.zeros_like(vec64)
            else:
                if centered_sum_sq is None:
                    raise RuntimeError("pod statistics are not initialized")
                delta = vec64 - running_mean
                running_mean += delta / float(idx + 1)
                centered_sum_sq += delta * (vec64 - running_mean)

        if running_mean is None or centered_sum_sq is None:
            raise ValueError("pod requires at least one training sample")

        mean = running_mean

        self._inner_product_effective = self._inner_product
        self._sqrt_weights = None
        self._inner_product_weights = None
        weights_type = "euclidean"
        if self._inner_product == "domain_weights":
            weights = domain_spec.integration_weights()
            if weights is None:
                raise ValueError("pod inner_product=domain_weights requires domain integration weights")
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
        variance_weights = 1.0 if self._sqrt_weights is None else self._sqrt_weights**2
        if float(np.sum(np.maximum(centered_sum_sq, 0.0) * variance_weights)) <= 0:
            self.mean, self.modes = mean, np.empty((len(mean), 0))
            self.weights_type = weights_type
            self._mask, self._mask_indices = mask, mask_indices
            self._grid_shape, self._n_samples = domain_spec.grid_shape, n_samples
            self._select_basis(np.empty(0), np.empty(0), (n_samples, n_features), 0.0, spectrum_complete=True)
            return self
        rank = min(n_samples - 1, n_features)
        if rank == 0:
            raise ValueError("pod training snapshots have zero numerical rank")
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

        def _batches():
            for start_idx in range(0, n_samples, batch_size):
                stop = min(n_samples, start_idx + batch_size)
                batch = np.empty((stop - start_idx, n_features), dtype=np.float64)
                for offset, idx in enumerate(range(start_idx, stop)):
                    field, _ = _sample_field_and_mask(idx)
                    vector = field.reshape(-1)
                    batch[offset] = vector if mask_indices is None else vector[mask_indices]
                batch -= mean
                if self._sqrt_weights is not None:
                    batch *= self._sqrt_weights
                yield batch

        # partial_fit may center its input in-place: each batch owns its memory.
        ipca = IncrementalPCA(n_components=int(target_modes), batch_size=batch_size, copy=False)
        total_energy = 0.0
        for batch in _batches():
            total_energy += float(np.einsum("ij,ij->", batch, batch))
            ipca.partial_fit(batch)

        self.mean = mean
        self.modes = ipca.components_.T
        self.weights_type = weights_type

        self._mask = mask
        self._mask_indices = mask_indices
        self._grid_shape = domain_spec.grid_shape
        self._n_samples = n_samples
        # Replay in bounded memory: IncrementalPCA's truncated covariance is
        # approximate, so its reported eigenvalues are not exact retained energy.
        mode_energy = np.zeros(self.modes.shape[1], dtype=np.float64)
        for batch in _batches():
            mode_energy += projection_energy(batch, self.modes)
        self._select_basis(
            mode_energy,
            np.asarray(ipca.singular_values_),
            (n_samples, n_features),
            total_energy,
            spectrum_complete=target_modes == rank,
        )
        return self

    def fit(self, dataset: Any | None = None, domain_spec: DomainSpec | None = None) -> "_PODScalarDecomposer":
        if dataset is None:
            raise ValueError("pod requires dataset for fit")
        if domain_spec is None:
            raise ValueError("pod requires domain_spec for fit")
        self._validate_backend()
        if self._backend == "sklearn" and self._solver == "incremental":
            return self._fit_incremental(dataset=dataset, domain_spec=domain_spec)
        field_arr, mask_arr = field_arrays(dataset, domain_spec, scalar=True)

        mask = self._resolve_mask(domain_spec, mask_arr)
        if mask is not None:
            mask_flat = mask.reshape(-1, order="C")
            mask_indices = np.flatnonzero(mask_flat)
            if mask_indices.size == 0:
                raise ValueError("pod mask has no valid entries")
        else:
            mask_indices = None

        n_samples = field_arr.shape[0]
        centered = snapshot_matrix(field_arr, mask_indices, method="pod")
        mean = centered.mean(axis=0)
        centered -= mean

        self._inner_product_effective = self._inner_product
        self._sqrt_weights = None
        self._inner_product_weights = None
        weights_type = "euclidean"
        if self._inner_product == "domain_weights":
            weights = domain_spec.integration_weights()
            if weights is None:
                raise ValueError("pod inner_product=domain_weights requires domain integration weights")
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
            centered *= self._sqrt_weights

        if all(np.array_equal(row, centered[0]) for row in centered[1:]):
            centered.fill(0)

        if not np.any(centered):
            self.mean, self.modes = mean, np.empty((len(mean), 0))
            self.weights_type = weights_type
            self._mask, self._mask_indices = mask, mask_indices
            self._grid_shape, self._n_samples = domain_spec.grid_shape, n_samples
            self._select_basis(np.empty(0), np.empty(0), centered.shape, 0.0, spectrum_complete=True)
            return self

        n_samples, n_features = centered.shape
        rank = min(n_samples - 1, n_features)
        if rank == 0:
            raise ValueError("pod training snapshots have zero numerical rank")
        target_modes = rank if self._n_modes is None else self._n_modes
        if self._n_modes is None and self._rank_select["enable"] and self._rank_select["max_modes"]:
            target_modes = min(int(self._rank_select["max_modes"]), rank)
        if target_modes > rank:
            raise ValueError("pod n_modes exceeds available rank")

        if self._backend == "sklearn":
            total_energy = float(np.einsum("ij,ij->", centered, centered))
            # Preserve PCA's second centering: a rounded physical mean can
            # otherwise create a spurious singular direction at large offsets.
            correction = centered.mean(axis=0)
            centered -= correction
            if self._solver in {"direct", "snapshots"}:
                singular, vh = snapshot_svd(centered)
                _, vh = svd_flip(None, vh, u_based_decision=False)
                self.modes = vh.T
                mode_energy = singular**2 + n_samples * (correction @ self.modes) ** 2
            elif self._solver == "randomized":
                _, singular, vh = randomized_svd(
                    centered,
                    n_components=int(target_modes),
                    n_oversamples=10,
                    n_iter="auto",
                    power_iteration_normalizer="auto",
                    flip_sign=False,
                    random_state=self.seed,
                )
                _, vh = svd_flip(None, vh, u_based_decision=False)
                self.modes = vh.T
                mode_energy = projection_energy(centered, self.modes, offset=correction @ self.modes)
            else:
                raise ValueError(f"Unsupported solver for sklearn backend: {self._solver}")
            self.mean = mean
        else:
            modred_backend = _modred
            if modred_backend is None:
                raise ImportError("pod backend=modred requires modred")
            weights = self._inner_product_weights
            result = modred_backend.pod.compute_POD_arrays_snaps_method(
                centered.T,
                inner_product_weights=weights,
                atol=0.0,
                rtol=np.finfo(float).eps * max(centered.shape),
            )
            eigvals, modes = result.eigvals, result.modes
            eigvals_arr = np.asarray(eigvals, dtype=np.float64).reshape(-1)
            modes_arr = np.asarray(modes, dtype=np.float64)
            if modes_arr.ndim == 1:
                modes_arr = modes_arr[:, None]
            if modes_arr.ndim != 2:
                raise ValueError(f"modred modes must be 2D, got shape {modes_arr.shape}")
            if modes_arr.shape[0] != n_features and modes_arr.shape[1] == n_features:
                modes_arr = modes_arr.T
            if modes_arr.shape[0] != n_features:
                raise ValueError(f"modred modes shape {modes_arr.shape} does not match features {n_features}")
            self.mean = mean
            self.modes = modes_arr
            singular = np.sqrt(np.maximum(eigvals_arr, 0.0))
            mode_energy = projection_energy(centered, self.modes, weights)
            total_energy = float(
                np.einsum("ij,ij->", centered, centered)
                if weights is None
                else np.einsum("ij,j,ij->", centered, weights, centered)
            )
        self.weights_type = weights_type

        self._mask = mask
        self._mask_indices = mask_indices
        self._grid_shape = domain_spec.grid_shape
        self._n_samples = n_samples
        self._select_basis(
            mode_energy,
            singular,
            centered.shape,
            total_energy,
            spectrum_complete=self._solver in {"direct", "snapshots"} or target_modes == rank,
        )
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
            raise ValueError(f"field shape {field_3d.shape[:2]} does not match domain {domain_spec.grid_shape}")
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
        projection_weights = None
        if self._inner_product_effective == "domain_weights":
            projection_weights = self._sqrt_weights if self._backend == "sklearn" else self._inner_product_weights
            if projection_weights is None:
                raise ValueError("pod weights are not initialized")
        coeff_vec = project_pod(vec, self.modes, mean=self.mean, projection_weights=projection_weights)
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
                "diagnostics": dict(self._diagnostics),
                "mode_weight": dict(self._mode_weight),
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
        if self.mean is None:
            raise ValueError("pod mean is not available")
        sqrt_weights = self._sqrt_weights if self._backend == "sklearn" else None
        if self._inner_product_effective == "domain_weights" and self._backend == "sklearn" and sqrt_weights is None:
            raise ValueError("pod weights are not initialized")
        vec = reconstruct_pod(coeff_vec, self.modes, mean=self.mean, sqrt_weights=sqrt_weights)

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


def _pod_channel_diagnostics(channels: list[dict[str, Any]]) -> dict[str, Any]:
    total = sum(float(item["total_training_energy"]) for item in channels)
    retained = sum(float(item["retained_training_energy"]) for item in channels)
    threshold = channels[0]["energy_threshold"]
    fraction = float(np.clip(retained / total, 0.0, 1.0)) if total else 1.0
    return {
        "effective_rank": sum(item["effective_rank"] for item in channels),
        "channel_effective_ranks": [item["effective_rank"] for item in channels],
        "selected_modes": sum(item["selected_modes"] for item in channels),
        "channel_selected_modes": [item["selected_modes"] for item in channels],
        "spectrum_complete": all(item["spectrum_complete"] for item in channels),
        "total_training_energy": total,
        "retained_training_energy": retained,
        "retained_energy_fraction": fraction,
        "energy_threshold": threshold,
        "energy_threshold_met": (
            bool(fraction >= threshold - 16 * np.finfo(float).eps) if threshold is not None else None
        ),
        "energy_metric": channels[0]["energy_metric"],
        "centered": True,
    }


class PODDecomposer(BaseDecomposer[np.ndarray]):
    """POD decomposer with optional channel-wise adapter for vector fields."""

    synthesis_kind = "affine"

    def __init__(
        self,
        *,
        cfg: Mapping[str, Any] | None = None,
        options: Any = None,
        n_modes: Any = None,
        seed: Any = None,
        mask_policy: Any = "ignore_masked_points",
        backend: Any = "sklearn",
        solver: Any = "direct",
        inner_product: Any = "euclidean",
        batch_size: Any = None,
    ) -> None:
        cfg = constructor_config(cfg, locals(), name="pod")
        super().__init__(cfg=cfg)
        self.name = "pod"
        self._impl: BaseDecomposer[np.ndarray] | None = None

    def _make_scalar(self) -> _PODScalarDecomposer:
        return _PODScalarDecomposer(cfg=dict(self._cfg))

    def _ensure_impl(
        self,
        *,
        dataset: Any | None = None,
        field: np.ndarray | None = None,
    ) -> BaseDecomposer[np.ndarray]:
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
                decomposer_factory=_PODScalarDecomposer,
                label=self.name,
            )
        else:
            self._impl = self._make_scalar()
        return self._impl

    def fit(self, dataset: Any | None = None, domain_spec: DomainSpec | None = None) -> "PODDecomposer":
        if dataset is None:
            raise ValueError("pod requires dataset for fit")
        impl = self._ensure_impl(dataset=dataset)
        self._component_blocks = True
        if isinstance(impl, ChannelwiseAdapter):
            impl._variable_blocks = True
        impl.fit(dataset=dataset, domain_spec=domain_spec)
        if isinstance(impl, ChannelwiseAdapter):
            scalar_decomposers: list[_PODScalarDecomposer] = []
            for dec in impl._decomposers:
                if not isinstance(dec, _PODScalarDecomposer):
                    raise TypeError("pod channelwise adapter must contain scalar POD decomposers")
                scalar_decomposers.append(dec)
            for dec in scalar_decomposers:
                dec._discard_candidate_basis()
        elif isinstance(impl, _PODScalarDecomposer):
            impl._discard_candidate_basis()
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
            diagnostics = [dict(dec.coeff_meta()["diagnostics"]) for dec in self._impl._decomposers]
            meta["diagnostics"] = _pod_channel_diagnostics(diagnostics)
            meta["n_modes"] = sum(item["selected_modes"] for item in diagnostics)
            meta["mean_centered"] = True
            meta["inner_product"] = diagnostics[0]["energy_metric"]
        self._coeff_meta = meta
        if getattr(self, "_component_blocks", False):
            counts = meta.get("component_modes", [coeff.size])
            self._coeff_meta = component_blocks_meta(meta, counts)
            return np.asarray(coeff).reshape(-1)
        return coeff

    def inverse_transform(self, coeff: np.ndarray, domain_spec: DomainSpec) -> np.ndarray:
        if self._impl is None:
            raise ValueError("pod transform must be called before inverse_transform")
        return self._impl.inverse_transform(coeff, domain_spec=domain_spec)


__all__ = ["PODDecomposer"]
