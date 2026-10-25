"""Gappy POD decomposer for masked/partial observations."""

from __future__ import annotations

from ml_platform_mode_decomp.config import constructor_config

from typing import Any, Mapping

import numpy as np
from ml_platform_mode_decomp.config import cfg_get

from ml_platform_mode_decomp.geometry import DomainSpec, validate_decomposer_compatibility

from .base import BaseDecomposer, ChannelwiseAdapter, FieldBatch, field_arrays
from .pod import _PODScalarDecomposer, _pod_channel_diagnostics
from .pod_utils import solve_ridge, component_blocks_meta, resolve_mode_weight as _resolve_mode_weight
from .base import _combine_masks, _normalize_field

_MASK_POLICIES = {"require"}


def _apply_mode_weight(coeff_vec: np.ndarray, eigvals: np.ndarray | None) -> tuple[np.ndarray, np.ndarray | None]:
    if eigvals is None:
        raise ValueError("gappy_pod mode_weight requires eigvals")
    scale = np.sqrt(np.asarray(eigvals, dtype=np.float64) + 1e-12)
    return coeff_vec * scale, scale


def _remove_mode_weight(coeff_vec: np.ndarray, scale: np.ndarray | None, eigvals: np.ndarray | None) -> np.ndarray:
    if scale is None:
        if eigvals is None:
            raise ValueError("gappy_pod mode_weight requires eigvals")
        scale = np.sqrt(np.asarray(eigvals, dtype=np.float64) + 1e-12)
    return coeff_vec / scale


class _GappyPODScalarDecomposer(BaseDecomposer[np.ndarray]):
    """Scalar Gappy POD decomposer (no channel branching)."""

    def __init__(self, *, cfg: Mapping[str, Any]) -> None:
        super().__init__(cfg=cfg)
        self.name = "gappy_pod"
        self._mask_policy = str(cfg["mask_policy"]).strip() or "require"
        if self._mask_policy not in _MASK_POLICIES:
            raise ValueError(f"decompose.mask_policy must be one of {_MASK_POLICIES}, got {self._mask_policy}")
        self._reg_lambda = float(cfg["reg_lambda"])
        if not np.isfinite(self._reg_lambda) or self._reg_lambda < 0:
            raise ValueError("decompose.reg_lambda must be >= 0 for gappy_pod")
        self._mode_weight = _resolve_mode_weight(cfg)
        self._pod: _PODScalarDecomposer | None = None  # read compatibility for old bundles only

        self.mean: np.ndarray | None = None
        self.modes: np.ndarray | None = None
        self.eigvals: np.ndarray | None = None
        self._grid_shape: tuple[int, int] | None = None
        self._coeff_shape: tuple[int, ...] | None = None
        self._field_ndim: int | None = None
        self._mode_weight_scale: np.ndarray | None = None

    def fit(self, dataset: Any | None = None, domain_spec: DomainSpec | None = None) -> "_GappyPODScalarDecomposer":
        if dataset is None:
            raise ValueError("gappy_pod requires dataset for fit")
        if domain_spec is None:
            raise ValueError("gappy_pod requires domain_spec for fit")
        # Basis fit expects complete fields. None and an explicit all-observed
        # mask have the same meaning, including for the shared FieldDataset API.
        fields, masks = field_arrays(dataset, domain_spec, scalar=True)
        domain = np.ones(domain_spec.grid_shape, dtype=bool) if domain_spec.mask is None else domain_spec.mask
        if masks is not None and not np.broadcast_to(masks, fields.shape)[:, domain].all():
            raise ValueError("gappy_pod basis fit requires complete observations inside the fixed domain")
        base_cfg = {**self._cfg, "mask_policy": "ignore_masked_points"}
        pod = _PODScalarDecomposer(cfg=base_cfg).fit(dataset=FieldBatch(fields), domain_spec=domain_spec)
        self._sync_pod_basis(pod)
        self._diagnostics = dict(pod._diagnostics)
        return self

    def _sync_pod_basis(self, pod: _PODScalarDecomposer) -> None:
        self.mean = pod.mean
        self.modes = pod.modes
        if self.modes is not None and pod._sqrt_weights is not None:
            # sklearn POD stores an orthonormal basis in sqrt(W)-scaled
            # coordinates; gappy least squares synthesizes physical fields.
            sqrt_weights = pod._sqrt_weights
            physical = np.zeros_like(self.modes)
            positive = sqrt_weights > 0
            physical[positive] = self.modes[positive] / sqrt_weights[positive, None]
            self.modes = physical
        elif self.modes is not None:
            self.modes = self.modes.copy()
        if pod._mask_indices is not None:
            # POD learns on the fixed domain only. Gappy queries index physical
            # grid locations, so keep one full-grid mean/basis with zero outside.
            assert self.mean is not None and self.modes is not None and pod._grid_shape is not None
            mean = np.zeros(int(np.prod(pod._grid_shape)), dtype=np.float64)
            modes = np.zeros((len(mean), self.modes.shape[1]), dtype=np.float64)
            mean[pod._mask_indices] = self.mean
            modes[pod._mask_indices] = self.modes
            self.mean, self.modes = mean, modes
        self.eigvals = pod.eigvals
        self._grid_shape = pod._grid_shape
        self._coeff_shape = pod._coeff_shape
        self._field_ndim = pod._field_ndim

    def _resolve_mask(self, mask: np.ndarray | None, domain_spec: DomainSpec) -> np.ndarray:
        combined = _combine_masks(mask, domain_spec.mask)
        if combined is None:
            return np.ones(domain_spec.grid_shape, dtype=bool)
        return combined.astype(bool)

    def _solve_coeffs(
        self,
        vec: np.ndarray,
        mask: np.ndarray,
        weights: np.ndarray | None,
    ) -> np.ndarray:
        if self.modes is None or self.mean is None:
            raise ValueError("gappy_pod fit must be called before transform")
        mask_flat = mask.reshape(-1, order="C")
        obs_idx = np.flatnonzero(mask_flat)
        if obs_idx.size == 0:
            raise ValueError("gappy_pod mask has no observed entries")
        phi = self.modes
        mean = self.mean
        phi_obs = phi[obs_idx, :]
        y_obs = vec[obs_idx] - mean[obs_idx].reshape((-1,) + (1,) * (vec.ndim - 1))
        if not np.isfinite(y_obs).all():
            raise ValueError("gappy_pod requires finite observed field values")
        if weights is not None:
            w_flat = weights.reshape(-1, order="C")[obs_idx]
            if not np.isfinite(w_flat).all() or np.any(w_flat < 0) or not np.any(w_flat > 0):
                raise ValueError("gappy_pod observation weights must be finite, nonnegative and nonempty")
            sqrt_w = np.sqrt(w_flat)
            phi_obs = phi_obs * sqrt_w[:, None]
            y_obs = y_obs * sqrt_w.reshape((-1,) + (1,) * (vec.ndim - 1))
        return solve_ridge(phi_obs, y_obs, self._reg_lambda, method="gappy_pod")

    def transform(self, field: np.ndarray, mask: np.ndarray | None, domain_spec: DomainSpec) -> np.ndarray:
        return self.transform_batch(np.asarray(field)[None, ...], mask, domain_spec)[0]

    def transform_batch(self, fields: np.ndarray, masks: np.ndarray | None, domain_spec: DomainSpec) -> np.ndarray:
        """Share one stable least-squares solve when the observed support is fixed."""
        validate_decomposer_compatibility(domain_spec, self._cfg)
        self._bind_geometry(domain_spec)
        if self.modes is None or self.mean is None or self._grid_shape is None:
            raise ValueError("gappy_pod fit must be called before transform")
        fields = np.asarray(fields)
        if fields.ndim not in {3, 4} or not len(fields) or fields.dtype.kind not in "fiu":
            raise ValueError("gappy_pod requires a nonempty real scalar field batch")
        field, was_2d = _normalize_field(fields[0])
        if field.shape[:2] != self._grid_shape or field.shape[-1] != 1:
            raise ValueError("gappy_pod requires the fitted scalar grid")
        if masks is not None:
            batch_masks: np.ndarray = np.broadcast_to(np.asarray(masks, dtype=bool), (len(fields), *self._grid_shape))
            if domain_spec.mask is not None:
                batch_masks = batch_masks & domain_spec.mask
            if any(not np.array_equal(mask, batch_masks[0]) for mask in batch_masks[1:]):
                return np.stack(
                    [self.transform(value, mask, domain_spec) for value, mask in zip(fields, batch_masks, strict=True)]
                )
            masks = batch_masks
        combined = self._resolve_mask(None if masks is None else masks[0], domain_spec)
        weights = None
        if str(cfg_get(self._cfg, "inner_product", "euclidean")) == "domain_weights":
            weights = domain_spec.integration_weights()
            if weights is None or weights.shape != domain_spec.grid_shape:
                raise ValueError("gappy_pod requires domain integration_weights matching its grid")
        vectors = fields.reshape(len(fields), -1).T
        coefficients = self._solve_coeffs(vectors, combined, weights).T
        self._mode_weight_scale = None
        if self._mode_weight["enable"]:
            coefficients, self._mode_weight_scale = _apply_mode_weight(coefficients, self.eigvals)
        self._coeff_shape = (1, coefficients.shape[1])
        self._field_ndim = 2 if was_2d else 3
        meta = self._coeff_meta_base(
            field_shape=field.shape[:2] if was_2d else field.shape,
            field_ndim=self._field_ndim,
            field_layout="HW" if was_2d else "HWC",
            channels=1,
            coeff_shape=self._coeff_shape,
            coeff_layout="CK",
            complex_format="real",
        )
        meta.update(
            projection="gappy_pod",
            mask_policy=self._mask_policy,
            mask_valid_count=int(np.count_nonzero(combined)),
            reg_lambda=float(self._reg_lambda),
            mode_weight=dict(self._mode_weight),
            eigvals=None if self.eigvals is None else self.eigvals.tolist(),
            diagnostics=dict(getattr(self, "_diagnostics", getattr(self._pod, "_diagnostics", {}))),
        )
        self._coeff_meta = meta
        return coefficients[:, None, :]

    def inverse_transform(self, coeff: np.ndarray, domain_spec: DomainSpec) -> np.ndarray:
        validate_decomposer_compatibility(domain_spec, self._cfg)
        if (
            self.modes is None
            or self.mean is None
            or self._grid_shape is None
            or self._field_ndim is None
            or self._coeff_shape is None
        ):
            raise ValueError("gappy_pod transform must be called before inverse_transform")
        coeff_tensor = self._reshape_coeff(coeff, self._coeff_shape, name=self.name)
        coeff_vec = coeff_tensor[0]
        if self._mode_weight["enable"]:
            coeff_vec = _remove_mode_weight(coeff_vec, self._mode_weight_scale, self.eigvals)
        vec = coeff_vec @ self.modes.T
        vec = vec + self.mean
        height, width = self._grid_shape
        field_c = vec.reshape(height, width, order="C")
        field_hat = field_c[..., None]
        if self._field_ndim == 2 and field_hat.shape[-1] == 1:
            return field_hat[..., 0]
        return field_hat


class GappyPODDecomposer(BaseDecomposer[np.ndarray]):
    """Gappy POD decomposer with optional channel-wise adapter for vector fields."""

    synthesis_kind = "affine"

    def __init__(
        self,
        *,
        cfg: Mapping[str, Any] | None = None,
        n_modes: Any = None,
        batch_size: Any = None,
        options: Any = None,
        reg_lambda: Any = 1e-06,
        mask_policy: Any = "require",
        backend: Any = "sklearn",
        inner_product: Any = "euclidean",
        seed: Any = None,
        solver: Any = "direct",
    ) -> None:
        cfg = constructor_config(cfg, locals(), name="gappy_pod")
        super().__init__(cfg=cfg)
        self.name = "gappy_pod"
        self._impl: BaseDecomposer[np.ndarray] | None = None

    def _make_scalar(self) -> _GappyPODScalarDecomposer:
        return _GappyPODScalarDecomposer(cfg=dict(self._cfg))

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
                raise ValueError("gappy_pod requires non-empty dataset for fit")
            sample = dataset[0]
            field_arr = np.asarray(sample.field)
        elif field is not None:
            field_arr = np.asarray(field)
        else:
            raise ValueError("gappy_pod requires dataset or field to infer channel layout")
        if field_arr.ndim not in {2, 3}:
            raise ValueError(f"gappy_pod expects 2D or 3D field, got shape {field_arr.shape}")
        if field_arr.ndim == 3 and field_arr.shape[-1] > 1:
            self._impl = ChannelwiseAdapter(
                cfg=dict(self._cfg),
                decomposer_factory=_GappyPODScalarDecomposer,
                label=self.name,
            )
        else:
            self._impl = self._make_scalar()
        return self._impl

    def fit(self, dataset: Any | None = None, domain_spec: DomainSpec | None = None) -> "GappyPODDecomposer":
        if dataset is None:
            raise ValueError("gappy_pod requires dataset for fit")
        impl = self._ensure_impl(dataset=dataset)
        self._component_blocks = True
        if isinstance(impl, ChannelwiseAdapter):
            impl._variable_blocks = True
        impl.fit(dataset=dataset, domain_spec=domain_spec)
        return self

    def transform(self, field: np.ndarray, mask: np.ndarray | None, domain_spec: DomainSpec) -> np.ndarray:
        if self._impl is None:
            raise ValueError("gappy_pod fit must be called before transform")
        coeff = self._impl.transform(field, mask=mask, domain_spec=domain_spec)
        self._coeff_meta = dict(self._impl.coeff_meta())
        if isinstance(self._impl, ChannelwiseAdapter):
            channels = [dict(dec.coeff_meta()["diagnostics"]) for dec in self._impl._decomposers]
            self._coeff_meta["diagnostics"] = _pod_channel_diagnostics(channels)
        if getattr(self, "_component_blocks", False):
            counts = self._coeff_meta.get("component_modes", [coeff.size])
            self._coeff_meta = component_blocks_meta(self._coeff_meta, counts)
            return np.asarray(coeff).reshape(-1)
        return coeff

    def transform_batch(self, fields: np.ndarray, masks: np.ndarray | None, domain_spec: DomainSpec) -> np.ndarray:
        if not getattr(self, "_component_blocks", False):
            return np.asarray(super().transform_batch(fields, masks, domain_spec))
        self._bind_geometry(domain_spec)
        if self._impl is None:
            raise ValueError("gappy_pod fit must be called before transform")
        fields = np.asarray(fields)
        if isinstance(self._impl, ChannelwiseAdapter):
            if fields.ndim != 4 or not len(fields) or fields.shape[-1] != self._impl._channels:
                raise ValueError("Gappy batch channels must match the fitted representation")
            parts = [
                decomposer.transform_batch(fields[..., channel], masks, domain_spec)[:, 0, :]
                for channel, decomposer in enumerate(self._impl._decomposers)
            ]
            counts = [part.shape[1] for part in parts]
            coefficients = np.concatenate(parts, axis=1)
            self._impl._coeff_shape = coefficients.shape[1:]
            self._impl._field_ndim = 3
            self._impl._channel_shapes = [(count,) for count in counts]
            meta = self._impl._coeff_meta_base(
                field_shape=fields.shape[1:],
                field_ndim=3,
                field_layout="HWC",
                channels=fields.shape[-1],
                coeff_shape=coefficients.shape[1:],
                coeff_layout="D",
                complex_format="real",
            )
            meta["base_method"] = "gappy_pod"
            self._impl._coeff_meta = component_blocks_meta(meta, counts)
            diagnostics = [decomposer.coeff_meta()["diagnostics"] for decomposer in self._impl._decomposers]
            meta["diagnostics"] = _pod_channel_diagnostics(diagnostics)
        else:
            raw = np.asarray(self._impl.transform_batch(fields, masks, domain_spec))
            coefficients = raw.reshape(len(raw), raw.shape[-1])
            counts = [coefficients.shape[1]]
            meta = dict(self._impl.coeff_meta())
        self._coeff_meta = component_blocks_meta(meta, counts)
        return coefficients

    def inverse_transform(self, coeff: np.ndarray, domain_spec: DomainSpec) -> np.ndarray:
        if self._impl is None:
            raise ValueError("gappy_pod transform must be called before inverse_transform")
        return self._impl.inverse_transform(coeff, domain_spec=domain_spec)


__all__ = ["GappyPODDecomposer"]
