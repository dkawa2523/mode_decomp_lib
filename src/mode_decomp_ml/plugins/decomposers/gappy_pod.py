"""Gappy POD decomposer for masked/partial observations."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from mode_decomp_ml.config import cfg_get

from mode_decomp_ml.domain import DomainSpec, validate_decomposer_compatibility
from mode_decomp_ml.plugins.registry import register_decomposer

from .base import BaseDecomposer, ChannelwiseAdapter, parse_bool
from .pod import _PODScalarDecomposer
from .base import _combine_masks, _normalize_field, _normalize_mask

_MASK_POLICIES = {"require"}



def parse_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
    raise ValueError(f"Invalid boolean value: {value}")


def _resolve_mode_weight(cfg: Mapping[str, Any]) -> dict[str, Any]:
    options = cfg_get(cfg, "options", None)
    mode_cfg = {}
    if isinstance(options, Mapping):
        mode_cfg = options.get("mode_weight") or {}
    if not isinstance(mode_cfg, Mapping):
        mode_cfg = {}
    enable = parse_bool(mode_cfg.get("enable", False), default=False)
    method = str(mode_cfg.get("method", "eigval_scale")).strip().lower() or "eigval_scale"
    if method != "eigval_scale":
        raise ValueError("options.mode_weight.method must be eigval_scale for gappy_pod")
    return {"enable": enable, "method": method}


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


class _GappyPODScalarDecomposer(BaseDecomposer):
    """Scalar Gappy POD decomposer (no channel branching)."""

    def __init__(self, *, cfg: Mapping[str, Any]) -> None:
        super().__init__(cfg=cfg)
        self.name = "gappy_pod"
        self._mask_policy = str(cfg_get(cfg, "mask_policy", "require")).strip() or "require"
        if self._mask_policy not in _MASK_POLICIES:
            raise ValueError(f"decompose.mask_policy must be one of {_MASK_POLICIES}, got {self._mask_policy}")
        self._reg_lambda = float(cfg_get(cfg, "reg_lambda", 1e-6))
        if self._reg_lambda < 0:
            raise ValueError("decompose.reg_lambda must be >= 0 for gappy_pod")
        self._mode_weight = _resolve_mode_weight(cfg)
        base_cfg = dict(cfg)
        base_cfg["mask_policy"] = "error"
        self._pod = _PODScalarDecomposer(cfg=base_cfg)

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
        # CONTRACT: gappy_pod basis fit expects complete fields (no masks).
        for idx in range(len(dataset)):
            sample = dataset[idx]
            if sample.mask is not None:
                raise ValueError("gappy_pod fit requires samples without masks")
        self._pod.fit(dataset=dataset, domain_spec=domain_spec)
        self.mean = self._pod.mean
        self.modes = self._pod.modes
        self.eigvals = self._pod.eigvals
        self._grid_shape = self._pod._grid_shape
        self._coeff_shape = self._pod._coeff_shape
        self._field_ndim = self._pod._field_ndim
        return self

    def _resolve_mask(self, mask: np.ndarray | None, domain_spec: DomainSpec) -> np.ndarray:
        combined = _combine_masks(mask, domain_spec.mask)
        if combined is None:
            raise ValueError("gappy_pod requires a mask for transform")
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
        y_obs = vec[obs_idx] - mean[obs_idx]
        if weights is not None:
            w_flat = weights.reshape(-1, order="C")[obs_idx]
            if not np.isfinite(w_flat).all():
                raise ValueError("gappy_pod weights must be finite")
            sqrt_w = np.sqrt(w_flat)
            phi_obs = phi_obs * sqrt_w[:, None]
            y_obs = y_obs * sqrt_w
        if self._reg_lambda > 0:
            lhs = phi_obs.T @ phi_obs + self._reg_lambda * np.eye(phi_obs.shape[1])
            rhs = phi_obs.T @ y_obs
            coeff = np.linalg.solve(lhs, rhs)
        else:
            coeff, _, _, _ = np.linalg.lstsq(phi_obs, y_obs, rcond=None)
        return coeff

    def transform(self, field: np.ndarray, mask: np.ndarray | None, domain_spec: DomainSpec) -> np.ndarray:
        validate_decomposer_compatibility(domain_spec, self._cfg)
        if self.modes is None or self.mean is None or self._grid_shape is None:
            raise ValueError("gappy_pod fit must be called before transform")
        field_3d, was_2d = _normalize_field(field)
        if field_3d.shape[:2] != self._grid_shape:
            raise ValueError("gappy_pod domain grid does not match fit")
        if field_3d.shape[-1] != 1:
            raise ValueError("gappy_pod expects scalar fields; use channelwise adapter for vectors")
        field_mask = _normalize_mask(mask, field_3d.shape[:2])
        combined = self._resolve_mask(field_mask, domain_spec)

        vec = field_3d[..., 0].reshape(-1, order="C")
        weights = None
        if str(cfg_get(self._cfg, "inner_product", "euclidean")) == "domain_weights":
            weights = domain_spec.integration_weights()
            if weights is None:
                raise ValueError("gappy_pod inner_product=domain_weights requires integration_weights")
            if weights.shape != domain_spec.grid_shape:
                raise ValueError("gappy_pod weights shape does not match domain")
        coeff_vec = self._solve_coeffs(vec, combined, weights)
        self._mode_weight_scale = None
        if self._mode_weight["enable"]:
            coeff_vec, self._mode_weight_scale = _apply_mode_weight(coeff_vec, self.eigvals)
        coeff_tensor = coeff_vec[None, :]
        self._coeff_shape = coeff_tensor.shape
        self._field_ndim = 2 if was_2d else 3
        valid_count = int(np.count_nonzero(combined))
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
                "projection": "gappy_pod",
                "mask_policy": self._mask_policy,
                "mask_valid_count": valid_count,
                "reg_lambda": float(self._reg_lambda),
                "mode_weight": dict(self._mode_weight),
                "eigvals": [float(val) for val in self.eigvals] if self.eigvals is not None else None,
            }
        )
        self._coeff_meta = meta
        return coeff_tensor

    def inverse_transform(self, coeff: np.ndarray, domain_spec: DomainSpec) -> np.ndarray:
        validate_decomposer_compatibility(domain_spec, self._cfg)
        if self.modes is None or self.mean is None or self._grid_shape is None or self._field_ndim is None:
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


@register_decomposer("gappy_pod")
class GappyPODDecomposer(BaseDecomposer):
    """Gappy POD decomposer with optional channel-wise adapter for vector fields."""

    def __init__(self, *, cfg: Mapping[str, Any]) -> None:
        super().__init__(cfg=cfg)
        self.name = "gappy_pod"
        self._impl: BaseDecomposer | None = None

    def _make_scalar(self) -> _GappyPODScalarDecomposer:
        return _GappyPODScalarDecomposer(cfg=dict(self._cfg))

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
                decomposer_factory=lambda cfg: _GappyPODScalarDecomposer(cfg=cfg),
                label=self.name,
            )
        else:
            self._impl = self._make_scalar()
        return self._impl

    def fit(self, dataset: Any | None = None, domain_spec: DomainSpec | None = None) -> "GappyPODDecomposer":
        if dataset is None:
            raise ValueError("gappy_pod requires dataset for fit")
        impl = self._ensure_impl(dataset=dataset)
        impl.fit(dataset=dataset, domain_spec=domain_spec)
        return self

    def transform(self, field: np.ndarray, mask: np.ndarray | None, domain_spec: DomainSpec) -> np.ndarray:
        if self._impl is None:
            raise ValueError("gappy_pod fit must be called before transform")
        coeff = self._impl.transform(field, mask=mask, domain_spec=domain_spec)
        self._coeff_meta = dict(self._impl.coeff_meta())
        return coeff

    def inverse_transform(self, coeff: np.ndarray, domain_spec: DomainSpec) -> np.ndarray:
        if self._impl is None:
            raise ValueError("gappy_pod transform must be called before inverse_transform")
        return self._impl.inverse_transform(coeff, domain_spec=domain_spec)


__all__ = ["GappyPODDecomposer"]
