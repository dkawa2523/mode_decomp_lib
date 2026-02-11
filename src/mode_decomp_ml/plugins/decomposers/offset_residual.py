"""Offset/residual split wrapper decomposer.

This wrapper optionally splits each sample field into:
  - offset: per-channel weighted mean on the valid mask
  - residual: field - offset (outside the valid mask is zeroed)

If enabled, residual is decomposed by an inner decomposer, while offset is
packed alongside residual coefficients via a specialized codec.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from mode_decomp_ml.config import cfg_get
from mode_decomp_ml.data import FieldSample
from mode_decomp_ml.domain import DomainSpec
from mode_decomp_ml.plugins.registry import build_decomposer, register_decomposer

from .base import BaseDecomposer


_ENABLED_OPTIONS = {"auto", "true", "false"}


def _normalize_field(field: np.ndarray) -> tuple[np.ndarray, bool]:
    arr = np.asarray(field)
    if arr.ndim == 2:
        return arr[..., None], True
    if arr.ndim == 3:
        return arr, False
    raise ValueError(f"field must be 2D or 3D, got shape {arr.shape}")


def _normalize_spatial_mask(mask: Any, spatial_shape: tuple[int, int], *, label: str) -> np.ndarray:
    m = np.asarray(mask).astype(bool)
    if m.shape == spatial_shape:
        return m
    # mesh domain mask/weights can be 1D with W=1 spatial shape.
    if m.ndim == 1 and spatial_shape[1] == 1 and m.shape[0] == spatial_shape[0]:
        return m[:, None]
    raise ValueError(f"{label} shape {m.shape} does not match field spatial shape {spatial_shape}")


def _normalize_weights(weights: Any, spatial_shape: tuple[int, int]) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64)
    if w.shape == spatial_shape:
        return w
    if w.ndim == 1 and spatial_shape[1] == 1 and w.shape[0] == spatial_shape[0]:
        return w[:, None]
    raise ValueError(f"weights shape {w.shape} does not match field spatial shape {spatial_shape}")


@dataclass(frozen=True)
class _OffsetResidualResult:
    offset: np.ndarray  # (C,)
    residual: np.ndarray  # same ndim as input field
    valid_mask: np.ndarray  # (H,W) boolean
    offset_rms: float
    residual_rms: float
    ratio: float


def _compute_offset_residual(
    field: np.ndarray,
    mask: np.ndarray | None,
    *,
    domain_spec: DomainSpec,
    min_residual_rms: float,
) -> _OffsetResidualResult:
    field_3d, was_2d = _normalize_field(field)
    spatial_shape = (int(field_3d.shape[0]), int(field_3d.shape[1]))

    valid: np.ndarray | None = None
    if domain_spec.mask is not None:
        valid = _normalize_spatial_mask(domain_spec.mask, spatial_shape, label="domain mask")
    if mask is not None:
        valid_field = _normalize_spatial_mask(mask, spatial_shape, label="field mask")
        valid = valid_field if valid is None else (valid & valid_field)
    if valid is None:
        valid = np.ones(spatial_shape, dtype=bool)
    if not np.any(valid):
        raise ValueError("offset_residual has no valid mask entries")

    weights = domain_spec.integration_weights()
    if weights is None:
        w = np.ones(spatial_shape, dtype=np.float64)
    else:
        w = _normalize_weights(weights, spatial_shape)
    wm = w * valid.astype(np.float64)
    denom = float(np.sum(wm))
    if denom <= 0.0:
        raise ValueError("offset_residual has zero total weight on the valid mask")

    # Weighted mean per channel on valid points.
    offset = (wm[..., None] * field_3d.astype(np.float64)).sum(axis=(0, 1)) / denom
    offset = np.asarray(offset, dtype=np.float32).reshape(-1)

    # Residual field, with invalid points zero-filled to keep downstream stable.
    residual_3d = field_3d.astype(np.float32, copy=False) - offset[None, None, :]
    if not valid.all():
        residual_3d = residual_3d.copy()
        residual_3d[~valid] = 0.0

    # RMS stats (vector magnitude for multi-channel).
    offset_rms = float(np.sqrt(np.sum(offset.astype(np.float64) ** 2)))
    res_mag2 = np.sum(residual_3d.astype(np.float64) ** 2, axis=-1)
    residual_rms = float(np.sqrt(np.sum(wm * res_mag2) / denom))
    denom_rms = max(float(min_residual_rms), 0.0)
    ratio = offset_rms / max(residual_rms, denom_rms)

    residual_out: np.ndarray
    if was_2d and residual_3d.shape[-1] == 1:
        residual_out = residual_3d[..., 0]
    else:
        residual_out = residual_3d
    return _OffsetResidualResult(
        offset=offset,
        residual=residual_out,
        valid_mask=valid,
        offset_rms=offset_rms,
        residual_rms=residual_rms,
        ratio=ratio,
    )


class _ResidualDataset:
    def __init__(
        self,
        dataset: Any,
        *,
        domain_spec: DomainSpec,
        min_residual_rms: float,
    ) -> None:
        self._dataset = dataset
        self._domain_spec = domain_spec
        self._min_residual_rms = float(min_residual_rms)
        self.name = getattr(dataset, "name", "dataset")

    def __len__(self) -> int:
        return int(len(self._dataset))

    def __getitem__(self, index: int) -> FieldSample:
        sample = self._dataset[int(index)]
        if not isinstance(sample, FieldSample):
            # Allow duck-typed samples.
            cond = np.asarray(getattr(sample, "cond"))
            field = np.asarray(getattr(sample, "field"))
            mask = getattr(sample, "mask", None)
            meta = dict(getattr(sample, "meta", {}) or {})
        else:
            cond = np.asarray(sample.cond)
            field = np.asarray(sample.field)
            mask = sample.mask
            meta = dict(sample.meta)
        out = _compute_offset_residual(
            field,
            None if mask is None else np.asarray(mask),
            domain_spec=self._domain_spec,
            min_residual_rms=self._min_residual_rms,
        )
        return FieldSample(cond=cond, field=out.residual, mask=mask, meta=meta)


def _parse_enabled(value: Any) -> str:
    if value is None:
        return "auto"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, np.integer)):
        return "true" if bool(value) else "false"
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in _ENABLED_OPTIONS:
            return lowered
    raise ValueError(f"offset_residual.enabled must be one of {_ENABLED_OPTIONS}, got {value}")


@register_decomposer("offset_residual")
class OffsetResidualDecomposer(BaseDecomposer):
    """Split (offset,residual) and delegate residual decomposition to an inner decomposer."""

    def __init__(self, *, cfg: Mapping[str, Any]) -> None:
        super().__init__(cfg=cfg)
        self.name = "offset_residual"
        self._enabled_mode = _parse_enabled(cfg_get(cfg, "enabled", "auto"))
        self._f_offset = float(cfg_get(cfg, "f_offset", 5.0))
        if self._f_offset <= 0.0:
            raise ValueError("offset_residual.f_offset must be > 0")
        self._max_samples = int(cfg_get(cfg, "max_samples", 128))
        self._seed = cfg_get(cfg, "seed", None)
        self._min_residual_rms = float(cfg_get(cfg, "min_residual_rms", 1e-8))
        if self._min_residual_rms < 0.0:
            raise ValueError("offset_residual.min_residual_rms must be >= 0")
        self._offset_def = str(cfg_get(cfg, "offset_def", "mean_per_channel")).strip().lower()
        if self._offset_def != "mean_per_channel":
            raise ValueError("offset_residual.offset_def must be mean_per_channel (v1)")
        self._agg = str(cfg_get(cfg, "agg", "median")).strip().lower()
        if self._agg != "median":
            raise ValueError("offset_residual.agg must be median (v1)")

        inner_cfg = cfg_get(cfg, "inner", None)
        if not isinstance(inner_cfg, Mapping):
            raise ValueError("offset_residual.inner config is required")
        self._inner_cfg = dict(inner_cfg)
        self._inner = build_decomposer(self._inner_cfg)

        self._split_enabled: bool | None = None
        self._ratio_median: float | None = None

    def _decide_split(self, dataset: Any, domain_spec: DomainSpec) -> tuple[bool, float]:
        n = int(len(dataset))
        if n <= 0:
            raise ValueError("offset_residual.fit requires a non-empty dataset")
        max_samples = max(1, min(int(self._max_samples), n))
        seed = None
        if self._seed is not None:
            try:
                seed = int(self._seed)
            except Exception:
                seed = None
        rng = np.random.default_rng(seed)
        if max_samples < n:
            idxs = rng.choice(n, size=max_samples, replace=False)
        else:
            idxs = np.arange(n, dtype=int)
        ratios: list[float] = []
        for idx in np.asarray(idxs, dtype=int).tolist():
            sample = dataset[int(idx)]
            field = np.asarray(getattr(sample, "field"))
            mask = getattr(sample, "mask", None)
            out = _compute_offset_residual(
                field,
                None if mask is None else np.asarray(mask),
                domain_spec=domain_spec,
                min_residual_rms=self._min_residual_rms,
            )
            if np.isfinite(out.ratio):
                ratios.append(float(out.ratio))
        if not ratios:
            raise ValueError("offset_residual.fit could not compute any finite ratios")
        ratio = float(np.median(np.asarray(ratios, dtype=float)))
        enabled = ratio >= float(self._f_offset)
        return enabled, ratio

    def fit(self, dataset: Any | None = None, domain_spec: DomainSpec | None = None) -> "OffsetResidualDecomposer":
        if domain_spec is None:
            raise ValueError("domain_spec is required for offset_residual.fit")
        if dataset is None:
            raise ValueError("dataset is required for offset_residual.fit")

        if self._enabled_mode == "false":
            enabled = False
            ratio = float("nan")
        elif self._enabled_mode == "true":
            enabled, ratio = self._decide_split(dataset, domain_spec)
            enabled = True  # force
        else:
            enabled, ratio = self._decide_split(dataset, domain_spec)

        self._split_enabled = bool(enabled)
        self._ratio_median = float(ratio)

        if self._split_enabled:
            residual_dataset = _ResidualDataset(
                dataset,
                domain_spec=domain_spec,
                min_residual_rms=self._min_residual_rms,
            )
            self._inner.fit(dataset=residual_dataset, domain_spec=domain_spec)
        else:
            self._inner.fit(dataset=dataset, domain_spec=domain_spec)
        return self

    def transform(self, field: np.ndarray, mask: np.ndarray | None, domain_spec: DomainSpec) -> Any:
        if self._split_enabled is None:
            raise ValueError("offset_residual.fit must be called before transform")
        if not self._split_enabled:
            raw_coeff = self._inner.transform(field, mask, domain_spec)
            meta = dict(self._inner.coeff_meta())
            meta["offset_split_enabled"] = False
            if self._ratio_median is not None:
                meta["offset_ratio_median"] = float(self._ratio_median)
                meta["f_offset"] = float(self._f_offset)
            self._coeff_meta = meta
            return raw_coeff

        out = _compute_offset_residual(
            field,
            mask,
            domain_spec=domain_spec,
            min_residual_rms=self._min_residual_rms,
        )
        raw_residual = self._inner.transform(out.residual, mask, domain_spec)
        residual_meta = dict(self._inner.coeff_meta())
        meta = dict(residual_meta)
        meta["method"] = self.name
        meta["coeff_format"] = "offset_residual_v1"
        meta["offset_dim"] = int(out.offset.size)
        meta["offset_kind"] = "mean_per_channel"
        meta["residual_raw_meta"] = residual_meta
        meta["split_enabled"] = True
        if self._ratio_median is not None:
            meta["split_ratio_median"] = float(self._ratio_median)
        meta["f_offset"] = float(self._f_offset)
        self._coeff_meta = meta
        return {"offset": out.offset, "residual": raw_residual}

    def inverse_transform(self, coeff: Any, domain_spec: DomainSpec) -> np.ndarray:
        if self._split_enabled is None:
            raise ValueError("offset_residual.fit must be called before inverse_transform")
        if not self._split_enabled:
            return self._inner.inverse_transform(coeff, domain_spec)
        if not isinstance(coeff, Mapping) or "offset" not in coeff or "residual" not in coeff:
            raise TypeError("offset_residual.inverse_transform expects a mapping with keys {'offset','residual'}")

        offset = np.asarray(coeff["offset"], dtype=np.float32).reshape(-1)
        residual_field = self._inner.inverse_transform(coeff["residual"], domain_spec)
        residual_3d, was_2d = _normalize_field(residual_field)
        if offset.size != residual_3d.shape[-1]:
            raise ValueError("offset dimension does not match residual channels")
        field_3d = residual_3d + offset[None, None, :]
        if domain_spec.mask is not None:
            mask2d = _normalize_spatial_mask(domain_spec.mask, residual_3d.shape[:2], label="domain mask")
            if not mask2d.all():
                field_3d = field_3d.copy()
                field_3d[~mask2d] = 0.0
        if was_2d and field_3d.shape[-1] == 1:
            return field_3d[..., 0]
        return field_3d


__all__ = ["OffsetResidualDecomposer"]
