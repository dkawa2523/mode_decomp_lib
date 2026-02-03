"""Shared decomposer base utilities."""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Sequence

import numpy as np

from mode_decomp_ml.config import cfg_get
from mode_decomp_ml.domain import DomainSpec


def _normalize_field(field: np.ndarray) -> tuple[np.ndarray, bool]:
    field = np.asarray(field)
    if field.ndim == 2:
        return field[..., None], True
    if field.ndim == 3:
        return field, False
    raise ValueError(f"field must be 2D or 3D, got shape {field.shape}")


def _normalize_mask(mask: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray | None:
    if mask is None:
        return None
    mask = np.asarray(mask)
    if mask.shape != shape:
        raise ValueError(f"mask shape {mask.shape} does not match {shape}")
    return mask.astype(bool)


def _combine_masks(
    field_mask: np.ndarray | None,
    domain_mask: np.ndarray | None,
) -> np.ndarray | None:
    if field_mask is None and domain_mask is None:
        return None
    if field_mask is None:
        return domain_mask
    if domain_mask is None:
        return field_mask
    return field_mask & domain_mask


def require_cfg(cfg: Mapping[str, Any], key: str, *, label: str) -> Any:
    value = cfg_get(cfg, key, None)
    if value is None:
        raise ValueError(f"{label}.{key} is required")
    if isinstance(value, str) and not value.strip():
        raise ValueError(f"{label}.{key} must be non-empty")
    return value


def require_cfg_stripped(cfg: Mapping[str, Any], key: str, *, label: str) -> str:
    value = require_cfg(cfg, key, label=label)
    return str(value).strip()


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


class BaseDecomposer:
    """Minimal decomposer interface."""

    name: str

    def __init__(self, *, cfg: Mapping[str, Any]) -> None:
        self._cfg = dict(cfg)
        self._coeff_meta: Dict[str, Any] | None = None

    def fit(self, dataset: Any | None = None, domain_spec: DomainSpec | None = None) -> "BaseDecomposer":
        return self

    def coeff_meta(self) -> Mapping[str, Any]:
        if self._coeff_meta is None:
            raise ValueError("coeff_meta is not available before transform")
        return self._coeff_meta

    def save_coeff_meta(self, run_dir: str | Path) -> Path:
        meta = self.coeff_meta()
        out_dir = Path(run_dir) / "states" / "decomposer"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "coeff_meta.json"
        # CONTRACT: coeff_meta must be persisted for comparability.
        with path.open("w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)
        return path

    def save_state(self, run_dir: str | Path) -> Path:
        out_dir = Path(run_dir) / "states" / "decomposer"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "state.pkl"
        with path.open("wb") as fh:
            pickle.dump(self, fh)
        return path

    @staticmethod
    def load_state(path: str | Path) -> "BaseDecomposer":
        obj = load_pickle_compat(path)
        if not isinstance(obj, BaseDecomposer):
            raise TypeError("Loaded decomposer state is not a BaseDecomposer")
        return obj

    def _prepare_field(
        self,
        field: np.ndarray,
        mask: np.ndarray | None,
        domain_spec: DomainSpec,
        *,
        allow_zero_fill: bool,
    ) -> tuple[np.ndarray, bool]:
        if domain_spec is None:
            raise ValueError("domain_spec is required")
        field_3d, was_2d = _normalize_field(field)
        if field_3d.shape[:2] != domain_spec.grid_shape:
            raise ValueError(
                f"field shape {field_3d.shape[:2]} does not match domain {domain_spec.grid_shape}"
            )
        field_mask = _normalize_mask(mask, field_3d.shape[:2])
        combined = _combine_masks(field_mask, domain_spec.mask)
        if combined is not None and not combined.all():
            if not allow_zero_fill:
                raise ValueError("mask has invalid entries for FFT/DCT without mask_zero_fill policy")
            # CONTRACT: only zero-fill when policy explicitly allows it.
            field_3d = field_3d.copy()
            field_3d[~combined] = 0.0
        return field_3d, was_2d

    @staticmethod
    def _reshape_coeff(coeff: np.ndarray, shape: Sequence[int], *, name: str | None = None) -> np.ndarray:
        coeff = np.asarray(coeff).reshape(-1)
        expected = int(np.prod(shape))
        if coeff.size > expected:
            label = name or "coeff"
            raise ValueError(f"{label} size {coeff.size} exceeds expected {expected}")
        if coeff.size < expected:
            padded = np.zeros(expected, dtype=coeff.dtype)
            padded[: coeff.size] = coeff
            coeff = padded
        return coeff.reshape(tuple(int(x) for x in shape), order="C")

    def _coeff_meta_base(
        self,
        *,
        field_shape: Sequence[int],
        field_ndim: int,
        field_layout: str,
        channels: int,
        coeff_shape: Sequence[int],
        coeff_layout: str,
        complex_format: str,
        flatten_order: str = "C",
    ) -> Dict[str, Any]:
        return {
            "method": self.name,
            "field_shape": [int(x) for x in field_shape],
            "field_ndim": int(field_ndim),
            "field_layout": str(field_layout),
            "channels": int(channels),
            "coeff_shape": [int(x) for x in coeff_shape],
            "coeff_layout": str(coeff_layout),
            "flatten_order": str(flatten_order),
            "complex_format": str(complex_format),
            "keep": "all",
        }


class GridDecomposerBase(BaseDecomposer):
    """Baseclass for grid decomposers with per-channel transforms."""

    def __init__(self, *, cfg: Mapping[str, Any]) -> None:
        super().__init__(cfg=cfg)
        self._coeff_shape: tuple[int, ...] | None = None
        self._field_ndim: int | None = None

    def _grid_transform(
        self,
        field: np.ndarray,
        mask: np.ndarray | None,
        domain_spec: DomainSpec,
        *,
        allow_zero_fill: bool,
        forward_fn: Callable[[np.ndarray], np.ndarray],
        coeff_layout: str,
        complex_format: str,
        extra_meta: Mapping[str, Any] | None = None,
    ) -> np.ndarray:
        from mode_decomp_ml.domain import validate_decomposer_compatibility

        validate_decomposer_compatibility(domain_spec, self._cfg)
        field_3d, was_2d = self._prepare_field(
            field, mask, domain_spec, allow_zero_fill=allow_zero_fill
        )
        coeffs = [forward_fn(field_3d[..., ch]) for ch in range(field_3d.shape[-1])]
        coeff_tensor = np.stack(coeffs, axis=0)
        self._coeff_shape = coeff_tensor.shape
        self._field_ndim = 2 if was_2d else 3
        meta = self._coeff_meta_base(
            field_shape=field_3d.shape[:2] if was_2d else field_3d.shape,
            field_ndim=self._field_ndim,
            field_layout="HW" if was_2d else "HWC",
            channels=int(field_3d.shape[-1]),
            coeff_shape=coeff_tensor.shape,
            coeff_layout=coeff_layout,
            complex_format=complex_format,
        )
        if extra_meta:
            meta.update(dict(extra_meta))
        self._coeff_meta = meta
        return coeff_tensor

    def _grid_inverse(
        self,
        coeff: np.ndarray,
        domain_spec: DomainSpec,
        *,
        inverse_fn: Callable[[np.ndarray], np.ndarray],
    ) -> np.ndarray:
        from mode_decomp_ml.domain import validate_decomposer_compatibility

        validate_decomposer_compatibility(domain_spec, self._cfg)
        if self._coeff_shape is None or self._field_ndim is None:
            raise ValueError("transform must be called before inverse_transform")
        coeff_tensor = self._reshape_coeff(coeff, self._coeff_shape, name=self.name)
        field_channels = [inverse_fn(coeff_tensor[ch]) for ch in range(coeff_tensor.shape[0])]
        field_hat = np.stack(field_channels, axis=-1)
        if self._field_ndim == 2 and field_hat.shape[-1] == 1:
            return field_hat[..., 0]
        return field_hat


class ZernikeFamilyBase(BaseDecomposer):
    """Shared Zernike-family least-squares projection helpers."""

    def __init__(self, *, cfg: Mapping[str, Any]) -> None:
        super().__init__(cfg=cfg)
        self._coeff_shape: tuple[int, ...] | None = None
        self._field_ndim: int | None = None

    def _prepare_zernike_inputs(
        self,
        field: np.ndarray,
        mask: np.ndarray | None,
        domain_spec: DomainSpec,
    ) -> tuple[np.ndarray, bool, np.ndarray, np.ndarray | None]:
        from mode_decomp_ml.domain import validate_decomposer_compatibility

        validate_decomposer_compatibility(domain_spec, self._cfg)
        field_3d, was_2d = _normalize_field(field)
        if field_3d.shape[:2] != domain_spec.grid_shape:
            raise ValueError(
                f"field shape {field_3d.shape[:2]} does not match domain {domain_spec.grid_shape}"
            )
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
            # REVIEW: masked points are ignored via zeroed weights to avoid silent fill.
            weights = np.where(combined_mask, weights, 0.0)
        return field_3d, was_2d, weights, combined_mask

    @staticmethod
    def _solve_weighted_least_squares(
        basis: np.ndarray,
        field_3d: np.ndarray,
        weights: np.ndarray,
    ) -> np.ndarray:
        weights_flat = weights.reshape(-1)
        valid = weights_flat > 0
        if not np.any(valid):
            raise ValueError("zernike weights are empty after masking")
        basis_flat = basis.reshape(basis.shape[0], -1).T
        design = basis_flat[valid]
        if design.shape[0] < design.shape[1]:
            raise ValueError("zernike basis has more modes than valid samples")
        field_flat = field_3d.reshape(-1, field_3d.shape[-1])
        if not np.isfinite(field_flat[valid]).all():
            raise ValueError("field has non-finite values within valid mask")
        sqrt_w = np.sqrt(weights_flat[valid])
        design_w = design * sqrt_w[:, None]
        field_w = field_flat[valid] * sqrt_w[:, None]
        coeffs, _, rank, _ = np.linalg.lstsq(design_w, field_w, rcond=None)
        if rank < coeffs.shape[0]:
            raise ValueError("zernike basis is rank-deficient; reduce n_max or check mask coverage")
        return coeffs.T

    @staticmethod
    def _reconstruct_from_basis(coeff_tensor: np.ndarray, basis: np.ndarray) -> np.ndarray:
        field_channels = []
        for idx in range(coeff_tensor.shape[0]):
            field_c = np.tensordot(coeff_tensor[idx], basis, axes=(0, 0))
            field_channels.append(field_c)
        return np.stack(field_channels, axis=-1)


class EigenBasisDecomposerBase(BaseDecomposer):
    """Baseclass for eigenbasis decomposers with sign-fix handling."""

    def __init__(self, *, cfg: Mapping[str, Any]) -> None:
        super().__init__(cfg=cfg)
        self._basis: np.ndarray | None = None
        self._eigenvalues: np.ndarray | None = None
        self._sign_fix_rule = "max_abs_component"

    def _fix_eigenvector_signs(self, eigvecs: np.ndarray) -> np.ndarray:
        fixed = np.asarray(eigvecs, dtype=np.float64).copy()
        if fixed.ndim != 2:
            raise ValueError("eigenvectors must be 2D (nodes x modes)")
        for idx in range(fixed.shape[1]):
            vec = fixed[:, idx]
            if vec.size == 0:
                continue
            anchor = int(np.argmax(np.abs(vec)))
            if vec[anchor] < 0:
                fixed[:, idx] = -vec
        return fixed

    def _set_basis(self, eigvecs: np.ndarray, eigvals: np.ndarray) -> None:
        self._basis = self._fix_eigenvector_signs(eigvecs)
        self._eigenvalues = np.asarray(eigvals, dtype=np.float64)

    def _eigen_meta(self, *, projection: str, mode_order: str = "ascending_eigenvalue") -> Dict[str, Any]:
        if self._eigenvalues is None:
            raise ValueError("eigenvalues are not available")
        return {
            "mode_order": mode_order,
            "eigenvalues": [float(val) for val in self._eigenvalues],
            "sign_fix_rule": self._sign_fix_rule,
            "projection": str(projection),
        }


class ChannelwiseAdapter(BaseDecomposer):
    """Apply a scalar decomposer channel-by-channel for vector fields."""

    def __init__(
        self,
        *,
        cfg: Mapping[str, Any],
        decomposer_factory: Callable[[Mapping[str, Any]], BaseDecomposer],
        label: str | None = None,
    ) -> None:
        super().__init__(cfg=cfg)
        self.name = label or "channelwise"
        self._factory = decomposer_factory
        self._decomposers: list[BaseDecomposer] = []
        self._channels: int | None = None
        self._coeff_shape: tuple[int, ...] | None = None
        self._field_ndim: int | None = None
        self._channel_shapes: list[tuple[int, ...]] | None = None

    def _ensure_decomposers(self, channels: int) -> None:
        if self._decomposers:
            if len(self._decomposers) != channels:
                raise ValueError("channelwise adapter channel count mismatch")
            return
        self._decomposers = [self._factory(dict(self._cfg)) for _ in range(channels)]
        self._channels = channels

    def fit(self, dataset: Any | None = None, domain_spec: DomainSpec | None = None) -> "ChannelwiseAdapter":
        if dataset is None:
            return self
        sample = dataset[0]
        field = np.asarray(sample.field)
        if field.ndim != 3:
            raise ValueError("channelwise adapter expects 3D fields in dataset")
        channels = int(field.shape[-1])
        self._ensure_decomposers(channels)
        from mode_decomp_ml.data.datasets import FieldSample

        class _ChannelDataset:
            def __init__(self, dataset: Any, channel: int) -> None:
                self._dataset = dataset
                self._channel = channel

            def __len__(self) -> int:
                return len(self._dataset)

            def __getitem__(self, index: int) -> FieldSample:
                sample = self._dataset[index]
                field = np.asarray(sample.field)
                if field.ndim != 3:
                    raise ValueError("channelwise adapter expects 3D fields in dataset")
                field_ch = field[..., self._channel]
                return FieldSample(
                    cond=sample.cond,
                    field=field_ch,
                    mask=sample.mask,
                    meta=dict(sample.meta),
                )

        for idx, decomposer in enumerate(self._decomposers):
            decomposer.fit(dataset=_ChannelDataset(dataset, idx), domain_spec=domain_spec)
        return self

    def transform(
        self,
        field: np.ndarray,
        mask: np.ndarray | None,
        domain_spec: DomainSpec,
    ) -> np.ndarray:
        field_arr = np.asarray(field)
        if field_arr.ndim == 2:
            field_arr = field_arr[..., None]
            was_2d = True
        elif field_arr.ndim == 3:
            was_2d = False
        else:
            raise ValueError(f"channelwise adapter expects 2D or 3D field, got {field_arr.shape}")
        channels = int(field_arr.shape[-1])
        self._ensure_decomposers(channels)
        coeffs: list[np.ndarray] = []
        channel_metas = []
        base_layout: str | None = None
        channel_shapes: list[tuple[int, ...]] = []
        for idx, decomposer in enumerate(self._decomposers):
            raw_coeff = decomposer.transform(field_arr[..., idx], mask=mask, domain_spec=domain_spec)
            meta = decomposer.coeff_meta()
            channel_metas.append(meta)
            raw_arr = np.asarray(raw_coeff)
            coeff_shape = meta.get("coeff_shape")
            if isinstance(coeff_shape, (list, tuple)) and coeff_shape:
                expected_shape = tuple(int(x) for x in coeff_shape)
                if raw_arr.shape != expected_shape:
                    raw_arr = raw_arr.reshape(expected_shape, order=str(meta.get("flatten_order", "C")))
            layout = str(meta.get("coeff_layout", "")).strip()
            if (
                int(meta.get("channels", 1)) == 1
                and raw_arr.ndim >= 1
                and raw_arr.shape[0] == 1
                and isinstance(coeff_shape, (list, tuple))
                and coeff_shape
                and int(coeff_shape[0]) == 1
            ):
                raw_arr = raw_arr[0]
                if layout.startswith("C"):
                    layout = layout[1:]
            if base_layout is None:
                base_layout = layout
            elif base_layout != layout:
                raise ValueError("channelwise adapter requires consistent coeff_layout across channels")
            coeffs.append(raw_arr)
            channel_shapes.append(raw_arr.shape)
        coeff_tensor = np.stack(coeffs, axis=0)
        self._coeff_shape = coeff_tensor.shape
        self._field_ndim = 2 if was_2d else 3
        self._channel_shapes = channel_shapes
        if base_layout is None:
            base_layout = "K"
        meta = self._coeff_meta_base(
            field_shape=field_arr.shape[:2] if was_2d else field_arr.shape,
            field_ndim=self._field_ndim,
            field_layout="HW" if was_2d else "HWC",
            channels=channels,
            coeff_shape=coeff_tensor.shape,
            coeff_layout=f"C{base_layout}",
            complex_format="channelwise",
        )
        meta.update({"base_method": channel_metas[0]["method"] if channel_metas else None})
        self._coeff_meta = meta
        return coeff_tensor

    def inverse_transform(self, coeff: np.ndarray, domain_spec: DomainSpec) -> np.ndarray:
        if self._coeff_shape is None or self._field_ndim is None or self._channel_shapes is None:
            raise ValueError("transform must be called before inverse_transform")
        coeff_arr = np.asarray(coeff)
        expected = int(np.prod(self._coeff_shape))
        if coeff_arr.shape != self._coeff_shape:
            coeff_flat = coeff_arr.reshape(-1)
            if coeff_flat.size > expected:
                raise ValueError(f"coeff size {coeff_flat.size} exceeds expected {expected}")
            if coeff_flat.size < expected:
                padded = np.zeros(expected, dtype=coeff_flat.dtype)
                padded[: coeff_flat.size] = coeff_flat
                coeff_flat = padded
            coeff_arr = coeff_flat.reshape(self._coeff_shape, order="C")
        fields = []
        for idx, (decomposer, shape) in enumerate(zip(self._decomposers, self._channel_shapes)):
            chunk = coeff_arr[idx]
            if shape and chunk.shape != shape:
                chunk = chunk.reshape(shape, order="C")
            fields.append(decomposer.inverse_transform(chunk, domain_spec=domain_spec))
        field_hat = np.stack(fields, axis=-1)
        if self._field_ndim == 2 and field_hat.shape[-1] == 1:
            return field_hat[..., 0]
        return field_hat


from mode_decomp_ml.pickle_compat import load_pickle_compat

__all__ = [
    "BaseDecomposer",
    "GridDecomposerBase",
    "ZernikeFamilyBase",
    "EigenBasisDecomposerBase",
    "ChannelwiseAdapter",
]
