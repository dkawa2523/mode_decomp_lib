"""Wavelet2D decomposer using PyWavelets."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
from mode_decomp_ml.config import cfg_get

from mode_decomp_ml.domain import DomainSpec, validate_decomposer_compatibility
from mode_decomp_ml.optional import require_dependency
from mode_decomp_ml.plugins.registry import register_decomposer

from .base import BaseDecomposer, require_cfg_stripped

try:  # optional dependency
    import pywt
except Exception:  # pragma: no cover - PyWavelets is optional
    pywt = None

_MASK_POLICIES = {"error", "zero_fill"}
_MASK_FILL_OPTIONS = {"zero", "mean"}
_KEEP_OPTIONS = {"all", "approx"}



def _require_pywt() -> None:
    require_dependency(pywt, name="wavelet2d decomposer", pip_name="pywt")


def _parse_level(value: Any) -> int | None:
    if value is None or value == "" or value == "null":
        return None
    if isinstance(value, (int, np.integer)):
        level = int(value)
        if level <= 0:
            raise ValueError("decompose.level must be positive for wavelet2d")
        return level
    raise ValueError("decompose.level must be an int or null for wavelet2d")


def _shape_list(shape: Sequence[int]) -> list[int]:
    return [int(x) for x in shape]


def _coeff_structure(coeffs: Sequence[Any]) -> dict[str, Any]:
    if not coeffs:
        raise ValueError("wavelet2d coeffs are empty")
    approx = np.asarray(coeffs[0])
    detail_shapes: list[list[list[int]]] = []
    for level in coeffs[1:]:
        if not isinstance(level, (list, tuple)) or len(level) != 3:
            raise ValueError("wavelet2d coeff detail must be a tuple of (cH, cV, cD)")
        bands = [np.asarray(band) for band in level]
        detail_shapes.append([_shape_list(band.shape) for band in bands])
    return {"approx": _shape_list(approx.shape), "details": detail_shapes}


def _coeff_size(structure: Mapping[str, Any]) -> int:
    approx = structure.get("approx")
    details = structure.get("details")
    if not isinstance(approx, (list, tuple)) or not isinstance(details, (list, tuple)):
        raise ValueError("wavelet2d coeff_structure is invalid")
    total = int(np.prod(approx))
    for level in details:
        if not isinstance(level, (list, tuple)) or len(level) != 3:
            raise ValueError("wavelet2d coeff_structure.details must contain 3 bands per level")
        for band in level:
            total += int(np.prod(band))
    return int(total)


def _validate_structure_match(reference: Mapping[str, Any], other: Mapping[str, Any]) -> None:
    if reference.get("approx") != other.get("approx") or reference.get("details") != other.get("details"):
        raise ValueError("wavelet2d coeff structure mismatch across channels")


def _trim_field(field: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if field.shape[0] < shape[0] or field.shape[1] < shape[1]:
        raise ValueError("wavelet2d reconstruction is smaller than expected grid shape")
    if field.shape[:2] == shape:
        return field
    return field[: shape[0], : shape[1]]


@register_decomposer("wavelet2d")
class Wavelet2DDecomposer(BaseDecomposer):
    """Wavelet2D decomposer for rectangle grids (PyWavelets)."""

    def __init__(self, *, cfg: Mapping[str, Any]) -> None:
        super().__init__(cfg=cfg)
        self.name = "wavelet2d"
        self._wavelet = require_cfg_stripped(cfg, "wavelet", label="decompose")
        self._mode = str(cfg_get(cfg, "mode", "symmetric")).strip() or "symmetric"
        self._level = _parse_level(cfg_get(cfg, "level", None))
        self._keep = str(cfg_get(cfg, "keep", "all")).strip().lower() or "all"
        if self._keep not in _KEEP_OPTIONS:
            raise ValueError(f"decompose.keep must be one of {_KEEP_OPTIONS}, got {self._keep}")
        self._mask_policy = require_cfg_stripped(cfg, "mask_policy", label="decompose")
        if self._mask_policy not in _MASK_POLICIES:
            raise ValueError(
                f"decompose.mask_policy must be one of {_MASK_POLICIES}, got {self._mask_policy}"
            )
        self._mask_fill = str(cfg_get(cfg, "mask_fill", "zero")).strip().lower() or "zero"
        if self._mask_fill not in _MASK_FILL_OPTIONS:
            raise ValueError(
                f"decompose.mask_fill must be one of {_MASK_FILL_OPTIONS}, got {self._mask_fill}"
            )
        self._coeff_shape: tuple[int, ...] | None = None
        self._field_ndim: int | None = None
        self._grid_shape: tuple[int, int] | None = None
        self._coeff_structure: dict[str, Any] | None = None

    def transform(
        self,
        field: np.ndarray,
        mask: np.ndarray | None,
        domain_spec: DomainSpec,
    ) -> Any:
        _require_pywt()
        validate_decomposer_compatibility(domain_spec, self._cfg)
        allow_zero_fill = self._mask_policy == "zero_fill"
        field_3d, was_2d = self._prepare_field(
            field, mask, domain_spec, allow_zero_fill=allow_zero_fill
        )

        # `mask_policy=zero_fill` uses 0 outside the mask, which introduces a hard discontinuity at
        # the boundary. For low-K wavelet projections (keep=approx), this can destroy reconstruction
        # quality *inside* the mask. Optionally fill masked-out pixels with the per-channel mean to
        # reduce boundary artifacts; domain masking is applied after inverse_transform.
        if self._mask_fill == "mean":
            combined_mask: np.ndarray | None = None
            if mask is not None:
                mask_arr = np.asarray(mask).astype(bool)
                if mask_arr.ndim == 3:
                    mask_arr = mask_arr[..., 0]
                if mask_arr.shape != domain_spec.grid_shape:
                    raise ValueError(
                        f"mask shape {mask_arr.shape} does not match domain {domain_spec.grid_shape}"
                    )
                combined_mask = mask_arr
            if domain_spec.mask is not None:
                combined_mask = domain_spec.mask if combined_mask is None else (combined_mask & domain_spec.mask)
            if combined_mask is not None and not combined_mask.all():
                if not np.any(combined_mask):
                    raise ValueError("wavelet2d mask has no valid entries")
                field_3d = field_3d.copy()
                for ch in range(field_3d.shape[-1]):
                    mean = float(np.mean(field_3d[..., ch][combined_mask]))
                    field_3d[..., ch] = np.where(combined_mask, field_3d[..., ch], mean)

        coeffs_per_channel: list[list[Any]] = []
        for ch in range(field_3d.shape[-1]):
            coeffs = pywt.wavedec2(
                field_3d[..., ch],
                wavelet=self._wavelet,
                mode=self._mode,
                level=self._level,
            )
            coeffs_per_channel.append(coeffs)

        structure = _coeff_structure(coeffs_per_channel[0])
        for coeffs in coeffs_per_channel[1:]:
            _validate_structure_match(structure, _coeff_structure(coeffs))

        channels = int(field_3d.shape[-1])
        self._field_ndim = 2 if was_2d else 3
        self._grid_shape = domain_spec.grid_shape
        self._coeff_structure = structure

        actual_level = int(len(structure.get("details", [])))
        if self._keep == "approx":
            approx_shape = tuple(int(x) for x in structure.get("approx", []))
            if not approx_shape:
                raise ValueError("wavelet2d approx_shape is missing")
            approx_len = int(np.prod(approx_shape))
            if approx_len <= 0:
                raise ValueError("wavelet2d approx_len must be positive")
            coeffs_flat = []
            for coeffs in coeffs_per_channel:
                cA = np.asarray(coeffs[0])
                coeffs_flat.append(cA.reshape(-1, order="C"))
            coeff_tensor = np.stack(coeffs_flat, axis=0)
            self._coeff_shape = (channels, approx_len)

            meta = self._coeff_meta_base(
                field_shape=field_3d.shape[:2] if was_2d else field_3d.shape,
                field_ndim=self._field_ndim,
                field_layout="HW" if was_2d else "HWC",
                channels=channels,
                coeff_shape=self._coeff_shape,
                coeff_layout="CK",
                complex_format="real",
                flatten_order="C",
            )
            meta.update(
                {
                    "wavelet": self._wavelet,
                    "mode": self._mode,
                    "level": actual_level,
                    "mask_policy": self._mask_policy,
                    "mask_fill": self._mask_fill,
                    "keep": "approx",
                    # Keep structure for reconstruction, but avoid triggering the wavelet_pack codec
                    # (it requires full wavedec2 coefficients).
                    "coeff_format": "wavelet_approx",
                    "detail_order": ["H", "V", "D"],
                    "coeff_structure": structure,
                    "approx_shape": _shape_list(approx_shape),
                    "detail_shapes": structure.get("details", []),
                    "coeff_len": approx_len,
                }
            )
            self._coeff_meta = meta
            return coeff_tensor

        total_coeffs = _coeff_size(structure)
        self._coeff_shape = (channels, total_coeffs)
        meta = self._coeff_meta_base(
            field_shape=field_3d.shape[:2] if was_2d else field_3d.shape,
            field_ndim=self._field_ndim,
            field_layout="HW" if was_2d else "HWC",
            channels=channels,
            coeff_shape=self._coeff_shape,
            coeff_layout="CK",
            complex_format="real",
            flatten_order="C",
        )
        meta.update(
            {
                "wavelet": self._wavelet,
                "mode": self._mode,
                "level": actual_level,
                "mask_policy": self._mask_policy,
                "mask_fill": self._mask_fill,
                "coeff_format": "wavedec2",
                "detail_order": ["H", "V", "D"],
                "coeff_structure": structure,
                "coeff_len": total_coeffs,
            }
        )
        self._coeff_meta = meta

        if was_2d and channels == 1:
            return coeffs_per_channel[0]
        return coeffs_per_channel

    def inverse_transform(self, coeff: Any, domain_spec: DomainSpec) -> np.ndarray:
        _require_pywt()
        validate_decomposer_compatibility(domain_spec, self._cfg)
        if self._coeff_shape is None or self._field_ndim is None or self._coeff_structure is None:
            raise ValueError("wavelet2d transform must be called before inverse_transform")
        if self._grid_shape is not None and domain_spec.grid_shape != self._grid_shape:
            raise ValueError("wavelet2d domain grid does not match cached shape")

        if self._keep == "approx":
            coeff_tensor = self._reshape_coeff(coeff, self._coeff_shape, name=self.name)
            structure = self._coeff_structure
            approx_shape = tuple(int(x) for x in structure.get("approx", []))
            details = structure.get("details", [])
            if not approx_shape or not isinstance(details, list):
                raise ValueError("wavelet2d coeff_structure is missing for keep=approx")

            channel_coeffs = []
            for ch in range(int(self._coeff_shape[0])):
                cA = np.asarray(coeff_tensor[ch], dtype=np.float64).reshape(approx_shape, order="C")
                coeffs_full: list[Any] = [cA]
                for level in details:
                    if not isinstance(level, (list, tuple)) or len(level) != 3:
                        raise ValueError("wavelet2d coeff_structure.details must have 3 bands per level")
                    bands = []
                    for band_shape in level:
                        shape = tuple(int(x) for x in band_shape)
                        bands.append(np.zeros(shape, dtype=np.float64))
                    coeffs_full.append(tuple(bands))
                channel_coeffs.append(coeffs_full)

            fields = []
            for coeffs_full in channel_coeffs:
                field_hat = pywt.waverec2(coeffs_full, wavelet=self._wavelet, mode=self._mode)
                field_hat = np.asarray(field_hat)
                field_hat = _trim_field(field_hat, domain_spec.grid_shape)
                fields.append(field_hat)

            field_stack = np.stack(fields, axis=-1)
            if domain_spec.mask is not None:
                field_stack = field_stack.copy()
                field_stack[~domain_spec.mask] = 0.0
            if self._field_ndim == 2 and field_stack.shape[-1] == 1:
                return field_stack[..., 0]
            return field_stack

        channels = int(self._coeff_shape[0])
        if channels == 1:
            channel_coeffs = [coeff]
        else:
            if not isinstance(coeff, (list, tuple)) or len(coeff) != channels:
                raise ValueError("wavelet2d expects list of coeffs per channel for vector fields")
            channel_coeffs = list(coeff)

        fields = []
        for channel_coeff in channel_coeffs:
            field_hat = pywt.waverec2(channel_coeff, wavelet=self._wavelet, mode=self._mode)
            field_hat = np.asarray(field_hat)
            field_hat = _trim_field(field_hat, domain_spec.grid_shape)
            fields.append(field_hat)

        field_stack = np.stack(fields, axis=-1)
        if domain_spec.mask is not None:
            field_stack = field_stack.copy()
            field_stack[~domain_spec.mask] = 0.0
        if self._field_ndim == 2 and field_stack.shape[-1] == 1:
            return field_stack[..., 0]
        return field_stack


__all__ = ["Wavelet2DDecomposer"]
