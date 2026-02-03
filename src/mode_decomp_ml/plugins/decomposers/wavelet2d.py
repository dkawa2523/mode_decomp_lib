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
        self._mask_policy = require_cfg_stripped(cfg, "mask_policy", label="decompose")
        if self._mask_policy not in _MASK_POLICIES:
            raise ValueError(
                f"decompose.mask_policy must be one of {_MASK_POLICIES}, got {self._mask_policy}"
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

        total_coeffs = _coeff_size(structure)
        channels = int(field_3d.shape[-1])
        self._coeff_shape = (channels, total_coeffs)
        self._field_ndim = 2 if was_2d else 3
        self._grid_shape = domain_spec.grid_shape
        self._coeff_structure = structure

        actual_level = int(len(structure.get("details", [])))
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
        if self._field_ndim == 2 and field_stack.shape[-1] == 1:
            return field_stack[..., 0]
        return field_stack


__all__ = ["Wavelet2DDecomposer"]
