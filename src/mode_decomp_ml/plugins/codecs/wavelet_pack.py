"""Wavelet coefficient pack/unpack codec."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from mode_decomp_ml.plugins.registry import register_coeff_codec

from mode_decomp_ml.config import cfg_get

from .basic import BaseCoeffCodec, _resolve_flatten_order


def _resolve_float_dtype(policy: str) -> np.dtype:
    value = str(policy or "").strip().lower()
    if value in {"float32", "fp32", "f32", ""}:
        return np.float32
    if value in {"float64", "fp64", "f64", "double"}:
        return np.float64
    raise ValueError(f"Unsupported dtype_policy: {policy}")


def _shape_tuple(shape: Sequence[Any], *, label: str) -> tuple[int, ...]:
    if not isinstance(shape, (list, tuple)) or not shape:
        raise ValueError(f"wavelet_pack_v1 expects {label} to be a list of ints")
    return tuple(int(x) for x in shape)


def _parse_structure(raw_meta: Mapping[str, Any]) -> tuple[tuple[int, ...], list[list[tuple[int, ...]]]]:
    structure = raw_meta.get("coeff_structure")
    if not isinstance(structure, Mapping):
        raise ValueError("wavelet_pack_v1 requires raw_meta.coeff_structure")
    approx = _shape_tuple(structure.get("approx"), label="coeff_structure.approx")
    details = structure.get("details")
    if not isinstance(details, (list, tuple)):
        raise ValueError("wavelet_pack_v1 requires coeff_structure.details list")
    detail_shapes: list[list[tuple[int, ...]]] = []
    for level in details:
        if not isinstance(level, (list, tuple)) or len(level) != 3:
            raise ValueError("wavelet_pack_v1 coeff_structure.details must have 3 bands per level")
        level_shapes = [_shape_tuple(band, label="coeff_structure.details") for band in level]
        detail_shapes.append(level_shapes)
    return approx, detail_shapes


def _coeff_size(approx: tuple[int, ...], details: Sequence[Sequence[tuple[int, ...]]]) -> int:
    total = int(np.prod(approx))
    for level in details:
        for band in level:
            total += int(np.prod(band))
    return int(total)


def _flatten_coeffs(
    coeffs: Any,
    approx: tuple[int, ...],
    details: Sequence[Sequence[tuple[int, ...]]],
    *,
    order: str,
) -> np.ndarray:
    if not isinstance(coeffs, (list, tuple)) or len(coeffs) != 1 + len(details):
        raise ValueError("wavelet_pack_v1 expects wavedec2 coeff list per channel")
    parts: list[np.ndarray] = []
    cA = np.asarray(coeffs[0])
    if np.iscomplexobj(cA):
        raise ValueError("wavelet_pack_v1 does not support complex coefficients")
    if cA.shape != approx:
        raise ValueError("wavelet_pack_v1 approx shape mismatch")
    parts.append(cA.reshape(-1, order=order))
    for level, shapes in enumerate(details, start=1):
        detail = coeffs[level]
        if not isinstance(detail, (list, tuple)) or len(detail) != 3:
            raise ValueError("wavelet_pack_v1 expects detail tuple (cH, cV, cD)")
        for band, shape in zip(detail, shapes):
            arr = np.asarray(band)
            if np.iscomplexobj(arr):
                raise ValueError("wavelet_pack_v1 does not support complex coefficients")
            if arr.shape != shape:
                raise ValueError("wavelet_pack_v1 detail shape mismatch")
            parts.append(arr.reshape(-1, order=order))
    if not parts:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(parts, axis=0)


def _unflatten_coeffs(
    vec: np.ndarray,
    approx: tuple[int, ...],
    details: Sequence[Sequence[tuple[int, ...]]],
    *,
    order: str,
) -> list[Any]:
    coeffs: list[Any] = []
    idx = 0
    size = int(np.prod(approx))
    cA = vec[idx : idx + size].reshape(approx, order=order)
    coeffs.append(cA)
    idx += size
    for shapes in details:
        bands = []
        for shape in shapes:
            size = int(np.prod(shape))
            band = vec[idx : idx + size].reshape(shape, order=order)
            bands.append(band)
            idx += size
        coeffs.append(tuple(bands))
    return coeffs


@register_coeff_codec("wavelet_pack_v1")
class WaveletPackCodecV1(BaseCoeffCodec):
    """Flatten wavedec2 coefficients to a vector and restore them."""

    def __init__(self, *, cfg: Mapping[str, Any]) -> None:
        super().__init__(cfg=cfg)
        self.name = "wavelet_pack_v1"
        self.dtype_policy = str(cfg_get(cfg, "dtype_policy", "float32"))
        self._float_dtype = _resolve_float_dtype(self.dtype_policy)
        self.is_lossless = True

    def encode(self, raw_coeff: Any, raw_meta: Mapping[str, Any]) -> np.ndarray:
        if raw_meta is None:
            raise ValueError("wavelet_pack_v1 requires raw_meta")
        approx, details = _parse_structure(raw_meta)
        order = _resolve_flatten_order(raw_meta)
        channels = int(raw_meta.get("channels", 1))
        if channels == 1:
            if (
                isinstance(raw_coeff, (list, tuple))
                and len(raw_coeff) == 1
                and isinstance(raw_coeff[0], (list, tuple))
            ):
                raw_coeff = raw_coeff[0]
            channel_coeffs = [raw_coeff]
        else:
            if not isinstance(raw_coeff, (list, tuple)) or len(raw_coeff) != channels:
                raise ValueError("wavelet_pack_v1 expects coeff list per channel")
            channel_coeffs = list(raw_coeff)

        vecs = []
        for coeffs in channel_coeffs:
            vecs.append(_flatten_coeffs(coeffs, approx, details, order=order))
        vector = np.concatenate(vecs, axis=0) if vecs else np.zeros(0, dtype=self._float_dtype)
        return np.asarray(vector, dtype=self._float_dtype).reshape(-1)

    def decode(self, vector_coeff: np.ndarray, raw_meta: Mapping[str, Any]) -> Any:
        if raw_meta is None:
            raise ValueError("wavelet_pack_v1 requires raw_meta")
        approx, details = _parse_structure(raw_meta)
        order = _resolve_flatten_order(raw_meta)
        channels = int(raw_meta.get("channels", 1))
        per_channel = _coeff_size(approx, details)
        expected = per_channel * max(channels, 1)

        vec = np.asarray(vector_coeff, dtype=self._float_dtype).reshape(-1)
        if vec.size > expected:
            raise ValueError("wavelet_pack_v1 vector_coeff is larger than expected")
        if vec.size < expected:
            padded = np.zeros(expected, dtype=vec.dtype)
            padded[: vec.size] = vec
            vec = padded

        if channels == 1:
            return _unflatten_coeffs(vec, approx, details, order=order)

        coeffs_out = []
        for ch in range(channels):
            start = ch * per_channel
            end = start + per_channel
            coeffs_out.append(_unflatten_coeffs(vec[start:end], approx, details, order=order))
        return coeffs_out

    def coeff_meta(
        self,
        raw_meta: Mapping[str, Any] | None,
        vector_coeff: np.ndarray,
    ) -> Mapping[str, Any]:
        base_raw = dict(raw_meta or {})
        meta = dict(base_raw)
        approx, details = _parse_structure(base_raw)
        channels = int(base_raw.get("channels", 1))
        meta["raw_meta"] = base_raw
        meta["codec"] = {
            "name": self.name,
            "is_lossless": bool(self.is_lossless),
            "dtype_policy": self.dtype_policy,
        }
        meta["vector_dim"] = int(np.asarray(vector_coeff).reshape(-1).shape[0])
        per_channel = _coeff_size(approx, details)
        meta["coeff_shape"] = [int(channels), int(per_channel)]
        meta["coeff_layout"] = "CK"
        meta["complex_format"] = "real"
        return meta


__all__ = ["WaveletPackCodecV1"]
