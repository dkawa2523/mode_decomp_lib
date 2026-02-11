"""Offset+residual coefficient packing codec.

raw_coeff format (mapping):
  {
    "offset": np.ndarray shape (C,),
    "residual": <inner raw coeff>,
  }

raw_meta must include:
  - coeff_format: "offset_residual_v1"
  - offset_dim: int
  - residual_raw_meta: mapping (inner raw_meta)

The encoded vector is:
  [offset_flat, residual_vector]
where residual_vector is encoded using the same dispatch rules as auto_codec_v1
(wavelet/complex/real).
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from mode_decomp_ml.config import cfg_get
from mode_decomp_ml.plugins.registry import register_coeff_codec

from .basic import BaseCoeffCodec, NoOpCoeffCodec
from .fft_complex import FFTComplexCodecV1
from .wavelet_pack import WaveletPackCodecV1


def _is_wavelet_meta(raw_meta: Mapping[str, Any] | None) -> bool:
    if raw_meta is None:
        return False
    if raw_meta.get("coeff_structure") is None:
        return False
    coeff_format = str(raw_meta.get("coeff_format", "")).strip().lower()
    return coeff_format == "wavedec2"


def _is_complex_meta(raw_meta: Mapping[str, Any] | None) -> bool:
    if raw_meta is None:
        return False
    fmt = str(raw_meta.get("complex_format", "")).strip().lower()
    return fmt in {"complex", "real_imag"}


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping, got {type(value)}")
    return value


def _require_int(value: Any, *, label: str) -> int:
    if value is None:
        raise ValueError(f"{label} is required")
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, str) and value.strip():
        return int(value)
    raise TypeError(f"{label} must be int, got {type(value)}")


@register_coeff_codec("offset_residual_pack_v1")
class OffsetResidualPackCodecV1(BaseCoeffCodec):
    """Lossless pack/unpack for offset_residual_v1 mapping coeffs."""

    def __init__(self, *, cfg: Mapping[str, Any]) -> None:
        super().__init__(cfg=cfg)
        self.name = "offset_residual_pack_v1"
        dtype_policy = str(cfg_get(cfg, "dtype_policy", "float32"))
        self.dtype_policy = dtype_policy
        complex_mode = str(cfg_get(cfg, "complex_mode", "real_imag"))

        self._none = NoOpCoeffCodec(cfg={"dtype_policy": dtype_policy})
        self._fft_complex = FFTComplexCodecV1(cfg={"dtype_policy": dtype_policy, "mode": complex_mode})
        self._wavelet = WaveletPackCodecV1(cfg={"dtype_policy": dtype_policy})

        self.is_lossless = True

    def _select_residual(self, raw_coeff: Any, residual_raw_meta: Mapping[str, Any] | None) -> BaseCoeffCodec:
        if _is_wavelet_meta(residual_raw_meta):
            return self._wavelet
        if _is_complex_meta(residual_raw_meta):
            return self._fft_complex
        # Fallback to actual dtype when meta is missing/misleading.
        try:
            arr = np.asarray(raw_coeff)
            if np.iscomplexobj(arr):
                return self._fft_complex
        except Exception:
            pass
        return self._none

    def _select_residual_for_decode(self, residual_raw_meta: Mapping[str, Any] | None) -> BaseCoeffCodec:
        if _is_wavelet_meta(residual_raw_meta):
            return self._wavelet
        if _is_complex_meta(residual_raw_meta):
            return self._fft_complex
        return self._none

    def encode(self, raw_coeff: Any, raw_meta: Mapping[str, Any]) -> np.ndarray:
        raw_meta = _require_mapping(raw_meta, label="raw_meta")
        coeff_format = str(raw_meta.get("coeff_format", "")).strip().lower()
        if coeff_format != "offset_residual_v1":
            raise ValueError("offset_residual_pack_v1 requires raw_meta.coeff_format == 'offset_residual_v1'")

        raw_coeff_map = _require_mapping(raw_coeff, label="raw_coeff")
        if "offset" not in raw_coeff_map or "residual" not in raw_coeff_map:
            raise ValueError("offset_residual_pack_v1 expects raw_coeff keys {'offset','residual'}")

        offset_dim = _require_int(raw_meta.get("offset_dim"), label="raw_meta.offset_dim")
        residual_raw_meta = _require_mapping(raw_meta.get("residual_raw_meta"), label="raw_meta.residual_raw_meta")

        offset = np.asarray(raw_coeff_map["offset"], dtype=np.float32).reshape(-1)
        if offset.size != int(offset_dim):
            raise ValueError("offset_residual_pack_v1 offset vector size does not match raw_meta.offset_dim")

        residual_coeff = raw_coeff_map["residual"]
        residual_codec = self._select_residual(residual_coeff, residual_raw_meta)
        residual_vec = residual_codec.encode(residual_coeff, residual_raw_meta)

        vec = np.concatenate([offset.reshape(-1), np.asarray(residual_vec).reshape(-1)], axis=0)
        return np.asarray(vec, dtype=np.float32).reshape(-1)

    def decode(self, vector_coeff: np.ndarray, raw_meta: Mapping[str, Any]) -> Any:
        raw_meta = _require_mapping(raw_meta, label="raw_meta")
        coeff_format = str(raw_meta.get("coeff_format", "")).strip().lower()
        if coeff_format != "offset_residual_v1":
            raise ValueError("offset_residual_pack_v1 requires raw_meta.coeff_format == 'offset_residual_v1'")

        offset_dim = _require_int(raw_meta.get("offset_dim"), label="raw_meta.offset_dim")
        residual_raw_meta = _require_mapping(raw_meta.get("residual_raw_meta"), label="raw_meta.residual_raw_meta")

        vec = np.asarray(vector_coeff, dtype=np.float32).reshape(-1)
        if vec.size < int(offset_dim):
            raise ValueError("offset_residual_pack_v1 vector_coeff is shorter than offset_dim")

        offset = vec[: int(offset_dim)].astype(np.float32, copy=False)
        residual_vec = vec[int(offset_dim) :].astype(np.float32, copy=False)
        residual_codec = self._select_residual_for_decode(residual_raw_meta)
        residual = residual_codec.decode(residual_vec, residual_raw_meta)
        return {"offset": offset, "residual": residual}

    def coeff_meta(self, raw_meta: Mapping[str, Any] | None, vector_coeff: np.ndarray) -> Mapping[str, Any]:
        base_raw = dict(raw_meta or {})
        coeff_format = str(base_raw.get("coeff_format", "")).strip().lower()
        if coeff_format != "offset_residual_v1":
            # Fallback to the base implementation for safety.
            return super().coeff_meta(raw_meta, vector_coeff)

        offset_dim = _require_int(base_raw.get("offset_dim"), label="raw_meta.offset_dim")
        residual_raw_meta = _require_mapping(base_raw.get("residual_raw_meta"), label="raw_meta.residual_raw_meta")

        vec = np.asarray(vector_coeff, dtype=np.float32).reshape(-1)
        residual_vec = vec[int(offset_dim) :].astype(np.float32, copy=False)
        residual_codec = self._select_residual_for_decode(residual_raw_meta)
        residual_coeff_meta = residual_codec.coeff_meta(residual_raw_meta, residual_vec)

        meta = dict(base_raw)
        meta["raw_meta"] = base_raw
        meta["codec"] = {
            "name": self.name,
            "is_lossless": bool(self.is_lossless),
            "dtype_policy": self.dtype_policy,
        }
        meta["vector_dim"] = int(vec.size)
        meta["coeff_format"] = "offset_residual_v1"
        meta["offset_dim"] = int(offset_dim)
        meta["residual_vector_dim"] = int(residual_vec.size)
        meta["residual_coeff_meta"] = dict(residual_coeff_meta)
        # Represent the packed coefficient as a flat real vector.
        meta["coeff_shape"] = [int(vec.size)]
        meta["coeff_layout"] = "K"
        meta["flatten_order"] = "C"
        meta["complex_format"] = "real"
        return meta


__all__ = ["OffsetResidualPackCodecV1"]

