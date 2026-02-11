"""Auto-dispatch codec for heterogeneous coefficient types.

This codec selects an underlying codec based on raw_meta/coeff dtype:
  - complex coefficients: fft_complex_codec_v1 (real/imag lossless)
  - wavelet wavedec2 coefficients (list structure): wavelet_pack_v1
  - otherwise: none (flatten real-valued arrays)
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from mode_decomp_ml.config import cfg_get
from mode_decomp_ml.plugins.registry import register_coeff_codec

from .basic import BaseCoeffCodec, NoOpCoeffCodec
from .fft_complex import FFTComplexCodecV1
from .offset_residual_pack import OffsetResidualPackCodecV1
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


def _is_offset_residual_meta(raw_meta: Mapping[str, Any] | None) -> bool:
    if raw_meta is None:
        return False
    coeff_format = str(raw_meta.get("coeff_format", "")).strip().lower()
    return coeff_format == "offset_residual_v1"


@register_coeff_codec("auto_codec_v1")
class AutoCodecV1(BaseCoeffCodec):
    """Dispatch codec based on raw_meta.

    Config:
      - complex_mode: fft_complex_codec_v1.mode (default: real_imag)
      - dtype_policy: float32/float64 (default: float32)
    """

    def __init__(self, *, cfg: Mapping[str, Any]) -> None:
        super().__init__(cfg=cfg)
        self.name = "auto_codec_v1"
        dtype_policy = str(cfg_get(cfg, "dtype_policy", "float32"))
        self.dtype_policy = dtype_policy
        complex_mode = str(cfg_get(cfg, "complex_mode", "real_imag"))

        # Underlying codecs.
        self._none = NoOpCoeffCodec(cfg={"dtype_policy": dtype_policy})
        self._fft_complex = FFTComplexCodecV1(cfg={"dtype_policy": dtype_policy, "mode": complex_mode})
        self._wavelet = WaveletPackCodecV1(cfg={"dtype_policy": dtype_policy})
        self._offset_residual = OffsetResidualPackCodecV1(
            cfg={"dtype_policy": dtype_policy, "complex_mode": complex_mode}
        )

        # "Lossless" depends on the chosen sub-codec; advertise conservative false.
        self.is_lossless = False

    def _select(self, raw_coeff: Any, raw_meta: Mapping[str, Any] | None) -> BaseCoeffCodec:
        if _is_offset_residual_meta(raw_meta):
            return self._offset_residual
        if _is_wavelet_meta(raw_meta):
            return self._wavelet
        if _is_complex_meta(raw_meta):
            return self._fft_complex
        # Fallback: detect actual complex arrays even if meta is missing/misleading.
        try:
            arr = np.asarray(raw_coeff)
            if np.iscomplexobj(arr):
                return self._fft_complex
        except Exception:
            pass
        return self._none

    def encode(self, raw_coeff: Any, raw_meta: Mapping[str, Any]) -> np.ndarray:
        return self._select(raw_coeff, raw_meta).encode(raw_coeff, raw_meta)

    def decode(self, vector_coeff: np.ndarray, raw_meta: Mapping[str, Any]) -> Any:
        # For decode we can select purely from meta.
        if _is_offset_residual_meta(raw_meta):
            return self._offset_residual.decode(vector_coeff, raw_meta)
        if _is_wavelet_meta(raw_meta):
            return self._wavelet.decode(vector_coeff, raw_meta)
        if _is_complex_meta(raw_meta):
            return self._fft_complex.decode(vector_coeff, raw_meta)
        return self._none.decode(vector_coeff, raw_meta)

    def coeff_meta(self, raw_meta: Mapping[str, Any] | None, vector_coeff: np.ndarray) -> Mapping[str, Any]:
        # Delegate to the selected codec so coeff_shape/layout stay consistent with decode.
        meta_source = raw_meta or {}
        if _is_offset_residual_meta(meta_source):
            return self._offset_residual.coeff_meta(raw_meta, vector_coeff)
        if _is_wavelet_meta(meta_source):
            return self._wavelet.coeff_meta(raw_meta, vector_coeff)
        if _is_complex_meta(meta_source):
            return self._fft_complex.coeff_meta(raw_meta, vector_coeff)
        return self._none.coeff_meta(raw_meta, vector_coeff)


__all__ = ["AutoCodecV1"]
