"""Slepian coefficient pack/unpack codec."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from mode_decomp_ml.plugins.registry import register_coeff_codec

from mode_decomp_ml.config import cfg_get

from .basic import BaseCoeffCodec, _devectorize_raw, _vectorize_raw


def _validate_slepian_meta(raw_meta: Mapping[str, Any]) -> int:
    coeff_shape = raw_meta.get("coeff_shape")
    if not isinstance(coeff_shape, (list, tuple)) or not coeff_shape:
        raise ValueError("slepian_pack_v1 requires raw_meta.coeff_shape")
    n_modes = int(coeff_shape[-1])
    eigenvalues = raw_meta.get("eigenvalues")
    if not isinstance(eigenvalues, (list, tuple)) or not eigenvalues:
        raise ValueError("slepian_pack_v1 requires raw_meta.eigenvalues")
    if len(eigenvalues) != n_modes:
        raise ValueError("slepian_pack_v1 eigenvalues length does not match coeff_shape")
    return n_modes


@register_coeff_codec("slepian_pack_v1")
class SlepianPackCodecV1(BaseCoeffCodec):
    """Lossless packing for spherical Slepian coefficients."""

    def __init__(self, *, cfg: Mapping[str, Any]) -> None:
        super().__init__(cfg=cfg)
        self.name = "slepian_pack_v1"
        self.is_lossless = True
        self.dtype_policy = str(cfg_get(cfg, "dtype_policy", "float32"))

    def encode(self, raw_coeff: Any, raw_meta: Mapping[str, Any]) -> np.ndarray:
        if raw_meta is None:
            raise ValueError("slepian_pack_v1 requires raw_meta")
        _validate_slepian_meta(raw_meta)
        vec = _vectorize_raw(raw_coeff, raw_meta)
        if np.iscomplexobj(vec):
            raise ValueError("slepian_pack_v1 does not support complex coefficients")
        return np.asarray(vec, dtype=np.float32).reshape(-1)

    def decode(self, vector_coeff: np.ndarray, raw_meta: Mapping[str, Any]) -> Any:
        if raw_meta is None:
            raise ValueError("slepian_pack_v1 requires raw_meta")
        _validate_slepian_meta(raw_meta)
        vec = np.asarray(vector_coeff, dtype=np.float32).reshape(-1)
        return _devectorize_raw(vec, raw_meta)


__all__ = ["SlepianPackCodecV1"]
