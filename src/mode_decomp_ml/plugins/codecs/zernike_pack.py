"""Zernike coefficient pack/unpack codec."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from mode_decomp_ml.plugins.registry import register_coeff_codec

from mode_decomp_ml.config import cfg_get

from .basic import BaseCoeffCodec, _devectorize_raw, _vectorize_raw


@register_coeff_codec("zernike_pack_v1")
class ZernikePackCodecV1(BaseCoeffCodec):
    """Lossless packing for Zernike-family coefficients."""

    def __init__(self, *, cfg: Mapping[str, Any]) -> None:
        super().__init__(cfg=cfg)
        self.name = "zernike_pack_v1"
        self.is_lossless = True
        self.dtype_policy = str(cfg_get(cfg, "dtype_policy", "float32"))

    def encode(self, raw_coeff: Any, raw_meta: Mapping[str, Any]) -> np.ndarray:
        if raw_meta is None:
            raise ValueError("zernike_pack_v1 requires raw_meta")
        if "nm_list" not in raw_meta:
            raise ValueError("zernike_pack_v1 requires raw_meta.nm_list")
        vec = _vectorize_raw(raw_coeff, raw_meta)
        if np.iscomplexobj(vec):
            raise ValueError("zernike_pack_v1 does not support complex coefficients")
        return np.asarray(vec, dtype=np.float32).reshape(-1)

    def decode(self, vector_coeff: np.ndarray, raw_meta: Mapping[str, Any]) -> Any:
        if raw_meta is None:
            raise ValueError("zernike_pack_v1 requires raw_meta")
        vec = np.asarray(vector_coeff, dtype=np.float32).reshape(-1)
        return _devectorize_raw(vec, raw_meta)
