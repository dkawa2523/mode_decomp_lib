"""Spherical harmonics coefficient pack/unpack codec."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from mode_decomp_ml.plugins.registry import register_coeff_codec

from mode_decomp_ml.config import cfg_get

from .basic import BaseCoeffCodec, _devectorize_raw, _vectorize_raw


def _validate_lm_kind_list(raw_meta: Mapping[str, Any]) -> list[Any]:
    lm_kind_list = raw_meta.get("lm_kind_list")
    if not isinstance(lm_kind_list, (list, tuple)) or not lm_kind_list:
        raise ValueError("sh_pack_v1 requires raw_meta.lm_kind_list")
    coeff_shape = raw_meta.get("coeff_shape")
    if isinstance(coeff_shape, (list, tuple)) and coeff_shape:
        n_modes = int(coeff_shape[-1])
        if n_modes != len(lm_kind_list):
            raise ValueError("sh_pack_v1 coeff_shape does not match lm_kind_list length")
    return list(lm_kind_list)


@register_coeff_codec("sh_pack_v1")
class SHPackCodecV1(BaseCoeffCodec):
    """Lossless packing for spherical harmonics coefficients."""

    def __init__(self, *, cfg: Mapping[str, Any]) -> None:
        super().__init__(cfg=cfg)
        self.name = "sh_pack_v1"
        self.is_lossless = True
        self.dtype_policy = str(cfg_get(cfg, "dtype_policy", "float32"))

    def encode(self, raw_coeff: Any, raw_meta: Mapping[str, Any]) -> np.ndarray:
        if raw_meta is None:
            raise ValueError("sh_pack_v1 requires raw_meta")
        _validate_lm_kind_list(raw_meta)
        vec = _vectorize_raw(raw_coeff, raw_meta)
        if np.iscomplexobj(vec):
            raise ValueError("sh_pack_v1 does not support complex coefficients")
        return np.asarray(vec, dtype=np.float32).reshape(-1)

    def decode(self, vector_coeff: np.ndarray, raw_meta: Mapping[str, Any]) -> Any:
        if raw_meta is None:
            raise ValueError("sh_pack_v1 requires raw_meta")
        _validate_lm_kind_list(raw_meta)
        vec = np.asarray(vector_coeff, dtype=np.float32).reshape(-1)
        return _devectorize_raw(vec, raw_meta)


__all__ = ["SHPackCodecV1"]
