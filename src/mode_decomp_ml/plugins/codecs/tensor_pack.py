"""Tensor coefficient pack/unpack codec."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from mode_decomp_ml.plugins.registry import register_coeff_codec

from mode_decomp_ml.config import cfg_get

from .basic import BaseCoeffCodec, _resolve_flatten_order, _resolve_coeff_shape


def _resolve_float_dtype(policy: str) -> np.dtype:
    value = str(policy or "").strip().lower()
    if value in {"float32", "fp32", "f32", ""}:
        return np.float32
    if value in {"float64", "fp64", "f64", "double"}:
        return np.float64
    raise ValueError(f"Unsupported dtype_policy: {policy}")


@register_coeff_codec("tensor_pack_v1")
class TensorPackCodecV1(BaseCoeffCodec):
    """Flatten tensor coefficients to a vector and restore them."""

    def __init__(self, *, cfg: Mapping[str, Any]) -> None:
        super().__init__(cfg=cfg)
        self.name = "tensor_pack_v1"
        self.dtype_policy = str(cfg_get(cfg, "dtype_policy", "float32"))
        self._float_dtype = _resolve_float_dtype(self.dtype_policy)
        self.is_lossless = True

    def encode(self, raw_coeff: Any, raw_meta: Mapping[str, Any]) -> np.ndarray:
        if raw_meta is None:
            raise ValueError("tensor_pack_v1 requires raw_meta")
        coeff_shape = _resolve_coeff_shape(raw_meta)
        order = _resolve_flatten_order(raw_meta)
        arr = np.asarray(raw_coeff)
        if np.iscomplexobj(arr):
            raise ValueError("tensor_pack_v1 does not support complex coefficients")
        if coeff_shape is not None:
            expected = int(np.prod(coeff_shape))
            if arr.size != expected:
                raise ValueError("tensor_pack_v1 raw_coeff size does not match raw_meta.coeff_shape")
            if tuple(arr.shape) != coeff_shape:
                arr = arr.reshape(coeff_shape, order=order)
        vec = arr.reshape(-1, order=order)
        return np.asarray(vec, dtype=self._float_dtype).reshape(-1)

    def decode(self, vector_coeff: np.ndarray, raw_meta: Mapping[str, Any]) -> Any:
        if raw_meta is None:
            raise ValueError("tensor_pack_v1 requires raw_meta")
        coeff_shape = _resolve_coeff_shape(raw_meta)
        order = _resolve_flatten_order(raw_meta)
        vec = np.asarray(vector_coeff, dtype=self._float_dtype).reshape(-1)
        if coeff_shape is None:
            return vec
        expected = int(np.prod(coeff_shape))
        if vec.size > expected:
            raise ValueError("tensor_pack_v1 vector_coeff is larger than expected")
        if vec.size < expected:
            padded = np.zeros(expected, dtype=vec.dtype)
            padded[: vec.size] = vec
            vec = padded
        return vec.reshape(coeff_shape, order=order)


__all__ = ["TensorPackCodecV1"]
