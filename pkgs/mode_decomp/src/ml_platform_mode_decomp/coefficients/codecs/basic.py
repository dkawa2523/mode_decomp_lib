"""Coeff codec implementations."""

from __future__ import annotations

from ml_platform_mode_decomp.config import constructor_config

from typing import Any, Literal, Mapping

import numpy as np


def _resolve_coeff_shape(raw_meta: Mapping[str, Any] | None) -> tuple[int, ...] | None:
    if raw_meta is None:
        return None
    coeff_shape = raw_meta.get("coeff_shape")
    if isinstance(coeff_shape, (list, tuple)) and coeff_shape:
        return tuple(int(x) for x in coeff_shape)
    return None


FlattenOrder = Literal["A", "C", "F"]


def _resolve_flatten_order(raw_meta: Mapping[str, Any] | None) -> FlattenOrder:
    if raw_meta is None:
        return "C"
    order = str(raw_meta.get("flatten_order", "C")).strip().upper()
    if not order or order == "C":
        return "C"
    if order == "F":
        return "F"
    if order == "A":
        return "A"
    raise ValueError(f"flatten_order must be one of A, C, F; got {order}")


def _vectorize_raw(raw_coeff: Any, raw_meta: Mapping[str, Any] | None) -> np.ndarray:
    if isinstance(raw_coeff, Mapping):
        raise TypeError("raw_coeff mapping requires a specialized codec")
    arr = np.asarray(raw_coeff)
    if np.iscomplexobj(arr):
        raise ValueError("raw_coeff must be real-valued for the none codec")
    order = _resolve_flatten_order(raw_meta)
    coeff_shape = _resolve_coeff_shape(raw_meta)
    if coeff_shape is not None:
        expected = int(np.prod(coeff_shape))
        if arr.size != expected:
            raise ValueError("raw_coeff size does not match raw_meta.coeff_shape")
        if tuple(arr.shape) != coeff_shape:
            arr = arr.reshape(coeff_shape, order=order)
    return arr.reshape(-1, order=order)


def _devectorize_raw(vector_coeff: np.ndarray, raw_meta: Mapping[str, Any] | None) -> np.ndarray:
    vec = np.asarray(vector_coeff).reshape(-1)
    coeff_shape = _resolve_coeff_shape(raw_meta)
    if coeff_shape is None:
        return vec
    expected = int(np.prod(coeff_shape))
    if vec.size > expected:
        raise ValueError("vector_coeff size exceeds raw_meta.coeff_shape")
    if vec.size < expected:
        raise ValueError("Coefficient size is smaller than the fitted layout")
    order = _resolve_flatten_order(raw_meta)
    return vec.reshape(coeff_shape, order=order)


class BaseCoeffCodec:
    """Minimal coeff codec interface."""

    name: str
    is_lossless: bool = False
    dtype_policy: str = "float32"

    def fit(self, raw_coeff: Any, raw_meta: Mapping[str, Any]) -> "BaseCoeffCodec":
        return self

    def encode(self, raw_coeff: Any, raw_meta: Mapping[str, Any]) -> np.ndarray:
        raise NotImplementedError

    def decode(self, vector_coeff: np.ndarray, raw_meta: Mapping[str, Any]) -> Any:
        raise NotImplementedError

    def encode_batch(self, raw_coefficients, raw_meta: Mapping[str, Any]) -> np.ndarray:
        return np.stack([self.encode(coeff, raw_meta) for coeff in raw_coefficients])

    def decode_batch(self, vector_coefficients: np.ndarray, raw_meta: Mapping[str, Any]) -> list[Any] | np.ndarray:
        return [self.decode(coeff, raw_meta) for coeff in vector_coefficients]

    def coeff_meta(
        self,
        raw_meta: Mapping[str, Any] | None,
        vector_coeff: np.ndarray,
    ) -> Mapping[str, Any]:
        meta = dict(raw_meta or {})
        meta["codec"] = {
            "name": self.name,
            "is_lossless": bool(self.is_lossless),
            "dtype_policy": self.dtype_policy,
        }
        meta["vector_dim"] = int(np.asarray(vector_coeff).reshape(-1).shape[0])
        return meta


class NoOpCoeffCodec(BaseCoeffCodec):
    """Flatten real coefficients using the configured float32/float64 dtype."""

    def __init__(self, *, cfg: Mapping[str, Any] | None = None, dtype_policy: Any = "float64") -> None:
        cfg = constructor_config(cfg, locals(), name="none")
        self.name = "none"
        self.is_lossless = True
        self.dtype_policy = str(cfg["dtype_policy"])
        if self.dtype_policy not in {"float32", "float64"}:
            raise ValueError("dtype_policy must be float32 or float64")
        self._float_dtype = np.dtype(self.dtype_policy)

    def encode(self, raw_coeff: Any, raw_meta: Mapping[str, Any]) -> np.ndarray:
        vec = _vectorize_raw(raw_coeff, raw_meta)
        return np.asarray(vec, dtype=self._float_dtype).reshape(-1)

    def decode(self, vector_coeff: np.ndarray, raw_meta: Mapping[str, Any]) -> Any:
        vec = np.asarray(vector_coeff, dtype=self._float_dtype).reshape(-1)
        return _devectorize_raw(vec, raw_meta)

    def encode_batch(self, raw_coefficients, raw_meta: Mapping[str, Any]) -> np.ndarray:
        if _resolve_flatten_order(raw_meta) != "C":
            return super().encode_batch(raw_coefficients, raw_meta)
        values = np.asarray(raw_coefficients)
        if np.iscomplexobj(values):
            raise ValueError("raw_coeff must be real-valued for the none codec")
        shape = _resolve_coeff_shape(raw_meta)
        if shape is not None and int(np.prod(values.shape[1:])) != int(np.prod(shape)):
            raise ValueError("raw_coeff size does not match raw_meta.coeff_shape")
        dimension = int(np.prod(values.shape[1:]))
        return np.asarray(values, dtype=self._float_dtype).reshape(len(values), dimension)

    def decode_batch(self, vector_coefficients: np.ndarray, raw_meta: Mapping[str, Any]):
        if _resolve_flatten_order(raw_meta) != "C":
            return super().decode_batch(vector_coefficients, raw_meta)
        values = np.asarray(vector_coefficients, dtype=self._float_dtype)
        shape = _resolve_coeff_shape(raw_meta)
        if shape is None:
            return values
        if values.ndim != 2 or values.shape[1] != int(np.prod(shape)):
            raise ValueError("Coefficient batch does not match the fitted layout")
        return values.reshape((len(values), *shape))


__all__ = [
    "BaseCoeffCodec",
    "NoOpCoeffCodec",
]
