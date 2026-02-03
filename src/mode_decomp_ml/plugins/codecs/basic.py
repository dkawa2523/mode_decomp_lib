"""Coeff codec implementations."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np

from mode_decomp_ml.config import cfg_get
from mode_decomp_ml.pipeline import write_json
from mode_decomp_ml.plugins.registry import register_coeff_codec


def _resolve_coeff_shape(raw_meta: Mapping[str, Any] | None) -> tuple[int, ...] | None:
    if raw_meta is None:
        return None
    coeff_shape = raw_meta.get("coeff_shape")
    if isinstance(coeff_shape, (list, tuple)) and coeff_shape:
        return tuple(int(x) for x in coeff_shape)
    return None


def _resolve_flatten_order(raw_meta: Mapping[str, Any] | None) -> str:
    if raw_meta is None:
        return "C"
    order = str(raw_meta.get("flatten_order", "C")).strip().upper()
    return order or "C"


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
        padded = np.zeros(expected, dtype=vec.dtype)
        padded[: vec.size] = vec
        vec = padded
    order = _resolve_flatten_order(raw_meta)
    return vec.reshape(coeff_shape, order=order)


class BaseCoeffCodec:
    """Minimal coeff codec interface."""

    name: str
    is_lossless: bool = False
    dtype_policy: str = "float32"

    def __init__(self, *, cfg: Mapping[str, Any]) -> None:
        self._cfg = dict(cfg)
        self._state: Dict[str, Any] | None = None

    def encode(self, raw_coeff: Any, raw_meta: Mapping[str, Any]) -> np.ndarray:
        raise NotImplementedError

    def decode(self, vector_coeff: np.ndarray, raw_meta: Mapping[str, Any]) -> Any:
        raise NotImplementedError

    def state(self) -> Mapping[str, Any]:
        if self._state is None:
            self._state = {
                "method": self.name,
                "is_lossless": bool(self.is_lossless),
                "dtype_policy": self.dtype_policy,
            }
        return self._state

    def save_state(self, run_dir: str | Path) -> Path:
        out_dir = Path(run_dir) / "states" / "coeff_codec"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "state.pkl"
        with path.open("wb") as fh:
            pickle.dump(self, fh)
        return path

    @staticmethod
    def load_state(path: str | Path) -> "BaseCoeffCodec":
        obj = load_pickle_compat(path)
        if not isinstance(obj, BaseCoeffCodec):
            raise TypeError("Loaded coeff codec state is not a BaseCoeffCodec")
        return obj

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

    def save_coeff_meta(
        self,
        run_dir: str | Path,
        raw_meta: Mapping[str, Any] | None,
        vector_coeff: np.ndarray,
    ) -> Path:
        meta = self.coeff_meta(raw_meta, vector_coeff)
        run_root = Path(run_dir)
        states_dir = run_root / "states"
        states_dir.mkdir(parents=True, exist_ok=True)
        primary = states_dir / "coeff_meta.json"
        write_json(primary, meta)
        codec_dir = states_dir / "coeff_codec"
        codec_dir.mkdir(parents=True, exist_ok=True)
        write_json(codec_dir / "coeff_meta.json", meta)
        return primary


@register_coeff_codec("none")
class NoOpCoeffCodec(BaseCoeffCodec):
    """No-op coeff codec (flatten + float32)."""

    def __init__(self, *, cfg: Mapping[str, Any]) -> None:
        super().__init__(cfg=cfg)
        self.name = "none"
        self.is_lossless = True
        self.dtype_policy = str(cfg_get(cfg, "dtype_policy", "float32"))

    def encode(self, raw_coeff: Any, raw_meta: Mapping[str, Any]) -> np.ndarray:
        vec = _vectorize_raw(raw_coeff, raw_meta)
        return np.asarray(vec, dtype=np.float32).reshape(-1)

    def decode(self, vector_coeff: np.ndarray, raw_meta: Mapping[str, Any]) -> Any:
        vec = np.asarray(vector_coeff, dtype=np.float32).reshape(-1)
        return _devectorize_raw(vec, raw_meta)


from mode_decomp_ml.pickle_compat import load_pickle_compat

__all__ = [
    "BaseCoeffCodec",
    "NoOpCoeffCodec",
]
