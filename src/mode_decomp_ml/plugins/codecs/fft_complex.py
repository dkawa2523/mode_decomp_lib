"""FFT complex coefficient codec."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from mode_decomp_ml.config import cfg_get
from mode_decomp_ml.plugins.registry import register_coeff_codec

from .basic import BaseCoeffCodec, _resolve_coeff_shape, _resolve_flatten_order


def _normalize_mode(mode: str) -> str:
    value = str(mode).strip().lower()
    aliases = {
        "ri": "real_imag",
        "realimag": "real_imag",
        "real-imag": "real_imag",
        "magphase": "mag_phase",
        "mag-phase": "mag_phase",
        "logmagphase": "logmag_phase",
        "logmag-phase": "logmag_phase",
    }
    value = aliases.get(value, value)
    if value not in {"real_imag", "mag_phase", "logmag_phase"}:
        raise ValueError(f"fft_complex_codec_v1.mode must be one of real_imag/mag_phase/logmag_phase, got {mode}")
    return value


def _resolve_float_dtype(policy: str) -> np.dtype:
    value = str(policy or "").strip().lower()
    if value in {"float32", "fp32", "f32", ""}:
        return np.float32
    if value in {"float64", "fp64", "f64", "double"}:
        return np.float64
    raise ValueError(f"Unsupported dtype_policy: {policy}")


def _reshape_vector(vec: np.ndarray, shape: tuple[int, ...], order: str, *, name: str) -> np.ndarray:
    flat = np.asarray(vec).reshape(-1)
    expected = int(np.prod(shape))
    if flat.size > expected:
        raise ValueError(f"{name} size {flat.size} exceeds expected {expected}")
    if flat.size < expected:
        padded = np.zeros(expected, dtype=flat.dtype)
        padded[: flat.size] = flat
        flat = padded
    return flat.reshape(shape, order=order)


def _ensure_raw_shape(raw_coeff: np.ndarray, raw_shape: tuple[int, ...] | None, order: str) -> np.ndarray:
    if raw_shape is None:
        return raw_coeff
    if raw_coeff.shape == raw_shape:
        return raw_coeff
    expected = int(np.prod(raw_shape))
    if raw_coeff.size != expected:
        raise ValueError("raw_coeff size does not match raw_meta.coeff_shape")
    return raw_coeff.reshape(raw_shape, order=order)


def _unwrap_phase(phase: np.ndarray) -> np.ndarray:
    if phase.ndim == 0:
        return phase
    unwrapped = np.unwrap(phase, axis=-1)
    if phase.ndim >= 2:
        unwrapped = np.unwrap(unwrapped, axis=-2)
    return unwrapped


def _complex_shape_from_meta(raw_meta: Mapping[str, Any] | None) -> tuple[tuple[int, ...], tuple[int, ...]]:
    raw_shape = _resolve_coeff_shape(raw_meta)
    if raw_shape is None:
        raise ValueError("raw_meta.coeff_shape is required for fft_complex_codec_v1")
    complex_format = str(raw_meta.get("complex_format", "")).strip().lower() if raw_meta else ""
    if complex_format == "real_imag":
        if not raw_shape or raw_shape[-1] != 2:
            raise ValueError("raw_meta.coeff_shape must end with 2 for real_imag format")
        return tuple(raw_shape[:-1]), tuple(raw_shape)
    return tuple(raw_shape), tuple(raw_shape)


@register_coeff_codec("fft_complex_codec_v1")
class FFTComplexCodecV1(BaseCoeffCodec):
    """Codec for FFT complex coefficients (real/imag or mag/phase)."""

    def __init__(self, *, cfg: Mapping[str, Any]) -> None:
        super().__init__(cfg=cfg)
        self.name = "fft_complex_codec_v1"
        self.mode = _normalize_mode(cfg_get(cfg, "mode", "real_imag"))
        self.phase_unwrap = bool(cfg_get(cfg, "phase_unwrap", False))
        phase_clip = cfg_get(cfg, "phase_clip", None)
        self.phase_clip = None if phase_clip in (None, "", "null") else float(phase_clip)
        self.log_epsilon = float(cfg_get(cfg, "log_epsilon", 1e-12))
        self.dtype_policy = str(cfg_get(cfg, "dtype_policy", "float32"))
        self._float_dtype = _resolve_float_dtype(self.dtype_policy)
        self._complex_dtype = np.complex64 if self._float_dtype == np.float32 else np.complex128
        self.is_lossless = self.mode == "real_imag"

    def _vector_shape_from_meta(self, raw_meta: Mapping[str, Any] | None) -> tuple[int, ...] | None:
        raw_shape = _resolve_coeff_shape(raw_meta)
        if raw_shape is None:
            return None
        complex_format = str(raw_meta.get("complex_format", "")).strip().lower() if raw_meta else ""
        if complex_format == "real_imag":
            return tuple(raw_shape)
        return tuple(raw_shape) + (2,)

    def _vector_layout(self, raw_layout: str, raw_meta: Mapping[str, Any] | None) -> str:
        layout = str(raw_layout or "")
        if not layout:
            return layout
        complex_format = str(raw_meta.get("complex_format", "")).strip().lower() if raw_meta else ""
        suffix = "RI" if self.mode == "real_imag" else "MP" if self.mode == "mag_phase" else "LP"
        if complex_format == "real_imag":
            if self.mode == "real_imag":
                return layout
            if layout.endswith("RI"):
                return f"{layout[:-2]}{suffix}"
            return f"{layout}{suffix}"
        if layout.endswith(suffix):
            return layout
        return f"{layout}{suffix}"

    def _codec_meta(self) -> dict[str, Any]:
        meta: dict[str, Any] = {
            "name": self.name,
            "is_lossless": bool(self.is_lossless),
            "dtype_policy": self.dtype_policy,
            "mode": self.mode,
            "phase_unwrap": bool(self.phase_unwrap),
            "phase_clip": self.phase_clip,
        }
        if self.mode == "logmag_phase":
            meta["log_epsilon"] = float(self.log_epsilon)
        return meta

    def encode(self, raw_coeff: Any, raw_meta: Mapping[str, Any]) -> np.ndarray:
        order = _resolve_flatten_order(raw_meta)
        raw_shape = _resolve_coeff_shape(raw_meta)
        raw_arr = np.asarray(raw_coeff)
        raw_arr = _ensure_raw_shape(raw_arr, raw_shape, order)

        complex_format = str(raw_meta.get("complex_format", "")).strip().lower() if raw_meta else ""
        if np.iscomplexobj(raw_arr):
            complex_arr = raw_arr
        else:
            if complex_format != "real_imag":
                raise ValueError("fft_complex_codec_v1 expects complex raw_coeff or real_imag format")
            if raw_arr.shape[-1] != 2:
                raise ValueError("real_imag raw_coeff must have last dimension 2")
            complex_arr = raw_arr[..., 0] + 1j * raw_arr[..., 1]

        if raw_shape is not None:
            if complex_format == "real_imag":
                complex_shape = tuple(raw_shape[:-1])
            else:
                complex_shape = tuple(raw_shape)
            if complex_arr.shape != complex_shape:
                expected = int(np.prod(complex_shape))
                if complex_arr.size != expected:
                    raise ValueError("raw_coeff size does not match raw_meta.coeff_shape")
                complex_arr = complex_arr.reshape(complex_shape, order=order)

        if self.mode == "real_imag":
            comp_a = complex_arr.real
            comp_b = complex_arr.imag
        else:
            comp_a = np.abs(complex_arr)
            if self.mode == "logmag_phase":
                comp_a = np.log(np.clip(comp_a, self.log_epsilon, None))
            comp_b = np.angle(complex_arr)
            if self.phase_unwrap:
                comp_b = _unwrap_phase(comp_b)
            if self.phase_clip is not None:
                comp_b = np.clip(comp_b, -self.phase_clip, self.phase_clip)

        vector_tensor = np.stack((comp_a, comp_b), axis=-1)
        return np.asarray(vector_tensor, dtype=self._float_dtype).reshape(-1, order=order)

    def decode(self, vector_coeff: np.ndarray, raw_meta: Mapping[str, Any]) -> Any:
        order = _resolve_flatten_order(raw_meta)
        complex_shape, raw_shape = _complex_shape_from_meta(raw_meta)
        vector_shape = complex_shape + (2,)
        vec = _reshape_vector(vector_coeff, vector_shape, order, name="vector_coeff")
        comp_a = vec[..., 0]
        comp_b = vec[..., 1]

        if self.mode == "real_imag":
            complex_arr = comp_a + 1j * comp_b
        else:
            mag = np.clip(comp_a, 0.0, None)
            if self.mode == "logmag_phase":
                mag = np.exp(comp_a)
            complex_arr = mag * np.exp(1j * comp_b)

        complex_arr = np.asarray(complex_arr, dtype=self._complex_dtype)
        complex_format = str(raw_meta.get("complex_format", "")).strip().lower() if raw_meta else ""
        if complex_format == "real_imag":
            raw_arr = np.stack((complex_arr.real, complex_arr.imag), axis=-1)
            if raw_arr.shape != raw_shape:
                raw_arr = _reshape_vector(raw_arr, raw_shape, order, name="raw_coeff")
            return raw_arr
        if complex_arr.shape != raw_shape:
            complex_arr = _reshape_vector(complex_arr, raw_shape, order, name="raw_coeff")
        return complex_arr

    def coeff_meta(
        self,
        raw_meta: Mapping[str, Any] | None,
        vector_coeff: np.ndarray,
    ) -> Mapping[str, Any]:
        base_raw = dict(raw_meta or {})
        meta = dict(base_raw)
        meta["raw_meta"] = base_raw
        meta["codec"] = self._codec_meta()
        meta["vector_dim"] = int(np.asarray(vector_coeff).reshape(-1).shape[0])
        vector_shape = self._vector_shape_from_meta(base_raw)
        if vector_shape is not None:
            meta["coeff_shape"] = [int(x) for x in vector_shape]
            meta["coeff_layout"] = self._vector_layout(base_raw.get("coeff_layout", ""), base_raw)
        meta["complex_format"] = self.mode
        return meta


__all__ = ["FFTComplexCodecV1"]
