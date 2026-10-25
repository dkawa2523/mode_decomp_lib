"""An isometric real coordinate system for conjugate-symmetric 2D FFTs."""

from __future__ import annotations

from ml_platform_mode_decomp.config import constructor_config

from functools import lru_cache
from typing import Any, Mapping

import numpy as np

from .basic import BaseCoeffCodec


@lru_cache(maxsize=64)
def _orbits(height: int, width: int):
    index = np.arange(height * width).reshape(height, width)
    partner = index[np.ix_((-np.arange(height)) % height, (-np.arange(width)) % width)].reshape(-1)
    own = np.arange(height * width)
    return own[own == partner], own[own < partner], partner[own < partner]


class RealFFTCodec(BaseCoeffCodec):
    """Keep DC/Nyquist real values and sqrt(2)*real/imag for one of each pair.

    The packed vector has exactly H*W*C real degrees of freedom. Every decoded
    vector is a valid FFT of a real field; a regressor cannot break conjugacy.
    """

    def __init__(self, *, cfg: Mapping[str, Any] | None = None, dtype_policy: Any = "float64") -> None:
        cfg = constructor_config(cfg, locals(), name="real_fft")
        self.name = "real_fft"
        self.is_lossless = True
        self.dtype_policy = str(cfg["dtype_policy"])
        if self.dtype_policy not in {"float32", "float64"}:
            raise ValueError("real_fft dtype_policy must be float32 or float64")

    @staticmethod
    def _shape(meta):
        shape = tuple(int(x) for x in meta["coeff_shape"])
        if len(shape) < 2 or meta.get("method") not in {"fft2", "fft2_lowpass"}:
            raise ValueError("real_fft requires Cartesian fft2/fft2_lowpass coefficients")
        return shape

    def encode(self, raw_coeff: Any, raw_meta: Mapping[str, Any]) -> np.ndarray:
        return self.encode_batch(np.asarray(raw_coeff)[None], raw_meta)[0]

    def encode_batch(self, raw_coefficients, raw_meta: Mapping[str, Any]) -> np.ndarray:
        shape = self._shape(raw_meta)
        values = np.asarray(raw_coefficients)
        if values.ndim != len(shape) + 1 or values.shape[1:] != shape or not np.isfinite(values).all():
            raise ValueError("FFT coefficients must match their finite raw shape")
        if raw_meta.get("fft_shift", False):
            values = np.fft.ifftshift(values, axes=(-2, -1))
        matrix = values.reshape(len(values), -1, shape[-2] * shape[-1])
        single, first, second = _orbits(*shape[-2:])
        scale = np.maximum(np.abs(matrix).max(axis=(1, 2)), np.finfo(float).tiny)
        tolerance = 2e-6 if values.dtype == np.complex64 else 1e-10
        if np.any(np.max(np.abs(matrix[..., single].imag), axis=(1, 2), initial=0) > tolerance * scale) or np.any(
            np.max(np.abs(matrix[..., first] - matrix[..., second].conj()), axis=(1, 2), initial=0) > tolerance * scale
        ):
            raise ValueError("real_fft requires a conjugate-symmetric real-field spectrum")
        packed = np.concatenate(
            [matrix[..., single].real, np.sqrt(2) * matrix[..., first].real, np.sqrt(2) * matrix[..., first].imag],
            axis=-1,
        )
        return packed.astype(self.dtype_policy, copy=False).reshape(len(values), -1)

    def decode(self, vector_coeff: np.ndarray, raw_meta: Mapping[str, Any]) -> np.ndarray:
        return self.decode_batch(np.asarray(vector_coeff).reshape(1, -1), raw_meta)[0]

    def decode_batch(self, vector_coefficients, raw_meta: Mapping[str, Any]) -> np.ndarray:
        shape = self._shape(raw_meta)
        vector = np.asarray(vector_coefficients)
        if (
            vector.dtype.kind not in "fiu"
            or vector.ndim != 2
            or vector.shape[1] != int(np.prod(shape))
            or not np.isfinite(vector).all()
        ):
            raise ValueError("real_fft vector must have the exact finite independent real dimension")
        matrix = vector.reshape(len(vector), -1, shape[-2] * shape[-1])
        single, first, second = _orbits(*shape[-2:])
        spectrum = np.zeros(matrix.shape, dtype=np.complex128)
        count, pairs = len(single), len(first)
        spectrum[..., single] = matrix[..., :count]
        spectrum[..., first] = (matrix[..., count : count + pairs] + 1j * matrix[..., count + pairs :]) / np.sqrt(2)
        spectrum[..., second] = spectrum[..., first].conj()
        spectrum = spectrum.reshape((len(vector), *shape))
        return np.fft.fftshift(spectrum, axes=(-2, -1)) if raw_meta.get("fft_shift", False) else spectrum

    @staticmethod
    def coordinate_metadata(raw_meta: Mapping[str, Any], indices: list[int]) -> list[dict[str, Any]]:
        """Describe requested packed coordinates, without constructing any basis fields.

        raw_index addresses the *stored* raw spectrum, including fftshift. A pair's
        imaginary coordinate synthesizes a negative sine under NumPy's inverse FFT.
        """
        shape = RealFFTCodec._shape(raw_meta)
        height, width = shape[-2:]
        per_channel = height * width
        single, first, second = _orbits(height, width)
        count, pairs = len(single), len(first)
        result = []

        def stored(flat, prefix):
            y, x = divmod(flat, width)
            if raw_meta.get("fft_shift", False):
                y, x = (y + height // 2) % height, (x + width // 2) % width
            return [*prefix, y, x]

        for index in indices:
            leading, local = divmod(index, per_channel)
            prefix = tuple(int(x) for x in np.unravel_index(leading, shape[:-2])) if shape[:-2] else ()
            if local < count:
                own, partner, component = int(single[local]), int(single[local]), "real"
            else:
                position = (local - count) % pairs
                own, partner = int(first[position]), int(second[position])
                component = "real" if local < count + pairs else "imag"

            result.append(
                {
                    "raw_index": stored(own, prefix),
                    "conjugate_raw_index": stored(partner, prefix),
                    "component": component,
                    "self_conjugate": own == partner,
                    "packing_scale": 1.0 if own == partner else float(np.sqrt(2)),
                }
            )
        return result

    def coeff_meta(self, raw_meta, vector_coeff):
        meta = dict(super().coeff_meta(raw_meta, vector_coeff))
        meta["codec"] = {
            **meta["codec"],
            "independent_real_coordinates": True,
            "norm_preserving": True,
            "coordinate_order": "channel_then_self_real_pair_real_pair_imag",
        }
        meta["raw_meta"] = dict(raw_meta or {})
        meta["complex_format"] = "real"
        meta["coeff_shape"] = [int(np.asarray(vector_coeff).size)]
        meta["coeff_layout"] = "K"
        return meta
