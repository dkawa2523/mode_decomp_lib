from __future__ import annotations

import numpy as np

from mode_decomp_ml.plugins.codecs import build_coeff_codec


def test_none_codec_roundtrip() -> None:
    raw = np.arange(12, dtype=np.float32).reshape(3, 4)
    raw_meta = {"coeff_shape": [3, 4], "flatten_order": "C"}
    codec = build_coeff_codec({"name": "none"})

    vec = codec.encode(raw, raw_meta)
    assert vec.ndim == 1
    assert vec.dtype == np.float32

    raw_back = codec.decode(vec, raw_meta)
    assert raw_back.shape == raw.shape
    assert np.allclose(raw_back, raw)


def test_fft_complex_codec_real_imag_roundtrip() -> None:
    rng = np.random.default_rng(0)
    raw = rng.normal(size=(2, 3)).astype(np.float32) + 1j * rng.normal(size=(2, 3)).astype(np.float32)
    raw_meta = {"coeff_shape": [2, 3], "flatten_order": "C", "complex_format": "complex"}
    codec = build_coeff_codec({"name": "fft_complex_codec_v1", "mode": "real_imag"})

    vec = codec.encode(raw, raw_meta)
    assert vec.ndim == 1
    assert vec.dtype == np.float32

    raw_back = codec.decode(vec, raw_meta)
    assert raw_back.shape == raw.shape
    assert np.allclose(raw_back, raw, atol=1e-6)


def test_sh_pack_codec_roundtrip() -> None:
    raw = np.arange(6, dtype=np.float32).reshape(2, 3)
    raw_meta = {
        "coeff_shape": [2, 3],
        "flatten_order": "C",
        "lm_kind_list": [[0, 0, "cos"], [1, 0, "cos"], [1, 1, "sin"]],
    }
    codec = build_coeff_codec({"name": "sh_pack_v1"})

    vec = codec.encode(raw, raw_meta)
    assert vec.ndim == 1
    assert vec.dtype == np.float32

    raw_back = codec.decode(vec, raw_meta)
    assert raw_back.shape == raw.shape
    assert np.allclose(raw_back, raw)


def test_slepian_pack_codec_roundtrip() -> None:
    raw = np.arange(8, dtype=np.float32).reshape(2, 4)
    raw_meta = {
        "coeff_shape": [2, 4],
        "flatten_order": "C",
        "eigenvalues": [0.9, 0.7, 0.5, 0.2],
    }
    codec = build_coeff_codec({"name": "slepian_pack_v1"})

    vec = codec.encode(raw, raw_meta)
    assert vec.ndim == 1
    assert vec.dtype == np.float32

    raw_back = codec.decode(vec, raw_meta)
    assert raw_back.shape == raw.shape
    assert np.allclose(raw_back, raw)


def test_tensor_pack_codec_roundtrip() -> None:
    raw = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    raw_meta = {"coeff_shape": [2, 3, 4], "flatten_order": "C"}
    codec = build_coeff_codec({"name": "tensor_pack_v1", "dtype_policy": "float32"})

    vec = codec.encode(raw, raw_meta)
    assert vec.ndim == 1
    assert vec.dtype == np.float32

    raw_back = codec.decode(vec, raw_meta)
    assert raw_back.shape == raw.shape
    assert np.allclose(raw_back, raw)
