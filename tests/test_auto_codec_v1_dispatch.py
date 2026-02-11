from __future__ import annotations

import numpy as np


def test_auto_codec_wavelet_dispatch_roundtrip() -> None:
    from mode_decomp_ml.plugins.codecs import build_coeff_codec

    codec = build_coeff_codec({"name": "auto_codec_v1"})
    raw_meta = {
        "coeff_format": "wavedec2",
        "flatten_order": "C",
        "channels": 1,
        "coeff_structure": {
            "approx": [2, 2],
            "details": [
                [[2, 2], [2, 2], [2, 2]],
            ],
        },
    }
    cA = np.arange(4, dtype=np.float32).reshape(2, 2)
    cH = np.ones((2, 2), dtype=np.float32) * 2
    cV = np.ones((2, 2), dtype=np.float32) * 3
    cD = np.ones((2, 2), dtype=np.float32) * 4
    raw_coeff = [cA, (cH, cV, cD)]

    vec = codec.encode(raw_coeff, raw_meta)
    decoded = codec.decode(vec, raw_meta)
    # wavelet_pack returns list: [cA, (cH,cV,cD)]
    assert isinstance(decoded, list)
    assert np.allclose(decoded[0], cA)
    assert isinstance(decoded[1], tuple) and len(decoded[1]) == 3
    assert np.allclose(decoded[1][0], cH)
    assert np.allclose(decoded[1][1], cV)
    assert np.allclose(decoded[1][2], cD)


def test_auto_codec_complex_dispatch_roundtrip() -> None:
    from mode_decomp_ml.plugins.codecs import build_coeff_codec

    codec = build_coeff_codec({"name": "auto_codec_v1"})
    raw_meta = {"coeff_shape": [2, 3], "complex_format": "complex", "flatten_order": "C"}
    raw_coeff = (np.arange(6, dtype=np.float32).reshape(2, 3) + 1j * np.ones((2, 3), dtype=np.float32)).astype(
        np.complex64
    )
    vec = codec.encode(raw_coeff, raw_meta)
    decoded = codec.decode(vec, raw_meta)
    assert np.iscomplexobj(decoded)
    assert np.allclose(decoded, raw_coeff)


def test_auto_codec_real_fallback_roundtrip() -> None:
    from mode_decomp_ml.plugins.codecs import build_coeff_codec

    codec = build_coeff_codec({"name": "auto_codec_v1"})
    raw_meta = {"coeff_shape": [4, 5], "flatten_order": "C"}
    raw_coeff = np.arange(20, dtype=np.float32).reshape(4, 5)
    vec = codec.encode(raw_coeff, raw_meta)
    decoded = codec.decode(vec, raw_meta)
    assert np.allclose(decoded, raw_coeff)

