import numpy as np

from mode_decomp_ml.viz import coeff_energy_spectrum


def test_coeff_energy_spectrum_1d_shape_returns_index() -> None:
    # Represents a 1D coefficient layout (e.g. autoencoder latent K).
    coeff_a = np.random.RandomState(0).randn(8, 16).astype(np.float64)
    meta = {"coeff_shape": [16], "flatten_order": "C", "complex_format": "real"}
    spec = coeff_energy_spectrum(coeff_a, meta)
    assert spec["kind"] == "index"
    assert np.asarray(spec["x"]).shape == (16,)
    assert np.asarray(spec["y"]).shape == (16,)


def test_coeff_energy_spectrum_2d_shape_returns_heatmap() -> None:
    coeff_a = np.random.RandomState(0).randn(8, 16).astype(np.float64)
    meta = {"coeff_shape": [4, 4], "flatten_order": "C", "complex_format": "real"}
    spec = coeff_energy_spectrum(coeff_a, meta)
    assert spec["kind"] == "heatmap"
    assert np.asarray(spec["data"]).shape == (4, 4)


def test_coeff_energy_spectrum_ck_returns_index_and_sums_channels() -> None:
    rs = np.random.RandomState(0)
    # 2 channels, 4 modes -> flattened vector length 8
    coeff_a = rs.randn(6, 8).astype(np.float64)
    meta = {
        "coeff_layout": "CK",
        "coeff_shape": [2, 4],
        "channels": 2,
        "flatten_order": "C",
        "complex_format": "real",
    }
    spec = coeff_energy_spectrum(coeff_a, meta)
    assert spec["kind"] == "index"
    y = np.asarray(spec["y"])
    assert y.shape == (4,)

    energy = np.mean(coeff_a**2, axis=0).reshape(2, 4, order="C").sum(axis=0)
    assert np.allclose(y, energy)
