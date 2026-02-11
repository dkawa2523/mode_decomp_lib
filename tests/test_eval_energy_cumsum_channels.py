from __future__ import annotations

import numpy as np

from mode_decomp_ml.evaluate import coeff_energy_cumsum


def test_coeff_energy_cumsum_aggregates_channels() -> None:
    rng = np.random.default_rng(0)
    n_samples = 5
    n_modes = 9

    coeff_1ch = rng.normal(size=(n_samples, n_modes)).astype(np.float64)
    meta_1ch = {
        "channels": 1,
        "coeff_shape": [1, n_modes],
        "coeff_layout": "CK",
        "complex_format": "real",
        "flatten_order": "C",
    }
    c1 = coeff_energy_cumsum(coeff_1ch, coeff_meta=meta_1ch)

    # Duplicate into 2 channels (vector field); energy cumsum should be identical after channel aggregation.
    coeff_2ch = np.concatenate([coeff_1ch, coeff_1ch], axis=1)
    meta_2ch = {
        "channels": 2,
        "coeff_shape": [2, n_modes],
        "coeff_layout": "CK",
        "complex_format": "real",
        "flatten_order": "C",
    }
    c2 = coeff_energy_cumsum(coeff_2ch, coeff_meta=meta_2ch)

    assert c1.shape == c2.shape
    assert np.allclose(c1, c2, atol=1e-12)

