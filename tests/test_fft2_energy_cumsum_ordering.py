from __future__ import annotations

import numpy as np


def _required_components(cumsum: np.ndarray, *, threshold: float) -> int:
    values = np.asarray(cumsum, dtype=float).reshape(-1)
    meets = np.where(values >= float(threshold))[0]
    if meets.size == 0:
        return int(values.size)
    return int(meets[0] + 1)


def test_fft2_energy_cumsum_uses_frequency_radius_order() -> None:
    # Two low-frequency modes: +2 and -2 on the ky axis.
    # In unshifted FFT indexing, the -2 mode lives near the end (ky=H-2),
    # so a row-major cumsum would need ~O(H*W) components to reach it.
    from mode_decomp_ml.evaluate import coeff_energy_cumsum

    h = w = 64
    coeff_shape = [1, h, w, 2]  # (C,H,W,RI)
    meta = {
        "method": "fft2",
        "channels": 1,
        "coeff_shape": coeff_shape,
        "coeff_layout": "CHWRI",
        "complex_format": "real_imag",
        "flatten_order": "C",
        "fft_shift": False,
    }

    vec = np.zeros((1, int(np.prod(coeff_shape))), dtype=np.float32)

    def _set_coeff(ky: int, kx: int, *, real: float = 1.0, imag: float = 0.0) -> None:
        base = (((0 * h + int(ky)) * w + int(kx)) * 2)
        vec[0, base + 0] = float(real)
        vec[0, base + 1] = float(imag)

    _set_coeff(0, 2)
    _set_coeff(h - 2, 0)  # ky=-2

    cumsum = coeff_energy_cumsum(vec, coeff_meta=meta)
    req = _required_components(cumsum, threshold=0.9)

    # With frequency-radius ordering, this should be O(1..100), not ~O(H*W).
    assert req < 200

