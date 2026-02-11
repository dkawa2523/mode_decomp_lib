from __future__ import annotations

import numpy as np

from mode_decomp_ml.domain import build_domain_spec
from mode_decomp_ml.plugins.decomposers import build_decomposer


def _disk_domain(height: int, width: int):
    cfg = {
        "name": "disk",
        "center": [0.0, 0.0],
        "radius": 1.0,
        "x_range": [-1.0, 1.0],
        "y_range": [-1.0, 1.0],
    }
    return build_domain_spec(cfg, (height, width))


def _ds_cfg(*, n_modes: int, freq_radius: int, solver: str):
    return {
        "name": "disk_slepian",
        "n_modes": int(n_modes),
        "freq_radius": int(freq_radius),
        "solver": str(solver),
        "dense_threshold": 512,
        "eigsh_tol": 1.0e-6,
        "eigsh_maxiter": 2000,
        "mask_policy": "allow",
        "boundary_condition": "dirichlet",
    }


def test_disk_slepian_eigs_and_roundtrip_constant() -> None:
    domain = _disk_domain(17, 17)
    cfg = _ds_cfg(n_modes=6, freq_radius=3, solver="dense")
    decomposer = build_decomposer(cfg)
    decomposer.fit(domain_spec=domain)
    # Initialize coeff shape / field ndim via a transform call.
    _ = decomposer.transform(np.zeros((17, 17, 1), dtype=np.float64), mask=None, domain_spec=domain)
    # Roundtrip in coefficient space: coeff -> field -> coeff.
    coeff_true = np.zeros((1, 6), dtype=np.float64)
    coeff_true[0, 0] = 1.0
    coeff_true[0, 3] = -0.25
    field_hat = decomposer.inverse_transform(coeff_true, domain_spec=domain)
    coeff = decomposer.transform(field_hat, mask=None, domain_spec=domain)
    meta = decomposer.coeff_meta()

    eigvals = np.asarray(meta["eigenvalues"], dtype=np.float64)
    assert eigvals.ndim == 1
    assert len(eigvals) == 6
    assert np.all(np.isfinite(eigvals))
    assert np.all(eigvals <= 1.0 + 1e-6)
    assert np.all(eigvals >= -1e-6)
    # descending order
    assert np.all(eigvals[:-1] >= eigvals[1:] - 1e-12)

    assert coeff.shape == coeff_true.shape
    max_coeff_err = float(np.max(np.abs(np.asarray(coeff) - coeff_true)))
    assert max_coeff_err < 1e-6
