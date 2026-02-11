from __future__ import annotations

import numpy as np

from mode_decomp_ml.domain import build_domain_spec
from mode_decomp_ml.plugins.decomposers import build_decomposer


def _rectangle_domain(height: int, width: int):
    cfg = {"name": "rectangle", "x_range": [0.0, 1.0], "y_range": [0.0, 1.0]}
    return build_domain_spec(cfg, (height, width))


def _ggf_cfg(*, n_modes: int, ridge_alpha: float):
    return {
        "name": "gappy_graph_fourier",
        "n_modes": int(n_modes),
        "connectivity": 4,
        "laplacian_type": "combinatorial",
        "mask_policy": "allow_full",
        "solver": "dense",
        "dense_threshold": 4096,
        "eigsh_tol": 1.0e-6,
        "eigsh_maxiter": 2000,
        "ridge_alpha": float(ridge_alpha),
    }


def test_gappy_graph_fourier_recovers_coeffs_under_mask() -> None:
    domain = _rectangle_domain(17, 17)
    cfg = _ggf_cfg(n_modes=6, ridge_alpha=1e-8)
    decomposer = build_decomposer(cfg)
    decomposer.fit(domain_spec=domain)
    # Initialize coeff shape.
    _ = decomposer.transform(np.zeros((17, 17, 1), dtype=np.float64), mask=None, domain_spec=domain)

    rng = np.random.default_rng(0)
    coeff_true = rng.normal(size=(1, 6)).astype(np.float64)
    field = decomposer.inverse_transform(coeff_true, domain_spec=domain)

    # Varying sample mask.
    mask = rng.random((17, 17)) > 0.3
    field_obs = np.asarray(field).copy()
    field_obs[~mask] = 0.0

    coeff_hat = decomposer.transform(field_obs, mask=mask, domain_spec=domain)
    max_err = float(np.max(np.abs(np.asarray(coeff_hat) - coeff_true)))
    assert max_err < 1e-3

