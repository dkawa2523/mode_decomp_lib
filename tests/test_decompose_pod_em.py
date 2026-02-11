from __future__ import annotations

import numpy as np

from mode_decomp_ml.domain import build_domain_spec
from mode_decomp_ml.plugins.decomposers import build_decomposer


def _rectangle_domain(height: int, width: int):
    cfg = {"name": "rectangle", "x_range": [0.0, 1.0], "y_range": [0.0, 1.0]}
    return build_domain_spec(cfg, (height, width))


def test_pod_em_fits_with_varying_masks_and_reconstructs() -> None:
    height, width = 17, 17
    domain = _rectangle_domain(height, width)
    F = height * width
    K = 3
    N = 8
    rng = np.random.default_rng(0)

    # Orthonormal spatial basis.
    Q, _ = np.linalg.qr(rng.normal(size=(F, K)))
    U = Q.astype(np.float64)
    A = rng.normal(size=(N, K)).astype(np.float64)
    X_full = (U @ A.T).T  # (N,F)

    masks = []
    fields = []
    for i in range(N):
        # Random varying density, but ensure enough observed entries.
        p = float(rng.uniform(0.75, 0.95))
        m = rng.random(size=F) < p
        # Guarantee at least K observed entries.
        if int(np.count_nonzero(m)) < K:
            m[:K] = True
        mask_hw = m.reshape(height, width)
        x_hw = X_full[i].reshape(height, width)
        field_hw = np.where(mask_hw, x_hw, 0.0).astype(np.float64)
        fields.append(field_hw[..., None])
        masks.append(mask_hw.astype(bool))

    class _Dataset:
        def __len__(self) -> int:
            return N

        def __getitem__(self, idx: int):
            return type("Sample", (), {"field": fields[idx], "mask": masks[idx], "cond": None, "meta": {}})

    cfg = {
        "name": "pod_em",
        "n_modes": K,
        "n_iter": 20,
        "ridge_alpha": 1.0e-6,
        "init": "mean_fill",
        "inner_product": "euclidean",
        "mask_policy": "allow",
    }
    decomposer = build_decomposer(cfg)
    decomposer.fit(dataset=_Dataset(), domain_spec=domain)

    coeff = decomposer.transform(fields[0], mask=masks[0], domain_spec=domain)
    recon = decomposer.inverse_transform(coeff, domain_spec=domain)
    assert recon.shape == fields[0].shape

    true0 = X_full[0].reshape(height, width)
    rmse = float(np.sqrt(np.mean((recon[..., 0] - true0) ** 2)))
    assert rmse < 1.0e-2
