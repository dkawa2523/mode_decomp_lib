from __future__ import annotations

import numpy as np

from mode_decomp_ml.domain import build_domain_spec
from mode_decomp_ml.plugins.decomposers import build_decomposer


def _rectangle_domain(height: int, width: int):
    cfg = {"name": "rectangle", "x_range": [0.0, 1.0], "y_range": [0.0, 1.0]}
    return build_domain_spec(cfg, (height, width))


def test_pod_joint_em_fits_with_varying_masks_and_reconstructs() -> None:
    height, width = 17, 17
    domain = _rectangle_domain(height, width)
    F = height * width
    K = 3
    N = 8
    rng = np.random.default_rng(1)

    # Shared spatial basis across channels.
    Q, _ = np.linalg.qr(rng.normal(size=(F, K)))
    U = Q.astype(np.float64)
    A_u = rng.normal(size=(N, K)).astype(np.float64)
    A_v = 0.5 * A_u + 0.1 * rng.normal(size=(N, K)).astype(np.float64)
    X_u = (U @ A_u.T).T
    X_v = (U @ A_v.T).T

    masks = []
    fields = []
    for i in range(N):
        p = float(rng.uniform(0.6, 0.9))
        m = rng.random(size=F) < p
        if int(np.count_nonzero(m)) < K:
            m[:K] = True
        mask_hw = m.reshape(height, width).astype(bool)
        u_hw = X_u[i].reshape(height, width)
        v_hw = X_v[i].reshape(height, width)
        field = np.stack([np.where(mask_hw, u_hw, 0.0), np.where(mask_hw, v_hw, 0.0)], axis=-1).astype(
            np.float64
        )
        fields.append(field)
        masks.append(mask_hw)

    class _Dataset:
        def __len__(self) -> int:
            return N

        def __getitem__(self, idx: int):
            return type("Sample", (), {"field": fields[idx], "mask": masks[idx], "cond": None, "meta": {}})

    cfg_joint = {
        "name": "pod_joint_em",
        "n_modes": K,
        "n_iter": 12,
        "ridge_alpha": 1.0e-6,
        "init": "mean_fill",
        "inner_product": "euclidean",
        "mask_policy": "allow",
    }
    joint = build_decomposer(cfg_joint)
    joint.fit(dataset=_Dataset(), domain_spec=domain)
    coeff_j = joint.transform(fields[0], mask=masks[0], domain_spec=domain)
    recon_j = joint.inverse_transform(coeff_j, domain_spec=domain)
    assert recon_j.shape == fields[0].shape
    true0 = np.stack([X_u[0].reshape(height, width), X_v[0].reshape(height, width)], axis=-1)
    rmse_joint = float(np.sqrt(np.mean((recon_j - true0) ** 2)))
    assert rmse_joint < 1.0e-2

    # Compare against channel-wise pod_em baseline (should be comparable or slightly worse).
    cfg_ch = {
        "name": "pod_em",
        "n_modes": K,
        "n_iter": 12,
        "ridge_alpha": 1.0e-6,
        "init": "mean_fill",
        "inner_product": "euclidean",
        "mask_policy": "allow",
    }
    ch = build_decomposer(cfg_ch)
    ch.fit(dataset=_Dataset(), domain_spec=domain)
    coeff_c = ch.transform(fields[0], mask=masks[0], domain_spec=domain)
    recon_c = ch.inverse_transform(coeff_c, domain_spec=domain)
    rmse_ch = float(np.sqrt(np.mean((recon_c - true0) ** 2)))
    assert rmse_joint <= rmse_ch + 1.0e-3
