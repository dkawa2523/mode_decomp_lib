from __future__ import annotations

import numpy as np

from mode_decomp_ml.domain import build_domain_spec
from mode_decomp_ml.plugins.decomposers import build_decomposer


def _rectangle_domain(height: int, width: int):
    cfg = {"name": "rectangle", "x_range": [0.0, 1.0], "y_range": [0.0, 1.0]}
    return build_domain_spec(cfg, (height, width))


def test_channelwise_adapter_fit_with_minimal_samples() -> None:
    rng = np.random.default_rng(0)
    fields = [rng.normal(size=(8, 6, 2)).astype(np.float64) for _ in range(5)]

    class _Dataset:
        def __len__(self) -> int:
            return len(fields)

        def __getitem__(self, idx: int):
            # Minimal sample: field+mask only (no cond/meta)
            return type("Sample", (), {"field": fields[idx], "mask": None})

    domain = _rectangle_domain(8, 6)
    decomposer = build_decomposer(
        {
            "name": "pod_em",
            "n_modes": 3,
            "n_iter": 2,
            "ridge_alpha": 1.0e-6,
            "init": "mean_fill",
            "inner_product": "euclidean",
            "mask_policy": "allow",
        }
    )
    decomposer.fit(dataset=_Dataset(), domain_spec=domain)
    coeff = decomposer.transform(fields[0], mask=None, domain_spec=domain)
    field_hat = decomposer.inverse_transform(coeff, domain_spec=domain)
    assert field_hat.shape == fields[0].shape

