from __future__ import annotations

import numpy as np

from mode_decomp_ml.domain import build_domain_spec
from mode_decomp_ml.plugins.decomposers import build_decomposer


def _rectangle_domain(height: int, width: int):
    cfg = {"name": "rectangle", "x_range": [0.0, 1.0], "y_range": [0.0, 1.0]}
    return build_domain_spec(cfg, (height, width))


class _Dataset:
    def __init__(self, fields: list[np.ndarray], masks: list[np.ndarray | None]):
        self._fields = fields
        self._masks = masks

    def __len__(self) -> int:
        return len(self._fields)

    def __getitem__(self, idx: int):
        return type("Sample", (), {"field": self._fields[idx], "mask": self._masks[idx]})


def test_pod_joint_vector_rank2_reconstruction() -> None:
    domain = _rectangle_domain(9, 9)
    rng = np.random.default_rng(0)
    n = 10
    # Two shared spatial patterns across both channels.
    xx = domain.coords["x"]
    yy = domain.coords["y"]
    p1 = np.sin(2 * np.pi * xx) * np.cos(2 * np.pi * yy)
    p2 = np.cos(2 * np.pi * xx) * np.sin(2 * np.pi * yy)
    fields = []
    for _ in range(n):
        a = rng.normal(size=2)
        u = a[0] * p1 + a[1] * p2
        # Make v a fixed linear mapping of u-coeffs so the joint state is rank-2.
        v = 0.3 * a[0] * p1 - 0.7 * a[1] * p2
        field = np.stack([u, v], axis=-1).astype(np.float64)
        fields.append(field)

    cfg = {"name": "pod_joint", "n_modes": 2, "mask_policy": "error", "inner_product": "euclidean"}
    decomposer = build_decomposer(cfg)
    decomposer.fit(dataset=_Dataset(fields, [None] * len(fields)), domain_spec=domain)
    coeff = decomposer.transform(fields[0], mask=None, domain_spec=domain)
    field_hat = decomposer.inverse_transform(coeff, domain_spec=domain)

    err = float(np.sqrt(np.mean((field_hat - fields[0]) ** 2)))
    assert err < 1e-8


def test_pod_joint_zero_fill_smoke() -> None:
    domain = _rectangle_domain(9, 9)
    rng = np.random.default_rng(1)
    fields = [rng.normal(size=(9, 9, 2)).astype(np.float64) for _ in range(4)]
    masks = [(rng.random((9, 9)) > 0.3) for _ in range(4)]
    fields_obs = []
    for f, m in zip(fields, masks):
        fo = f.copy()
        fo[~m] = 0.0
        fields_obs.append(fo)

    cfg = {"name": "pod_joint", "n_modes": 3, "mask_policy": "zero_fill", "inner_product": "euclidean"}
    decomposer = build_decomposer(cfg)
    decomposer.fit(dataset=_Dataset(fields_obs, masks), domain_spec=domain)
    coeff = decomposer.transform(fields_obs[0], mask=masks[0], domain_spec=domain)
    field_hat = decomposer.inverse_transform(coeff, domain_spec=domain)
    assert field_hat.shape == fields_obs[0].shape
