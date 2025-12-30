from __future__ import annotations

import numpy as np

from mode_decomp_ml.data.datasets import FieldSample
from mode_decomp_ml.decompose import build_decomposer
from mode_decomp_ml.domain import build_domain_spec


def _rectangle_domain(height: int, width: int):
    cfg = {"name": "rectangle", "x_range": [0.0, 1.0], "y_range": [0.0, 1.0]}
    return build_domain_spec(cfg, (height, width))


class _MaskDataset:
    def __init__(self, fields: list[np.ndarray], mask: np.ndarray) -> None:
        self._fields = fields
        self._mask = mask

    def __len__(self) -> int:
        return len(self._fields)

    def __getitem__(self, index: int) -> FieldSample:
        field = self._fields[index]
        return FieldSample(cond=np.zeros(1, dtype=np.float32), field=field, mask=self._mask, meta={})


def _graph_fourier_cfg(n_modes: int) -> dict[str, object]:
    return {
        "name": "graph_fourier",
        "n_modes": int(n_modes),
        "connectivity": 4,
        "laplacian_type": "combinatorial",
        "mask_policy": "require_mask",
        "solver": "dense",
        "dense_threshold": 512,
        "eigsh_tol": 1.0e-6,
        "eigsh_maxiter": 10000,
    }


def test_graph_fourier_roundtrip_masked() -> None:
    height, width = 6, 6
    mask = np.zeros((height, width), dtype=bool)
    mask[1:5, 1:5] = True
    rng = np.random.default_rng(0)
    field = rng.normal(size=(height, width, 1)).astype(np.float32)
    field[~mask] = 0.0

    domain = _rectangle_domain(height, width)
    decomposer = build_decomposer(_graph_fourier_cfg(int(mask.sum())))
    decomposer.fit(dataset=_MaskDataset([field], mask), domain_spec=domain)

    coeff = decomposer.transform(field, mask=mask, domain_spec=domain)
    field_hat = decomposer.inverse_transform(coeff, domain_spec=domain)

    assert field_hat.shape == field.shape
    assert np.allclose(field_hat, field, atol=1e-5)
    meta = decomposer.coeff_meta()
    assert meta["method"] == "graph_fourier"
    assert meta["coeff_shape"] == [1, int(mask.sum())]
    eigenvalues = np.asarray(meta["eigenvalues"], dtype=np.float64)
    assert eigenvalues.shape[0] == int(mask.sum())
    assert np.all(np.diff(eigenvalues) >= -1e-10)
