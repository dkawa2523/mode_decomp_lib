from __future__ import annotations

import numpy as np

from ml_platform_mode_decomp.data.schema import FieldSample
from ml_platform_mode_decomp.decomposition import build_decomposer
from ml_platform_mode_decomp.geometry import build_domain_spec


def _rectangle_domain(height: int, width: int):
    cfg = {"name": "rectangle", "x_range": [0.0, 1.0], "y_range": [0.0, 1.0]}
    return build_domain_spec(cfg, (height, width))


class _ArrayDataset:
    def __init__(self, fields: list[np.ndarray]) -> None:
        self._fields = fields

    def __len__(self) -> int:
        return len(self._fields)

    def __getitem__(self, index: int) -> FieldSample:
        field = self._fields[index]
        return FieldSample(cond=np.zeros(1, dtype=np.float32), field=field, mask=None, meta={})


def test_dict_learning_sparse_coeffs() -> None:
    rng = np.random.default_rng(0)
    fields = [rng.normal(size=(8, 8, 1)).astype(np.float32) for _ in range(6)]
    domain = _rectangle_domain(8, 8)

    decomposer = build_decomposer(
        {
            "name": "dict_learning",
            "mask_policy": "error",
            "n_components": 6,
            "alpha": 1.0,
            "max_iter": 50,
            "tol": 1.0e-6,
            "fit_algorithm": "lars",
            "transform_algorithm": "omp",
            "transform_n_nonzero_coefs": 2,
            "positive_code": False,
            "positive_dict": False,
            "seed": 0,
        }
    )
    decomposer.fit(dataset=_ArrayDataset(fields), domain_spec=domain)
    coeff = decomposer.transform(fields[0], mask=None, domain_spec=domain)
    field_hat = decomposer.inverse_transform(coeff, domain_spec=domain)

    assert field_hat.shape == fields[0].shape
    assert np.isfinite(field_hat).all()

    meta = decomposer.coeff_meta()
    assert meta["method"] == "dict_learning"
    coeff_shape = tuple(meta["coeff_shape"])
    coeff_tensor = np.asarray(coeff)
    if coeff_tensor.shape != coeff_shape:
        coeff_tensor = coeff_tensor.reshape(coeff_shape, order="C")
    nonzero = np.count_nonzero(np.abs(coeff_tensor) > 1.0e-8)
    assert nonzero <= coeff_tensor.shape[0] * 2


def test_requested_omp_remains_sparse_with_more_atoms_than_training_samples():
    fields = np.random.default_rng(19).normal(size=(8, 4, 4, 1))
    domain = _rectangle_domain(4, 4)
    decomposer = build_decomposer(
        {
            "name": "dict_learning",
            "n_components": 12,
            "max_iter": 10,
            "transform_algorithm": "omp",
            "transform_n_nonzero_coefs": 2,
            "seed": 8,
        }
    ).fit(_ArrayDataset(list(fields)), domain)
    codes = np.stack([decomposer.transform(field, None, domain) for field in fields])
    assert decomposer.coeff_meta()["transform_algorithm"] == "omp"
    assert np.isfinite(codes).all()
    assert np.max(np.count_nonzero(np.abs(codes) > 1e-10, axis=-1)) <= 2
