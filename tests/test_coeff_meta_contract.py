from __future__ import annotations

import numpy as np

from mode_decomp_ml.domain import build_domain_spec
from mode_decomp_ml.plugins.decomposers import build_decomposer


_REQUIRED_KEYS = [
    "method",
    "field_shape",
    "field_ndim",
    "field_layout",
    "channels",
    "coeff_shape",
    "coeff_layout",
    "flatten_order",
    "complex_format",
    "keep",
]


def _assert_required(meta: dict) -> None:
    missing = [key for key in _REQUIRED_KEYS if key not in meta]
    assert not missing, f"coeff_meta missing keys: {missing}"


def _rectangle_domain(height: int, width: int):
    cfg = {"name": "rectangle", "x_range": [0.0, 1.0], "y_range": [0.0, 1.0]}
    return build_domain_spec(cfg, (height, width))


def test_coeff_meta_contract_fft2() -> None:
    domain = _rectangle_domain(4, 4)
    field = np.zeros((4, 4, 1), dtype=np.float32)
    decomposer = build_decomposer({"name": "fft2", "disk_policy": "error"})
    _ = decomposer.transform(field, mask=None, domain_spec=domain)
    meta = decomposer.coeff_meta()
    _assert_required(meta)


def test_coeff_meta_contract_pod_svd() -> None:
    domain = _rectangle_domain(4, 4)
    fields = [np.zeros((4, 4, 1), dtype=np.float32) for _ in range(3)]

    class _Dataset:
        def __len__(self) -> int:
            return len(fields)

        def __getitem__(self, idx: int):
            return type("Sample", (), {"field": fields[idx], "mask": None})

    decomposer = build_decomposer({"name": "pod_svd", "mask_policy": "error"})
    decomposer.fit(dataset=_Dataset(), domain_spec=domain)
    _ = decomposer.transform(fields[0], mask=None, domain_spec=domain)
    meta = decomposer.coeff_meta()
    _assert_required(meta)


def test_coeff_meta_contract_graph_fourier() -> None:
    domain = _rectangle_domain(4, 4)
    field = np.zeros((4, 4, 1), dtype=np.float32)
    cfg = {
        "name": "graph_fourier",
        "n_modes": 4,
        "connectivity": 4,
        "laplacian_type": "combinatorial",
        "mask_policy": "allow_full",
        "solver": "dense",
        "dense_threshold": 64,
        "eigsh_tol": 1.0e-6,
        "eigsh_maxiter": 1000,
    }
    decomposer = build_decomposer(cfg)
    decomposer.fit(domain_spec=domain)
    _ = decomposer.transform(field, mask=None, domain_spec=domain)
    meta = decomposer.coeff_meta()
    _assert_required(meta)


def test_coeff_meta_contract_laplace_beltrami() -> None:
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    domain = build_domain_spec({"name": "mesh", "vertices": vertices, "faces": faces}, (4, 1, 1))
    field = np.zeros((4, 1, 1), dtype=np.float64)
    cfg = {
        "name": "laplace_beltrami",
        "n_modes": 4,
        "laplacian_type": "cotangent",
        "mass_type": "lumped",
        "boundary_condition": "neumann",
        "mask_policy": "allow",
        "solver": "dense",
        "dense_threshold": 32,
        "eigsh_tol": 1.0e-6,
        "eigsh_maxiter": 1000,
    }
    decomposer = build_decomposer(cfg)
    decomposer.fit(domain_spec=domain)
    _ = decomposer.transform(field, mask=None, domain_spec=domain)
    meta = decomposer.coeff_meta()
    _assert_required(meta)


def test_coeff_meta_contract_gappy_pod() -> None:
    domain = _rectangle_domain(4, 4)
    base = np.arange(16, dtype=np.float32).reshape(4, 4, 1)
    fields = [base + float(idx) for idx in range(3)]
    mask = np.ones((4, 4), dtype=bool)

    class _Dataset:
        def __len__(self) -> int:
            return len(fields)

        def __getitem__(self, idx: int):
            return type("Sample", (), {"field": fields[idx], "mask": None})

    cfg = {
        "name": "gappy_pod",
        "mask_policy": "require",
        "backend": "sklearn",
        "solver": "direct",
        "n_modes": 2,
    }
    decomposer = build_decomposer(cfg)
    decomposer.fit(dataset=_Dataset(), domain_spec=domain)
    _ = decomposer.transform(fields[0], mask=mask, domain_spec=domain)
    meta = decomposer.coeff_meta()
    _assert_required(meta)


def test_coeff_meta_contract_wavelet2d() -> None:
    import pytest
    pytest.importorskip("pywt")
    domain = _rectangle_domain(4, 4)
    field = np.zeros((4, 4, 1), dtype=np.float32)
    cfg = {
        "name": "wavelet2d",
        "wavelet": "db1",
        "mask_policy": "error",
        "level": 1,
        "mode": "symmetric",
    }
    decomposer = build_decomposer(cfg)
    _ = decomposer.transform(field, mask=None, domain_spec=domain)
    meta = decomposer.coeff_meta()
    _assert_required(meta)
