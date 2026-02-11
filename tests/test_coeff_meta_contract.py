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

def test_coeff_meta_contract_fourier_jacobi() -> None:
    # Build a disk domain matching the field shape for the decomposer.
    disk = build_domain_spec(
        {"name": "disk", "center": [0.0, 0.0], "radius": 1.0, "x_range": [-1.0, 1.0], "y_range": [-1.0, 1.0]},
        (9, 9),
    )
    field = np.zeros((9, 9, 1), dtype=np.float32)
    cfg = {
        "name": "fourier_jacobi",
        "m_max": 1,
        "k_max": 1,
        "ordering": "m_then_k",
        "normalization": "orthonormal",
        "boundary_condition": "dirichlet",
        "mask_policy": "ignore_masked_points",
    }
    decomposer = build_decomposer(cfg)
    _ = decomposer.transform(field, mask=None, domain_spec=disk)
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


def test_coeff_meta_contract_polar_fft() -> None:
    disk = build_domain_spec(
        {"name": "disk", "center": [0.0, 0.0], "radius": 1.0, "x_range": [-1.0, 1.0], "y_range": [-1.0, 1.0]},
        (9, 9),
    )
    field = np.zeros((9, 9, 1), dtype=np.float32)
    cfg = {
        "name": "polar_fft",
        "n_r": 16,
        "n_theta": 32,
        "radial_transform": "dct",
        "angular_transform": "fft",
        "interpolation": "bilinear",
        "boundary_condition": "dirichlet",
        "mask_policy": "ignore_masked_points",
    }
    decomposer = build_decomposer(cfg)
    _ = decomposer.transform(field, mask=None, domain_spec=disk)
    meta = decomposer.coeff_meta()
    _assert_required(meta)


def test_coeff_meta_contract_disk_slepian() -> None:
    disk = build_domain_spec(
        {"name": "disk", "center": [0.0, 0.0], "radius": 1.0, "x_range": [-1.0, 1.0], "y_range": [-1.0, 1.0]},
        (17, 17),
    )
    field = np.zeros((17, 17, 1), dtype=np.float64)
    cfg = {
        "name": "disk_slepian",
        "n_modes": 4,
        "freq_radius": 3,
        "solver": "eigsh",
        "dense_threshold": 512,
        "eigsh_tol": 1.0e-6,
        "eigsh_maxiter": 2000,
        "mask_policy": "allow",
        "boundary_condition": "dirichlet",
    }
    decomposer = build_decomposer(cfg)
    decomposer.fit(domain_spec=disk)
    _ = decomposer.transform(field, mask=None, domain_spec=disk)
    meta = decomposer.coeff_meta()
    _assert_required(meta)


def test_coeff_meta_contract_rbf_expansion() -> None:
    domain = build_domain_spec({"name": "rectangle", "x_range": [0.0, 1.0], "y_range": [0.0, 1.0]}, (9, 9))
    field = np.zeros((9, 9, 1), dtype=np.float32)
    cfg = {
        "name": "rbf_expansion",
        "kernel": "gaussian",
        "centers": "stride",
        "stride": 3,
        "length_scale": 0.25,
        "ridge_alpha": 1.0e-6,
        "mask_policy": "ignore_masked_points",
    }
    decomposer = build_decomposer(cfg)
    _ = decomposer.transform(field, mask=None, domain_spec=domain)
    meta = decomposer.coeff_meta()
    _assert_required(meta)


def test_coeff_meta_contract_pod_joint() -> None:
    domain = build_domain_spec({"name": "rectangle", "x_range": [0.0, 1.0], "y_range": [0.0, 1.0]}, (9, 9))
    rng = np.random.default_rng(0)
    fields = [rng.normal(size=(9, 9, 2)).astype(np.float64) for _ in range(3)]
    masks = [None, None, None]

    class _Dataset:
        def __len__(self) -> int:
            return len(fields)

        def __getitem__(self, idx: int):
            return type("Sample", (), {"field": fields[idx], "mask": masks[idx]})

    cfg = {"name": "pod_joint", "n_modes": 2, "mask_policy": "error", "inner_product": "euclidean"}
    decomposer = build_decomposer(cfg)
    decomposer.fit(dataset=_Dataset(), domain_spec=domain)
    _ = decomposer.transform(fields[0], mask=None, domain_spec=domain)
    meta = decomposer.coeff_meta()
    _assert_required(meta)


def test_coeff_meta_contract_gappy_graph_fourier() -> None:
    domain = build_domain_spec({"name": "rectangle", "x_range": [0.0, 1.0], "y_range": [0.0, 1.0]}, (9, 9))
    field = np.zeros((9, 9, 1), dtype=np.float64)
    cfg = {
        "name": "gappy_graph_fourier",
        "n_modes": 4,
        "connectivity": 4,
        "laplacian_type": "combinatorial",
        "mask_policy": "allow_full",
        "solver": "dense",
        "dense_threshold": 4096,
        "eigsh_tol": 1.0e-6,
        "eigsh_maxiter": 2000,
        "ridge_alpha": 1.0e-6,
    }
    decomposer = build_decomposer(cfg)
    decomposer.fit(domain_spec=domain)
    _ = decomposer.transform(field, mask=None, domain_spec=domain)
    meta = decomposer.coeff_meta()
    _assert_required(meta)


def test_coeff_meta_contract_helmholtz_poisson() -> None:
    domain = build_domain_spec({"name": "rectangle", "x_range": [0.0, 1.0], "y_range": [0.0, 1.0]}, (9, 9))
    field = np.zeros((9, 9, 2), dtype=np.float64)
    cfg = {"name": "helmholtz_poisson", "boundary_condition": "periodic", "mask_policy": "error"}
    decomposer = build_decomposer(cfg)
    _ = decomposer.transform(field, mask=None, domain_spec=domain)
    meta = decomposer.coeff_meta()
    _assert_required(meta)


def test_coeff_meta_contract_pod_em() -> None:
    domain = build_domain_spec({"name": "rectangle", "x_range": [0.0, 1.0], "y_range": [0.0, 1.0]}, (9, 9))
    rng = np.random.default_rng(2)
    fields = [rng.normal(size=(9, 9, 1)).astype(np.float64) for _ in range(4)]
    masks = []
    for _ in range(4):
        m = rng.random(size=(9, 9)) < 0.8
        m[:2, :2] = True
        masks.append(m.astype(bool))

    class _Dataset:
        def __len__(self) -> int:
            return len(fields)

        def __getitem__(self, idx: int):
            return type("Sample", (), {"field": fields[idx], "mask": masks[idx]})

    cfg = {
        "name": "pod_em",
        "n_modes": 2,
        "n_iter": 2,
        "ridge_alpha": 1.0e-6,
        "init": "mean_fill",
        "inner_product": "euclidean",
        "mask_policy": "allow",
    }
    decomposer = build_decomposer(cfg)
    decomposer.fit(dataset=_Dataset(), domain_spec=domain)
    _ = decomposer.transform(fields[0], mask=masks[0], domain_spec=domain)
    meta = decomposer.coeff_meta()
    _assert_required(meta)


def test_coeff_meta_contract_pod_joint_em() -> None:
    domain = build_domain_spec({"name": "rectangle", "x_range": [0.0, 1.0], "y_range": [0.0, 1.0]}, (9, 9))
    rng = np.random.default_rng(3)
    fields = [rng.normal(size=(9, 9, 2)).astype(np.float64) for _ in range(4)]
    masks = []
    for _ in range(4):
        m = rng.random(size=(9, 9)) < 0.8
        m[:2, :2] = True
        masks.append(m.astype(bool))

    class _Dataset:
        def __len__(self) -> int:
            return len(fields)

        def __getitem__(self, idx: int):
            return type("Sample", (), {"field": fields[idx], "mask": masks[idx]})

    cfg = {
        "name": "pod_joint_em",
        "n_modes": 2,
        "n_iter": 2,
        "ridge_alpha": 1.0e-6,
        "init": "mean_fill",
        "inner_product": "euclidean",
        "mask_policy": "allow",
    }
    decomposer = build_decomposer(cfg)
    decomposer.fit(dataset=_Dataset(), domain_spec=domain)
    _ = decomposer.transform(fields[0], mask=masks[0], domain_spec=domain)
    meta = decomposer.coeff_meta()
    _assert_required(meta)
