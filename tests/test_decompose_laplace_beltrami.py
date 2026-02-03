from __future__ import annotations

import numpy as np

from mode_decomp_ml.plugins.decomposers import build_decomposer
from mode_decomp_ml.domain import build_domain_spec


def _mesh_domain(n_vertices: int, n_channels: int):
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    assert vertices.shape[0] == n_vertices
    cfg = {"name": "mesh", "vertices": vertices, "faces": faces}
    return build_domain_spec(cfg, (n_vertices, 1, n_channels))


def _laplace_cfg(n_modes: int) -> dict[str, object]:
    return {
        "name": "laplace_beltrami",
        "n_modes": int(n_modes),
        "laplacian_type": "cotangent",
        "mass_type": "lumped",
        "boundary_condition": "neumann",
        "mask_policy": "allow",
        "solver": "dense",
        "dense_threshold": 128,
        "eigsh_tol": 1.0e-6,
        "eigsh_maxiter": 10000,
    }


def test_laplace_beltrami_roundtrip() -> None:
    n_vertices = 4
    n_channels = 1
    rng = np.random.default_rng(0)
    field = rng.normal(size=(n_vertices, 1, n_channels)).astype(np.float64)
    domain = _mesh_domain(n_vertices, n_channels)
    decomposer = build_decomposer(_laplace_cfg(n_vertices))

    decomposer.fit(domain_spec=domain)
    coeff = decomposer.transform(field, mask=None, domain_spec=domain)
    field_hat = decomposer.inverse_transform(coeff, domain_spec=domain)

    assert field_hat.shape == field.shape
    assert np.allclose(field_hat, field, atol=1e-5)

    meta = decomposer.coeff_meta()
    assert meta["method"] == "laplace_beltrami"
    assert meta["coeff_shape"] == [n_channels, n_vertices]
    eigenvalues = np.asarray(meta["eigenvalues"], dtype=np.float64)
    assert eigenvalues.shape[0] == n_vertices
    assert np.all(np.diff(eigenvalues) >= -1e-10)


def test_laplace_beltrami_auto_n_modes() -> None:
    n_vertices = 4
    n_channels = 1
    rng = np.random.default_rng(2)
    field = rng.normal(size=(n_vertices, 1, n_channels)).astype(np.float64)
    domain = _mesh_domain(n_vertices, n_channels)
    cfg = {
        "name": "laplace_beltrami",
        "n_modes": "auto",
        "laplacian_type": "cotangent",
        "mass_type": "lumped",
        "boundary_condition": "neumann",
        "mask_policy": "allow",
        "solver": "auto",
        "dense_threshold": 128,
        "eigsh_tol": 1.0e-6,
        "eigsh_maxiter": 10000,
    }
    decomposer = build_decomposer(cfg)

    decomposer.fit(domain_spec=domain)
    coeff = decomposer.transform(field, mask=None, domain_spec=domain)
    field_hat = decomposer.inverse_transform(coeff, domain_spec=domain)

    assert field_hat.shape == field.shape
    meta = decomposer.coeff_meta()
    assert meta["n_modes"] == n_vertices
    assert meta["n_modes_config"] == "auto"
