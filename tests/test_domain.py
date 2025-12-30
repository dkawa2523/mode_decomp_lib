from __future__ import annotations

import numpy as np
import pytest

from mode_decomp_ml.domain import build_domain_spec, validate_decomposer_compatibility


def test_rectangle_coords() -> None:
    cfg = {"name": "rectangle", "x_range": [-1.0, 1.0], "y_range": [-1.0, 1.0]}
    domain = build_domain_spec(cfg, (4, 6))

    assert domain.coords["x"].shape == (4, 6)
    assert domain.coords["y"].shape == (4, 6)
    assert np.isclose(domain.coords["x"][0, 0], -1.0)
    assert np.isclose(domain.coords["x"][0, -1], 1.0)
    assert np.isclose(domain.coords["y"][0, 0], -1.0)
    assert np.isclose(domain.coords["y"][-1, 0], 1.0)


def test_disk_coords_r_normalized() -> None:
    cfg = {
        "name": "disk",
        "center": [0.0, 0.0],
        "radius": 1.0,
        "x_range": [-1.0, 1.0],
        "y_range": [-1.0, 1.0],
    }
    domain = build_domain_spec(cfg, (5, 5))

    r = domain.coords["r"]
    assert domain.mask is not None
    assert r.shape == domain.mask.shape
    assert float(r[domain.mask].max()) <= 1.0 + 1e-6


def test_arbitrary_mask_from_path(tmp_path) -> None:
    mask = np.array([[1, 0, 1], [0, 1, 0]], dtype=bool)
    mask_path = tmp_path / "mask.npy"
    np.save(mask_path, mask)
    cfg = {
        "name": "arbitrary_mask",
        "x_range": [0.0, 1.0],
        "y_range": [0.0, 1.0],
        "mask_path": str(mask_path),
    }
    domain = build_domain_spec(cfg, (2, 3))

    assert domain.mask is not None
    assert np.array_equal(domain.mask, mask)
    assert domain.weights is not None
    assert np.all(domain.weights[~domain.mask] == 0.0)
    assert domain.meta["mask_source"] == "file"


def test_decomposer_compatibility() -> None:
    rect_cfg = {"name": "rectangle", "x_range": [-1.0, 1.0], "y_range": [-1.0, 1.0]}
    disk_cfg = {
        "name": "disk",
        "center": [0.0, 0.0],
        "radius": 1.0,
        "x_range": [-1.0, 1.0],
        "y_range": [-1.0, 1.0],
    }
    rect_domain = build_domain_spec(rect_cfg, (4, 4))
    disk_domain = build_domain_spec(disk_cfg, (4, 4))

    with pytest.raises(ValueError):
        validate_decomposer_compatibility(rect_domain, {"name": "zernike"})

    with pytest.raises(ValueError):
        validate_decomposer_compatibility(rect_domain, {"name": "fourier_bessel"})

    with pytest.raises(ValueError):
        validate_decomposer_compatibility(disk_domain, {"name": "fft2", "disk_policy": "error"})

    validate_decomposer_compatibility(disk_domain, {"name": "fourier_bessel"})

    validate_decomposer_compatibility(disk_domain, {"name": "fft2", "disk_policy": "mask_zero_fill"})

    with pytest.raises(ValueError):
        validate_decomposer_compatibility(disk_domain, {"name": "dct2", "disk_policy": "unknown"})

    validate_decomposer_compatibility(rect_domain, {"name": "helmholtz"})
    with pytest.raises(ValueError):
        validate_decomposer_compatibility(disk_domain, {"name": "helmholtz"})


def test_mesh_domain_and_compatibility() -> None:
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    mesh_cfg = {"name": "mesh", "vertices": vertices, "faces": faces}
    domain = build_domain_spec(mesh_cfg, (vertices.shape[0], 1))

    assert domain.meta["vertex_count"] == 4
    assert domain.meta["face_count"] == 2
    validate_decomposer_compatibility(domain, {"name": "laplace_beltrami"})

    rect_cfg = {"name": "rectangle", "x_range": [0.0, 1.0], "y_range": [0.0, 1.0]}
    rect_domain = build_domain_spec(rect_cfg, (2, 2))
    with pytest.raises(ValueError):
        validate_decomposer_compatibility(rect_domain, {"name": "laplace_beltrami"})
