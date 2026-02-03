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


def test_rectangle_integration_weights() -> None:
    cfg = {"name": "rectangle", "x_range": [-1.0, 1.0], "y_range": [-1.0, 1.0]}
    domain = build_domain_spec(cfg, (4, 6))

    weights = domain.integration_weights()
    assert weights is not None
    assert weights.shape == (4, 6)
    x = domain.coords["x"]
    y = domain.coords["y"]
    dx = float((x[0, -1] - x[0, 0]) / max(x.shape[1] - 1, 1))
    dy = float((y[-1, 0] - y[0, 0]) / max(y.shape[0] - 1, 1))
    assert np.allclose(weights, dx * dy)


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


def test_disk_integration_weights_masked() -> None:
    cfg = {
        "name": "disk",
        "center": [0.0, 0.0],
        "radius": 1.0,
        "x_range": [-1.0, 1.0],
        "y_range": [-1.0, 1.0],
    }
    domain = build_domain_spec(cfg, (5, 5))

    weights = domain.integration_weights()
    assert weights is not None
    assert domain.mask is not None
    assert weights.shape == domain.mask.shape
    assert np.all(weights[~domain.mask] == 0.0)
    assert np.any(weights[domain.mask] > 0.0)


def test_annulus_coords_r_normalized() -> None:
    cfg = {
        "name": "annulus",
        "center": [0.0, 0.0],
        "r_inner": 0.4,
        "r_outer": 1.0,
        "x_range": [-1.0, 1.0],
        "y_range": [-1.0, 1.0],
    }
    domain = build_domain_spec(cfg, (5, 5))

    r = domain.coords["r"]
    assert domain.mask is not None
    assert r.shape == domain.mask.shape
    assert float(r[domain.mask].max()) <= 1.0 + 1e-6
    assert float(r[domain.mask].min()) >= domain.meta["r_inner_norm"] - 1e-6


def test_sphere_grid_coords() -> None:
    cfg = {
        "name": "sphere_grid",
        "lat_range": [-90.0, 90.0],
        "lon_range": [0.0, 360.0],
        "angle_unit": "deg",
        "radius": 1.0,
    }
    domain = build_domain_spec(cfg, (6, 8))

    assert domain.coords["lat"].shape == (6, 8)
    assert domain.coords["lon"].shape == (6, 8)
    assert domain.coords["theta"].shape == (6, 8)
    assert domain.coords["phi"].shape == (6, 8)
    assert domain.weights is not None
    assert domain.weights.shape == (6, 8)
    assert np.isfinite(domain.weights).all()


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


def test_arbitrary_mask_integration_weights(tmp_path) -> None:
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

    weights = domain.integration_weights()
    assert weights is not None
    assert domain.mask is not None
    assert weights.shape == domain.mask.shape
    assert np.all(weights[~domain.mask] == 0.0)
    assert np.any(weights[domain.mask] > 0.0)


def test_decomposer_compatibility() -> None:
    rect_cfg = {"name": "rectangle", "x_range": [-1.0, 1.0], "y_range": [-1.0, 1.0]}
    disk_cfg = {
        "name": "disk",
        "center": [0.0, 0.0],
        "radius": 1.0,
        "x_range": [-1.0, 1.0],
        "y_range": [-1.0, 1.0],
    }
    annulus_cfg = {
        "name": "annulus",
        "center": [0.0, 0.0],
        "r_inner": 0.3,
        "r_outer": 1.0,
        "x_range": [-1.0, 1.0],
        "y_range": [-1.0, 1.0],
    }
    rect_domain = build_domain_spec(rect_cfg, (4, 4))
    disk_domain = build_domain_spec(disk_cfg, (4, 4))
    annulus_domain = build_domain_spec(annulus_cfg, (4, 4))
    sphere_domain = build_domain_spec(
        {"name": "sphere_grid", "lat_range": [-90.0, 90.0], "lon_range": [0.0, 360.0]},
        (4, 8),
    )

    with pytest.raises(ValueError):
        validate_decomposer_compatibility(rect_domain, {"name": "zernike"})

    with pytest.raises(ValueError):
        validate_decomposer_compatibility(rect_domain, {"name": "fourier_bessel"})

    with pytest.raises(ValueError):
        validate_decomposer_compatibility(rect_domain, {"name": "annular_zernike"})

    with pytest.raises(ValueError):
        validate_decomposer_compatibility(disk_domain, {"name": "annular_zernike"})

    with pytest.raises(ValueError):
        validate_decomposer_compatibility(disk_domain, {"name": "fft2", "disk_policy": "error"})

    validate_decomposer_compatibility(disk_domain, {"name": "fourier_bessel"})

    validate_decomposer_compatibility(annulus_domain, {"name": "annular_zernike"})

    validate_decomposer_compatibility(disk_domain, {"name": "fft2", "disk_policy": "mask_zero_fill"})

    with pytest.raises(ValueError):
        validate_decomposer_compatibility(disk_domain, {"name": "dct2", "disk_policy": "unknown"})

    validate_decomposer_compatibility(rect_domain, {"name": "helmholtz"})
    with pytest.raises(ValueError):
        validate_decomposer_compatibility(disk_domain, {"name": "helmholtz"})

    validate_decomposer_compatibility(sphere_domain, {"name": "spherical_harmonics"})
    with pytest.raises(ValueError):
        validate_decomposer_compatibility(rect_domain, {"name": "spherical_harmonics"})

    validate_decomposer_compatibility(sphere_domain, {"name": "spherical_slepian"})
    with pytest.raises(ValueError):
        validate_decomposer_compatibility(rect_domain, {"name": "spherical_slepian"})

    validate_decomposer_compatibility(rect_domain, {"name": "pswf2d_tensor"})
    with pytest.raises(ValueError):
        validate_decomposer_compatibility(disk_domain, {"name": "pswf2d_tensor"})


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
    assert domain.weights is not None
    assert domain.weights.shape == (4,)
    assert np.isclose(domain.weights.sum(), 1.0)
    assert domain.mass_matrix() is None
    validate_decomposer_compatibility(domain, {"name": "laplace_beltrami"})

    mesh_cfg_mass = {
        "name": "mesh",
        "vertices": vertices,
        "faces": faces,
        "mass_matrix": True,
    }
    domain_mass = build_domain_spec(mesh_cfg_mass, (vertices.shape[0], 1))
    mass_matrix = domain_mass.mass_matrix()
    assert mass_matrix is not None
    assert np.allclose(np.diag(mass_matrix), domain_mass.weights)

    rect_cfg = {"name": "rectangle", "x_range": [0.0, 1.0], "y_range": [0.0, 1.0]}
    rect_domain = build_domain_spec(rect_cfg, (2, 2))
    with pytest.raises(ValueError):
        validate_decomposer_compatibility(rect_domain, {"name": "laplace_beltrami"})
