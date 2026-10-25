from dataclasses import replace
import json

import numpy as np
import pytest

from ml_platform_mode_decomp.catalog import build_decomposer
from ml_platform_mode_decomp.data import FieldSample, import_legacy_npy_dataset, load_field_dataset
from ml_platform_mode_decomp.geometry import build_domain_spec
from ml_platform_mode_decomp.geometry.sphere_grid import fill_sphere_grid_ranges, sphere_grid_domain_cfg
from ml_platform_mode_decomp.decomposition.polar_fft import _bilinear_sample_cart


def test_canonical_sphere_helper_uses_the_full_periodic_measure():
    config = sphere_grid_domain_cfg(361, 72, radius=2)
    geometry = build_domain_spec(config, (361, 72, 1))
    assert geometry.meta["periodic_axes"] == ["lon"]
    assert geometry.integration_weights().sum() == pytest.approx(4 * np.pi * 4, rel=1e-5)
    filled = fill_sphere_grid_ranges({"kind": "sphere_grid", "n_lat": 361, "n_lon": 72, "radius": 2})
    alternative = build_domain_spec(filled, (361, 72, 1))
    np.testing.assert_allclose(geometry.weights, alternative.weights)
    legacy_selector = fill_sphere_grid_ranges({"type": "sphere_grid", "n_lat": 361, "n_lon": 72, "radius": 2})
    np.testing.assert_allclose(build_domain_spec(legacy_selector, (361, 72, 1)).weights, geometry.weights)


def test_sh_cache_uses_coordinates_and_mask_not_just_shape_and_bounds():
    config = {
        "kind": "sphere_grid",
        "axes": {
            "lat": {"values": [-60, -20, 10, 60]},
            "lon": {"values": [-170, -80, 0, 75, 170]},
        },
    }
    first = build_domain_spec(config, (4, 5, 1))
    second = build_domain_spec(
        {**config, "axes": {**config["axes"], "lon": {"values": [-170, -50, 0, 55, 170]}}}, (4, 5, 1)
    )
    field = np.sin(second.coords["phi"]) * np.cos(second.coords["theta"])
    decomposer = build_decomposer({"name": "spherical_harmonics", "l_max": 1})
    decomposer.transform(field, None, first)
    actual = decomposer.transform(field, None, second)
    fresh = build_decomposer({"name": "spherical_harmonics", "l_max": 1})
    np.testing.assert_allclose(actual, fresh.transform(field, None, second), atol=1e-12)
    mask = np.ones((4, 5), dtype=bool)
    mask[1, 1] = False
    masked = replace(second, mask=mask)
    basis = decomposer._get_basis(masked)
    np.testing.assert_array_equal(basis[:, ~mask], 0)


def mesh_domain(**settings):
    return build_domain_spec(
        {
            "kind": "mesh",
            "vertices": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            "faces": [[0, 1, 2], [0, 2, 3]],
            **settings,
        },
        (4, 1, 1),
    )


def test_mesh_domain_and_laplace_use_linear_size_integration_measure():
    geometry = mesh_domain(vertex_mask=[1, 1, 1, 0])
    np.testing.assert_array_equal(geometry.coords["faces"], [[0, 1, 2]])
    np.testing.assert_allclose(geometry.integration_weights(), [1 / 6, 1 / 6, 1 / 6, 0])
    assert geometry.mass_matrix() is None
    with pytest.raises(ValueError, match="mass_matrix is retired"):
        mesh_domain(mass_matrix=True)
    decomposer = build_decomposer(
        {
            "name": "laplace_beltrami",
            "n_modes": 3,
            "laplacian_type": "cotangent",
            "mass_type": "lumped",
            "boundary_condition": "neumann",
            "mask_policy": "allow",
            "solver": "dense",
            "dense_threshold": 128,
            "eigsh_tol": 1e-8,
            "eigsh_maxiter": 1000,
        }
    )
    field = np.array([2.0, 4.0, 6.0, np.nan])[:, None, None]
    decomposer.fit(domain_spec=geometry)
    coefficients = decomposer.transform(field, None, geometry)
    restored = decomposer.inverse_transform(coefficients, geometry)
    np.testing.assert_allclose(restored[:3], field[:3], atol=1e-12)
    np.testing.assert_allclose(decomposer._mass, geometry.weights[:3])
    np.testing.assert_allclose(
        decomposer._basis.T @ np.diag(decomposer._mass) @ decomposer._basis, np.eye(3), atol=1e-12
    )


def test_mesh_observation_masks_do_not_redefine_physical_mass():
    geometry = mesh_domain()
    decomposer = build_decomposer(
        {
            "name": "laplace_beltrami",
            "n_modes": 1,
            "laplacian_type": "cotangent",
            "mass_type": "lumped",
            "boundary_condition": "neumann",
            "mask_policy": "allow",
            "solver": "dense",
            "dense_threshold": 128,
            "eigsh_tol": 1e-8,
            "eigsh_maxiter": 1000,
        }
    )
    samples = [
        FieldSample(None, np.full((4, 1, 1), 3.0), np.asarray(mask, dtype=bool)[:, None])
        for mask in ([1, 1, 1, 0], [0, 1, 1, 1])
    ]
    decomposer.fit(samples, geometry)
    mass = decomposer._mass.copy()
    field = np.array([3.0, 3.0, 3.0, np.nan])[:, None, None]
    coefficient = decomposer.transform(field, samples[0].mask, geometry)
    np.testing.assert_allclose(decomposer.inverse_transform(coefficient, geometry), 3.0, atol=1e-12)
    np.testing.assert_array_equal(decomposer._mass, mass)
    assert mass.sum() == pytest.approx(1.0)


def test_cartesian_interpolation_uses_nonuniform_coordinates_exactly():
    x = np.array([-1.0, -0.8, -0.1, 0.2, 1.0])
    y = np.array([-1.0, -0.4, 0.1, 0.9, 1.0])
    xx, yy = np.meshgrid(x, y)
    xq, yq = np.meshgrid(np.linspace(-0.9, 0.8, 13), np.linspace(-0.8, 0.8, 11))
    sampled = _bilinear_sample_cart(2 * xx - 3 * yy + 1, xq, yq, x_axis=x, y_axis=y)
    np.testing.assert_allclose(sampled, 2 * xq - 3 * yq + 1, atol=1e-14)


@pytest.mark.parametrize("mask_aware", [True, False])
def test_polar_nonuniform_axes_ignore_masked_nan_before_arithmetic(mask_aware):
    axis = np.sign(np.linspace(-1, 1, 25)) * np.abs(np.linspace(-1, 1, 25)) ** 1.4
    domain = build_domain_spec(
        {"kind": "disk", "center": [0, 0], "radius": 1, "axes": {"x": {"values": axis}, "y": {"values": axis}}},
        (25, 25, 1),
    )
    field = np.where(domain.mask, 1.0, np.nan)[..., None]
    cfg = {
        "name": "polar_fft",
        "n_r": 24,
        "n_theta": 64,
        "radial_transform": "dct",
        "angular_transform": "fft",
        "interpolation": "bilinear",
        "boundary_condition": "dirichlet",
        "mask_policy": "ignore_masked_points",
        "mask_aware_sampling": mask_aware,
    }
    decomposer = build_decomposer(cfg)
    with np.errstate(invalid="raise", over="raise"):
        coefficients = decomposer.transform(field, None, domain)
        restored = decomposer.inverse_transform(coefficients, domain)
    assert np.isfinite(coefficients).all() and np.isfinite(restored).all()
    if mask_aware:
        np.testing.assert_allclose(restored[..., 0][domain.mask], 1.0, atol=1e-12)


def legacy_input(root):
    root.mkdir()
    np.save(root / "field.npy", np.ones((3, 4, 4, 1)))
    np.save(root / "cond.npy", np.ones((3, 2)))
    manifest = {"grid": {"H": 4, "W": 4, "x_range": [-1, 1], "y_range": [-1, 1]}, "domain": {"type": "rectangle"}}
    return manifest


def test_legacy_import_respects_explicit_manifest_axes_and_override(tmp_path):
    source = tmp_path / "legacy"
    manifest = legacy_input(source)
    np.save(source / "x.npy", [-1.0, -0.5, 0.25, 1.0])
    manifest["geometry"] = {"axes": {"x": {"path": "x.npy"}}}
    (source / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    first = tmp_path / "first"
    import_legacy_npy_dataset(source, first)
    np.testing.assert_allclose(load_field_dataset(first).geometry.coords["x"][0], [-1, -0.5, 0.25, 1])
    assert (first / "x.npy").read_bytes() == (source / "x.npy").read_bytes()
    second = tmp_path / "second"
    import_legacy_npy_dataset(source, second, geometry_override={"axes": {"x": {"values": [-1, -0.75, 0.5, 2]}}})
    np.testing.assert_allclose(load_field_dataset(second).geometry.coords["x"][0], [-1, -0.75, 0.5, 2])
