"""Discrete-complex invariants, manufactured fields and honest node sampling."""

from dataclasses import replace

import numpy as np
import pytest

from ml_platform_mode_decomp.catalog import build_decomposer
from ml_platform_mode_decomp.data.schema import FieldDataset
from ml_platform_mode_decomp.decomposition.hodge import HodgeDecomposition
from ml_platform_mode_decomp.geometry import build_domain_spec
from ml_platform_mode_decomp.representation import FieldRepresentation


def _geometry(name="rectangle", size=15):
    cfg = {"name": name, "x_range": [-1.0, 1.0], "y_range": [-1.0, 1.0]}
    if name in {"disk", "annulus"}:
        cfg.update(center=[0.0, 0.0])
        cfg.update(radius=1.0) if name == "disk" else cfg.update(r_inner=0.3, r_outer=1.0)
    if name == "arbitrary_mask":
        mask = np.ones((size, size), dtype=bool)
        mask[:, size // 2] = False
        mask[size // 2, size // 4] = False
        cfg.update(mask_source="inline", mask=mask)
    return build_domain_spec(cfg, (size, size, 2))


@pytest.mark.parametrize(
    "name,components,holes",
    [
        ("rectangle", 1, 0),
        ("disk", 1, 0),
        ("annulus", 1, 1),
        ("arbitrary_mask", 2, 1),
    ],
)
def test_native_complex_and_orthogonal_weighted_projection(name, components, holes):
    hodge = HodgeDecomposition(_geometry(name))
    assert (hodge.d1 @ hodge.d0).nnz == 0
    assert hodge.connected_components == components and hodge.harmonic_dimension == holes
    flow = np.random.default_rng(7).normal(size=len(hodge.edge_weights))
    parts = hodge.decompose_edges(flow)
    np.testing.assert_allclose(hodge.reconstruct_edges(parts), flow, atol=1e-14)
    norm = np.sqrt(hodge.energy(flow))
    np.testing.assert_allclose(hodge.d1 @ parts["exact"], 0, atol=1e-10 * norm)
    np.testing.assert_allclose(hodge.d0.T @ (hodge.edge_weights * parts["coexact"]), 0, atol=1e-10 * norm)
    np.testing.assert_allclose(hodge.d1 @ parts["harmonic"], 0, atol=1e-8 * norm)
    np.testing.assert_allclose(hodge.d0.T @ (hodge.edge_weights * parts["harmonic"]), 0, atol=1e-8 * norm)
    for first, second in (("exact", "coexact"), ("exact", "harmonic"), ("coexact", "harmonic")):
        assert abs(np.dot(hodge.edge_weights * parts[first], parts[second])) < 1e-8 * norm**2
    assert sum(hodge.energy(value) for value in parts.values()) == pytest.approx(hodge.energy(flow), rel=1e-9)
    for name, value in parts.items():
        again = hodge.decompose_edges(value)
        np.testing.assert_allclose(again[name], value, atol=1e-8 * norm)
        np.testing.assert_allclose(hodge.sample_nodes(hodge.lift_edges(value)), value, atol=1e-12 * norm)


@pytest.mark.parametrize("scale", [1e-8, 1.0, 1e8])
def test_manufactured_exact_and_coexact_are_separated_at_relative_scales(scale):
    hodge = HodgeDecomposition(_geometry("annulus"))
    rng = np.random.default_rng(81)
    exact = scale * (hodge.d0 @ rng.normal(size=hodge.d0.shape[1]))
    coexact = scale * (hodge.d1.T @ rng.normal(size=hodge.d1.shape[0])) / hodge.edge_weights
    parts = hodge.decompose_edges(exact + coexact)
    np.testing.assert_allclose(parts["exact"], exact, atol=1e-8 * scale)
    np.testing.assert_allclose(parts["coexact"], coexact, atol=1e-8 * scale)
    np.testing.assert_allclose(parts["harmonic"], 0, atol=1e-8 * scale)


def _square_circulation(hodge, flow, size=15, radius=3):
    center = size // 2
    low, high = center - radius, center + radius
    nodes = [(low, column) for column in range(low, high)]
    nodes += [(row, high) for row in range(low, high)]
    nodes += [(high, column) for column in range(high, low, -1)]
    nodes += [(row, low) for row in range(high, low, -1)]
    global_edges = hodge.node_indices[hodge.edge_vertices]
    lookup = {}
    for index, (tail, head) in enumerate(global_edges):
        lookup[int(tail), int(head)] = flow[index]
        lookup[int(head), int(tail)] = -flow[index]
    flat = [row * size + column for row, column in nodes]
    return sum(lookup[first, second] for first, second in zip(flat, [*flat[1:], flat[0]], strict=True))


def test_annulus_harmonic_preserves_hole_circulation_while_log_radius_is_exact():
    hodge = HodgeDecomposition(_geometry("annulus"))
    tail, head = hodge.vertex_coordinates[hodge.edge_vertices].transpose(1, 0, 2)
    # Exact angular increments, not an approximate node-average of a continuous field.
    angle = np.arctan2(tail[:, 0] * head[:, 1] - tail[:, 1] * head[:, 0], np.sum(tail * head, axis=1))
    np.testing.assert_allclose(hodge.d1 @ angle, 0, atol=1e-15)
    parts = hodge.decompose_edges(angle)
    np.testing.assert_allclose(parts["coexact"], 0, atol=1e-9)
    assert hodge.energy(parts["harmonic"]) > 1
    assert _square_circulation(hodge, parts["harmonic"]) == pytest.approx(2 * np.pi, abs=1e-9)
    np.testing.assert_allclose(hodge.d0.T @ (hodge.edge_weights * parts["harmonic"]), 0, atol=1e-8)
    log_radius = np.log(np.linalg.norm(hodge.vertex_coordinates, axis=1))
    radial = hodge.decompose_edges(hodge.d0 @ log_radius)
    np.testing.assert_allclose(radial["harmonic"], 0, atol=1e-8)
    assert _square_circulation(hodge, radial["exact"]) == pytest.approx(0, abs=1e-12)


def test_nonuniform_coordinate_lengths_and_actual_node_area_define_the_metric():
    x = np.array([0.0, 0.1, 0.4, 1.0, 1.8])
    y = np.array([0.0, 0.3, 0.5, 1.1])
    geometry = build_domain_spec({"name": "rectangle", "axes": {"x": {"values": x}, "y": {"values": y}}}, (4, 5, 2))
    hodge = HodgeDecomposition(geometry)
    constant = np.broadcast_to([2.0, -3.0], (4, 5, 2))
    flow = hodge.sample_nodes(constant)
    assert hodge.energy(flow) == pytest.approx(13 * geometry.weights.sum())
    scaled = HodgeDecomposition(replace(geometry, weights=7 * geometry.weights))
    np.testing.assert_allclose(scaled.edge_weights, 7 * hodge.edge_weights)
    assert scaled.energy(flow) == pytest.approx(7 * hodge.energy(flow))
    np.testing.assert_allclose(hodge.decompose_edges(flow)["exact"], flow, atol=1e-9)


def test_node_sampling_nullspace_is_residual_not_harmonic_and_roundtrips():
    geometry = _geometry()
    field = np.zeros((*geometry.grid_shape, 2))
    field[..., 0] = (-1.0) ** np.arange(geometry.grid_shape[1])
    decomposer = build_decomposer({"name": "hodge"})
    coefficients = decomposer.transform(field, None, geometry)
    np.testing.assert_array_equal(coefficients[:3], 0)
    np.testing.assert_array_equal(coefficients[3], field)
    np.testing.assert_array_equal(decomposer.inverse_transform(coefficients, geometry), field)
    assert decomposer.coeff_meta()["parts"][-1] == "sampling_residual"
    assert decomposer.coeff_meta()["orthogonality_scope"] == "native_edge_cochains_only"


def test_edge_lift_uses_the_minimum_node_area_norm_gauge():
    geometry = _geometry("annulus")
    hodge = HodgeDecomposition(geometry)
    flow = np.random.default_rng(6).normal(size=len(hodge.edge_weights))
    lifted = hodge.lift_edges(flow)
    null = np.zeros_like(lifted)
    null[..., 0] = (-1.0) ** np.arange(geometry.grid_shape[1])
    np.testing.assert_array_equal(hodge.sample_nodes(null), 0)
    values = lifted.reshape(-1, 2)[hodge.node_indices]
    direction = null.reshape(-1, 2)[hodge.node_indices]
    assert abs(np.sum(hodge.node_weights[:, None] * values * direction)) < 1e-12
    original = np.sum(hodge.node_weights[:, None] * values**2)
    shifted = np.sum(hodge.node_weights[:, None] * (values + direction) ** 2)
    assert shifted > original


def test_node_quadratic_potential_and_masked_representation_use_the_same_cached_complex():
    geometry = _geometry("annulus")
    x, y = geometry.coords["x"], geometry.coords["y"]
    gradient = np.stack((2 * x, 2 * y), axis=-1)
    fields = np.stack([gradient, 2 * gradient])
    fields[:, ~geometry.mask] = np.nan
    dataset = FieldDataset(
        fields, geometry, np.array(["first", "second"]), components=("u_x", "u_y"), field_kind="vector"
    )
    representation = FieldRepresentation({"decomposition": {"name": "hodge"}}).fit(dataset)
    complex_id = id(representation.decomposer.hodge)
    values = representation.transform(dataset, space="a")
    reconstructed = representation.inverse_transform(values, space="a")
    np.testing.assert_allclose(reconstructed[:, geometry.mask], fields[:, geometry.mask], atol=1e-12)
    raw = representation.codec.decode(values[0], representation.raw_meta)
    np.testing.assert_allclose(raw[1:3], 0, atol=1e-8)
    assert id(representation.decomposer.hodge) == complex_id
    partial = geometry.mask.copy()
    partial[np.argwhere(partial)[0][0], np.argwhere(partial)[0][1]] = False
    with pytest.raises(ValueError, match="every fixed domain node"):
        representation.decomposer.transform(gradient, partial, geometry)
    with pytest.raises(ValueError, match="cannot change"):
        representation.decomposer.transform(gradient, None, replace(geometry, weights=2 * geometry.weights))


def test_unconverged_native_solve_is_not_published_as_a_harmonic_residual():
    hodge = HodgeDecomposition(_geometry(), maxiter=1)
    flow = np.random.default_rng(4).normal(size=len(hodge.edge_weights))
    with pytest.raises(RuntimeError, match="did not converge"):
        hodge.decompose_edges(flow)


def test_declared_periodic_geometry_cannot_be_silently_cut_open():
    geometry = _geometry()
    with pytest.raises(ValueError, match="nonperiodic"):
        HodgeDecomposition(replace(geometry, meta={**geometry.meta, "periodic_axes": ["x", "y"]}))
