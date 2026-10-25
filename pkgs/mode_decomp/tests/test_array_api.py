"""Small array-only entry points preserve numerical inputs and physical geometry."""

from dataclasses import replace

import numpy as np
import pytest

from ml_platform_mode_decomp.data.schema import FieldDataset
from ml_platform_mode_decomp.geometry import DomainSpec, build_domain_spec
from ml_platform_mode_decomp.representation import FieldRepresentation
from ml_platform_mode_decomp.surrogate import FieldSurrogate


def test_arrays_alone_fit_fft_representation_and_condition_surrogate_without_files():
    conditions = np.random.default_rng(12).normal(size=(16, 2)).astype(np.float32)
    x, y = np.meshgrid(np.arange(4, dtype=np.float32), np.arange(3, dtype=np.float32))
    fields = np.einsum("nk,khw->nhw", conditions, np.stack((x, y)))[..., None]
    data = FieldDataset.from_arrays(fields, conditions)
    assert np.shares_memory(data.fields, fields) and np.shares_memory(data.conditions, conditions)
    assert data.fields.dtype == data.conditions.dtype == np.float32
    assert data.condition_columns == ("condition_0", "condition_1")
    assert data.field_kind == "scalar" and data.field_units == ()
    assert data.components == ("component_0",) and data.sample_ids[0] == "sample_000000"
    assert data.geometry.meta["coordinate_source"] == {"x": "index", "y": "index"}
    representation = FieldRepresentation({"decomposition": {"name": "fft2"}}).fit(data)
    np.testing.assert_allclose(representation.inverse_transform(representation.transform(data)), fields, atol=1e-6)
    model = FieldSurrogate(
        {
            "representation": {"decomposition": {"name": "pod_svd", "params": {"n_modes": 2}}},
            "regression": {"name": "ridge", "params": {"alpha": 0.0}},
        }
    ).fit(data)
    np.testing.assert_allclose(model.predict(conditions[:3]), fields[:3], atol=1e-6)


@pytest.mark.parametrize("field_kind", [None, "vector"])
def test_explicit_hwc_single_vector_field_uses_real_area_and_roundtrips_hodge_parts(field_kind):
    x, y = np.asarray([0.0, 0.2, 0.7, 2.0]), np.asarray([-1.0, 0.0, 2.0])
    xx, yy = np.meshgrid(x, y)
    field = np.stack((2 * xx + (-1.0) ** np.arange(len(x)), 2 * yy), axis=-1)
    data = FieldDataset.from_arrays(
        field,
        layout="HWC",
        x=x,
        y=y,
        components=["u_x", "u_y"],
        field_kind=field_kind,
        field_units=["m/s", "m/s"],
    )
    assert data.fields.shape == (1, 3, 4, 2) and np.shares_memory(data.fields, field)
    assert data.geometry.weights.sum() == pytest.approx(6)
    assert data.geometry.meta["coordinate_source"] == {"x": "provided", "y": "provided"}
    assert data.field_kind == ("multichannel" if field_kind is None else "vector")
    if field_kind is None:
        with pytest.raises(ValueError, match="Hodge requires fields.kind: vector"):
            FieldRepresentation({"decomposition": {"name": "hodge"}}).fit(data)
        return
    model = FieldRepresentation({"decomposition": {"name": "hodge"}}).fit(data)
    coefficients = model.transform(data, space="a")
    parts = model.codec.decode(coefficients[0], model.raw_meta)
    assert parts.shape == (4, 3, 4, 2)
    assert np.linalg.norm(parts[-1]) > 0  # Sampling nullspace remains a separate fourth part.
    np.testing.assert_allclose(parts.sum(axis=0), field, atol=1e-12)
    np.testing.assert_allclose(model.inverse_transform(coefficients, space="a"), data.fields, atol=1e-12)


def test_geometry_and_shared_observation_mask_are_used_without_batch_expansion():
    geometry = build_domain_spec({"name": "rectangle", "x_range": [0, 2], "y_range": [0, 3]}, (3, 4, 1))
    domain_mask = np.ones((3, 4), dtype=bool)
    domain_mask[0, 0] = False
    geometry = replace(geometry, mask=domain_mask, weights=2 * geometry.weights)
    observed = np.ones((3, 4), dtype=bool)
    observed[1, 1] = False
    fields = np.ones((3, 3, 4, 1))
    fields[:, 0, 0] = fields[:, 1, 1] = np.nan
    data = FieldDataset.from_arrays(
        fields,
        geometry=geometry,
        observation_mask=observed,
        sample_ids=["a", "b", "c"],
        sample_metadata={"group": [1, 1, 2]},
    )
    assert data.geometry is geometry and data.observation_mask is observed
    assert data.observation_mask.shape == (3, 4)
    assert data.geometry.weights.sum() == geometry.weights.sum()
    np.testing.assert_array_equal(data.effective_mask, np.broadcast_to(domain_mask & observed, (3, 3, 4)))
    assert data[1].meta["group"] == 1 and data[1].sample_id == "b"


@pytest.mark.parametrize("shape,measure", [((1, 1), 1), ((1, 4), 3), ((4, 1), 3)])
def test_unspecified_singleton_axes_have_documented_unit_index_measure(shape, measure):
    data = FieldDataset.from_arrays(np.arange(4.0)[:, None, None, None] * np.ones((1, *shape, 1)))
    assert data.geometry.grid_shape == shape
    assert data.geometry.weights.sum() == pytest.approx(measure)
    assert data.geometry.meta["singleton_index_measure"] == 1
    assert data.geometry.meta["weight_measure"] == "index_coordinate_measure"
    assert "no physical units" in data.geometry.meta["coordinate_note"]


@pytest.mark.parametrize("method", ["fft2", "dct2"])
def test_singleton_index_axis_still_preserves_the_other_axis_signal(method):
    fields = (3 + np.cos(2 * np.pi * np.arange(8) / 8)).reshape(1, 1, 8, 1)
    dataset = FieldDataset.from_arrays(fields)
    model = FieldRepresentation({"decomposition": {"name": method}}).fit(dataset)
    np.testing.assert_allclose(model.reconstruct(dataset), fields, atol=1e-12)


@pytest.mark.parametrize("layout", ["NVC", "VC"])
def test_mesh_layout_requires_explicit_geometry_and_keeps_original_vertex_measure(layout):
    geometry = build_domain_spec(
        {"name": "mesh", "vertices": [[0, 0], [2, 0], [0, 1]], "faces": [[0, 1, 2]]}, (3, 1, 1)
    )
    fields = np.asarray([[1.0], [2.0], [3.0]])
    if layout == "NVC":
        fields = np.stack((fields, 2 * fields))
    with pytest.raises(ValueError, match="requires explicit mesh geometry"):
        FieldDataset.from_arrays(fields, layout=layout)
    data = FieldDataset.from_arrays(fields, layout=layout, geometry=geometry)
    assert data.geometry is geometry and np.shares_memory(data.fields, fields)
    assert data.fields.shape == ((2, 3, 1) if layout == "NVC" else (1, 3, 1))
    assert data.geometry.integration_weights().sum() == pytest.approx(1)


def test_layout_coordinate_and_condition_ambiguity_are_rejected():
    field = np.zeros((3, 4, 2))
    with pytest.raises(ValueError, match="explicit layout"):
        FieldDataset.from_arrays(field)
    geometry = DomainSpec("rectangle", {"x": np.zeros((3, 4)), "y": np.zeros((3, 4))}, None, np.ones((3, 4)), {})
    with pytest.raises(ValueError, match="not both"):
        FieldDataset.from_arrays(field, layout="HWC", geometry=geometry, x=np.arange(4))
    with pytest.raises(ValueError, match="singleton x coordinates cannot define area"):
        FieldDataset.from_arrays(np.zeros((2, 3, 1, 1)), x=[0])
    with pytest.raises(ValueError, match="conditions must have shape"):
        FieldDataset.from_arrays(np.zeros((2, 3, 4, 1)), conditions=[0, 1])
    with pytest.raises(ValueError, match="disagrees with geometry"):
        FieldDataset.from_arrays(np.zeros((1, 4, 3, 2)), geometry=geometry)
    for kind in ("scalar", "unknown"):
        with pytest.raises(ValueError, match="fields.kind"):
            FieldDataset.from_arrays(field, layout="HWC", field_kind=kind)


def test_array_and_file_geometry_share_identity_but_fitted_snapshot_owns_its_arrays(tmp_path):
    from ml_platform_mode_decomp.data import load_field_dataset
    from ml_platform_mode_decomp.geometry.identity import geometry_fingerprint, snapshot_geometry

    field = np.arange(12.0).reshape(1, 3, 4, 1)
    np.save(tmp_path / "field.npy", field)
    arrays = FieldDataset.from_arrays(field, x=np.arange(4), y=np.arange(3))
    files = load_field_dataset(
        {
            "local_path": tmp_path,
            "fields": {"path": "field.npy"},
            "geometry": {
                "kind": "rectangle",
                "axes": {"x": {"values": list(range(4))}, "y": {"values": list(range(3))}},
            },
        }
    )
    assert geometry_fingerprint(arrays.geometry) == geometry_fingerprint(files.geometry)
    fitted = snapshot_geometry(arrays.geometry)
    before = geometry_fingerprint(fitted)
    arrays.geometry.coords["x"][0, 0] = -1
    assert geometry_fingerprint(fitted) == before
    assert geometry_fingerprint(arrays.geometry) != before


@pytest.mark.parametrize("units", [[" "], [1], ["K", "K"]])
def test_array_units_are_validated_at_construction(units):
    with pytest.raises(ValueError, match="units"):
        FieldDataset.from_arrays(np.ones((1, 2, 2, 1)), field_units=units)
