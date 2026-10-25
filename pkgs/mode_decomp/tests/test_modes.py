"""Synthesis modes share coordinate selection, without a universal energy claim."""

from collections.abc import Iterator

import numpy as np
import pytest

from ml_platform_mode_decomp.data.schema import FieldDataset
from ml_platform_mode_decomp.geometry import build_domain_spec
from ml_platform_mode_decomp.representation import FieldRepresentation


def _dataset(height=6, width=8, channels=2):
    geometry = build_domain_spec(
        {
            "name": "rectangle",
            "axes": {
                "x": {"start": 2.0, "step": 0.25, "count": width, "periodic": True},
                "y": {"start": -1.0, "step": 0.5, "count": height, "periodic": True},
            },
            "coordinate_units": {"x": "m", "y": "s"},
        },
        (height, width, channels),
    )
    rng = np.random.default_rng(15)
    fields = rng.normal(size=(12, height, width, channels))
    return FieldDataset(
        fields,
        geometry,
        np.array([f"s{i}" for i in range(len(fields))]),
        manifest={"fields": {"units": ["K"] * channels}},
        components=tuple(f"c{i}" for i in range(channels)),
    )


def _representation(dataset, method="fft2", **params):
    return FieldRepresentation({"decomposition": {"name": method, "params": params}}).fit(dataset)


@pytest.mark.parametrize("height,width", [(1, 8), (8, 1), (1, 1)])
def test_singleton_fft_axis_has_only_zero_frequency_and_no_invented_period(height, width):
    dataset = FieldDataset.from_arrays(np.arange(float(height * width)).reshape(1, height, width, 1))
    representation = _representation(dataset)
    modes = representation.list_modes()
    assert len(modes) == height * width
    axis = "x" if width == 1 else "y"
    assert all(mode[f"frequency_{axis}"] == 0 and mode[f"period_{axis}"] is None for mode in modes)
    np.testing.assert_allclose(representation.reconstruct(dataset), dataset.fields, atol=1e-12)


@pytest.mark.parametrize("shape,block", [((5, 7), None), ((6, 8), None), ((6, 8), 2), ((6, 8), 6), ((5, 7), 5)])
def test_fft_discrete_labels_match_actual_requested_real_basis(shape, block):
    dataset = _dataset(*shape)
    representation = (
        _representation(dataset) if block is None else _representation(dataset, "fft2_lowpass", block_size=block)
    )
    descriptions = representation.list_modes()
    basis = representation.mode_fields(range(representation.packed_dim))
    assert len(descriptions) == representation.packed_dim
    np.testing.assert_allclose(
        basis.reshape(len(basis), -1) @ basis.reshape(len(basis), -1).T, np.eye(len(basis)), atol=1e-12
    )
    x, y = dataset.geometry.coords["x"], dataset.geometry.coords["y"]
    for item, actual in zip(descriptions, basis, strict=True):
        phase = 2 * np.pi * (item["frequency_x"] * (x - x[0, 0]) + item["frequency_y"] * (y - y[0, 0]))
        quadrature = np.sin(phase) * item["phase_sign"] if item["kind"] == "sine" else np.cos(phase)
        expected = np.zeros(dataset.fields.shape[1:])
        expected[..., item["channel"]] = item["basis_amplitude_before_field_inverse"] * quadrature
        np.testing.assert_allclose(actual, expected, atol=1e-12)
        assert item["interpretation"] == "fourier_mode" and item["additive"]
        assert item["coordinate_unit_x"] == "m" and item["frequency_unit_y"] == "cycles/s"
        assert item["period_x"] == shape[1] * 0.25
        assert item["period_y"] == shape[0] * 0.5
        assert item["component_name"] == f"c{item['channel']}"
        if item["self_conjugate"]:
            assert item["component"] == "real"
            assert item["kind"] in {"dc", "nyquist"}
    if block is None and shape[0] % 2 == shape[1] % 2 == 0:
        assert sum(row["kind"] == "nyquist" for row in descriptions) == 3 * dataset.fields.shape[-1]
        # A bin on one Nyquist axis may still have a nonzero sine on the other axis.
        assert any(row["nyquist_axes"] and row["kind"] == "sine" for row in descriptions)


def test_metadata_listing_builds_no_fields_and_selection_keeps_hermitian_symmetry(monkeypatch):
    dataset = _dataset(5, 8, 1)
    representation = _representation(dataset, "fft2_lowpass", block_size=2)
    packed = representation.transform(dataset, space="a")
    expected = representation.inverse_transform(packed, space="a")
    single = representation.reconstruct_modes(packed, [0, 2, 8])
    np.testing.assert_allclose(
        representation.transform(single, space="a")[:, [0, 2, 8]], packed[:, [0, 2, 8]], atol=1e-12
    )
    selected = representation.transform(single, space="a")
    selected[:, [0, 2, 8]] = 0
    np.testing.assert_allclose(selected, 0, atol=1e-12)
    np.testing.assert_allclose(
        representation.reconstruct_modes(packed, range(representation.packed_dim)), expected, atol=1e-12
    )

    def unexpected_inverse(*args, **kwargs):
        raise AssertionError("listing coordinates must not synthesize fields")

    monkeypatch.setattr(representation, "inverse_transform", unexpected_inverse)
    assert [row["index"] for row in representation.list_modes([8, 0])] == [8, 0]


@pytest.mark.parametrize("method", ["pod", "pod_svd", "pod_joint", "gappy_pod"])
def test_linear_modes_subtract_baseline_once_and_cumulative_matches_selection(method):
    dataset = _dataset(channels=2)
    representation = FieldRepresentation(
        {
            "decomposition": {"name": method, "params": {"n_modes": 2}},
            "field_preprocess": {"name": "basic", "params": {"ops": ["field_standardize"]}},
        }
    ).fit(dataset)
    packed = representation.transform(dataset, space="a")
    order = list(reversed(range(representation.packed_dim)))
    basis = representation.mode_fields(order)
    baseline = representation.inverse_transform(np.zeros((1, representation.packed_dim)), space="a")[0]
    predicted = baseline + np.einsum("nk,k...->n...", packed[:, order], basis)
    np.testing.assert_allclose(predicted, representation.inverse_transform(packed, space="a"), atol=1e-11)
    assert not np.allclose(baseline, 0)
    cumulative = representation.cumulative_reconstruct(packed, order)
    assert isinstance(cumulative, Iterator)
    for count, fields in enumerate(cumulative, start=1):
        np.testing.assert_allclose(fields, representation.reconstruct_modes(packed, order[:count]), atol=1e-12)
    descriptions = representation.list_modes([0])
    assert descriptions[0]["additive"]
    if method == "pod_joint":
        assert "channel" not in descriptions[0]  # One joint mode can span every field channel.
    else:
        assert descriptions[0]["channel"] == 0


def test_pca_targets_are_latent_directions_not_original_fourier_modes():
    dataset = _dataset(channels=1)
    representation = FieldRepresentation(
        {
            "decomposition": {"name": "fft2"},
            "coeff_post": {"name": "pca", "params": {"n_components": 3}},
        }
    ).fit(dataset)
    assert representation.list_modes([0])[0]["interpretation"] == "fourier_mode"
    row = representation.list_modes([0], space="z")[0]
    assert row["interpretation"] == "latent_coordinate" and row["additive"]
    assert row["coefficient_post"] == "pca" and "kx" not in row
    targets = representation.transform(dataset)
    basis = representation.mode_fields(range(3), space="z")
    baseline = representation.inverse_transform(np.zeros((1, 3)))[0]
    np.testing.assert_allclose(
        baseline + np.einsum("nk,k...->n...", targets, basis), representation.inverse_transform(targets), atol=1e-12
    )


def test_nonlinear_target_inverse_requires_reference_and_returns_finite_perturbation():
    dataset = _dataset(channels=1)
    representation = FieldRepresentation(
        {
            "decomposition": {"name": "fft2"},
            "coeff_post": {"name": "quantile", "params": {"n_quantiles": 8}},
        }
    ).fit(dataset)
    assert representation.list_modes([0])[0]["additive"]  # A is unchanged by its target transform.
    row = representation.list_modes([0], space="z")[0]
    assert row["interpretation"] == "latent_coordinate" and not row["additive"]
    with pytest.raises(ValueError, match="reference"):
        representation.mode_fields([0], space="z")
    reference = np.full(representation.target_dim, 0.5)
    changed = reference.copy()
    changed[0] += 0.1
    actual = representation.mode_fields([0], space="z", reference=reference, amplitude=0.1)
    expected = representation.inverse_transform(changed[None, :]) - representation.inverse_transform(reference[None, :])
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_composite_is_not_labelled_an_independent_fixed_mode_system():
    dataset = _dataset(channels=1)
    representation = _representation(dataset, "offset_residual", enabled=True)
    row = representation.list_modes([0])[0]
    assert row["interpretation"] == "composite_coordinate" and not row["additive"]
    with pytest.raises(ValueError, match="reference"):
        representation.mode_fields([0])
    assert representation.mode_fields([0], reference=np.zeros(representation.packed_dim)).shape == (
        1,
        *representation.field_shape,
    )


def test_mode_order_is_explicit_unique_and_within_actual_packed_dimension():
    representation = _representation(_dataset(channels=1), "fft2_lowpass", block_size=2)
    assert representation.packed_dim == 9
    for invalid in ([9], [-1], [0, 0], [True]):
        with pytest.raises(ValueError):
            representation.list_modes(invalid)


def test_integer_reference_does_not_truncate_fractional_coordinates_or_perturbations():
    representation = _representation(_dataset(channels=1), "fft2_lowpass", block_size=2)
    reference = np.zeros(representation.packed_dim, dtype=int)
    values = np.full((1, representation.packed_dim), 0.125)
    expected = representation.inverse_transform(values, space="a")
    np.testing.assert_allclose(
        representation.reconstruct_modes(values, range(representation.packed_dim), reference=reference), expected
    )
    np.testing.assert_allclose(
        representation.mode_fields([0], reference=reference, amplitude=0.125),
        0.125 * representation.mode_fields([0]),
        atol=1e-12,
    )
