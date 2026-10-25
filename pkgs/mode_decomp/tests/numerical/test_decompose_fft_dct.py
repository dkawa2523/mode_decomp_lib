from __future__ import annotations

from pathlib import Path

from ml_platform_core.io import dump_joblib, load_joblib

import numpy as np
import pytest

from ml_platform_mode_decomp.decomposition import build_decomposer
from ml_platform_mode_decomp.geometry import DomainSpec, build_domain_spec


def _rectangle_domain(height: int, width: int):
    xx, yy = np.meshgrid(np.arange(width, dtype=float), np.arange(height, dtype=float))
    return DomainSpec(name="rectangle", coords={"x": xx, "y": yy}, mask=None, weights=None, meta={})


def _basis(size, method):
    frequency, coordinate = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    if method == "fft2":
        return np.exp(-2j * np.pi * frequency * coordinate / size) / np.sqrt(size)
    basis = np.sqrt(2 / size) * np.cos(np.pi * frequency * (coordinate + 0.5) / size)
    basis[0] /= np.sqrt(2)
    return basis


@pytest.mark.parametrize("method", ["fft2", "dct2"])
@pytest.mark.parametrize("shape", [(5, 8, 2), (1, 7, 2), (6, 1, 3)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_batch_transform_matches_explicit_spatial_basis_and_preserves_each_sample_channel(method, shape, dtype):
    height, width, channels = shape
    # A strided field batch also catches assumptions about contiguous NHWC memory.
    fields = np.random.default_rng(42).normal(size=(3, 2 * height, width, channels)).astype(dtype)[:, ::2, ::-1]
    fields *= np.arange(1, channels + 1)
    domain = _rectangle_domain(height, width)
    decomposer = build_decomposer({"name": method})
    coefficients = decomposer.transform_batch(fields, None, domain)
    expected = np.einsum("yh,nhwc,xw->ncyx", _basis(height, method), fields, _basis(width, method))
    tolerance = 4e-6 if dtype == np.float32 else 1e-12
    np.testing.assert_allclose(coefficients, expected, atol=tolerance, rtol=tolerance)
    np.testing.assert_allclose(
        np.sum(abs(coefficients) ** 2, axis=(1, 2, 3)),
        np.sum(fields.astype(float) ** 2, axis=(1, 2, 3)),
        rtol=tolerance,
    )
    np.testing.assert_allclose(decomposer.inverse_transform_batch(coefficients, domain), fields, atol=tolerance)
    np.testing.assert_allclose(decomposer.transform(fields[1], None, domain), expected[1], atol=tolerance)
    np.testing.assert_allclose(decomposer.inverse_transform(coefficients[1], domain), fields[1], atol=tolerance)


@pytest.mark.parametrize("method", ["fft2", "dct2"])
def test_batch_disk_zero_fill_combines_domain_and_sample_masks_without_mutating_fields(method):
    domain = build_domain_spec(
        {"name": "disk", "radius": 1, "center": [0, 0], "x_range": [-1, 1], "y_range": [-1, 1]}, (7, 9)
    )
    fields = np.random.default_rng(1).normal(size=(3, 7, 9, 2))
    masks = np.ones(fields.shape[:-1], dtype=bool)
    masks[0, 3, 4] = False
    masks[1, 2:4, 3:5] = False
    combined = masks & domain.mask
    fields[~combined] = np.nan
    snapshot = fields.copy()
    decomposer = build_decomposer({"name": method, "disk_policy": "mask_zero_fill"})
    coefficients = decomposer.transform_batch(fields, masks, domain)
    expected = np.where(combined[..., None], fields, 0)
    np.testing.assert_allclose(decomposer.inverse_transform_batch(coefficients, domain), expected, atol=1e-12)
    np.testing.assert_array_equal(fields, snapshot)
    with pytest.raises(ValueError, match="mask_zero_fill"):
        build_decomposer({"name": method}).transform_batch(fields, masks, domain)


@pytest.mark.parametrize("method", ["fft2", "dct2"])
def test_representation_chunks_preserve_packed_order_fit_targets_and_reconstruction(monkeypatch, method):
    import ml_platform_mode_decomp.representation as representation
    from ml_platform_mode_decomp import FieldDataset

    fields = np.random.default_rng(2).normal(size=(9, 5, 8, 2))
    dataset = FieldDataset(fields, _rectangle_domain(5, 8), np.asarray([str(i) for i in range(len(fields))]))
    config = {"decomposition": {"name": method}, "coeff_post": {"name": "standardize"}}
    reference = representation.FieldRepresentation(config).fit(dataset)
    expected_a = reference.transform(dataset, space="a")
    expected_z = reference.transform(dataset)
    # Exercise several chunks and a shorter final chunk using the same public API.
    monkeypatch.setattr(representation, "_TRANSFORM_WORK_BYTES", fields[0].size * 16 * 6 * 2)
    model = representation.FieldRepresentation(config)
    targets = model.fit_transform(dataset)
    np.testing.assert_allclose(targets, expected_z, atol=1e-12)
    np.testing.assert_allclose(model.transform(dataset, space="a"), expected_a, atol=1e-12)
    np.testing.assert_allclose(model.inverse_transform(targets), fields, atol=1e-12)
    np.testing.assert_allclose(model.fit_transform(dataset, space="a"), expected_a, atol=1e-12)
    if method == "fft2":
        np.testing.assert_allclose(np.sum(expected_a**2, axis=1), np.sum(fields**2, axis=(1, 2, 3)), atol=1e-12)


def test_fft_metadata_and_transform_survive_core_joblib(tmp_path: Path) -> None:
    field = np.random.default_rng(1).normal(size=(4, 5, 1))
    domain = _rectangle_domain(4, 5)
    decomposer = build_decomposer({"name": "fft2", "disk_policy": "error"})
    coeff = decomposer.transform(field, mask=None, domain_spec=domain)
    path = dump_joblib(decomposer, tmp_path / "decomposer.joblib")
    restored = load_joblib(path)
    assert restored.coeff_meta() == decomposer.coeff_meta()
    assert np.allclose(restored.transform(field, mask=None, domain_spec=domain), coeff)
    assert np.allclose(restored.inverse_transform(coeff, domain_spec=domain), field)
