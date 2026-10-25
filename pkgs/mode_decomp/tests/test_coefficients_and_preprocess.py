"""Numerical boundaries and regressions for the coefficient/field adapters."""

from __future__ import annotations

import json
import subprocess
import sys
import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from ml_platform_mode_decomp.catalog import (
    build_coeff_codec,
    build_coeff_post,
    build_preprocess,
    build_regressor,
    list_decomposers,
)
from ml_platform_mode_decomp.config import component_config


def test_catalog_lists_builtins_without_loading_optional_backends():
    code = """
import json, sys
from ml_platform_mode_decomp.catalog import list_decomposers, list_regressors
print(json.dumps([list_decomposers(), list_regressors(),
                 [x for x in ('torch', 'gpytorch', 'xgboost', 'lightgbm', 'catboost', 'pywt', 'pyshtools') if x in sys.modules]]))
"""
    output = subprocess.check_output([sys.executable, "-B", "-c", code], text=True)
    decomposers, regressors, imported = json.loads(output)
    assert {"pod_svd", "fft2", "hodge"}.issubset(decomposers)
    assert "ridge" in regressors
    assert imported == []
    assert "pod_svd" in list_decomposers()


@pytest.mark.parametrize("reserved", ["name", "id", "method", "params"])
def test_component_parameters_cannot_replace_identity(reserved):
    with pytest.raises(ValueError, match="identity"):
        component_config({"name": "ridge", "params": {reserved: "different"}})


def test_component_configuration_does_not_mutate_input_or_silently_merge_duplicates():
    cfg = {"name": "ridge", "params": {"alpha": 2.0}}
    build_regressor(cfg)
    assert cfg == {"name": "ridge", "params": {"alpha": 2.0}}
    with pytest.raises(ValueError, match="twice"):
        component_config({"name": "ridge", "alpha": 1, "params": {"alpha": 2}})
    with pytest.raises(ValueError, match="candidate"):
        component_config({"name": "ridge", "id": "candidate"})


def test_auto_codec_alias_and_complex_roundtrip():
    raw = np.array([[1 + 2j, -3j, 4], [5j, -2 + 1j, 0]], dtype=np.complex128)
    meta = {"coeff_shape": [2, 3], "coeff_layout": "CK", "complex_format": "complex", "flatten_order": "C"}
    codec = build_coeff_codec({"name": "auto", "params": {"dtype_policy": "float64"}})
    packed = codec.encode(raw, meta)
    assert packed.shape == (12,)
    assert not np.iscomplexobj(packed)
    np.testing.assert_allclose(codec.decode(packed, meta), raw, atol=1e-12)


@pytest.mark.parametrize(
    "options",
    [
        {"phase_unwrap": True},
        {"mode": "mag_phase", "phase_clip": -1.0},
        {"mode": "logmag_phase", "log_epsilon": 0.0},
        {"mode": "mag_phase", "log_epsilon": 0.01},
    ],
)
def test_phase_options_cannot_be_ignored_or_change_sign(options):
    with pytest.raises(ValueError):
        build_coeff_codec({"name": "fft_complex_codec_v1", **options})


@pytest.mark.parametrize("name,raw", [("none", np.ones((2, 2))), ("fft_complex_codec_v1", np.ones((2, 2)) + 1j)])
def test_short_coefficients_are_rejected_instead_of_repaired(name, raw):
    codec = build_coeff_codec({"name": name})
    meta = {"coeff_shape": [2, 2], "complex_format": "complex" if np.iscomplexobj(raw) else "real"}
    packed = codec.encode(raw, meta)
    with pytest.raises(ValueError, match="smaller"):
        codec.decode(packed[:-1], meta)


def test_real_and_composite_codecs_preserve_requested_float64_precision():
    codec = build_coeff_codec({"name": "auto", "params": {"dtype_policy": "float64"}})
    raw = np.array([[1.0000000001, 2.0000000001]])
    meta = {"coeff_shape": [1, 2], "coeff_layout": "CK", "complex_format": "real"}
    packed = codec.encode(raw, meta)
    assert packed.dtype == np.float64
    np.testing.assert_array_equal(codec.decode(packed, meta), raw)
    composite = {"offset": np.array([3.0000000001]), "residual": raw}
    composite_meta = {"coeff_format": "offset_residual_v1", "offset_dim": 1, "residual_raw_meta": meta}
    packed = codec.encode(composite, composite_meta)
    recovered = codec.decode(packed, composite_meta)
    assert packed.dtype == np.float64
    np.testing.assert_array_equal(recovered["offset"], composite["offset"])
    np.testing.assert_array_equal(recovered["residual"], raw)


@pytest.mark.parametrize("name", ["quantile", "power_yeojohnson"])
def test_inplace_backend_options_never_modify_caller_arrays(name):
    cfg = {"name": name, "params": {"copy": False}}
    if name == "quantile":
        cfg["params"]["n_quantiles"] = 4
    post = build_coeff_post(cfg)
    original = np.array([[0.0], [10.0], [20.0], [30.0]])
    a = original.copy()
    post.fit(a)
    np.testing.assert_array_equal(a, original)
    z = post.transform(a)
    z_snapshot = z.copy()
    recovered = post.inverse_transform(z)
    np.testing.assert_array_equal(a, original)
    np.testing.assert_array_equal(z, z_snapshot)
    np.testing.assert_allclose(recovered, original, rtol=1e-10, atol=1e-10)
    assert not np.shares_memory(z, recovered)


@pytest.mark.parametrize("solver", ["full", "randomized"])
def test_pca_full_energy_is_full_rank_roundtrip(solver):
    a = np.random.default_rng(4).normal(size=(12, 4))
    a[:, -1] = 3.0
    post = build_coeff_post({"name": "pca", "params": {"energy_threshold": 1.0, "svd_solver": solver}})
    post.fit(a)
    np.testing.assert_allclose(post.inverse_transform(post.transform(a)), a, atol=1e-12)
    assert post.latent_dim == 4
    with pytest.raises(ValueError, match="complete spectrum"):
        build_coeff_post({"name": "pca", "params": {"energy_threshold": 1.0, "svd_solver": "arpack"}})


@pytest.mark.parametrize("options", [{"per_channel": "false"}, {"per_chanel": False}, {"eps": True}])
def test_nested_preprocess_options_cannot_silently_change_standardization(options):
    with pytest.raises(ValueError):
        build_preprocess({"name": "basic", "ops": [{"name": "field_standardize", **options}]})


def test_coefficient_dictionary_rejects_a_field_mask_policy_without_effect():
    with pytest.raises(ValueError, match="Unknown dict_learning parameters.*mask_policy"):
        build_coeff_post({"name": "dict_learning", "mask_policy": "error"})


@pytest.mark.parametrize("invalid", [np.array([1 + 2j, 3 + 4j]), np.array([1.0, np.nan])])
def test_packed_coordinates_cannot_discard_invalid_values(invalid):
    codec = build_coeff_codec({"name": "none"})
    meta = {"coeff_shape": [2], "complex_format": "real"}
    with pytest.raises(ValueError, match="finite real"):
        codec.decode(invalid, meta)
    with pytest.raises(ValueError, match="finite real"):
        codec.decode_batch(invalid[None], meta)
    for name in ("none", "standardize"):
        post = build_coeff_post({"name": name})
        with pytest.raises(ValueError, match="finite real"):
            post.fit(invalid[None])
        post.fit(np.array([[1.0, 2.0], [3.0, 4.0]]))
        with pytest.raises(ValueError, match="finite real"):
            post.inverse_transform(invalid[None])


def test_power_inverse_domain_failure_is_explicit():
    post = build_coeff_post({"name": "power_yeojohnson", "params": {"standardize": False}})
    post.fit(np.exp(np.arange(8.0))[:, None])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with pytest.raises(ValueError, match="inverse transformation domain"):
            post.inverse_transform(np.array([[100.0]]))


@pytest.mark.parametrize("per_channel", [True, False])
def test_field_statistics_ignore_masked_nan_and_broadcast_common_mask(per_channel):
    mask = np.array([[True, True], [False, False]])
    fields = np.array([[[[1.0, 10.0], [3.0, 30.0]], [[np.nan, np.nan], [np.nan, np.nan]]]])
    fields = np.repeat(fields, 3, axis=0)
    alternate = fields.copy()
    alternate[:, 1, :, :] = 1000.0
    cfg = {"name": "basic", "params": {"ops": [{"name": "field_standardize", "per_channel": per_channel, "eps": 0}]}}
    first, second = build_preprocess(cfg), build_preprocess(cfg)
    first.fit(fields, mask)
    second.fit(alternate, mask)
    expected_mean = np.array([2.0, 20.0]).reshape(1, 1, 1, 2) if per_channel else 11.0
    np.testing.assert_allclose(first.state()["ops_state"]["field_standardize"]["mean"], expected_mean)
    np.testing.assert_allclose(
        first.state()["ops_state"]["field_standardize"]["scale"],
        second.state()["ops_state"]["field_standardize"]["scale"],
    )
    transformed, batch_mask = first.transform(fields, mask)
    assert batch_mask.shape == (3, 2, 2)
    assert np.isfinite(transformed).all()
    np.testing.assert_array_equal(transformed[:, 1, :, :], 0)
    recovered, _ = first.inverse_transform(transformed, batch_mask)
    np.testing.assert_allclose(recovered[batch_mask], fields[batch_mask], atol=1e-12)
    assert np.isnan(fields[:, 1, :, :]).all()


def test_field_preprocessing_rejects_invalid_observations_and_empty_statistics():
    post = build_preprocess({"name": "basic", "params": {"ops": ["field_standardize"]}})
    fields = np.ones((2, 2, 2, 1))
    with pytest.raises(ValueError, match="observed training"):
        post.fit(fields, np.zeros((2, 2), dtype=bool))
    fields[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        post.fit(fields, np.ones((2, 2), dtype=bool))


@pytest.mark.parametrize("per_channel", [True, False])
def test_field_statistics_are_stable_for_strided_readonly_values_with_large_offset(monkeypatch, per_channel):
    import ml_platform_mode_decomp.field_preprocessing as preprocessing

    rng = np.random.default_rng(9)
    origin = 2.0**40
    fields = (origin + rng.integers(-16, 17, size=(7, 9, 16, 6)) / 16)[:, :, ::-2, ::2]
    masks = rng.random(fields.shape[:-1]) > 0.3
    fields[~masks] = np.nan
    original = fields.copy()
    fields.flags.writeable = False
    centered = fields[masks] - origin
    expected_mean = origin + (centered.mean(axis=0) if per_channel else centered.mean())
    expected_std = centered.std(axis=0) if per_channel else centered.std()
    # Several short blocks exercise merging statistics independently of sample/spatial layout.
    monkeypatch.setattr(preprocessing, "_VALUE_BLOCK_ITEMS", 31)
    pre = build_preprocess(
        {"name": "basic", "ops": [{"name": "field_standardize", "per_channel": per_channel, "eps": 0}]}
    ).fit(fields, masks)
    state = pre.state()["ops_state"]["field_standardize"]
    np.testing.assert_array_equal(np.asarray(state["mean"]).reshape(-1), np.asarray(expected_mean).reshape(-1))
    np.testing.assert_allclose(np.asarray(state["scale"]).reshape(-1), np.asarray(expected_std).reshape(-1), rtol=1e-14)
    transformed, _ = pre.transform(fields, masks)
    restored, _ = pre.inverse_transform(transformed, masks)
    np.testing.assert_allclose(restored[masks], fields[masks], rtol=0, atol=2**-12)
    np.testing.assert_array_equal(transformed[~masks], 0)
    np.testing.assert_array_equal(fields, original)


def test_representation_preprocesses_chunks_and_matches_direct_standardized_dct(monkeypatch):
    from scipy.fft import dctn

    import ml_platform_mode_decomp.representation as representation
    from ml_platform_mode_decomp import FieldDataset
    from ml_platform_mode_decomp.geometry import DomainSpec

    fields = np.random.default_rng(10).normal(size=(9, 5, 16, 4))[:, :, ::-2, ::2]
    fields.flags.writeable = False
    original = fields.copy()
    xx, yy = np.meshgrid(np.arange(8, dtype=float), np.arange(5, dtype=float))
    domain = DomainSpec(name="rectangle", coords={"x": xx, "y": yy}, mask=None, weights=None, meta={})
    dataset = FieldDataset(fields, domain, np.asarray([str(i) for i in range(len(fields))]))
    model = representation.FieldRepresentation(
        {"decomposition": {"name": "dct2"}, "field_preprocess": {"name": "basic", "ops": ["field_standardize"]}}
    ).fit(dataset)
    mean = fields.mean(axis=(0, 1, 2), keepdims=True)
    scale = fields.std(axis=(0, 1, 2), keepdims=True) + 1e-8
    expected = dctn((fields - mean) / scale, axes=(1, 2), type=2, norm="ortho")
    expected = np.moveaxis(expected, -1, 1).reshape(len(fields), -1)
    preprocess_transform = model.preprocess.transform

    def bounded_transform(batch, masks):
        assert len(batch) <= 2
        return preprocess_transform(batch, masks)

    monkeypatch.setattr(representation, "_TRANSFORM_WORK_BYTES", fields[0].size * 16 * 6 * 2)
    monkeypatch.setattr(model.preprocess, "transform", bounded_transform)
    coefficients = model.transform(dataset, space="a")
    np.testing.assert_allclose(coefficients, expected, atol=1e-12)
    np.testing.assert_allclose(model.inverse_transform(coefficients, space="a"), fields, atol=1e-12)
    np.testing.assert_array_equal(fields, original)


def test_physical_inverse_rejects_overflow_even_when_predicted_coefficients_are_finite():
    from ml_platform_mode_decomp import FieldDataset, FieldSurrogate

    conditions = np.arange(8.0)[:, None]
    fields = (1e150 * conditions[:, 0])[:, None, None, None] * np.ones((1, 2, 2, 1))
    model = FieldSurrogate(
        {
            "representation": {
                "decomposition": {"name": "dct2"},
                "field_preprocess": {"name": "basic", "ops": ["field_standardize"]},
            },
            "regression": {"name": "ridge", "alpha": 0.0},
        }
    ).fit(FieldDataset.from_arrays(fields, conditions))
    query = np.array([[1e200]])
    assert np.isfinite(model.predict_coefficients(query)).all()
    with np.errstate(over="ignore"), pytest.raises(ValueError, match="Physical field inverse.*non-finite"):
        model.predict(query)


def test_lgbm_fraction_enables_subsampling_and_rejects_ineffective_pair(monkeypatch):
    monkeypatch.setitem(sys.modules, "lightgbm", SimpleNamespace(LGBMRegressor=lambda **kwargs: kwargs))
    model = build_regressor({"name": "lgbm", "params": {"subsample": 0.6}})
    effective = model._base_factory()
    assert effective["subsample"] == 0.6
    assert effective["subsample_freq"] == 1
    with pytest.raises(ValueError, match="subsample_freq"):
        build_regressor({"name": "lgbm", "params": {"subsample": 0.6, "subsample_freq": 0}})


def test_gpr_multioutput_mean_std_are_in_target_units():
    x = np.arange(5.0)[:, None]
    y = np.column_stack([np.sin(x[:, 0]), 100 * np.sin(x[:, 0]) + 7])
    model = build_regressor({"name": "gpr", "params": {"optimizer": None, "white_noise": False}})
    model.fit(x, y)
    mean, std = model.predict_with_std(np.array([[0.5], [1.5]]))
    assert mean.shape == std.shape == (2, 2)
    np.testing.assert_allclose(std[:, 1], 100 * std[:, 0], rtol=1e-6)


def test_mtgp_fit_preserves_global_random_states_when_available():
    torch = pytest.importorskip("torch")
    pytest.importorskip("gpytorch")
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state().clone()
    x = np.arange(6.0, dtype=np.float32)[:, None]
    y = np.column_stack([x[:, 0], x[:, 0] ** 2])
    model = build_regressor({"name": "mtgp", "params": {"seed": 3, "training_iter": 2}})
    model.fit(x, y)
    after = np.random.get_state()
    assert after[0] == numpy_state[0]
    np.testing.assert_array_equal(after[1], numpy_state[1])
    assert after[2:] == numpy_state[2:]
    assert torch.equal(torch.random.get_rng_state(), torch_state)


@pytest.mark.parametrize("seed_key", ["seed", "random_state"])
def test_randomized_pca_reproduces_with_local_seed(seed_key):
    data = np.random.default_rng(28).normal(size=(80, 64))

    def fitted(seed):
        post = build_coeff_post(
            {"name": "pca", "params": {"n_components": 5, "svd_solver": "randomized", seed_key: seed}}
        )
        return post.fit(data)

    first, same, different = fitted(13), fitted(13), fitted(29)
    np.testing.assert_array_equal(first.transform(data), same.transform(data))
    assert not np.allclose(first.transform(data), different.transform(data), atol=1e-7, rtol=1e-7)
    assert first.state()["random_state"] == 13
