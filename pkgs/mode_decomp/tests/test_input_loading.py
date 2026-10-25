"""Identified coefficient inputs and independent source case alignment."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from ml_platform_core.artifacts import file_hash

from ml_platform_mode_decomp import FieldDataset, FieldRepresentation, FieldSurrogate
from ml_platform_mode_decomp.data import load_condition_batch
from ml_platform_mode_decomp.inputs import (
    CoefficientBatch,
    align_inputs,
    load_surrogate_inputs,
    normalize_input_settings,
    subset_inputs,
)
from ml_platform_mode_decomp.model_artifact import save_model
from ml_platform_mode_decomp.policy import candidate_config, resolve_config


def _source(root, *, surrogate=False):
    fields = np.arange(12.0).reshape(3, 2, 2, 1)
    data = FieldDataset.from_arrays(fields, np.arange(3.0)[:, None], sample_ids=["alpha", "beta", "gamma"])
    bundle = (
        FieldSurrogate({"representation": {"decomposition": {"name": "dct2"}}}).fit(data)
        if surrogate
        else FieldRepresentation({"decomposition": {"name": "dct2"}}).fit(data)
    )
    representation = bundle.representation if surrogate else bundle
    artifact, _ = save_model(bundle, root / "source.joblib")
    identity = file_hash(artifact)
    coefficients = representation.transform(data)
    archive = root / "source.npz"
    np.savez_compressed(
        archive, coefficients=coefficients, sample_ids=data.sample_ids, space="z", model_sha256=identity
    )
    settings = {
        "local_path": str(root),
        "input": {"kind": "coefficients", "path": archive.name, "model_artifact": artifact.name},
    }
    return data, representation, artifact, coefficients, settings


def test_condition_id_column_is_not_ignored_for_arrays_or_manifests(tmp_path):
    np.save(tmp_path / "conditions.npy", np.array([[1.0], [2.0]]))
    for empty_manifest in ("", {}):
        with pytest.raises(ValueError):
            load_condition_batch(
                {"local_path": str(tmp_path / "conditions.npy"), "source_manifest": empty_manifest},
                condition_columns=["x"],
            )
    for selection in ({"dataset_file": "conditions.npy"}, {"source_manifest": "conditions.yaml"}):
        with pytest.raises(ValueError, match="data.id_column"):
            load_condition_batch(
                {"local_path": str(tmp_path), "id_column": "case_id", **selection}, condition_columns=["x"]
            )
    (tmp_path / "conditions.csv").write_text("case_id,x\none,1\ntwo,2\n", encoding="utf-8")
    batch = load_condition_batch(
        {"local_path": str(tmp_path), "dataset_file": "conditions.csv", "id_column": "case_id"},
        condition_columns=["x"],
    )
    np.testing.assert_array_equal(batch.sample_ids, ["one", "two"])


@pytest.mark.parametrize("surrogate", [False, True])
def test_coefficient_loader_uses_identified_output_representation_and_aligns_cases(tmp_path, surrogate):
    data, representation, artifact, coefficients, settings = _source(tmp_path, surrogate=surrogate)
    target = data.subset([2, 0, 1])
    loaded = load_surrogate_inputs(settings, target=target)
    assert isinstance(loaded, CoefficientBatch)
    np.testing.assert_array_equal(loaded.sample_ids, target.sample_ids)
    np.testing.assert_array_equal(loaded.coefficients, coefficients[[2, 0, 1]])
    np.testing.assert_allclose(loaded.representation.inverse_transform(coefficients), data.fields, atol=1e-12)
    assert loaded.model_sha256 == file_hash(artifact)
    assert loaded.revision == file_hash(tmp_path / "source.npz")
    subset = subset_inputs(loaded, [1, 2])
    assert subset.representation is loaded.representation and subset.revision == loaded.revision
    np.testing.assert_array_equal(subset.sample_ids, ["alpha", "beta"])
    assert representation.target_dim == loaded.representation.target_dim


def test_saved_coefficient_model_needs_no_original_representation_artifact(tmp_path):
    data, representation, artifact, coefficients, settings = _source(tmp_path)
    saved = SimpleNamespace(
        input_kind="coefficients", input_representation=representation, input_model_sha256=file_hash(artifact)
    )
    artifact.unlink()
    settings["input"].pop("model_artifact")
    query = load_surrogate_inputs(settings, model=saved)
    np.testing.assert_array_equal(query.coefficients, coefficients)
    assert query.representation is representation
    with pytest.raises(ValueError, match="saved surrogate input kind"):
        load_surrogate_inputs(settings, model=SimpleNamespace(input_kind="field"))
    saved.input_model_sha256 = "0" * 64
    with pytest.raises(ValueError, match="saved input representation"):
        load_surrogate_inputs(settings, model=saved)
    assert load_surrogate_inputs({"local_path": str(tmp_path)}, target=data, model=saved) is None


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"space": "a"}, "only Z space"),
        ({"model_sha256": "0" * 64}, "model_sha256 differs"),
        ({"model_sha256": "unidentified"}, "complete SHA256"),
        ({"coefficients": np.ones((3, 3))}, "dimension differs"),
        ({"coefficients": np.ones((3, 2, 2))}, "finite real matrix"),
        ({"coefficients": np.full((3, 4), np.nan)}, "finite real matrix"),
        ({"sample_ids": np.array(["alpha", "alpha", "beta"])}, "unique"),
    ],
)
def test_coefficient_loader_rejects_wrong_coordinates_and_corrupt_declarations(tmp_path, changes, match):
    data, _, artifact, coefficients, settings = _source(tmp_path)
    payload = dict(coefficients=coefficients, sample_ids=data.sample_ids, space="z", model_sha256=file_hash(artifact))
    payload.update(changes)
    np.savez_compressed(tmp_path / "source.npz", **payload)
    with pytest.raises(ValueError, match=match):
        load_surrogate_inputs(settings)


def test_alignment_rejects_missing_extra_and_duplicate_cases_instead_of_inner_join():
    source = FieldDataset.from_arrays(np.ones((3, 2, 2, 1)), sample_ids=["a", "b", "c"])
    target = FieldDataset.from_arrays(np.ones((3, 2, 2, 1)), sample_ids=["b", "c", "d"])
    with pytest.raises(ValueError, match="match exactly"):
        align_inputs(source, target)
    source.sample_ids[1] = "a"  # Borrowed arrays are rechecked at the alignment boundary.
    with pytest.raises(ValueError, match="unique"):
        align_inputs(source, target)
    assert align_inputs(None, target) is None and subset_inputs(None, [0]) is None


def test_field_loader_preserves_own_geometry_and_checks_saved_source_schema(tmp_path):
    fields = np.arange(18.0).reshape(3, 2, 3, 1)
    np.save(tmp_path / "source.npy", fields)
    settings = {
        "input": {"kind": "field", "local_path": str(tmp_path), "fields": {"path": "source.npy", "layout": "NHWC"}}
    }
    source = load_surrogate_inputs(settings)
    np.testing.assert_array_equal(source.fields, fields)
    representation = FieldRepresentation({"decomposition": {"name": "dct2"}}).fit(source)
    saved = SimpleNamespace(input_kind="field", input_representation=representation)
    np.testing.assert_array_equal(load_surrogate_inputs(settings, model=saved).fields, fields)
    np.save(tmp_path / "wrong_shape.npy", np.ones((3, 3, 2, 1)))
    settings["input"]["fields"]["path"] = "wrong_shape.npy"
    with pytest.raises(ValueError, match="geometry differs"):
        load_surrogate_inputs(settings, model=saved)


def _pipeline_config(root):
    return {
        "task": "mode_decomp_pipeline",
        "data": {"local_path": str(root), "fields": {"path": "target.npy"}},
        "model": {"candidates": [{"id": "candidate"}]},
    }


def test_policy_preserves_legacy_candidate_shape_and_checks_explicit_source_roles(tmp_path):
    assert set(candidate_config({"id": "legacy"}, 42)) == {"id", "representation", "regression"}
    cfg = _pipeline_config(tmp_path)
    cfg["data"]["input"] = {"kind": "field", "fields": {"path": "source.npy"}}
    with pytest.raises(ValueError, match="requires input_representation"):
        resolve_config(cfg)
    cfg["model"]["candidates"][0]["input_representation"] = {"decomposition": {"name": "pod_svd"}}
    resolved = resolve_config(cfg)
    assert resolved["data"]["input"]["local_path"] == str(tmp_path)
    assert resolved["model"]["candidates"][0]["input_representation"]["decomposition"]["name"] == "pod_svd"
    cfg["data"]["input"] = {"kind": "coefficients", "path": "coefficients.npz", "model_artifact": "model.joblib"}
    with pytest.raises(ValueError, match="do not also set input_representation"):
        resolve_config(cfg)
    cfg["data"].pop("input")
    with pytest.raises(ValueError, match="requires data.input.kind"):
        resolve_config(cfg)


def test_policy_tunes_source_representation_only_when_it_will_be_fitted(tmp_path):
    cfg = _pipeline_config(tmp_path)
    cfg["data"]["input"] = {"kind": "field", "fields": {"path": "source.npy"}}
    cfg["model"]["candidates"][0]["input_representation"] = {"decomposition": {"name": "pod_svd"}}
    cfg["tuning"] = {
        "enabled": True,
        "n_trials": 2,
        "search_spaces": {"candidate": {"input_representation.decomposition.params.n_modes": [1, 2]}},
    }
    assert resolve_config(cfg)["tuning"]["n_trials"] == 2
    invalid = deepcopy(cfg)
    invalid["model"]["candidates"][0].pop("input_representation")
    with pytest.raises(ValueError, match="tuning requires a field input representation"):
        resolve_config(invalid)


def test_input_settings_reject_unknown_keys_and_unidentified_arrays_and_admit_stage_sources(tmp_path):
    with pytest.raises(ValueError, match="Unknown data.input"):
        normalize_input_settings({"input": {"kind": "field", "fileds": {"path": "typo.npy"}}})
    with pytest.raises(ValueError, match="model_artifact"):
        normalize_input_settings({"input": {"kind": "coefficients", "path": "unidentified.npz"}})
    cfg = _pipeline_config(tmp_path)
    cfg["task"] = "mode_decomp_stage"
    cfg["run"] = {"stage": "prepare_fields"}
    cfg["data"]["input"] = {"kind": "coefficients", "path": "source.npz", "model_artifact": "source.joblib"}
    assert resolve_config(cfg)["data"]["input"]["kind"] == "coefficients"
