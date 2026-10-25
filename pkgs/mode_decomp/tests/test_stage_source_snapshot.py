"""External Stages own both sides of a supervised field mapping."""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from ml_platform_core.artifacts import file_hash
from ml_platform_core.result import RunError

from ml_platform_mode_decomp import FieldDataset, FieldRepresentation, load_model, run_task
from ml_platform_mode_decomp.input_preparation import stage_dataset_config
from ml_platform_mode_decomp.model_artifact import save_model


@pytest.mark.parametrize("kind", ["field", "coefficients"])
def test_source_stages_are_self_contained_and_reject_other_source_candidates_or_modified_data(tmp_path, kind):
    values = np.linspace(-1, 1, 60)
    ids = np.array([f"case_{i:03d}" for i in range(len(values))])
    source_root, target_root = tmp_path / "raw_source", tmp_path / "raw_target"
    source_root.mkdir()
    target_root.mkdir()
    sx, sy = np.meshgrid(np.linspace(0, 2, 4), np.linspace(-1, 1, 3))
    tx, ty = np.meshgrid(np.linspace(0, 2, 3), np.linspace(-1, 1, 2))
    source_fields = (3 + values[:, None, None] * (1 + sx))[..., None]
    target_fields = (5 + values[:, None, None] * (2 + ty + 0.2 * tx))[..., None]
    for root, fields in ((source_root, source_fields), (target_root, target_fields)):
        np.save(root / "field.npy", fields)
        order = np.arange(len(values))[::-1] if root == source_root else np.arange(len(values))
        pd.DataFrame({"sample_id": ids[order], "array_index": order}).to_csv(root / "samples.csv", index=False)
    field_input = {
        "kind": "field",
        "local_path": str(source_root),
        "fields": {"path": "field.npy", "layout": "NHWC"},
        "samples": {"path": "samples.csv"},
    }
    source_artifact_hash = None
    if kind == "coefficients":
        source = FieldDataset.from_arrays(source_fields, sample_ids=ids)
        representation = FieldRepresentation({"decomposition": {"name": "pod_svd", "params": {"n_modes": 2}}})
        coordinates = representation.fit_transform(source)
        artifact, _ = save_model(representation, source_root / "source_model.joblib")
        source_artifact_hash = file_hash(artifact)
        np.savez_compressed(
            source_root / "coefficients.npz",
            coefficients=coordinates,
            sample_ids=ids,
            space="z",
            model_sha256=source_artifact_hash,
        )
        field_input = {
            "kind": "coefficients",
            "local_path": str(source_root),
            "path": "coefficients.npz",
            "model_artifact": artifact.name,
        }
    candidates = []
    for name, alpha in (("small", 1e-10), ("regularized", 0.1)):
        candidate = {
            "id": name,
            "representation": {"decomposition": {"name": "pod_svd", "params": {"n_modes": 2}}},
            "regression": {"name": "ridge", "params": {"alpha": alpha, "solver": "svd"}},
        }
        if kind == "field":
            candidate["input_representation"] = {"decomposition": {"name": "pod_svd", "params": {"n_modes": 2}}}
        candidates.append(candidate)
    config = {
        "task": "mode_decomp_stage",
        "run": {"stage": "prepare_fields"},
        "runtime": {"output_dir": str(tmp_path / "runs")},
        "data": {
            "local_path": str(target_root),
            "fields": {"path": "field.npy", "layout": "NHWC"},
            "samples": {"path": "samples.csv"},
            "input": field_input,
        },
        "model": {"candidates": candidates},
        "split": {"valid_size": 0.2, "selection_size": 0.25},
        "uncertainty": {"enabled": True, "calibration_size": 0.3},
    }
    if kind == "coefficients":
        config["tuning"] = {"enabled": True, "n_trials": 1, "cv": {"n_splits": 3}}
        config["split"].pop("selection_size")
    prepared = run_task(config)
    snapshot = prepared.artifacts["canonical_dataset"]
    record = json.loads((snapshot / "stage_input.json").read_text(encoding="utf-8"))
    assert record["schema"] == "canonical_stage_input_v2"
    assert not Path(record["data"]["input"]["local_path"]).is_absolute()
    if kind == "coefficients":
        assert file_hash(snapshot / "source" / "model.joblib") == source_artifact_hash
    source_root.rename(tmp_path / "hidden_source")
    target_root.rename(tmp_path / "hidden_target")
    config["stage_inputs"] = {"canonical_dataset": str(snapshot), "split": str(prepared.artifacts["split"])}
    candidate_results = []
    for candidate in candidates:
        trial = deepcopy(config)
        trial["run"]["stage"] = "train_candidate"
        trial["candidate"] = candidate
        result = run_task(trial)
        payload = json.loads(result.artifacts["candidate_result"].read_text(encoding="utf-8"))
        assert payload["input_revision"] == record["input_revision"]
        candidate_results.append(str(result.artifacts["candidate_result"]))
        effective = yaml.safe_load(result.artifacts["config"].read_text(encoding="utf-8"))["data"]
        assert Path(effective["input"]["local_path"]).is_relative_to(snapshot)
    final_config = deepcopy(config)
    final_config["run"]["stage"] = "finalize_surrogate"
    final_config["stage_inputs"]["candidate_results"] = candidate_results
    final = run_task(final_config)
    assert final.metrics["field_rmse"] < 1e-8
    assert load_model(final.artifacts["model"]).calibrator is not None
    candidate_path = Path(candidate_results[0])
    payload = json.loads(candidate_path.read_text(encoding="utf-8"))
    original_revision = payload["input_revision"]
    payload["input_revision"] = "0" * 64
    candidate_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RunError, match="another data"):
        run_task(final_config)
    payload["input_revision"] = original_revision
    candidate_path.write_text(json.dumps(payload), encoding="utf-8")
    effective, _ = stage_dataset_config(snapshot)
    source = effective["input"]
    if kind == "field":
        path = Path(source["local_path"]) / "field.npy"
        fields = np.load(path, allow_pickle=False)
        fields[0, 0, 0, 0] += 1
        np.save(path, fields)
    else:
        path = Path(source["local_path"]) / source["path"]
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        arrays["coefficients"][0, 0] += 1
        np.savez_compressed(path, **arrays)
    with pytest.raises(RunError, match="source input content differs"):
        run_task(final_config)
