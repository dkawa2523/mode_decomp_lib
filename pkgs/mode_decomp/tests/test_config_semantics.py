"""Explicit settings cannot silently select another operation or numerical meaning."""

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from ml_platform_mode_decomp.config import load_config
from ml_platform_mode_decomp.policy import candidate_config, resolve_config
from ml_platform_mode_decomp.regressors.sklearn import RidgeRegressor


@pytest.mark.parametrize(
    "settings,section",
    [
        ({"ouput": {}}, "run config"),
        ({"run": {"sead": 7}}, "run"),
        ({"runtime": {"output_dr": "ignored"}}, "runtime"),
        ({"model": {"canddiates": [{"id": "misspelled"}]}}, "model"),
        ({"metrics": {"channel_scalse": [100]}}, "metrics"),
        ({"analysis": {"bootstap_samples": 100}}, "analysis"),
        ({"output": {"predictions_name": "wrong.npy"}}, "output"),
    ],
)
def test_unknown_run_settings_are_rejected_before_defaults_can_hide_them(settings, section):
    with pytest.raises(ValueError, match=f"Unknown {section}"):
        resolve_config({"task": "mode_decomp_pipeline", **settings})


@pytest.mark.parametrize(
    "task,settings",
    [
        ("mode_decomp_representation", {"model": {"candidates": [{"id": "unused"}]}}),
        ("mode_decomp_representation", {"output": {"prediction_name": "requested.npy"}}),
        ("mode_decomp_representation", {"analysis": {"kind": "surrogate"}}),
        ("mode_decomp_representation", {"analysis": {"bootstrap_samples": 100}}),
        ("mode_decomp_pipeline", {"analysis": {"components": {"enabled": True}}}),
        ("mode_decomp_pipeline", {"representation": {"decomposition": {"name": "dct2"}}}),
        ("mode_decomp_infer", {"representation": {"decomposition": {"name": "dct2"}}}),
        ("mode_decomp_infer", {"metrics": {"names": ["field_rmse"]}}),
        ("mode_decomp_analysis", {"model": {"selection_metric": "field_rmse"}}),
        ("mode_decomp_analysis", {"analysis": {"kind": "surrogate", "observed_fraction": 0.1}}),
        ("mode_decomp_import", {"metrics": {"names": ["field_rmse"]}}),
        (
            "mode_decomp_import",
            {"import": {"format": "csv", "output_dir": "converted", "condition_columns": ["ignored"]}},
        ),
        ("mode_decomp_pipeline", {"stage_inputs": {"split": "unused.json"}}),
    ],
)
def test_settings_for_another_task_do_not_produce_a_successful_different_operation(task, settings):
    with pytest.raises(ValueError):
        resolve_config({"task": task, **settings})


@pytest.mark.parametrize("value", [False, [], 0, "", None])
def test_falsy_representation_is_not_replaced_by_default_pod(value):
    with pytest.raises((TypeError, ValueError), match="mapping"):
        candidate_config({"id": "invalid", "representation": value}, 42)


@pytest.mark.parametrize("section", ["analysis", "metrics", "model", "output", "run", "runtime"])
def test_explicit_null_sections_are_not_silently_defaulted(section):
    with pytest.raises(ValueError, match="mapping"):
        resolve_config({"task": "mode_decomp_pipeline", section: None})


@pytest.mark.parametrize("entry", ["field_preprocess", "decomposition", "codec", "coeff_post"])
def test_explicit_null_representation_components_are_not_an_implicit_method(entry):
    with pytest.raises(TypeError, match="not null"):
        candidate_config({"id": "invalid", "representation": {entry: None}}, 42)


@pytest.mark.parametrize("value", ["false", "true", 0, 1, None])
def test_boolean_component_defaults_require_actual_boolean_values_on_both_entries(value):
    with pytest.raises(ValueError, match="fit_intercept must be boolean"):
        candidate_config({"id": "invalid", "regression": {"name": "ridge", "params": {"fit_intercept": value}}}, 42)
    with pytest.raises(ValueError, match="fit_intercept must be boolean"):
        RidgeRegressor(fit_intercept=value)


def test_constructor_identity_and_conflicting_candidate_choices_are_explicit():
    with pytest.raises(ValueError, match="same component name"):
        RidgeRegressor(cfg={"name": "gpr"})
    with pytest.raises(ValueError, match="candidate or model.candidates"):
        resolve_config(
            {"task": "mode_decomp_pipeline", "candidate": {"id": "one"}, "model": {"candidates": [{"id": "two"}]}}
        )
    resolved = resolve_config({"task": "mode_decomp_pipeline", "candidate": {"id": "single"}})
    assert resolved["model"]["candidates"][0]["id"] == "single" and "candidate" not in resolved
    assert resolve_config(resolved) == resolved


@pytest.mark.parametrize("root_value", [False, [], 0, "", None])
def test_nonmapping_yaml_does_not_fall_back_to_an_active_profile(tmp_path, root_value):
    profile, run = tmp_path / "profile.yaml", tmp_path / "run.yaml"
    profile.write_text("task: mode_decomp_pipeline\n", encoding="utf-8")
    run.write_text(yaml.safe_dump(root_value), encoding="utf-8")
    with pytest.raises(ValueError, match="must contain a mapping"):
        load_config(run, profile)


def test_shipped_yaml_is_replayable_without_mutating_user_choices():
    root = Path(__file__).resolve().parents[3]
    for path in (root / "config" / "tasks").glob("*.yaml"):
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        before = deepcopy(raw)
        resolved = resolve_config(raw)
        assert raw == before, path
        assert resolve_config(resolved) == resolved, path


def test_conflicting_analysis_split_seeds_are_not_overwritten():
    with pytest.raises(ValueError, match="split_seeds conflicts"):
        resolve_config({"task": "mode_decomp_analysis", "split": {"seed": 7}, "analysis": {"split_seeds": [8, 9]}})
