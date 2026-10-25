"""Validate business meaning after transport has resolved artifacts to paths."""

from pathlib import Path

from ml_platform_core.io import read_json

from .data.splitting import cross_validation_folds
from .training.preparation import PreparedExecution, needs_selection
from .training.calibration import validate_calibration_split

from .data import load_field_dataset, split_dataset
from .inputs import load_surrogate_inputs


def load_stage_execution(cfg):
    refs = cfg.get("stage_inputs", {})
    split = read_json(refs["split"])
    reference = refs.get("canonical_dataset")
    if reference is not None:
        from .input_preparation import stage_dataset_config

        data, revision = stage_dataset_config(reference)
        dataset = load_field_dataset(data)
        if dataset.revision != revision:
            raise ValueError("Canonical Dataset content differs from its prepared revision")
        # The resolved canonical settings also serve original-observation evaluation.
        cfg["data"] = data
    else:
        dataset = load_field_dataset(cfg["data"])
    inputs = load_surrogate_inputs(cfg["data"], target=dataset)
    if reference is not None:
        record = read_json(Path(reference) / "stage_input.json")
        if record["schema"] == "canonical_stage_input_v2" and (
            inputs is None or inputs.revision != record["input_revision"]
        ):
            raise ValueError("Canonical source input content differs from its prepared revision")
    expected_split = split_dataset(
        dataset,
        cfg["split"],
        cfg["run"]["seed"],
        nested=needs_selection(cfg) and not cfg.get("tuning", {}).get("enabled", False),
        calibration=cfg.get("uncertainty"),
    )
    if split != expected_split:
        raise ValueError("Split artifact differs from the fixed Dataset and split configuration")
    validate_calibration_split(dataset, split, cfg["uncertainty"])
    return PreparedExecution(dataset, split, inputs)


def load_stage_dataset(cfg):
    """Keep the former target/split helper for condition-only callers."""
    prepared = load_stage_execution(cfg)
    if prepared.inputs is not None:
        raise ValueError("Source-backed Stage execution requires load_stage_execution, which retains the source inputs")
    return prepared.dataset, prepared.split


def _candidate_context(context):
    # Candidate and final evaluation are different Tasks within one admitted DAG.
    # Workspace routing is validated by the SDK; compare it here when supplied.
    return {
        key: (context or {}).get(key)
        for key in (
            "source_pipeline_task_id",
            "source_release_id",
            "execution_release_id",
            "runtime_id",
            "workspace_key",
        )
    }


def load_candidate_results(cfg, dataset, split, *, inputs=None):
    from .training.candidate_training import selection_options, selection_scope
    from .training.search import fold_training_ids, trial_candidates

    payloads = [read_json(path) for path in cfg["stage_inputs"]["candidate_results"]]
    expected = {item["id"]: item for item in cfg["model"]["candidates"]}
    ids = [item.get("candidate_id") for item in payloads]
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise ValueError("Candidate results do not match the planned candidate IDs")
    if cfg["data"].get("input") is not None and inputs is None:
        raise ValueError("Source candidate validation requires the prepared source inputs")
    for item in payloads:
        candidate = expected[item["candidate_id"]]
        tuned = cfg.get("tuning", {}).get("enabled", False)
        expected_fit_ids = split.get("fit_ids")
        if tuned:
            search = item.get("search")
            if not isinstance(search, dict) or search.get("settings") != cfg["tuning"]:
                raise ValueError("Candidate result uses another tuning configuration")
            trial = search.get("best_trial")
            if isinstance(trial, bool) or not isinstance(trial, int) or not 0 <= trial < cfg["tuning"]["n_trials"]:
                raise ValueError("Candidate result has no valid selected trial")
            candidate = trial_candidates(candidate, cfg["tuning"], cfg["run"]["seed"])[trial]
            training = dataset._subset_validated(split["train_indices"])
            expected_fit_ids = fold_training_ids(training, cross_validation_folds(training, cfg["tuning"]["cv"]))
        if (
            item.get("schema_version") != 1
            or item.get("dataset_revision") != dataset.revision
            or item.get("split_fingerprint") != split["fingerprint"]
            or item.get("input_revision") != (None if inputs is None else inputs.revision)
            or item.get("candidate") != candidate
            or item.get("fit_ids") != expected_fit_ids
            or item.get("evaluation") != selection_options(cfg)
            or item.get("metric_scope") != selection_scope(cfg)
            or _candidate_context(item.get("execution_context"))
            != _candidate_context(cfg["runtime"].get("execution_context"))
        ):
            raise ValueError("Candidate result belongs to another data, split, candidate, or execution context")
    return payloads
