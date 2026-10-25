"""Execute one declared business Stage with local paths and plain settings."""

from importlib import import_module
from typing import Any

from ml_platform_core.result import RunResult

from .manifest import MODE_DECOMP_MANIFEST
from .stage_result import execute_stage

_OPERATIONS = {
    "prepare_fields": (".training.preparation", "prepare_fields"),
}


def train_candidate(cfg, result):
    from .stage_inputs import load_stage_dataset
    from .training.candidate_training import evaluate_candidate, selection_options, write_candidate
    from .training.preparation import needs_selection
    from .training.search import prepare_folds

    dataset, split = load_stage_dataset(cfg)
    if not needs_selection(cfg):
        raise ValueError("A fixed single candidate does not require a train_candidate Stage")
    tuning = cfg.get("tuning", {}).get("enabled", False)
    training = dataset._subset_validated(split["train_indices"] if tuning else split["fit_indices"])
    selection = None if tuning else dataset._subset_validated(split["selection_indices"])
    folds = (
        prepare_folds(
            training,
            cfg["tuning"]["cv"],
            selection_options(cfg),
            source_indices=split["train_indices"],
            data=cfg["data"],
        )
        if tuning
        else None
    )
    payload, report, searched = evaluate_candidate(cfg, cfg["candidate"], training, selection, folds=folds)
    if payload is not None:
        payload.update(
            dataset_revision=dataset.revision,
            split_fingerprint=split["fingerprint"],
            execution_context=cfg["runtime"].get("execution_context", {}),
        )
    write_candidate(cfg, result, payload, report, searched)


def finalize_surrogate(cfg, result):
    from .stage_inputs import load_candidate_results, load_stage_dataset
    from .training.candidate_training import selection_scope
    from .training.finalization import finalize_selected

    dataset, split = load_stage_dataset(cfg)
    candidates = load_candidate_results(cfg, dataset, split) if selection_scope(cfg) != "fixed_configuration" else []
    finalize_selected(cfg, result, dataset, split, candidates)


def run_stage(cfg: dict[str, Any]) -> RunResult:
    stage = cfg.get("run", {}).get("stage")
    MODE_DECOMP_MANIFEST.stage(stage)
    if stage == "train_candidate":
        operation = train_candidate
    elif stage == "finalize_surrogate":
        operation = finalize_surrogate
    else:
        module, name = _OPERATIONS[stage]
        operation = getattr(import_module(module, __package__), name)
    return execute_stage(cfg, stage, operation)
