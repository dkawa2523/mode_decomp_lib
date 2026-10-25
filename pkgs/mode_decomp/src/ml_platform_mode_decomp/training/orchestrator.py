"""Ordinary local coordination; external Stage transport is a separate boundary."""

from dataclasses import replace

from ml_platform_core.io import write_json

from ..evaluation.objectives import prepare_objective
from ..stage_result import execute_stage
from .candidate_training import evaluate_candidate, selection_options, write_candidate
from .finalization import finalize_selected
from .preparation import needs_selection, prepare_execution
from .search import prepare_folds


def _select_candidates(cfg, result, dataset, split):
    candidates = []
    if needs_selection(cfg):
        tuning = cfg.get("tuning", {}).get("enabled", False)
        training = dataset._subset_validated(split["train_indices"] if tuning else split["fit_indices"])
        selection = None if tuning else dataset._subset_validated(split["selection_indices"])
        options = selection_options(cfg)
        reference = (
            None if selection is None else prepare_objective(selection, options, training=training, data=cfg["data"])
        )
        folds = (
            prepare_folds(
                training, cfg["tuning"]["cv"], options, source_indices=split["train_indices"], data=cfg["data"]
            )
            if tuning
            else None
        )
        for candidate in cfg["model"]["candidates"]:
            # Only the fitting roles, never the coordinator's whole Dataset, enter here.
            payload, report, searched = evaluate_candidate(
                cfg, candidate, training, selection, prepared=reference, folds=folds
            )
            directory = result.run_dir / "candidates" / candidate["id"]
            directory.mkdir(parents=True)
            child = replace(result, run_dir=directory, artifacts={}, tables={}, metrics={}, extra={})
            if payload is not None:
                payload.update(
                    dataset_revision=dataset.revision,
                    split_fingerprint=split["fingerprint"],
                    execution_context=cfg["runtime"].get("execution_context", {}),
                )
            # Keep durable comparison evidence, without reading it back to continue.
            try:
                write_candidate({**cfg, "candidate": candidate}, child, payload, report, searched)
            finally:
                result.artifacts.update({f"{candidate['id']}_{key}": path for key, path in child.artifacts.items()})
                result.tables.update({f"{candidate['id']}_{key}": path for key, path in child.tables.items()})
                if payload is None:
                    result.tables.update(child.tables)
                    result.extra["failed_candidate"] = candidate["id"]
            candidates.append(payload)
    return candidates


def _pipeline(cfg, result):
    prepared = prepare_execution(cfg)
    dataset, split = prepared.dataset, prepared.split
    result.artifacts["split"] = write_json(split, result.run_dir / "split.json")
    candidates = _select_candidates(cfg, result, dataset, split)
    finalize_selected(cfg, result, dataset, split, candidates)


def run_pipeline(cfg):
    return execute_stage(cfg, "pipeline", _pipeline)
