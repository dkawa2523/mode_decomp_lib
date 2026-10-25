"""Budgeted discrete search on shared CV folds inside an outer training set."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import math
import random
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from ..data.schema import FieldDataset
from ..data.splitting import cross_validation_folds
from ..inputs import subset_inputs
from ..evaluation import EvaluationReport, score_surrogate, mean_field_prediction
from ..policy import METRIC_DIRECTIONS, candidate_config
from ..surrogate import FieldSurrogate
from ..evaluation.objectives import PreparedObjective, prepare_objective


@dataclass
class SearchResult:
    candidate: dict[str, Any] | None
    report: EvaluationReport | None
    trials: pd.DataFrame
    folds: list[dict[str, Any]]
    best_trial: int | None


def fold_training_ids(training: FieldDataset, folds: list[dict]) -> list[str]:
    """Actual fit coverage; the last time-validation block is never fitted."""
    fitted = {sample_id for fold in folds for sample_id in fold["fit_ids"]}
    return [str(sample_id) for sample_id in training.sample_ids if sample_id in fitted]


def trial_candidates(candidate: dict, settings: dict, model_seed: int) -> list[dict]:
    """Enumerate equal unique trial budgets without constructing a large product."""
    space = settings["search_spaces"].get(candidate["id"], {})
    paths = sorted(space)
    lengths = [len(space[path]) for path in paths]
    total = math.prod(lengths)
    count = settings["n_trials"]
    indices = (
        range(count) if settings["method"] == "grid" else random.Random(settings["seed"]).sample(range(total), count)
    )
    candidates = []
    for index in indices:
        selected = deepcopy(candidate)
        choices = []
        for length in reversed(lengths):
            index, choice = divmod(index, length)
            choices.append(choice)
        for path, choice in zip(paths, reversed(choices), strict=True):
            node = selected
            keys = path.split(".")
            for key in keys[:-1]:
                child = node.setdefault(key, {})
                if not isinstance(child, dict):
                    raise ValueError(f"Search path traverses a non-mapping parameter: {path}")
                node = child
            node[keys[-1]] = deepcopy(space[path][choice])
        candidates.append(candidate_config(selected, model_seed))
    return candidates


@dataclass
class PreparedFold:
    record: dict
    fit_positions: list[int]
    validation_positions: list[int]
    reference: FieldDataset
    evaluation: PreparedObjective


@dataclass
class PreparedSearch:
    folds: list[PreparedFold]
    pooled: PreparedObjective


def prepare_folds(training, config, options, *, source_indices=None, data=None):
    """Persist parent-Dataset indices; local positions stay private to this role."""
    source = np.arange(len(training)) if source_indices is None else np.asarray(source_indices)
    local_folds = cross_validation_folds(training, config)
    positions = [position for fold in local_folds for position in fold["validation_indices"]]
    # Gather held-out values once; fold references below are contiguous slices of it.
    pooled_reference = training._subset_validated(positions)
    prepared = []
    start = 0
    for local in local_folds:
        record = dict(local)
        for key in ("fit_indices", "validation_indices", "purged_indices"):
            if key in record:
                record[key] = source[record[key]].tolist()
        record["index_space"] = "dataset"
        fit_positions, validation_positions = local["fit_indices"], local["validation_indices"]
        stop = start + len(validation_positions)
        validation = pooled_reference._subset_validated(slice(start, stop))
        baseline = mean_field_prediction(training, validation, indices=fit_positions)
        evaluation = prepare_objective(
            validation,
            {**options, "names": [], "primary_metric": None},
            baseline=baseline,
            data=data,
            allow_zero_weight=True,
        )
        prepared.append(PreparedFold(record, fit_positions, validation_positions, validation, evaluation))
        start = stop
    return PreparedSearch(prepared, prepare_objective(pooled_reference, options, data=data))


def _trial_report(
    candidate: dict, training: FieldDataset, prepared: PreparedSearch, options: dict, *, inputs=None
) -> EvaluationReport:
    folds = prepared.folds
    reports = []
    for fold in folds:
        fit = training._subset_validated(fold.fit_positions)
        fit_inputs = subset_inputs(inputs, fold.fit_positions)
        validation_inputs = subset_inputs(inputs, fold.validation_positions)
        model = FieldSurrogate(candidate)
        if fit_inputs is None:
            model.fit(fit)
        else:
            model.fit(fit, inputs=fit_inputs)
        evaluated = score_surrogate(
            model, fit, fold.reference, options, prepared=fold.evaluation, inputs=validation_inputs
        )
        # Retain sufficient sample losses, not fold predictions or A/Z arrays.
        evaluated.predictions = None
        reports.append(evaluated)
        del model
    report = prepared.pooled.pool(reports)
    report.per_sample["fold"] = np.concatenate(
        [np.full(len(fold.validation_positions), fold.record["fold"]) for fold in folds]
    )
    report.summary["analysis_kind"] = "surrogate_cv"
    if inputs is not None:
        report.summary["input_representation_fit_scope"] = (
            "fixed_input_representation" if hasattr(inputs, "coefficients") else "training_partition"
        )

    return report


def search_candidate(
    cfg: dict, candidate: dict, training: FieldDataset, options: dict, *, folds=None, inputs=None
) -> SearchResult:
    """The caller supplies outer train only; no final-test Dataset is accepted."""
    settings = cfg["tuning"]
    folds = folds if folds is not None else prepare_folds(training, settings["cv"], options, data=cfg["data"])
    metric = cfg["model"]["selection_metric"]
    sign = 1 if METRIC_DIRECTIONS[metric] == "min" else -1
    rows = []
    best_candidate = best_report = best_trial = None
    best_score = np.inf
    for trial, proposed in enumerate(trial_candidates(candidate, settings, cfg["run"]["seed"])):
        started = perf_counter()
        row = {"trial": trial, "candidate_id": candidate["id"], "configuration": json.dumps(proposed, sort_keys=True)}
        try:
            report = _trial_report(proposed, training, folds, options, inputs=inputs)
            score = report.metrics.get(metric, np.nan)
            if not np.isfinite(score):
                raise ValueError(f"CV selection metric {metric} must be finite")
            row.update(status="ok", **report.metrics)
            if sign * score < best_score:
                best_score, best_candidate, best_report, best_trial = sign * score, proposed, report, trial
        except (ValueError, RuntimeError, ImportError, np.linalg.LinAlgError) as exc:
            row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        row["elapsed_seconds"] = perf_counter() - started
        rows.append(row)
    if best_report is not None:
        best_report.summary.update(
            best_trial=best_trial,
            search_method=settings["method"],
            trial_budget=settings["n_trials"],
            cv=settings["cv"],
            folds=[fold.record for fold in folds.folds],
            outer_train_sample_count=len(training),
        )
    return SearchResult(
        best_candidate, best_report, pd.DataFrame(rows), [fold.record for fold in folds.folds], best_trial
    )
