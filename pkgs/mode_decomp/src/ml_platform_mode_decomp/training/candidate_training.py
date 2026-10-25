from copy import deepcopy

from ml_platform_core.io import write_json, write_table

from ..evaluation import prepare_objective, score_surrogate
from ..model_artifact import evaluation_identity
from ..surrogate import FieldSurrogate
from .preparation import needs_selection
from .search import fold_training_ids, search_candidate


def selection_scope(cfg):
    if cfg.get("tuning", {}).get("enabled", False):
        return "inner_cv"
    return "selection_holdout" if needs_selection(cfg) else "fixed_configuration"


def selection_options(cfg, *, scope=None):
    options = deepcopy(cfg["metrics"])
    options["names"] = sorted(set(options["names"]) | {cfg["model"]["selection_metric"]})
    options["reference_scope"] = scope or selection_scope(cfg)
    options["primary_metric"] = cfg["model"]["selection_metric"]
    return options


def evaluate_candidate(cfg, candidate, training, selection=None, *, prepared=None, folds=None):
    """Accept fit/selection roles only; final-test truth cannot enter this kernel."""
    evaluation = selection_options(cfg)
    search = None
    if cfg.get("tuning", {}).get("enabled", False):
        searched = search_candidate(cfg, candidate, training, evaluation, folds=folds)
        if searched.candidate is None or searched.report is None:
            return None, None, searched
        candidate, report = searched.candidate, searched.report
        fit_ids = fold_training_ids(training, searched.folds)
        search = {"settings": cfg["tuning"], "best_trial": searched.best_trial}
    else:
        if selection is None or selection.conditions is None:
            raise ValueError("Candidate selection requires held-out selection conditions")
        if prepared is None:
            prepared = prepare_objective(selection, evaluation, training=training, data=cfg["data"])
        model = FieldSurrogate(candidate).fit(training)
        report = score_surrogate(model, training, selection, evaluation, prepared=prepared)
        fit_ids = list(model.fit_sample_ids)
        searched = None
    payload = {
        "schema_version": 1,
        "candidate_id": candidate["id"],
        "candidate": candidate,
        "metrics": report.metrics,
        "evaluation": evaluation,
        "evaluation_identity": evaluation_identity(candidate),
        "metric_scope": selection_scope(cfg),
        "fit_ids": fit_ids,
        "search": search,
    }
    return payload, report, searched


def write_candidate(cfg, result, payload, report, searched):
    if searched is not None:
        result.tables["search_trials"] = write_table(searched.trials, result.run_dir / "search_trials.csv")
        result.artifacts["cv_folds"] = write_json({"folds": searched.folds}, result.run_dir / "cv_folds.json")
    if payload is None or report is None:
        raise ValueError(f"All tuning trials failed for candidate {cfg['candidate']['id']!r}; see search_trials.csv")
    result.artifacts["candidate_result"] = write_json(payload, result.run_dir / "candidate_result.json")
    result.artifacts["analysis_summary"] = write_json(report.summary, result.run_dir / "analysis_summary.json")
    result.tables["per_sample"] = write_table(report.per_sample, result.run_dir / "per_sample.csv")
    result.metrics.update(report.metrics)
    result.extra.update(metric_scope=payload["metric_scope"], candidate_id=payload["candidate_id"])
