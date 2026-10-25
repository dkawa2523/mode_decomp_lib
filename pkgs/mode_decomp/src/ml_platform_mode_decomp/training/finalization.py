import numpy as np
import pandas as pd
from ml_platform_core.io import write_json, write_table

from ..evaluation import score_surrogate, prepare_objective
from ..model_artifact import evaluation_identity, fingerprint, save_model
from ..policy import METRIC_DIRECTIONS
from ..stage_result import write_prediction_batches
from ..surrogate import FieldSurrogate
from ..uncertainty_reporting import add_uncertainty_report
from .calibration import calibrate_from_split
from .candidate_training import selection_options, selection_scope


def _prediction_batches(model, predictions):
    batch_size = model._query_batch_size()
    for start in range(0, len(predictions), batch_size):
        prediction = predictions[start : start + batch_size]
        if model.calibrator is None:
            yield prediction
        else:
            yield prediction, *model.calibrator.intervals(prediction)


def finalize_selected(cfg, result, dataset, split, candidates):
    metric = cfg["model"]["selection_metric"]
    sign = 1 if METRIC_DIRECTIONS[metric] == "min" else -1
    for item in candidates:
        if not np.isfinite(item["metrics"].get(metric, np.nan)):
            raise ValueError("Candidate selection metric must be finite")
    ranked = sorted(candidates, key=lambda item: (sign * item["metrics"][metric], item["candidate_id"]))
    winner = (
        ranked[0]
        if ranked
        else {
            "candidate_id": cfg["model"]["candidates"][0]["id"],
            "candidate": cfg["model"]["candidates"][0],
            "metrics": {},
            "evaluation_identity": evaluation_identity(cfg["model"]["candidates"][0]),
        }
    )
    train, test = dataset._subset_validated(split["train_indices"]), dataset._subset_validated(split["test_indices"])
    model = FieldSurrogate(winner["candidate"]).fit(train)
    calibrate_from_split(model, dataset, split, cfg["uncertainty"])
    if test.conditions is None:
        raise ValueError("Final test samples require conditions")
    options = selection_options(cfg, scope="final_test")
    prepared = prepare_objective(test, options, training=train, data=cfg["data"])
    report = score_surrogate(model, train, test, options, prepared=prepared)
    add_uncertainty_report(report, model, test, options)
    metrics = report.metrics
    provenance = {
        **cfg["runtime"].get("execution_context", {}),
        "dataset_revision": dataset.revision,
        "split_fingerprint": split["fingerprint"],
        "candidate_fingerprint": fingerprint(winner["candidate"]),
        "evaluation_identity": winner["evaluation_identity"],
    }
    result.metrics.update(metrics)
    result.artifacts["metrics"] = write_json(metrics, result.run_dir / "metrics.json")
    result.artifacts["analysis_summary"] = write_json(report.summary, result.run_dir / "analysis_summary.json")
    result.tables["per_sample"] = write_table(report.per_sample, result.run_dir / "per_sample.csv")
    summary = {
        "best_candidate": winner["candidate_id"],
        "selection_metric": metric,
        "selection_score": winner["metrics"].get(metric),
        "selection_scope": selection_scope(cfg),
        "metric_scope": "final_test",
        "metrics": metrics,
        "evaluation": options,
        "field_components": list(model.representation.components),
        "field_units": list(model.representation.field_units),
        "provenance": provenance,
        "fit_ids": list(model.fit_sample_ids),
        "test_ids": split["test_ids"],
        "calibration_ids": split.get("calibration_ids", []),
    }
    result.artifacts["result_summary"] = write_json(summary, result.run_dir / "result_summary.json")
    rows = [
        {
            "candidate_id": item["candidate_id"],
            "rank": rank,
            "scope": selection_scope(cfg),
            "selection_metric": metric,
            **item["metrics"],
        }
        for rank, item in enumerate(ranked, 1)
    ]
    if rows:
        result.tables["leaderboard"] = write_table(pd.DataFrame(rows), result.run_dir / "leaderboard.csv")
    model_path, info_path = save_model(model, result.run_dir / "model.joblib", provenance=provenance)
    result.artifacts.update(model=model_path, model_info=info_path)
    write_prediction_batches(
        cfg,
        result,
        test.sample_ids,
        _prediction_batches(model, report.predictions),
        model.representation,
        uncertainty=None if model.calibrator is None else model.calibrator.summary(),
        observation_mask=test.observation_mask,
        calibration_support=None if model.calibrator is None else model.calibrator.support,
    )
    result.metrics.update(metrics)
    result.extra.update(
        metric_scope="final_test",
        best_candidate=winner["candidate_id"],
        model_publications=[{"artifact_key": "model", "info_key": "model_info"}],
    )
