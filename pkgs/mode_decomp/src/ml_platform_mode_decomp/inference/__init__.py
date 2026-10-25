"""Condition-only inference from a verified self-contained surrogate."""

from ml_platform_core.io import write_json

from ..data import load_condition_batch
from ..model_artifact import load_model
from ..stage_result import execute_stage, write_prediction_batches
from ..surrogate import FieldSurrogate


def _predict(cfg, result):
    model_cfg = cfg["model"]
    model = load_model(
        model_cfg["artifact_path"], model_cfg.get("info_path"), expected_context=cfg["runtime"].get("execution_context")
    )
    if not isinstance(model, FieldSurrogate):
        raise ValueError("Inference requires a surrogate bundle")
    batch = load_condition_batch(cfg["data"], condition_columns=model.condition_columns)
    uncertainty = None
    if cfg["uncertainty"]["enabled"]:
        if model.calibrator is None:
            raise ValueError("uncertainty.enabled requires a model calibrated on independent held-out fields")
        uncertainty = model.calibrator.summary()
        for key in ("coverage", "scope", "normalization"):
            if key in cfg["uncertainty"] and cfg["uncertainty"][key] != uncertainty[key]:
                raise ValueError(f"uncertainty.{key} differs from the saved calibration; retrain to change it")
    write_prediction_batches(
        cfg,
        result,
        batch.sample_ids,
        model.iter_predict(batch.conditions, intervals=uncertainty is not None),
        model.representation,
        uncertainty=uncertainty,
    )
    result.artifacts["result_summary"] = write_json(
        {
            "sample_count": len(batch),
            "condition_columns": list(batch.condition_columns),
            "field_shape": list(model.representation.field_shape),
            "components": list(model.representation.components),
            "field_units": list(model.representation.field_units),
            "field_uncertainty": uncertainty,
        },
        result.run_dir / "result_summary.json",
    )


def run_inference(cfg):
    return execute_stage(cfg, "inference", _predict)
