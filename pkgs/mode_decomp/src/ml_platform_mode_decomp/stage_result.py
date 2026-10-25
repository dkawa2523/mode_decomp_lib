"""Use core runs and outcomes; expose only business publication decisions."""

import numpy as np

from ml_platform_core.artifacts import finish_run, prepare_run_dir, write_config_snapshot
from ml_platform_core.result import RunError, RunResult
from ml_platform_core.paths import resolve_path_within


def write_prediction_batches(
    cfg,
    result,
    sample_ids,
    batches,
    representation,
    *,
    uncertainty=None,
    observation_mask=None,
    calibration_support=None,
):
    from .data.schema import FieldSpec
    from .field_output import write_field_batches

    path = resolve_path_within(
        result.run_dir,
        cfg["output"]["prediction_name"],
        setting="output.prediction_name",
        root_description="the run directory",
        allow_nested=False,
    )
    spec = getattr(representation, "field_spec", None)
    if spec is None:
        spec = FieldSpec(
            tuple(representation.field_shape),
            representation.field_kind,
            tuple(representation.components),
            tuple(representation.field_units),
        )
    if uncertainty is not None and calibration_support is None:
        spatial = tuple(representation.field_shape[:-1])
        area = representation.geometry.integration_weights()
        calibration_support = np.ones(spatial, dtype=bool) if area is None else np.asarray(area).reshape(spatial) > 0
        if representation.geometry.mask is not None:
            calibration_support &= np.asarray(representation.geometry.mask, dtype=bool).reshape(spatial)
    paths = write_field_batches(
        path,
        sample_ids,
        batches,
        spec,
        representation.geometry,
        observation_mask=observation_mask,
        calibration_support=calibration_support,
        uncertainty=uncertainty,
    )
    result.artifacts["predictions"] = paths.pop("predictions")
    result.artifacts.update({"prediction_" + key: value for key, value in paths.items()})


def execute_stage(cfg, stage, operation):
    run_dir = prepare_run_dir(cfg["runtime"]["output_dir"], f"{cfg['run']['name']}_{stage}")
    result = RunResult(run_dir, extra={"pipeline_stage": stage, "model_publications": []})
    phase = "input_preparation"
    try:
        result.artifacts["config"] = write_config_snapshot(cfg, run_dir)
        from .input_preparation import prepare_field_input

        prepared = prepare_field_input(cfg, result)
        if prepared is not cfg:
            result.artifacts["config"] = write_config_snapshot(prepared, run_dir)
        cfg = prepared
        phase = "operation"
        operation(cfg, result)
        phase = "finalization"
        return finish_run(
            cfg,
            run_dir,
            metrics=result.metrics,
            artifacts=result.artifacts,
            tables=result.tables,
            plots=result.plots,
            extra=result.extra,
        )
    except Exception as exc:
        result.extra["failed_stage"] = stage
        result.extra["failed_phase"] = phase
        result.extra["error"] = str(exc)
        result.extra["model_publications"] = []
        raise RunError(f"{stage} failed: {exc}", result) from exc
