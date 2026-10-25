"""Fixed-candidate studies with shared holdouts and purpose-specific field evaluation."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from time import perf_counter

import numpy as np
import pandas as pd
from ml_platform_core.io import write_json, write_table

from .data import load_field_dataset, split_dataset
from .data.schema import FieldSpec
from .evaluation import (
    analyze_imputation,
    analyze_representation,
    analyze_surrogate,
    mean_field_prediction,
    observation_mask,
)
from .evaluation.comparison import paired_comparisons
from .model_artifact import evaluation_identity, save_model
from .mode_reporting import mode_options, write_mode_report
from .policy import candidate_config
from .representation import FieldRepresentation
from .inputs import load_surrogate_inputs, subset_inputs
from .stage_result import execute_stage
from .surrogate import FieldSurrogate
from .scattered_reporting import add_scattered_report, prepare_scattered_report
from .evaluation.objectives import prepare_objective
from .evaluation.analysis import ImputationReference, prepare_imputation_reference
from .uncertainty_reporting import add_uncertainty_report
from .training.calibration import calibrate_from_split, validate_calibration_split


def _linear_baseline(train, reference):
    """Fit raw-field OLS with bounded groups of shared observation patterns."""
    x = np.asarray(train.conditions, dtype=float)
    center, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale == 0] = 1
    design = np.column_stack((np.ones(len(train)), (x - center) / scale))
    new = np.column_stack((np.ones(len(reference)), (np.asarray(reference.conditions) - center) / scale))
    values = np.asarray(train.fields).reshape(len(train), -1, train.fields.shape[-1])
    valid = train.effective_mask.reshape(len(train), -1)
    predicted = np.full((len(reference), values.shape[1], values.shape[2]), np.nan)
    channels = np.arange(values.shape[-1])
    block_size = max(1, 262_144 // (max(len(train), len(reference), design.shape[1]) * len(channels)))
    for start in range(0, values.shape[1], block_size):
        block = valid[:, start : start + block_size]
        if block.all():
            patterns = np.ones((1, len(train)), dtype=bool)
            inverse = np.zeros(block.shape[1], dtype=int)
        else:
            # np.unique needs a contiguous sort buffer. Restrict that buffer to
            # this spatial block rather than copying the whole N-by-field mask.
            patterns, inverse = np.unique(block.T, axis=0, return_inverse=True)
        for number, available in enumerate(patterns):
            if not available.any():
                continue
            rows = np.flatnonzero(available)
            selected = start + np.flatnonzero(inverse == number)
            targets = values[np.ix_(rows, selected, channels)].reshape(len(rows), -1)
            coefficients, *_ = np.linalg.lstsq(design[available], targets, rcond=None)
            predicted[:, selected] = (new @ coefficients).reshape(len(reference), len(selected), -1)
    return predicted.reshape(reference.fields.shape), int(values.shape[1] * values.shape[2] * design.shape[1])


def _baseline_reports(train, reference, kind, prepared, *, condition_inputs=True):
    evaluator = prepared.hidden if kind == "imputation" else prepared
    mean = evaluator.baseline
    candidates = [("baseline_mean", mean, 0, int(np.prod(train.fields.shape[1:])))]
    if kind == "surrogate" and condition_inputs and train.conditions is not None and reference.conditions is not None:
        linear, parameters = _linear_baseline(train, reference)
        candidates.append(("baseline_linear", linear, 0, parameters))
    for name, predictions, target_dim, parameters in candidates:
        try:
            report = evaluator.score(predictions)
        except ValueError:
            continue
        report.summary.update(
            analysis_kind=kind,
            packed_dimension=target_dim,
            target_dimension=target_dim,
            field_dimension=int(np.prod(train.fields.shape[1:])),
            fitted_parameters=parameters,
            coefficient_to_field_ratio=0.0,
            decomposition="baseline",
            prediction_input_kind="conditions" if name == "baseline_linear" else "none",
            prediction_input_dimension=train.conditions.shape[1] if name == "baseline_linear" else 0,
        )
        yield name, report


def _append_report(report, metadata, rows, per_sample, details, preview, reference):
    if "field_uncertainty" in report.summary:
        calibration = report.summary["field_uncertainty"]["calibration"]
        metadata = {
            **metadata,
            "interval_scope": calibration["scope"],
            "interval_coverage_target": calibration["coverage"],
        }
    metadata = {
        **metadata,
        **{
            k: report.summary.get(k)
            for k in (
                "field_dimension",
                "coefficient_to_field_ratio",
                "prediction_input_kind",
                "prediction_input_dimension",
                "basis_inner_product",
                "method_role",
            )
        },
    }
    rows.append({**metadata, "status": "ok", **report.metrics})
    table = report.per_sample.copy()
    for key in ("candidate_id", "split_seed", "model_seed"):
        table[key] = metadata[key]
    if "group" in metadata:
        table["group"] = metadata["group"]
    per_sample.append(table)
    details.append({**{k: v for k, v in metadata.items() if k != "group"}, **report.summary})
    if preview is not None and metadata["candidate_id"] not in preview:
        worst = int(np.nanargmax(report.per_sample["mse"].to_numpy()))
        from .analysis_reporting import prepare_preview

        preview[metadata["candidate_id"]] = prepare_preview(reference, report, worst)


def _study(cfg, result):
    from .analysis_reporting import write_analysis_report

    dataset = load_field_dataset(cfg["data"])
    inputs = load_surrogate_inputs(cfg["data"], target=dataset)
    prediction_source = "condition_predictions" if inputs is None else "source_predictions"
    study = cfg["analysis"]
    kind = study["kind"]
    if study["components"]["enabled"]:
        from .decomposition.physical import validate_components

        validate_components(
            FieldSpec.from_dataset(dataset),
            dataset.geometry,
            method=study["components"].get("method", "hodge"),
            params=study["components"].get("params"),
        )
    selected_modes = mode_options(study.get("modes"))
    if kind == "surrogate" and inputs is None and dataset.conditions is None:
        raise ValueError("Surrogate comparison requires conditions; use representation for field-only data")
    if inputs is not None and kind != "surrogate":
        raise ValueError("data.input is supported by surrogate analysis only")
    if study["scattered"]["enabled"] and not dataset.manifest.get("scattered"):
        raise ValueError(
            "analysis.scattered.enabled requires an imported scattered dataset retaining original observations"
        )
    rows, samples, details, failures, protocols = [], [], [], [], []
    output_failures = []
    previews = {} if cfg["report"]["enabled"] else None
    scope = f"heldout_{kind}" if kind != "imputation" else "heldout_hidden_reference"
    options = {**cfg["metrics"], "reference_scope": scope}
    explicit = None
    if kind == "imputation" and study.get("observation_mask"):
        from .evaluation.analysis import load_observation_mask

        explicit = load_observation_mask(cfg["data"], dataset, study["observation_mask"])
    group_column = study.get("bootstrap_group_column")
    if group_column is None and cfg["split"]["method"] == "group":
        group_column = cfg["split"].get("group_column")
    independent_samples = cfg["split"]["method"] != "time" or group_column is not None
    for split_seed in study["split_seeds"]:
        split_cfg = deepcopy(cfg["split"])
        split_cfg["seed"] = split_seed
        split = split_dataset(dataset, split_cfg, seed=split_seed, nested=False, calibration=cfg["uncertainty"])
        validate_calibration_split(dataset, split, cfg["uncertainty"])
        train, reference = (
            dataset._subset_validated(split["train_indices"]),
            dataset._subset_validated(split["test_indices"]),
        )
        train_inputs = subset_inputs(inputs, split["train_indices"])
        reference_inputs = subset_inputs(inputs, split["test_indices"])
        observed = None
        if kind == "imputation":
            if explicit is not None:
                observed = observation_mask(reference, explicit=explicit[split["test_indices"]])
            else:
                observed = observation_mask(
                    reference,
                    study["observed_fraction"],
                    study["mask_seed"] + split_seed,
                    pattern=study["observation_pattern"],
                )
        prepared = (
            prepare_imputation_reference(train, reference, observed, options)
            if kind == "imputation"
            else prepare_objective(
                reference, options, training=train, data=cfg["data"], baseline=mean_field_prediction(train, reference)
            )
        )
        scattered = None
        if study["scattered"]["enabled"]:
            scattered = getattr(prepared, "observations", None) or prepare_scattered_report(cfg["data"], reference)
        baseline_reports = list(_baseline_reports(train, reference, kind, prepared, condition_inputs=inputs is None))
        if scattered is not None:
            for _, report in baseline_reports:
                add_scattered_report(cfg["data"], report, reference, options, prepared=scattered)
        for model_seed in study["model_seeds"]:
            group = {}
            if group_column:
                if group_column not in reference.sample_metadata:
                    raise ValueError("bootstrap_group_column must name reference sample metadata")
                values = np.asarray(reference.sample_metadata[group_column])
                if pd.isna(values).any() or any(not str(value).strip() for value in values):
                    raise ValueError("bootstrap_group_column must not contain missing group labels")
                group["group"] = values.astype(str)
            base = {"split_seed": split_seed, "model_seed": model_seed, "scope": scope}
            for candidate_id, baseline_report in baseline_reports:
                report = replace(
                    baseline_report,
                    metrics=dict(baseline_report.metrics),
                    summary=dict(baseline_report.summary),
                    per_sample=baseline_report.per_sample.copy(),
                    fields=dict(baseline_report.fields),
                )
                metadata = {
                    **base,
                    **group,
                    "candidate_id": candidate_id,
                    "method": "baseline",
                    "packed_dimension": 0,
                    "target_dimension": 0,
                    "fit_seconds": None,
                    "prediction_seconds": None,
                    "model_bytes": None,
                }
                _append_report(report, metadata, rows, samples, details, previews, reference)
            for raw in cfg["model"]["candidates"]:
                candidate = candidate_config(raw, model_seed)
                metadata = {
                    **base,
                    "candidate_id": candidate["id"],
                    "method": candidate["representation"]["decomposition"]["name"],
                }
                report = None
                phase = "numerical"
                try:
                    started = perf_counter()
                    if kind == "surrogate":
                        model = FieldSurrogate(candidate)
                        if train_inputs is None:
                            model.fit(train)
                        else:
                            model.fit(train, inputs=train_inputs)
                        calibrate_from_split(model, dataset, split, cfg["uncertainty"], inputs=inputs)
                        representation = model.representation
                    else:
                        model = representation = FieldRepresentation(candidate["representation"]).fit(train)
                    fit_seconds = perf_counter() - started
                    if isinstance(model, FieldSurrogate):
                        report = analyze_surrogate(
                            model,
                            train,
                            reference,
                            options,
                            prepared=prepared,
                            retain_coefficients=selected_modes is not None,
                            retain_fields=False,
                            inputs=reference_inputs,
                        )
                        add_uncertainty_report(report, model, reference, options)
                    elif kind == "representation":
                        report = analyze_representation(
                            representation,
                            train,
                            reference,
                            options,
                            prepared=prepared,
                            retain_coefficients=selected_modes is not None,
                            retain_fields=False,
                        )
                    else:
                        report = analyze_imputation(
                            representation,
                            train,
                            reference,
                            observed,
                            options,
                            prepared=prepared,
                            retain_coefficients=selected_modes is not None,
                        )
                    if study["scattered"]["enabled"]:
                        add_scattered_report(cfg["data"], report, reference, options, prepared=scattered)
                    metadata.update(
                        packed_dimension=representation.packed_dim,
                        target_dimension=representation.target_dim,
                        fit_seconds=fit_seconds,
                        prediction_seconds=report.summary.get("prediction_seconds", 0.0),
                        model_bytes=None,
                    )
                    report.summary["candidate"] = candidate
                    report.summary["execution_identity"] = evaluation_identity(candidate)
                    phase = "requested_outputs"
                    if study["save_models"]:
                        directory = result.run_dir / "models" / f"{candidate['id']}_{split_seed}_{model_seed}"
                        directory.mkdir(parents=True)
                        path, info = save_model(model, directory / "model.joblib")
                        metadata["model_bytes"] = path.stat().st_size
                        report.summary["saved_model"] = str(path.relative_to(result.run_dir))
                        report.summary["saved_model_info"] = str(info.relative_to(result.run_dir))
                    if study["components"]["enabled"]:
                        from .component_reporting import write_component_report

                        prefix = f"{candidate['id']}_{split_seed}_{model_seed}"
                        outputs = write_component_report(
                            representation,
                            report.predictions if isinstance(model, FieldSurrogate) else reference.fields,
                            reference.sample_ids,
                            result.run_dir / "components" / prefix,
                            method=study["components"].get("method", "hodge"),
                            params=study["components"].get("params"),
                        )
                        for name, path in outputs.items():
                            collection = result.tables if path.suffix == ".csv" else result.artifacts
                            collection[f"{name}_{prefix}"] = path
                        report.summary["component_analysis"] = {
                            "status": "ok",
                            "method": study["components"].get("method", "hodge"),
                            "source": prediction_source if isinstance(model, FieldSurrogate) else "input_fields",
                            "outputs": {
                                name: path.relative_to(result.run_dir).as_posix() for name, path in outputs.items()
                            },
                        }
                    if selected_modes is not None:
                        evaluation_mask = None
                        if isinstance(model, FieldSurrogate):
                            coefficient_source = prediction_source
                        elif observed is not None:
                            evaluation_mask = reference.effective_mask & ~observed
                            coefficient_source = "input_observations_only"
                        else:
                            coefficient_source = "heldout_fields"
                        prefix = f"{candidate['id']}_{split_seed}_{model_seed}"
                        outputs = write_mode_report(
                            representation,
                            report.packed_coefficients,
                            reference,
                            options,
                            selected_modes,
                            result.run_dir / "modes" / prefix,
                            evaluation_mask=evaluation_mask,
                            full_predictions=report.predictions if isinstance(model, FieldSurrogate) else None,
                            prepared=prepared.hidden if isinstance(prepared, ImputationReference) else prepared,
                        )
                        result.artifacts[f"mode_report_{prefix}"] = outputs["mode_report"]
                        result.tables.update(
                            {f"{name}_{prefix}": path for name, path in outputs.items() if name != "mode_report"}
                        )
                        report.summary["mode_analysis"] = {
                            "status": "ok",
                            "coefficient_source": coefficient_source,
                            "report": outputs["mode_report"].relative_to(result.run_dir).as_posix(),
                        }
                    _append_report(report, {**metadata, **group}, rows, samples, details, previews, reference)
                except (
                    ValueError,
                    TypeError,
                    RuntimeError,
                    ImportError,
                    ArithmeticError,
                    OSError,
                    np.linalg.LinAlgError,
                ) as exc:
                    failure = {**metadata, "status": "failed", "phase": phase, "error": f"{type(exc).__name__}: {exc}"}
                    if phase == "numerical":
                        failures.append(failure)
                        rows.append(failure)
                    else:
                        output_failures.append(failure)
                        assert report is not None
                        report.summary["output_error"] = failure["error"]
                        _append_report(report, {**metadata, **group}, rows, samples, details, previews, reference)
            protocols.append(
                {
                    **base,
                    "split": split,
                    "candidates": [candidate_config(raw, model_seed) for raw in cfg["model"]["candidates"]],
                }
            )
    rmse_name = {"grid": "field_rmse", "observations": "observation_rmse", "hidden": "hidden_rmse"}[
        options.get("reference", "grid")
    ]
    columns = [
        "candidate_id",
        "split_seed",
        "model_seed",
        "scope",
        "method",
        "status",
        "phase",
        "error",
        "packed_dimension",
        "target_dimension",
        "fit_seconds",
        "prediction_seconds",
        "model_bytes",
        *options.get("names", []),
        rmse_name,
    ]
    comparison = pd.DataFrame(rows).reindex(columns=list(dict.fromkeys(columns + [key for row in rows for key in row])))
    # Group labels live only in sample tables, not a list-valued comparison CSV cell.
    comparison = comparison.drop(columns=["group"], errors="ignore")
    per_sample = (
        pd.concat(samples, ignore_index=True)
        if samples
        else pd.DataFrame(columns=["sample_id", "candidate_id", "split_seed", "model_seed", "mse"])
    )
    paired = paired_comparisons(
        per_sample,
        draws=study["bootstrap_samples"] if independent_samples else 0,
        confidence=study["confidence_level"],
        seed=cfg["run"]["seed"],
        independent_samples=independent_samples,
    )
    good = comparison[comparison.status == "ok"]
    summary_rows = []
    for candidate_id, frame in good.groupby("candidate_id", sort=True):
        summary_rows.append(
            {
                "candidate_id": candidate_id,
                "repetitions": len(frame),
                "rmse_mean": float(frame[rmse_name].mean()),
                "rmse_std": float(frame[rmse_name].std(ddof=0)),
                "target_dimension_mean": float(frame.target_dimension.mean()),
                "fit_seconds_mean": float(frame.fit_seconds.mean()) if frame.fit_seconds.notna().any() else None,
            }
        )
    summary = {
        "study": "fixed_candidates_shared_outer_holdouts",
        "analysis_kind": kind,
        "metric_scope": scope,
        "selection": "No test-based candidate selection or model publication",
        "evaluation": options,
        "complete": not failures and not output_failures,
        "failures": failures,
        "output_failures": output_failures,
        "candidate_summary": summary_rows,
        "repetitions": len(protocols),
        "details": details,
        "uncertainty": {
            "paired_interval_scope": "conditional_on_fitted_models_and_one_test_split",
            "independent_unit": group_column or ("sample" if independent_samples else "not_assumed_for_time_series"),
            "multiple_comparisons": "unadjusted exploratory intervals; no family-wise significance claim",
            "across_repetitions": "descriptive mean/std; overlapping holdouts are not independent replicates",
        },
        "execution_identity": evaluation_identity({}),
    }
    result.tables["comparison"] = write_table(comparison, result.run_dir / "comparison.csv")
    result.tables["per_sample"] = write_table(per_sample, result.run_dir / "per_sample.csv")
    result.tables["paired_comparisons"] = write_table(paired, result.run_dir / "paired_comparisons.csv")
    result.artifacts["protocol"] = write_json(
        {
            "dataset_revision": dataset.revision,
            **({"input_revision": inputs.revision} if inputs is not None else {}),
            "repetitions": protocols,
        },
        result.run_dir / "protocol.json",
    )
    result.artifacts["analysis_summary"] = write_json(summary, result.run_dir / "analysis_summary.json")
    result.metrics.update(successful_evaluations=float(len(good)), failed_evaluations=float(len(failures)))
    result.extra.update(metric_scope=scope, analysis_kind=kind, analysis_complete=summary["complete"])
    if cfg["report"]["enabled"]:
        try:
            result.artifacts["analysis_report"] = write_analysis_report(
                summary, comparison, previews, dataset, result.run_dir / "analysis.html"
            )
        except (OSError, ValueError, RuntimeError) as exc:
            summary.update(complete=False, report_error=f"{type(exc).__name__}: {exc}")
            write_json(summary, result.run_dir / "analysis_summary.json")
            result.extra["analysis_complete"] = False
            raise RuntimeError("Analysis report rendering failed; numeric artifacts remain available") from exc
    candidate_successes = good[good.method != "baseline"]
    if candidate_successes.empty:
        raise ValueError("No declared candidate produced a valid numerical result; inspect analysis_summary.json")
    if output_failures:
        raise RuntimeError("Some requested analysis outputs failed; numeric artifacts remain available")
    if failures and not study["allow_failures"]:
        raise ValueError(
            "Some predeclared candidates failed; inspect the partial analysis report before comparing results"
        )


def run_analysis(cfg):
    return execute_stage(cfg, "analysis", _study)
