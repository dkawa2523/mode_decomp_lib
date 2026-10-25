"""Direct decomposition of every supplied field, without a holdout or model selection."""

from html import escape

import numpy as np
from ml_platform_core.artifacts import file_hash
from ml_platform_core.io import write_json, write_table

from .data import load_field_dataset
from .evaluation.objectives import prepare_objective
from .field_output import write_field_batches
from .data.schema import FieldSpec
from .evaluation.analysis import representation_size
from .model_artifact import save_model
from .representation import FieldRepresentation
from .stage_result import execute_stage

_SCOPE = "fitted_input_reconstruction"


def _additional_analysis(cfg, result, model, dataset, packed, options, prepared):
    settings = cfg.get("analysis", {})
    summary = {}
    if settings.get("components", {}).get("enabled", False):
        from .component_reporting import write_component_report

        paths = write_component_report(
            model,
            dataset.fields,
            dataset.sample_ids,
            result.run_dir / "components",
            method=settings["components"].get("method", "hodge"),
            params=settings["components"].get("params"),
        )
        for key, path in paths.items():
            (result.tables if path.suffix == ".csv" else result.artifacts)[key] = path
        summary["component_analysis"] = {
            "status": "ok",
            "method": settings["components"].get("method", "hodge"),
            "source": "input_fields",
            "outputs": {key: path.relative_to(result.run_dir).as_posix() for key, path in paths.items()},
        }
    from .mode_reporting import mode_options, write_mode_report

    modes = mode_options(settings.get("modes"))
    if modes is not None:
        try:
            paths = write_mode_report(
                model, packed, dataset, options, modes, result.run_dir / "modes", prepared=prepared
            )
            for key, path in paths.items():
                (result.tables if path.suffix == ".csv" else result.artifacts)[key] = path
            summary["mode_analysis"] = {
                "status": "ok",
                "coefficient_source": "input_fields",
                "report": paths["mode_report"].relative_to(result.run_dir).as_posix(),
            }
        except NotImplementedError as exc:
            summary["mode_analysis"] = {"status": "unavailable", "reason": str(exc)}
    return summary


def _write_report(result, summary):
    metrics = "".join(f"<tr><th>{escape(name)}</th><td>{value:.6g}</td></tr>" for name, value in result.metrics.items())
    links = []
    for key, path in {**result.artifacts, **result.tables}.items():
        if not path.is_relative_to(result.run_dir):
            continue
        links.append(
            f'<li><a href="{escape(path.relative_to(result.run_dir).as_posix(), quote=True)}">{escape(key)}</a></li>'
        )
    unavailable = "".join(
        f"<p>{escape(key)}: {escape(value['reason'])}</p>"
        for key, value in summary.items()
        if isinstance(value, dict) and value.get("status") == "unavailable"
    )
    document = '<!doctype html><html lang="en"><meta charset="utf-8"><title>Direct field decomposition</title>'
    document += (
        "<body><h1>Direct field decomposition</h1><p>All supplied samples were used to fit this representation. "
    )
    document += "These metrics measure fitted-input reconstruction, not held-out or unknown-sample performance.</p>"
    document += f"<p>Scope: {_SCOPE}</p><table>{metrics}</table>{unavailable}<h2>Outputs</h2><ul>{''.join(links)}</ul></body></html>"
    path = result.run_dir / "representation.html"
    path.write_text(document, encoding="utf-8")
    return path


def _fit(cfg, result):
    dataset = load_field_dataset(cfg["data"])
    if cfg["analysis"]["components"]["enabled"]:
        from .decomposition.physical import validate_components

        validate_components(
            FieldSpec.from_dataset(dataset),
            dataset.geometry,
            method=cfg["analysis"]["components"].get("method", "hodge"),
            params=cfg["analysis"]["components"].get("params"),
        )
    model = FieldRepresentation(cfg.get("representation", {}), seed=cfg["run"]["seed"])
    packed, targets = model._fit(dataset)
    reconstructed = model.inverse_transform(targets)
    options = {**cfg["metrics"], "reference_scope": _SCOPE}
    prepared = prepare_objective(dataset, options, training=dataset, data=cfg["data"])
    report = prepared.score(reconstructed)
    if cfg["analysis"]["scattered"]["enabled"]:
        from .scattered_reporting import add_scattered_report

        add_scattered_report(cfg["data"], report, dataset, options, prepared=prepared.observations)
    report.summary.update(representation_size(model))
    report.summary.update(
        metric_scope=_SCOPE,
        interpretation="All input samples were fitted; reconstruction is not unknown-sample performance",
        fit_sample_ids=list(model.fit_sample_ids),
        raw_coefficient_layout=model.raw_meta.get("coeff_layout"),
        raw_coefficient_shape=model.raw_meta.get("coeff_shape"),
    )
    result.metrics.update(report.metrics)
    result.extra["metric_scope"] = _SCOPE
    result.artifacts["metrics"] = write_json(report.metrics, result.run_dir / "metrics.json")
    result.artifacts["analysis_summary"] = write_json(report.summary, result.run_dir / "analysis_summary.json")
    result.tables["per_sample"] = write_table(report.per_sample, result.run_dir / "per_sample.csv")
    try:
        model_path, info_path = save_model(
            model, result.run_dir / "model.joblib", provenance=cfg["runtime"].get("execution_context")
        )
        result.artifacts.update(model=model_path, model_info=info_path)
        coefficients = result.run_dir / "coefficients.npz"
        np.savez_compressed(
            coefficients,
            sample_ids=dataset.sample_ids,
            coefficients=targets,
            space=np.asarray("z"),
            packed_a=packed,
            model_sha256=np.asarray(file_hash(model_path)),
        )
        result.artifacts["coefficients"] = coefficients
        reconstruction_path = result.run_dir / "reconstructed_fields.npz"
        paths = write_field_batches(
            reconstruction_path,
            dataset.sample_ids,
            [reconstructed],
            FieldSpec.from_dataset(dataset),
            dataset.geometry,
            array_name="reconstruction",
            observation_mask=dataset.observation_mask,
        )
        result.artifacts["reconstructed_fields"] = paths["reconstruction"]
        report.summary.update(_additional_analysis(cfg, result, model, dataset, packed, options, prepared))
        write_json(report.summary, result.run_dir / "analysis_summary.json")
    except (ValueError, TypeError, RuntimeError, ImportError, ArithmeticError, OSError) as exc:
        failure = {"phase": "requested_outputs", "error": f"{type(exc).__name__}: {exc}"}
        report.summary.update(complete=False, output_failures=[failure])
        result.extra["output_failures"] = [failure]
        write_json(report.summary, result.run_dir / "analysis_summary.json")
        raise
    if cfg.get("report", {}).get("enabled", False):
        result.artifacts["representation_report"] = _write_report(result, report.summary)


def run_representation(cfg):
    return execute_stage(cfg, "representation", _fit)
