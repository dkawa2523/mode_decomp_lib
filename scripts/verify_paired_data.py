"""Measure one paired-field pipeline without fitting or selecting on its final test.

Run from the same working directory as the supplied YAML. Data provenance is an
explicit user declaration; synthetic runs must never be reported as real data.
"""

from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import yaml

from ml_platform_mode_decomp import load_config, load_model, run_task
from ml_platform_mode_decomp.data import load_field_dataset
from ml_platform_mode_decomp.evaluation import analyze_representation, analyze_surrogate, prepare_objective
from ml_platform_mode_decomp.inputs import load_surrogate_inputs


def _peak_memory_bytes():
    """OS high-water mark, not Python allocations or a capacity extrapolation."""
    if sys.platform == "win32":
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                *[
                    (name, ctypes.c_size_t)
                    for name in (
                        "PeakWorkingSetSize",
                        "WorkingSetSize",
                        "QuotaPeakPagedPoolUsage",
                        "QuotaPagedPoolUsage",
                        "QuotaPeakNonPagedPoolUsage",
                        "QuotaNonPagedPoolUsage",
                        "PagefileUsage",
                        "PeakPagefileUsage",
                    )
                ],
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return int(counters.PeakWorkingSetSize)
    import resource

    measured = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(measured if sys.platform == "darwin" else measured * 1024)


def _physical_report(reference, predictions, *, training):
    # Per-channel values retain physical units. The aggregate mixes channels and
    # is dimensionally physical only when they share a unit.
    objective = prepare_objective(reference, {"reference": "grid"}, training=training)
    report = objective.score(predictions)
    signal = objective.score(np.zeros_like(predictions))
    relative = []
    for channel in range(reference.fields.shape[-1]):
        denominator = signal.metrics[f"channel_{channel}_rmse"]
        relative.append(None if denominator == 0 else report.metrics[f"channel_{channel}_rmse"] / denominator)
    baseline_status = report.summary["baseline_status"]
    baseline = (
        objective.score(objective.baseline).metrics
        if baseline_status in {"defined", "zero_error_skill_undefined"}
        else None
    )
    return {
        "metrics": report.metrics,
        "relative_l2_by_component": relative,
        "relative_l2_definition": "physical RMSE / RMS of held-out truth, using the same sample and geometry measure",
        "zero_truth_norm": "relative error is null; absolute RMSE remains defined",
        "training_mean_baseline": {"status": baseline_status, "metrics": baseline},
        "components": list(reference.components),
        "units": list(reference.field_units or ()),
        "aggregate_unit": "shared_component_unit"
        if len(set(reference.field_units or ())) == 1
        else "mixed_or_unspecified",
        "support": "known held-out grid locations with positive geometry weights",
        "measure": report.summary["spatial_measure"],
    }


def verify(config_path, work_dir, *, data_kind, provenance):
    if data_kind not in {"measured", "simulation", "synthetic"} or not provenance.strip():
        raise ValueError("Declare measured/simulation/synthetic data and a nonblank provenance description")
    cfg = load_config(config_path)
    if cfg.get("task") != "mode_decomp_pipeline" or cfg.get("data", {}).get("input", {}).get("kind") != "field":
        raise ValueError("Paired acceptance requires a mode_decomp_pipeline YAML with data.input.kind: field")
    work_dir = Path(work_dir).resolve()
    if work_dir.exists():
        raise ValueError(f"Use a new acceptance directory; existing results are never overwritten: {work_dir}")
    cfg.setdefault("runtime", {})["output_dir"] = str(work_dir / "pipeline")
    started = perf_counter()
    result = run_task(cfg)
    pipeline_seconds = perf_counter() - started
    pipeline_peak = _peak_memory_bytes()
    effective = yaml.safe_load(Path(result.artifacts["config"]).read_text(encoding="utf-8"))
    split = json.loads(Path(result.artifacts["split"]).read_text(encoding="utf-8"))
    target = load_field_dataset(effective["data"])
    source = load_surrogate_inputs(effective["data"], target=target)
    if source is None or not hasattr(source, "fields"):
        raise ValueError("Paired residual evaluation requires source truth fields")
    model_path = Path(result.artifacts["model"])
    model = load_model(
        model_path,
        expected_context={
            "dataset_revision": target.revision,
            "input_revision": source.revision,
            "split_fingerprint": split["fingerprint"],
        },
    )
    train_indices, test_indices = split["train_indices"], split["test_indices"]
    training, reference = target.subset(train_indices), target.subset(test_indices)
    source_training, source_reference = source.subset(train_indices), source.subset(test_indices)
    fit_ids = tuple(training.sample_ids)
    if model.fit_sample_ids != fit_ids or model.representation.fit_sample_ids != fit_ids:
        raise ValueError("Saved model was not fitted on exactly the declared training partition")
    if model.input_representation.fit_sample_ids != fit_ids:
        raise ValueError("Source representation was not fitted on exactly the declared training partition")
    started = perf_counter()
    options = effective["metrics"]
    prepared = prepare_objective(reference, options, training=training, data=effective["data"])
    predicted = analyze_surrogate(
        model,
        training,
        reference,
        options,
        prepared=prepared,
        inputs=source_reference,
        retain_fields=False,
    )
    target_residual = analyze_representation(
        model.representation,
        training,
        reference,
        options,
        prepared=prepared,
        retain_fields=False,
    )
    source_residual = analyze_representation(
        model.input_representation,
        source_training,
        source_reference,
        retain_fields=False,
    )
    physical = _physical_report(reference, predicted.predictions, training=training)
    physical_target = _physical_report(reference, target_residual.predictions, training=training)
    physical_source = _physical_report(source_reference, source_residual.predictions, training=source_training)
    diagnostics_seconds = perf_counter() - started
    per_sample = predicted.per_sample.copy()
    per_sample["source_representation_mse"] = source_residual.per_sample["representation_mse"].to_numpy()
    per_sample.to_csv(work_dir / "per_sample.csv", index=False)
    report = {
        "success": True,
        "success_meaning": "pipeline and measurements completed; no application accuracy threshold was declared",
        "quality_acceptance": "not_assessed_without_predeclared_accuracy_and_resource_limits",
        "scope": "untouched final test; model selection uses only the configured training/selection partitions",
        "data_kind": data_kind,
        "provenance": provenance,
        "provenance_verification": "user-declared; inspect source dataset documentation independently",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": str(Path(config_path).resolve()),
        "run_dir": str(result.run_dir),
        "selected_candidate": model.candidate["id"],
        "split": split,
        "split_method": effective["split"]["method"],
        "unknown_case_scope": "held-out sample IDs; unseen groups require an explicit group or fixed split",
        "dataset_revisions": {"target": target.revision, "source": source.revision},
        "shapes": {"source": list(source.fields.shape), "target": list(target.fields.shape)},
        "target_prediction": {"configured_metrics": predicted.metrics, "physical_grid": physical},
        "target_representation_residual": {
            "configured_metrics": target_residual.metrics,
            "physical_grid": physical_target,
        },
        "source_representation_residual": physical_source,
        "capacity": {
            "source_field_array_bytes": int(source.fields.nbytes),
            "target_field_array_bytes": int(target.fields.nbytes),
            "source_z_dimension": model.input_dimension,
            "target_z_dimension": model.representation.target_dim,
            "model_file_bytes": model_path.stat().st_size,
            "pipeline_seconds": pipeline_seconds,
            "held_out_prediction_seconds": predicted.summary["prediction_seconds"],
            "acceptance_diagnostics_seconds": diagnostics_seconds,
            "pipeline_process_peak_resident_bytes": pipeline_peak,
            "acceptance_process_peak_resident_bytes": _peak_memory_bytes(),
            "memory_scope": "OS process high-water mark, including imports and diagnostics; not minimum required memory",
            "limits": "Measured for these dimensions and this machine; no automatic scaling guarantee",
        },
        "runtime": {name: version(name) for name in ("ml-platform-mode-decomp", "numpy", "scipy", "scikit-learn")},
        "artifacts": {"model": str(model_path), "per_sample": str(work_dir / "per_sample.csv")},
    }
    (work_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--data-kind", choices=("measured", "simulation", "synthetic"), required=True)
    parser.add_argument(
        "--provenance",
        required=True,
        help="Dataset identifier, source and pairing meaning; never imply synthetic data are measurements",
    )
    args = parser.parse_args()
    report = verify(args.config, args.work_dir, data_kind=args.data_kind, provenance=args.provenance)
    print(json.dumps({"success": report["success"], "report": str(args.work_dir.resolve() / "report.json")}, indent=2))


if __name__ == "__main__":
    main()
