"""Resolve business defaults once, before planning or executing a task."""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import re
from typing import Any

from ml_platform_core.config import deep_merge
from ml_platform_core.config_validation import validate_run_config

from .data.splitting import validate_split_settings
from .config import resolve_component, resolve_representation

TASKS = (
    "mode_decomp_pipeline",
    "mode_decomp_stage",
    "mode_decomp_infer",
    "mode_decomp_representation",
    "mode_decomp_analysis",
    "mode_decomp_import",
)
STAGES = ("prepare_fields", "train_candidate", "finalize_surrogate")
METRIC_DIRECTIONS = {
    **{
        f"{prefix}_{metric}": direction
        for prefix in ("field", "hidden")
        for metric, direction in (("rmse", "min"), ("mae", "min"), ("r2", "max"), ("skill", "max"))
    },
    "observation_rmse": "min",
    "observation_mae": "min",
    "observation_skill": "max",
}


def candidate_config(value: dict, seed: int) -> dict:
    if not isinstance(value, dict) or set(value) - {"id", "representation", "regression"}:
        raise ValueError("Each candidate requires id, representation, and regression settings")
    candidate_id = value.get("id")
    if not isinstance(candidate_id, str) or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", candidate_id):
        raise ValueError("Candidate id must contain only letters, digits, underscores, and hyphens")
    rep = value.get("representation") or {}
    return {
        "id": candidate_id,
        "representation": resolve_representation(rep, seed=seed),
        "regression": resolve_component(value.get("regression"), "regression", seed),
    }


def _resolve_tuning(cfg: dict, seed: int) -> dict:
    settings = cfg.get("tuning", {"enabled": False})
    if not isinstance(settings, dict) or not isinstance(settings.get("enabled", False), bool):
        raise ValueError("tuning must be a mapping with a boolean enabled")
    allowed = {"enabled", "method", "n_trials", "seed", "cv", "search_spaces"}
    if set(settings) - allowed:
        raise ValueError(f"Unknown tuning settings: {sorted(set(settings) - allowed)}")
    if not settings.get("enabled", False):
        return {"enabled": False}
    if cfg["task"] not in {"mode_decomp_pipeline", "mode_decomp_stage"}:
        raise ValueError("tuning is supported only by the deployment pipeline and its Stages")
    settings = deep_merge(
        {"method": "grid", "n_trials": 1, "seed": seed, "cv": {"n_splits": 3}, "search_spaces": {}}, settings
    )
    if settings["method"] not in {"grid", "random"}:
        raise ValueError("tuning.method must be grid or random")
    count = settings["n_trials"]
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("tuning.n_trials must be a positive integer shared by all candidates")
    trial_seed = settings["seed"]
    if isinstance(trial_seed, bool) or not isinstance(trial_seed, int) or not 0 <= trial_seed < 2**32:
        raise ValueError("tuning.seed must be an integer in [0,2**32)")
    cv = settings["cv"]
    if not isinstance(cv, dict):
        raise ValueError("tuning.cv must be a mapping")
    outer_method = cfg["split"].get("method", "random")
    method = cv.get("method", outer_method)
    if method not in {"random", "group", "time"}:
        raise ValueError("A fixed outer split requires explicit tuning.cv.method: random, group, or time")
    if outer_method in {"group", "time"} and method != outer_method:
        raise ValueError("tuning.cv.method must preserve the outer group/time separation")
    folds = cv["n_splits"]
    if isinstance(folds, bool) or not isinstance(folds, int) or folds < 2:
        raise ValueError("tuning.cv.n_splits must be an integer of at least two")
    cv = {**cv, "method": method, "seed": cv.get("seed", trial_seed)}
    if isinstance(cv["seed"], bool) or not isinstance(cv["seed"], int) or not 0 <= cv["seed"] < 2**32:
        raise ValueError("tuning.cv.seed must be an integer in [0,2**32)")
    if method in {"group", "time"}:
        column = f"{method}_column"
        cv[column] = cv.get(column, cfg["split"].get(column))
        if not isinstance(cv[column], str) or not cv[column].strip():
            raise ValueError(f"tuning.cv.{column} is required")
        if outer_method == method and cv[column] != cfg["split"].get(column):
            raise ValueError(f"tuning.cv.{column} must preserve the outer {method} column")
    if method == "time":
        outer_gap = cfg["split"].get("gap_seconds", 0)
        cv.setdefault("gap_seconds", outer_gap)
        if validate_split_settings(cv, cross_validation=True) < outer_gap:
            raise ValueError("tuning.cv.gap_seconds cannot be smaller than the outer time gap")
    else:
        validate_split_settings(cv, cross_validation=True)
    settings["cv"] = cv
    spaces = settings["search_spaces"]
    candidates = {c["id"] for c in cfg["model"]["candidates"]}
    if not isinstance(spaces, dict) or set(spaces) - candidates:
        raise ValueError("tuning.search_spaces must map planned candidate IDs to parameter choices")
    path_pattern = r"(?:regression|representation\.(?:field_preprocess|decomposition|codec|coeff_post))\.params\.[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*"
    for candidate_id in candidates:
        space = spaces.get(candidate_id, {})
        if not isinstance(space, dict):
            raise ValueError("Each tuning search space must map parameter paths to lists of choices")
        for path, choices in space.items():
            if not isinstance(path, str) or not re.fullmatch(path_pattern, path):
                raise ValueError(f"Search path must name a component parameter: {path!r}")
            if not isinstance(choices, list) or not choices:
                raise ValueError(f"Search choices for {path} must be a non-empty list")
            try:
                encoded = [json.dumps(value, sort_keys=True, allow_nan=False) for value in choices]
            except (TypeError, ValueError) as exc:
                raise ValueError("Search choices must be finite JSON values") from exc
            if len(set(encoded)) != len(encoded):
                raise ValueError(f"Search choices for {path} must be distinct")
        if any(first != second and second.startswith(first + ".") for first in space for second in space):
            raise ValueError("Search paths cannot overwrite another search parameter's parent")
        if math.prod(len(choices) for choices in space.values()) < count:
            raise ValueError(f"Candidate {candidate_id!r} has fewer unique configurations than tuning.n_trials")
    return settings


def _resolve_uncertainty(cfg: dict, seed: int) -> dict:
    settings = cfg.get("uncertainty", {"enabled": False})
    if not isinstance(settings, dict) or not isinstance(settings.get("enabled", False), bool):
        raise ValueError("uncertainty must be a mapping with a boolean enabled")
    allowed = {
        "enabled",
        "method",
        "coverage",
        "scope",
        "normalization",
        "calibration_size",
        "calibration_seed",
        "calibration_method",
    }
    if set(settings) - allowed:
        raise ValueError(f"Unknown uncertainty settings: {sorted(set(settings) - allowed)}")
    if not settings.get("enabled", False):
        return {"enabled": False}
    if cfg["task"] == "mode_decomp_infer":
        if set(settings) - {"enabled", "coverage", "scope", "normalization"}:
            raise ValueError("Inference uses the model's saved calibration; configure calibration during training")
        return deepcopy(settings)
    if cfg["task"] not in {"mode_decomp_pipeline", "mode_decomp_stage", "mode_decomp_analysis"} or (
        cfg["task"] == "mode_decomp_analysis" and cfg["analysis"]["kind"] != "surrogate"
    ):
        raise ValueError("Field uncertainty is supported for surrogate training, analysis and inference")
    outer = cfg["split"].get("method", "random")
    method = outer if outer != "fixed" else cfg["split"].get("selection_method")
    settings = deep_merge(
        {
            "method": "split_conformal",
            "coverage": 0.9,
            "scope": "simultaneous",
            "normalization": "channel",
            "calibration_size": 0.2,
            "calibration_seed": (seed + 7) % 2**32,
            "calibration_method": method,
        },
        settings,
    )
    if settings["method"] != "split_conformal":
        raise ValueError("uncertainty.method must be split_conformal")
    from .uncertainty import FieldCalibrator

    FieldCalibrator(**{key: settings[key] for key in ("coverage", "scope", "normalization")})
    fraction = settings["calibration_size"]
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not 0 < fraction < 1:
        raise ValueError("uncertainty.calibration_size must be a fraction strictly between zero and one")
    split_seed = settings["calibration_seed"]
    if isinstance(split_seed, bool) or not isinstance(split_seed, int) or not 0 <= split_seed < 2**32:
        raise ValueError("uncertainty.calibration_seed must be an integer in [0,2**32)")
    method = settings["calibration_method"]
    if method not in {"random", "group", "time"}:
        raise ValueError("Fixed outer splits need uncertainty.calibration_method: random, group, or time")
    if outer in {"group", "time"} and method != outer:
        raise ValueError("Calibration must preserve the outer group/time split method")
    if method in {"group", "time"} and not cfg["split"].get(f"{method}_column"):
        raise ValueError(f"Calibration requires split.{method}_column")
    return settings


def _additional_analysis(cfg: dict) -> None:
    analysis = cfg.setdefault("analysis", {})
    if not isinstance(analysis, dict):
        raise ValueError("analysis must be a mapping")
    if cfg["task"] == "mode_decomp_representation" and set(analysis) - {"modes", "components", "scattered"}:
        raise ValueError(
            "Direct decomposition uses analysis.modes/components/scattered; kind and holdout settings belong to mode_decomp_analysis"
        )
    for name in ("scattered", "components"):
        options = analysis.get(name, {})
        allowed = {"enabled", "method", "params"} if name == "components" else {"enabled"}
        if (
            not isinstance(options, dict)
            or set(options) - allowed
            or not isinstance(options.get("enabled", False), bool)
        ):
            raise ValueError(f"analysis.{name} accepts a boolean enabled")
        analysis[name] = {"enabled": options.get("enabled", False)}
        if name == "components":
            method, params = options.get("method", "hodge"), options.get("params", {})
            if method not in {"hodge", "helmholtz"} or not isinstance(params, dict):
                raise ValueError("analysis.components needs method: hodge/helmholtz and a params mapping")
            # Validate the selected constructor once; numerical defaults stay there.
            if options.get("enabled", False) or "method" in options or "params" in options:
                from .catalog import build_decomposer

                build_decomposer({"name": method, "params": params})
            analysis[name].update(method=method, params=deepcopy(params))
        if analysis[name]["enabled"] and analysis.get("kind") == "imputation":
            raise ValueError(
                f"analysis.{name} is for complete-field representation/surrogate; imputation scores hidden reference locations"
            )


def resolve_config(cfg: dict) -> dict:
    if not isinstance(cfg, dict) or cfg.get("task") not in TASKS:
        raise ValueError(f"task must be one of {TASKS}")
    validate_run_config(cfg)
    task = cfg["task"]
    training = task in {"mode_decomp_pipeline", "mode_decomp_stage"}
    comparing = task == "mode_decomp_analysis"
    direct = task == "mode_decomp_representation"
    for section, key in (("runtime", "use_clearml"), ("output", "upload_plots")):
        if key in cfg.get(section, {}):
            raise ValueError(
                f"{section}.{key} was removed because it had no effect; configure external publication in the platform adapter"
            )
    supplied_metrics = cfg.get("metrics", {})
    if not isinstance(supplied_metrics, dict):
        raise ValueError("metrics must be a mapping")
    kind = cfg.get("analysis", {}).get("kind", "surrogate") if comparing else None
    reference = supplied_metrics.get("reference", "hidden" if kind == "imputation" else "grid")
    if reference not in {"grid", "observations", "hidden"}:
        raise ValueError("metrics.reference must be grid, observations, or hidden")
    if (reference == "hidden") != (kind == "imputation"):
        raise ValueError("metrics.reference: hidden is required for imputation and is available only there")
    explicit_mask = cfg.get("analysis", {}).get("observation_mask")
    if explicit_mask is not None and (not isinstance(explicit_mask, (str, Path)) or not str(explicit_mask).strip()):
        raise ValueError("analysis.observation_mask must be a non-empty NPY path")
    if kind == "imputation" and explicit_mask is not None:
        conflicts = {"observed_fraction", "observation_pattern", "mask_seed"} & set(cfg["analysis"])
        if conflicts:
            raise ValueError(
                f"analysis.observation_mask cannot be combined with explicit random mask settings: {sorted(conflicts)}"
            )
    prefix = {"grid": "field", "observations": "observation", "hidden": "hidden"}[reference]
    legacy_hidden = reference == "hidden" and supplied_metrics.get(
        "_legacy_hidden_aliases", "reference" not in supplied_metrics
    )
    requested_names = supplied_metrics.get("names", [f"{prefix}_rmse", f"{prefix}_mae"])
    selection_metric = cfg.get("model", {}).get("selection_metric", f"{prefix}_rmse")
    allowed_names = {name for name in METRIC_DIRECTIONS if name.startswith(prefix + "_")}
    if legacy_hidden:
        allowed_names |= {name for name in METRIC_DIRECTIONS if name.startswith("field_")}  # Legacy hidden scores.
    if (
        not isinstance(requested_names, list)
        or not requested_names
        or any(name not in allowed_names for name in requested_names)
    ):
        raise ValueError(f"metrics.names must select metrics for reference: {reference}")
    if training and selection_metric not in allowed_names:
        raise ValueError(f"model.selection_metric must use the metrics.reference: {reference} namespace")
    defaults: dict[str, Any] = {
        "run": {"name": "mode_decomp", "seed": 42},
        "runtime": {"output_dir": "outputs"},
        "data": {"source_manifest": None},
        "output": {},
    }
    if training or comparing:
        defaults["split"] = {"method": "random", "valid_size": 0.2, "selection_size": 0.2}
        defaults["model"] = {"selection_metric": selection_metric} if training else {}
    if training or comparing or direct:
        metric_defaults: dict[str, Any] = {"reference": reference, "names": requested_names}
        if reference == "hidden":
            metric_defaults["_legacy_hidden_aliases"] = legacy_hidden
        defaults["metrics"] = metric_defaults
    if training or task == "mode_decomp_infer":
        defaults["output"]["prediction_name"] = "predictions.npz"
    cfg = deep_merge(defaults, cfg)
    seed = cfg["run"]["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("run.seed must be an integer in [0, 2**32)")
    if training or comparing:
        validate_split_settings(cfg["split"])
    if training and cfg["model"]["selection_metric"] not in METRIC_DIRECTIONS:
        raise ValueError("Unsupported field selection metric")
    if (training or comparing or direct) and set(cfg["metrics"]["names"]) - set(METRIC_DIRECTIONS):
        raise ValueError("Unsupported field evaluation metric")
    report = cfg.get("report", {})
    if not isinstance(report, dict) or set(report) - {"enabled"} or not isinstance(report.get("enabled", False), bool):
        raise ValueError("report accepts a boolean enabled")
    cfg["report"] = {"enabled": report.get("enabled", cfg["task"] == "mode_decomp_analysis")}
    if cfg["report"]["enabled"] and not (comparing or direct):
        raise ValueError("report.enabled is available for mode_decomp_analysis or mode_decomp_representation")
    if cfg["task"] == "mode_decomp_stage" and cfg["run"].get("stage") not in STAGES:
        raise ValueError(f"run.stage must be one of {STAGES}")
    for section, key in (("runtime", "output_dir"), ("data", "local_path")):
        if cfg[section].get(key):
            cfg[section][key] = str(Path(cfg[section][key]).expanduser().resolve())
    if cfg["task"] == "mode_decomp_analysis":
        analysis_defaults = {
            "kind": "surrogate",
            "split_seeds": [cfg["split"].get("seed", seed)],
            "model_seeds": [seed],
            "bootstrap_samples": 1000,
            "confidence_level": 0.95,
            "allow_failures": False,
            "save_models": False,
        }
        if kind == "imputation" and explicit_mask is None:
            analysis_defaults.update(observed_fraction=0.3, observation_pattern="shared", mask_seed=seed)
        cfg["analysis"] = deep_merge(analysis_defaults, cfg.get("analysis", {}))
        analysis = cfg["analysis"]
        if analysis["kind"] not in {"representation", "surrogate", "imputation"}:
            raise ValueError("analysis.kind must be representation, surrogate, or imputation")
        if analysis.get("observation_mask") is not None and analysis["kind"] != "imputation":
            raise ValueError("analysis.observation_mask is available only for imputation")
        for key in ("split_seeds", "model_seeds"):
            values = analysis[key]
            if not isinstance(values, list) or not values or len(set(values)) != len(values):
                raise ValueError(f"analysis.{key} must be a non-empty list of distinct seeds")
            if any(isinstance(v, bool) or not isinstance(v, int) or not 0 <= v < 2**32 for v in values):
                raise ValueError(f"analysis.{key} must contain integer seeds in [0,2**32)")
        count = analysis["bootstrap_samples"]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("analysis.bootstrap_samples must be a nonnegative integer")
        if not 0 < analysis["confidence_level"] < 1:
            raise ValueError("analysis.confidence_level must be in (0,1)")
        raw = cfg["model"].get("candidates", [{"id": "pod_ridge"}])
        if not isinstance(raw, list) or not raw:
            raise ValueError("model.candidates must be a non-empty list fixed before analysis")
        normalized = [candidate_config(c, seed) for c in raw]
        ids = [c["id"].lower().replace("-", "_") for c in normalized]
        if len(set(ids)) != len(ids) or any(c["id"].startswith("baseline_") for c in normalized):
            raise ValueError("Analysis candidate IDs must be unique and cannot use the baseline_ prefix")
        # Resolve defaults anew per model seed; explicit per-component seeds retain precedence.
        cfg["model"]["candidates"] = deepcopy(raw)
    if direct:
        cfg["representation"] = resolve_representation(cfg.get("representation", {}), seed=seed)
    if comparing or direct:
        _additional_analysis(cfg)
    if cfg["task"] in {"mode_decomp_pipeline", "mode_decomp_stage"}:
        raw = cfg["model"].get("candidates", [cfg.get("candidate", {"id": "pod_ridge"})])
        if not isinstance(raw, list) or not raw:
            raise ValueError("model.candidates must be a non-empty list")
        cfg["model"]["candidates"] = [candidate_config(c, seed) for c in raw]
        ids = [c["id"].lower().replace("-", "_") for c in cfg["model"]["candidates"]]
        if len(set(ids)) != len(ids):
            raise ValueError("Candidate ids collide after Stage name normalization")
        cfg["tuning"] = _resolve_tuning(cfg, seed)
        if cfg["task"] == "mode_decomp_stage" and cfg["run"]["stage"] == "train_candidate":
            candidates = {item["id"]: item for item in cfg["model"]["candidates"]}
            selected = cfg.get("candidate")
            if selected is None:
                key = cfg.get("stage_inputs", {}).get("candidate_id")
                if key is None and len(candidates) == 1:
                    key = next(iter(candidates))
                if key not in candidates:
                    raise ValueError("train_candidate needs a planned stage_inputs.candidate_id or candidate config")
                selected = candidates[key]
            else:
                selected = candidate_config(selected, seed)
                if candidates.get(selected["id"]) != selected:
                    raise ValueError("Candidate Stage settings differ from model.candidates")
            cfg["candidate"] = selected
    else:
        cfg["tuning"] = _resolve_tuning(cfg, seed)
    cfg["uncertainty"] = _resolve_uncertainty(cfg, seed)
    if cfg["task"] == "mode_decomp_import":
        settings = cfg.get("import", {})
        if not isinstance(settings, dict) or set(settings) - {"format", "output_dir", "condition_columns", "geometry"}:
            raise ValueError("import accepts format, output_dir, condition_columns and geometry")
        if settings.get("format") not in {"scattered", "csv", "legacy-npy"} or not settings.get("output_dir"):
            raise ValueError("Import requires import.format: scattered/csv/legacy-npy and a new import.output_dir")
        if not cfg["data"].get("local_path"):
            raise ValueError("Import requires data.local_path")
        cfg["import"] = {**settings, "output_dir": str(Path(settings["output_dir"]).expanduser().resolve())}
    return cfg
