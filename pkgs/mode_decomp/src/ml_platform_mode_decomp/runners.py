"""Explicit, lazy task dispatch; no SDK imports."""

from importlib import import_module
from typing import Any

from ml_platform_core.result import RunResult

_TASKS = {
    "mode_decomp_pipeline": (".training.orchestrator", "run_pipeline"),
    "mode_decomp_stage": (".stage", "run_stage"),
    "mode_decomp_infer": (".inference", "run_inference"),
    "mode_decomp_representation": (".representation_task", "run_representation"),
    "mode_decomp_analysis": (".analysis_task", "run_analysis"),
    "mode_decomp_import": (".import_task", "run_import"),
}


def run_task(cfg: dict[str, Any]) -> RunResult:
    from .policy import resolve_config

    resolved = resolve_config(cfg)
    module, function = _TASKS[resolved["task"]]
    return getattr(import_module(module, __package__), function)(resolved)
