"""Explicit YAML entry point for one-time conversion to the canonical Dataset."""

from .input_preparation import convert_field_input
from .stage_result import execute_stage


def _import(cfg, result):
    settings = cfg["import"]
    manifest = convert_field_input(
        cfg["data"],
        settings["output_dir"],
        format_name=settings["format"],
        condition_columns=settings.get("condition_columns"),
        geometry=settings.get("geometry"),
    )
    result.artifacts["dataset_manifest"] = manifest
    result.extra.update(dataset_root=str(manifest.parent), import_format=settings["format"])


def run_import(cfg):
    return execute_stage(cfg, "import", _import)
