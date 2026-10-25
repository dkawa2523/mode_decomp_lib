"""Prepare a local execution once; persisted records belong to external boundaries."""

from dataclasses import dataclass

from ml_platform_core.io import write_json

from ..data import load_field_dataset, split_dataset
from ..data.schema import FieldDataset
from ..inputs import SurrogateInputs, load_surrogate_inputs


def needs_selection(cfg):
    return cfg.get("tuning", {}).get("enabled", False) or len(cfg["model"]["candidates"]) > 1


@dataclass
class PreparedExecution:
    # Owned by the coordinator. Never pass this object to a fitting kernel.
    dataset: FieldDataset
    split: dict
    inputs: SurrogateInputs | None = None


def prepare_execution(cfg):
    dataset = load_field_dataset(cfg["data"])
    return _prepare_dataset(cfg, dataset)


def _prepare_dataset(cfg, dataset, *, input_paths=None):
    inputs = load_surrogate_inputs(cfg["data"], target=dataset, _source_paths=input_paths)
    if inputs is None and dataset.conditions is None:
        raise ValueError("Surrogate training requires conditions or data.input; use representation for fields only")
    split = split_dataset(
        dataset,
        cfg["split"],
        cfg["run"]["seed"],
        nested=needs_selection(cfg) and not cfg.get("tuning", {}).get("enabled", False),
        calibration=cfg.get("uncertainty"),
    )
    from .calibration import validate_calibration_split

    validate_calibration_split(dataset, split, cfg.get("uncertainty", {}))
    return PreparedExecution(dataset, split, inputs)


def prepare_fields(cfg, result):
    paths, input_paths = [], []
    dataset = load_field_dataset(cfg["data"], _source_paths=paths)
    prepared = _prepare_dataset(cfg, dataset, input_paths=input_paths)
    from ..input_preparation import _copy_snapshot

    result.artifacts["canonical_dataset"] = _copy_snapshot(
        cfg["data"],
        result.run_dir / "canonical_dataset",
        paths,
        dataset.revision,
        input_paths=input_paths,
        input_revision=None if prepared.inputs is None else prepared.inputs.revision,
    )
    result.artifacts["split"] = write_json(prepared.split, result.run_dir / "split.json")
