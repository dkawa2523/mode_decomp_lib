"""Prepare optional field or identified coefficient inputs for a surrogate."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import numpy as np
from ml_platform_core.artifacts import file_hash
from ml_platform_core.paths import resolve_path_within

from .data import load_field_dataset
from .data.input_spec import _SPEC_KEYS, field_input_spec
from .data.schema import FieldDataset, _ids
from .representation import FieldRepresentation


def _model_hash(value: Any) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{64}", value) is None:
        raise ValueError("Coefficient inputs require a complete SHA256 model_sha256")
    return value.lower()


@dataclass(frozen=True)
class CoefficientBatch:
    """Real Z coordinates in one frozen, identified representation.

    Numerical arrays are borrowed, as for FieldDataset; callers must not change
    them during an operation. Internal subsets retain the representation and
    source-file revision without refitting or rescanning the parent matrix.
    """

    coefficients: np.ndarray
    sample_ids: np.ndarray
    representation: FieldRepresentation
    model_sha256: str
    revision: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.representation, FieldRepresentation):
            raise ValueError("Coefficient inputs need a fitted FieldRepresentation")
        self.representation._require_fitted()
        values = np.asarray(self.coefficients)
        if values.dtype.kind not in "fiu" or values.ndim != 2 or not len(values) or not np.isfinite(values).all():
            raise ValueError("Coefficient inputs must be a nonempty finite real matrix in Z space")
        if values.shape[1] != self.representation.target_dim:
            raise ValueError("Coefficient input dimension differs from the saved representation's Z space")
        object.__setattr__(self, "coefficients", values)
        object.__setattr__(self, "sample_ids", _ids(self.sample_ids, len(values)))
        object.__setattr__(self, "model_sha256", _model_hash(self.model_sha256))

    def __len__(self) -> int:
        return len(self.sample_ids)

    def _subset_validated(self, indices: Any) -> CoefficientBatch:
        if not isinstance(indices, slice):
            selected = np.asarray(indices)
            if selected.ndim != 1 or selected.dtype.kind not in "iu":
                raise ValueError("Coefficient subset indices must be a one-dimensional integer array")
            indices = selected
        coefficients, ids = self.coefficients[indices], self.sample_ids[indices]
        if not len(ids):
            raise ValueError("A coefficient subset must contain at least one sample")
        result = object.__new__(CoefficientBatch)
        for name, value in (
            ("coefficients", coefficients),
            ("sample_ids", ids),
            ("representation", self.representation),
            ("model_sha256", self.model_sha256),
            ("revision", self.revision),
        ):
            object.__setattr__(result, name, value)
        return result


SurrogateInputs = FieldDataset | CoefficientBatch


def normalize_input_settings(data: Mapping[str, Any], *, require_artifact: bool = True) -> dict[str, Any] | None:
    """Validate the optional input section without changing legacy data settings."""
    settings = data.get("input")
    if settings is None:
        return None
    if (
        not isinstance(settings, Mapping)
        or not isinstance(settings.get("kind"), str)
        or settings["kind"] not in {"field", "coefficients"}
    ):
        raise ValueError("data.input must select kind: field or coefficients")
    kind = settings["kind"]
    allowed = (
        {"kind", "local_path", "format", "source_manifest", "clearml_dataset_id", *_SPEC_KEYS}
        if kind == "field"
        else {"kind", "local_path", "path", "model_artifact", "info_path"}
    )
    unknown = set(settings) - allowed
    if unknown:
        raise ValueError(f"Unknown data.input settings: {sorted(unknown)}")
    normalized = dict(settings)
    root = normalized.get("local_path", data.get("local_path"))
    if root is None:
        root = "."
    if not isinstance(root, (str, Path)) or not str(root).strip():
        raise ValueError("data.input.local_path must be a non-empty path")
    root = Path(root).expanduser().resolve()
    if "local_path" not in settings and root.is_file():
        root = root.parent
    normalized["local_path"] = str(root)
    if kind == "field":
        field_input_spec({key: value for key, value in normalized.items() if key != "kind"})
    else:
        for name in ("path", "model_artifact", "info_path"):
            value = normalized.get(name)
            if value is not None and (not isinstance(value, (str, Path)) or not str(value).strip()):
                raise ValueError(f"data.input.{name} must be a non-empty path")
        if not normalized.get("path"):
            raise ValueError("Coefficient inputs require data.input.path to an identified coefficients.npz")
        if require_artifact and not normalized.get("model_artifact"):
            raise ValueError("Coefficient training inputs require data.input.model_artifact")
        if normalized.get("info_path") and not normalized.get("model_artifact"):
            raise ValueError("data.input.info_path requires model_artifact")
        for name in ("model_artifact", "info_path"):
            if normalized.get(name):
                reference = Path(normalized[name]).expanduser()
                normalized[name] = str((reference if reference.is_absolute() else root / reference).resolve())
    return normalized


def _scalar_text(archive, name: str) -> str:
    value = archive[name]
    if value.ndim != 0 or value.dtype.kind not in "SU":
        raise ValueError(f"Coefficient archive {name} must be a scalar string")
    text = value.item()
    return text.decode("utf-8") if isinstance(text, bytes) else text


def _load_coefficients(
    settings: dict[str, Any], model: Any | None, source_paths: list[Path] | None
) -> CoefficientBatch:
    root = Path(settings["local_path"])
    path = resolve_path_within(
        root, settings["path"], setting="data.input.path", root_description="the coefficient input root"
    )
    if path.suffix.lower() != ".npz":
        raise ValueError("Coefficient inputs require an identified NPZ archive, not an unlabelled array")
    loaded = np.load(path, allow_pickle=False)
    if not isinstance(loaded, np.lib.npyio.NpzFile):
        raise ValueError("Coefficient inputs must be an NPZ archive")
    with loaded as archive:
        if {"coefficients", "sample_ids", "space", "model_sha256"} - set(archive.files):
            raise ValueError("Coefficient archive requires coefficients, sample_ids, space and model_sha256")
        if _scalar_text(archive, "space") != "z":
            raise ValueError(
                "Coefficient inputs support only Z space (space: z); A/raw coordinates cannot be substituted"
            )
        identity = _model_hash(_scalar_text(archive, "model_sha256"))
        coefficients, ids = archive["coefficients"], archive["sample_ids"]
    artifact = settings.get("model_artifact")
    if artifact is not None and file_hash(Path(artifact)) != identity:
        raise ValueError("Coefficient model_sha256 differs from data.input.model_artifact")
    if model is None:
        from .model_artifact import load_model

        if artifact is None:
            raise ValueError("Coefficient training inputs require data.input.model_artifact")
        bundle = load_model(artifact, settings.get("info_path"))
        representation = bundle if isinstance(bundle, FieldRepresentation) else getattr(bundle, "representation", None)
    else:
        representation = getattr(model, "input_representation", None)
        if identity != getattr(model, "input_model_sha256", None):
            raise ValueError("Coefficient model_sha256 differs from the surrogate's saved input representation")
    if not isinstance(representation, FieldRepresentation):
        raise ValueError("Coefficient model_artifact must contain a fitted field representation")
    if source_paths is not None:
        source_paths.append(path)
        if artifact is not None:
            from .model_artifact import _info_path

            source_paths.extend((Path(artifact), Path(settings.get("info_path") or _info_path(Path(artifact)))))
    return CoefficientBatch(coefficients, ids, representation, identity, revision=file_hash(path))


def load_surrogate_inputs(
    data: Mapping[str, Any],
    *,
    target: FieldDataset | None = None,
    model: Any | None = None,
    _source_paths: list[Path] | None = None,
) -> SurrogateInputs | None:
    """Load an optional source; saved coefficient models need only the new NPZ."""
    settings = normalize_input_settings(data, require_artifact=model is None)
    if settings is None:
        return None
    if model is not None and settings["kind"] != getattr(model, "input_kind", "conditions"):
        raise ValueError("data.input.kind differs from the saved surrogate input kind")
    if settings["kind"] == "field":
        source_settings = {key: value for key, value in settings.items() if key != "kind"}
        if source_settings.get("format", "npy") != "npy":
            raise ValueError("Source fields must be canonical NPY data; use the normal task input conversion first")
        inputs = load_field_dataset(source_settings, _source_paths=_source_paths)
        if model is not None:
            representation = getattr(model, "input_representation", None)
            if not isinstance(representation, FieldRepresentation):
                raise ValueError("Saved field-input model has no input representation")
            representation._require_fitted()
            representation._inputs(inputs)
    else:
        inputs = _load_coefficients(settings, model, _source_paths)
    return align_inputs(inputs, target) if target is not None else inputs


def align_inputs(inputs: SurrogateInputs | None, target: FieldDataset) -> SurrogateInputs | None:
    """Align complete independent source IDs to target order without dropping cases."""
    if inputs is None:
        return None
    if not isinstance(inputs, (FieldDataset, CoefficientBatch)):
        raise ValueError("Surrogate inputs must be a FieldDataset or CoefficientBatch")
    source_ids, target_ids = _ids(inputs.sample_ids, len(inputs)), _ids(target.sample_ids, len(target))
    if set(source_ids) != set(target_ids):
        raise ValueError("Input and target sample IDs must match exactly; missing or surplus cases cannot be dropped")
    if np.array_equal(source_ids, target_ids):
        return inputs
    positions = {identity: position for position, identity in enumerate(source_ids)}
    return inputs._subset_validated(np.asarray([positions[identity] for identity in target_ids], dtype=np.int64))


def subset_inputs(inputs: SurrogateInputs | None, indices: Any) -> SurrogateInputs | None:
    return None if inputs is None else inputs._subset_validated(indices)


__all__ = ["CoefficientBatch", "load_surrogate_inputs", "align_inputs", "subset_inputs"]
