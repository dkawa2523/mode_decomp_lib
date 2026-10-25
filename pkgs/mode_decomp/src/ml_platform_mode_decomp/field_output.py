"""Portable physical field outputs, independent of runs and model persistence."""

from contextlib import ExitStack
import json
from numbers import Integral
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from .data.schema import _field_units, _ids
from .data._sources import _binary_mask
from .geometry import DomainSpec


def _field_schema(shape, kind, components, units):
    if (
        not isinstance(shape, (tuple, list))
        or len(shape) not in (2, 3)
        or any(isinstance(size, bool) or not isinstance(size, Integral) or size < 1 for size in shape)
    ):
        raise ValueError("Field output shape must be positive VC or HWC dimensions")
    shape = tuple(shape)
    if (
        not isinstance(kind, str)
        or kind not in {"scalar", "vector", "multichannel"}
        or (kind == "scalar" and shape[-1] != 1)
    ):
        raise ValueError("Field output kind must be scalar (C=1), vector, or multichannel")
    if (
        not isinstance(components, (tuple, list))
        or len(components) != shape[-1]
        or any(not isinstance(name, str) or not name.strip() for name in components)
        or len(set(components)) != len(components)
    ):
        raise ValueError("Field output components must uniquely name each channel")
    if not isinstance(units, (tuple, list)):
        raise ValueError("Field output units must be empty or specify one unit per component")
    _field_units(tuple(units), shape[-1])
    return shape


def _validate_geometry(metadata, arrays, shape):
    declared = metadata.get("geometry")
    if declared is None:
        if "geometry_weights" in arrays:
            raise ValueError("Field output geometry_weights require declared geometry")
        return
    if (
        not isinstance(declared, dict)
        or not isinstance(declared.get("name"), str)
        or not declared["name"].strip()
        or not isinstance(declared.get("coordinates"), dict)
        or not isinstance(declared.get("metadata"), dict)
    ):
        raise ValueError("Field output geometry must declare its coordinate arrays")
    coordinates = declared["coordinates"]
    if not coordinates or any(not isinstance(key, str) or key not in arrays for key in coordinates.values()):
        raise ValueError("Field output geometry is missing coordinate arrays")
    spatial = shape[:-1]
    if declared.get("name") == "mesh":
        vertices = arrays.get(coordinates.get("vertices"))
        if len(shape) != 2 or vertices is None or vertices.ndim != 2 or vertices.shape[0] != spatial[0]:
            raise ValueError("Field output geometry shape differs from the declared mesh field")
        count = declared.get("metadata", {}).get("vertex_count", spatial[0])
        if count != spatial[0]:
            raise ValueError("Field output geometry vertex count differs from the declared field")
    elif len(shape) != 3 or any(arrays[key].shape != spatial for key in coordinates.values()):
        raise ValueError("Field output geometry shape differs from the declared grid field")
    if "geometry_weights" in arrays:
        if declared.get("weights") != "geometry_weights":
            raise ValueError("Field output geometry_weights must declare their geometry role")
        geometry = DomainSpec(
            declared["name"],
            {name: arrays[key] for name, key in coordinates.items()},
            arrays.get("domain_mask"),
            arrays["geometry_weights"],
            declared["metadata"],
        )
        weights = geometry.integration_weights()
        if weights is None or not np.any(weights > 0):
            raise ValueError("Field output geometry_weights must have positive active mass")


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


def field_descriptor(field_spec, geometry=None, *, observation_mask=None, calibration_support=None):
    """Describe declared coordinates and distinguish domain, observed and calibrated support."""
    _field_schema(field_spec.shape, field_spec.kind, field_spec.components, field_spec.units)
    metadata: dict[str, Any] = {
        "schema": "field_output_v1",
        "field_shape": list(field_spec.shape),
        "field_kind": field_spec.kind,
        "components": list(field_spec.components),
        "field_units": list(field_spec.units),
        "geometry": None,
        "support": {"output": "finite_values_per_sample_and_component"},
    }
    arrays = {}
    if geometry is not None:
        metadata["geometry"] = {
            "name": geometry.name,
            "metadata": _json_value(geometry.meta),
            "coordinates": {name: f"geometry_{name}" for name in geometry.coords},
        }
        arrays.update({f"geometry_{name}": np.asarray(value) for name, value in geometry.coords.items()})
        if geometry.mask is not None:
            arrays["domain_mask"] = _binary_mask(geometry.mask, "domain_mask").reshape(field_spec.shape[:-1])
            metadata["support"]["domain"] = "domain_mask"
        if geometry.weights is not None:
            arrays["geometry_weights"] = np.asarray(geometry.weights).reshape(field_spec.shape[:-1])
            metadata["geometry"]["weights"] = "geometry_weights"
        _validate_geometry(metadata, arrays, tuple(field_spec.shape))
    if observation_mask is not None:
        arrays["observation_mask"] = _binary_mask(observation_mask, "observation_mask")
        metadata["support"]["observations"] = "observation_mask"
    if calibration_support is not None:
        arrays["calibration_support"] = _binary_mask(calibration_support, "calibration_support")
        metadata["support"]["calibration"] = "calibration_support"
    return metadata, arrays


def _validate_support_shapes(arrays, count, shape):
    spatial = shape[:-1]
    for name in ("domain_mask", "calibration_support", "geometry_weights"):
        if name in arrays and arrays[name].shape != spatial:
            raise ValueError(f"Field output {name} must match the declared spatial shape")
    if "observation_mask" in arrays and arrays["observation_mask"].shape not in (spatial, (count, *spatial)):
        raise ValueError("Field output observation_mask must be spatial or sample-by-spatial")


def _validate_support(metadata, arrays):
    roles = {"domain": "domain_mask", "observations": "observation_mask", "calibration": "calibration_support"}
    support = metadata.get("support")
    if support is None and metadata["schema"] == "field_output_v1":
        raise ValueError("Field output must declare its output support")
    if support is not None:
        if (
            not isinstance(support, dict)
            or set(support) - {"output", *roles}
            or support.get("output") != "finite_values_per_sample_and_component"
        ):
            raise ValueError("Field output support must declare the supported output and mask roles")
        for role, name in roles.items():
            if role in support and (support[role] != name or name not in arrays):
                raise ValueError(f"Field output {role} support is missing its declared {name}")
            if name in arrays and support.get(role) != name:
                raise ValueError(f"Field output {name} must declare its {role} support role")
    for name in roles.values():
        if name in arrays:
            _binary_mask(arrays[name], name)
    geometry = metadata.get("geometry")
    if isinstance(geometry, dict) and "weights" in geometry:
        if geometry["weights"] != "geometry_weights" or "geometry_weights" not in arrays:
            raise ValueError("Field output geometry is missing its declared geometry_weights")


def _validate_field_arrays(metadata, arrays):
    """Check declarations without scanning or copying mmap field values."""
    if not isinstance(metadata, dict) or metadata.get("schema") not in {"field_output_v1", "field_predictions_v1"}:
        raise ValueError("Expected field_output_v1 or legacy field_predictions_v1 metadata")
    metadata.setdefault("geometry", None)
    primary = metadata.get("primary_array", "predictions")
    if primary not in arrays or "sample_ids" not in arrays:
        raise ValueError("Field output must include its primary array and sample IDs")
    shape = _field_schema(
        metadata.get("field_shape"),
        metadata.get("field_kind"),
        metadata.get("components"),
        metadata.get("field_units", ()),
    )
    count = metadata.get("sample_count")
    fields, ids = arrays[primary], arrays["sample_ids"]
    if (
        isinstance(count, bool)
        or not isinstance(count, Integral)
        or count < 1
        or fields.shape != (count, *shape)
        or fields.dtype.kind not in "fiu"
    ):
        raise ValueError("Field array differs from its declared sample count or field shape")
    if ids.shape != (count,):
        raise ValueError("Field output sample IDs must have one value per field")
    _ids(ids, int(count))
    components, units = metadata.get("components", []), metadata.get("field_units", [])
    if len(components) != shape[-1] or (units and len(units) != shape[-1]):
        raise ValueError("Field output components and units must match the channel axis")
    for name, declared in (("components", components), ("field_units", units)):
        if name in arrays and not np.array_equal(arrays[name], np.asarray(declared, dtype=str)):
            raise ValueError(f"Field output {name} differs from its metadata")
    bands = {"lower", "upper"} & arrays.keys()
    if bands and bands != {"lower", "upper"}:
        raise ValueError("Field output requires both lower and upper bands")
    if metadata.get("field_uncertainty") is not None and not bands:
        raise ValueError("Calibrated field output is missing its saved bands")
    if any(arrays[name].shape != fields.shape for name in bands):
        raise ValueError("Prediction bands must match the primary field shape")
    _validate_support_shapes(arrays, count, shape)
    _validate_support(metadata, arrays)
    _validate_geometry(metadata, arrays, shape)


def write_field_batches(
    path,
    sample_ids,
    batches,
    field_spec,
    geometry=None,
    *,
    array_name="predictions",
    observation_mask=None,
    calibration_support=None,
    uncertainty=None,
):
    """Write a complete NPZ or NPY collection and return paths after atomic publication.

    NPY metadata is installed last. A failed write removes only files created by
    this operation; existing output paths are never replaced.
    """
    path = Path(path)
    if path.suffix not in {".npz", ".npy"}:
        raise ValueError("Field output must end in .npz or .npy")
    if path.exists():
        raise FileExistsError(f"Field output already exists: {path}")
    raw_ids = np.asarray(sample_ids)
    if raw_ids.ndim != 1 or not len(raw_ids):
        raise ValueError("Field output requires a nonempty one-dimensional sample ID array")
    ids = _ids(raw_ids, len(raw_ids))
    count, shape = len(ids), tuple(field_spec.shape)
    names = (array_name,) if uncertainty is None else (array_name, "lower", "upper")
    metadata, small = field_descriptor(
        field_spec, geometry, observation_mask=observation_mask, calibration_support=calibration_support
    )
    reserved = {"lower", "upper", "field_metadata", "sample_ids", "components", "field_units", *small}
    if (
        not isinstance(array_name, str)
        or not array_name.isidentifier()
        or array_name in reserved
        or array_name.startswith("interval_")
    ):
        raise ValueError("Field output array_name must be a non-reserved identifier")
    _validate_support_shapes(small, count, shape)
    metadata.update(sample_count=count, field_uncertainty=uncertainty, primary_array=array_name)
    small.update(
        sample_ids=ids,
        components=np.asarray(field_spec.components, dtype=str),
        field_units=np.asarray(field_spec.units, dtype=str),
    )
    if uncertainty is not None:
        small.update(
            {
                f"interval_{key}": np.asarray(uncertainty[key])
                for key in ("method", "coverage", "scope", "normalization")
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".fields_", dir=path.parent) as temporary:
        root = Path(temporary)
        files = {name: root / f"{name}.npy" for name in names}
        written, dtypes = 0, None
        with ExitStack() as stack:
            handles = {name: stack.enter_context(file.open("xb")) for name, file in files.items()}
            for batch in batches:
                parts = batch if isinstance(batch, tuple) else (batch,)
                if len(parts) != len(names):
                    raise ValueError("Field batches and calibration metadata must include the same bands")
                parts = tuple(np.asarray(part) for part in parts)
                rows = len(parts[0])
                if not rows or written + rows > count or any(part.shape != (rows, *shape) for part in parts):
                    raise ValueError("Field batches must preserve declared sample count and field shape")
                if dtypes is None:
                    dtypes = tuple(part.dtype for part in parts)
                    for name, dtype in zip(names, dtypes, strict=True):
                        if dtype.kind not in "fiu":
                            raise ValueError("Fields must contain real numeric values")
                        np.lib.format.write_array_header_2_0(
                            handles[name],
                            {
                                "descr": np.lib.format.dtype_to_descr(dtype),
                                "fortran_order": False,
                                "shape": (count, *shape),
                            },
                        )
                for name, dtype, part in zip(names, dtypes, parts, strict=True):
                    if part.dtype != dtype:
                        raise ValueError("Field dtype changed between batches")
                    part.tofile(handles[name])
                written += rows
        if written != count:
            raise ValueError("Field batches must contain one field per sample ID")
        if path.suffix == ".npz":
            small["field_metadata"] = np.asarray(json.dumps(_json_value(metadata), ensure_ascii=False))
            archive_path = root / "fields.npz"
            with ZipFile(archive_path, "w", compression=ZIP_DEFLATED, allowZip64=True) as archive:
                for name, file in files.items():
                    archive.write(file, arcname=f"{name}.npy")
                for name, array in small.items():
                    with archive.open(f"{name}.npy", "w", force_zip64=True) as handle:
                        np.lib.format.write_array(handle, array, allow_pickle=False)
            # A hard link publishes atomically without overwriting a competing writer.
            path.hardlink_to(archive_path)
            return {array_name: path}
        for name, array in small.items():
            files[name] = root / f"{name}.npy"
            np.save(files[name], array, allow_pickle=False)
        paths = {name: path if name == array_name else path.with_name(f"{path.stem}_{name}.npy") for name in files}
        metadata["arrays"] = {name: target.name for name, target in paths.items()}
        metadata_path = path.with_name(f"{path.stem}_metadata.json")
        temporary_metadata = root / "metadata.json"
        temporary_metadata.write_text(json.dumps(_json_value(metadata), ensure_ascii=False, indent=2), encoding="utf-8")
        published = []
        try:
            for name, target in paths.items():
                target.hardlink_to(files[name])
                published.append(target)
            metadata_path.hardlink_to(temporary_metadata)
        except (OSError, ValueError):
            for target in published:
                target.unlink(missing_ok=True)
            raise
        return {**paths, "metadata": metadata_path}


def read_field_output(path):
    """Read portable output; legacy arrays remain readable with unknown geometry."""
    path = Path(path)
    metadata = {"schema": "legacy_field_output", "geometry": None}
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files if name != "field_metadata"}
            if "field_metadata" in archive:
                metadata = json.loads(str(archive["field_metadata"].item()))
                _validate_field_arrays(metadata, arrays)
        return {"metadata": metadata, "arrays": arrays}
    if path.suffix == ".npy":
        metadata_path = path.with_name(f"{path.stem}_metadata.json")
    elif path.suffix == ".json":
        metadata_path = path
    else:
        raise ValueError("Field output must be NPZ, NPY or its metadata JSON")
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict) or not isinstance(metadata.get("arrays"), dict):
            raise ValueError("Field metadata must declare its associated array files")
        if path.suffix == ".npy":
            reference = metadata["arrays"].get(metadata.get("primary_array", "predictions"))
            if not isinstance(reference, str) or (metadata_path.parent / reference).resolve() != path.resolve():
                raise ValueError("Field metadata primary array differs from the requested NPY file")
        arrays = {}
        try:
            for name, reference in metadata["arrays"].items():
                target = (metadata_path.parent / reference).resolve()
                if not target.is_relative_to(metadata_path.parent.resolve()):
                    raise ValueError("Field array reference must remain within the output directory")
                value = np.load(target, allow_pickle=False, mmap_mode="r")
                if not isinstance(value, np.ndarray):
                    value.close()
                    raise ValueError("Field metadata array files must be ordinary NPY arrays")
                arrays[name] = value
            _validate_field_arrays(metadata, arrays)
        except Exception:
            # Ownership transfers only when the complete collection is returned.
            # Failed loads must not lock partial output files on Windows.
            for value in arrays.values():
                mapping = getattr(value, "_mmap", None)
                if mapping is not None:
                    mapping.close()
            raise
    else:
        arrays = {"predictions": np.load(path, allow_pickle=False, mmap_mode="r")}
    return {"metadata": metadata, "arrays": arrays}
