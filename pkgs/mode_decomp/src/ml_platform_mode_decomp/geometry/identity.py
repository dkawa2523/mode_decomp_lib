"""Numerical geometry identity independent of dataset paths and bundle locations."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

import numpy as np

_LOCATION_KEYS = {"mask_path", "mesh_path", "source_path", "dataset_root", "local_path", "mask_source", "mesh_source"}


def _semantic_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _semantic_value(item) for key, item in value.items() if key not in _LOCATION_KEYS}
    if isinstance(value, (list, tuple)):
        return [_semantic_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return {"dtype": value.dtype.str, "shape": list(value.shape), "values": value.tolist()}
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported geometry metadata value: {type(value).__name__}")


def geometry_fingerprint_v1(geometry: Any) -> str:
    """Original 0.7.x identity, used only while reading schema-1 bundles."""
    digest = hashlib.sha256()
    digest.update(str(geometry.name).encode())
    arrays = {
        **geometry.coords,
        "__mask__": geometry.mask,
        "__weights__": geometry.weights,
        "__mass_matrix__": geometry.mass_matrix(),
    }
    for key, value in sorted(arrays.items()):
        digest.update(key.encode())
        if value is None:
            digest.update(b"null")
            continue
        array = np.ascontiguousarray(value)
        if array.dtype.hasobject:
            raise ValueError("Geometry arrays must not contain Python objects.")
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(array.tobytes())
    semantics = json.dumps(_semantic_value(geometry.meta), sort_keys=True, separators=(",", ":"), allow_nan=False)
    digest.update(semantics.encode())
    return digest.hexdigest()


# Only values that affect numerical geometry belong to its identity. Free-form
# descriptions, source paths, and generated/provided coordinate provenance do not.
_NUMERICAL_META = {
    "units",
    "coordinate_units",
    "angle_unit",
    "periodic_axes",
    "center",
    "radius",
    "r_inner",
    "r_outer",
    "r_inner_norm",
    "component_basis",
    "boundary_condition",
}


def geometry_fingerprint(geometry: Any) -> str:
    """Exact numerical identity after canonical dtype normalization."""
    from . import _domain_mask

    _domain_mask(geometry)
    digest = hashlib.sha256()
    name = "arbitrary_mask" if geometry.name == "mask" else geometry.name
    digest.update(str(name).encode())
    arrays = {
        **geometry.coords,
        "__mask__": geometry.mask,
        "__weights__": geometry.weights,
        "__mass_matrix__": geometry.mass_matrix(),
    }
    for key, value in sorted(arrays.items()):
        digest.update(key.encode())
        if value is None:
            digest.update(b"null")
            continue
        array = np.asarray(value)
        if array.dtype.kind not in "biuf":
            raise ValueError("Geometry arrays must contain real numeric values.")
        if key == "__mask__" and np.asarray(array, dtype=bool).all():
            digest.update(b"null")
            continue
        dtype = np.bool_ if key == "__mask__" else np.int64 if key == "faces" else np.float64
        normalized = np.ascontiguousarray(array, dtype=dtype)
        digest.update(str(normalized.shape).encode())
        digest.update(normalized.tobytes())
    semantics = {key: _semantic_value(value) for key, value in geometry.meta.items() if key in _NUMERICAL_META}
    digest.update(json.dumps(semantics, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
    return digest.hexdigest()


def snapshot_geometry(geometry: Any) -> Any:
    """Own fitted geometry without freezing or retaining borrowed input buffers."""
    from copy import deepcopy
    from dataclasses import replace

    return replace(
        geometry,
        coords={name: np.array(value, copy=True) for name, value in geometry.coords.items()},
        mask=None if geometry.mask is None else np.array(geometry.mask, copy=True),
        weights=None if geometry.weights is None else np.array(geometry.weights, copy=True),
        meta=deepcopy(geometry.meta),
        _mass_matrix=None if geometry.mass_matrix() is None else np.array(geometry.mass_matrix(), copy=True),
    )
