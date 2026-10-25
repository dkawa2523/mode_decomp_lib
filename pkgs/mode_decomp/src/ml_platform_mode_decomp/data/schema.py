"""Small in-memory inputs shared by field workflows and numerical methods."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from operator import index as integer_index
from typing import Any, Self, overload

import numpy as np
from numpy.typing import ArrayLike

from ..geometry import DomainSpec, build_domain_spec


@dataclass(frozen=True)
class FieldSpec:
    """Physical field meaning; geometry is owned by the fitted representation."""

    shape: tuple[int, ...]
    kind: str
    components: tuple[str, ...]
    units: tuple[str, ...]

    @classmethod
    def from_dataset(cls, dataset):
        return cls(
            tuple(dataset.fields.shape[1:]),
            dataset.field_kind,
            tuple(dataset.components),
            tuple(dataset.field_units or ()),
        )

    def require_vector(self, method: str):
        label = "Hodge" if method == "hodge" else "Helmholtz" if method == "helmholtz" else method
        if self.kind != "vector" or self.shape[-1] != 2:
            raise ValueError(f"{label} requires fields.kind: vector with x/y components")
        if self.units and len(set(self.units)) != 1:
            raise ValueError(f"{label} vector components must use the same physical unit; convert them before fitting")


@dataclass(frozen=True)
class FieldSample:
    cond: np.ndarray | None
    field: np.ndarray
    mask: np.ndarray | None
    meta: dict[str, Any] = field(default_factory=dict)
    sample_id: str = ""


def _ids(value: Any, count: int) -> np.ndarray:
    ids = np.asarray(value)
    if ids.ndim != 1 or len(ids) != count:
        raise ValueError("sample_ids must have one ID per sample.")
    if any(item is None or (isinstance(item, (float, np.floating)) and not np.isfinite(item)) for item in ids):
        raise ValueError("sample_ids must not contain missing values.")
    ids = ids.astype(str)
    if any(not value.strip() for value in ids) or len(set(ids.tolist())) != count:
        raise ValueError("sample_ids must be non-empty and unique.")
    return ids


def _condition_array(value: Any, count: int, columns: Sequence[str]) -> np.ndarray:
    array = np.asarray(value)
    if count == 0 or array.ndim != 2 or array.shape[0] != count or array.shape[1] == 0:
        raise ValueError("conditions must have shape (N,P), with P >= 1.")
    if array.dtype.kind not in "fiu" or not np.isfinite(array).all():
        raise ValueError("conditions must contain finite real numbers.")
    if len(columns) != array.shape[1] or len(set(columns)) != len(columns) or any(not col for col in columns):
        raise ValueError("condition_columns must uniquely name all condition columns in order.")
    return array


def _field_units(value: Any, channels: int) -> tuple[str, ...]:
    if value is None or (isinstance(value, tuple) and not value):
        return ()
    if (
        not isinstance(value, (list, tuple))
        or len(value) != channels
        or any(not isinstance(unit, str) or not unit.strip() for unit in value)
    ):
        raise ValueError("fields.units must specify one non-empty unit string per component.")
    return tuple(value)


def _array_geometry(shape: tuple[int, ...], x: ArrayLike | None, y: ArrayLike | None) -> DomainSpec:
    height, width = shape[:2]
    if height < 1 or width < 1:
        raise ValueError("Field spatial axes must be nonempty")
    for name, values, count in (("x", x, width), ("y", y, height)):
        if values is not None and count == 1:
            raise ValueError(f"Explicit singleton {name} coordinates cannot define area; supply geometry with weights")
    # Two index nodes have total measure one. Collapsing only an unspecified
    # singleton axis below gives it unit index measure, without a physical width.
    padded = (max(height, 2), max(width, 2))
    geometry = build_domain_spec(
        {
            "name": "rectangle",
            "axes": {
                "x": {"values": np.arange(padded[1]) if x is None else x},
                "y": {"values": np.arange(padded[0]) if y is None else y},
            },
        },
        padded,
    )
    weights = geometry.weights
    assert weights is not None  # Rectangle construction always supplies quadrature.
    if height == 1:
        weights = weights.sum(axis=0, keepdims=True)
    if width == 1:
        weights = weights.sum(axis=1, keepdims=True)
    coords = {name: values[:height, :width] for name, values in geometry.coords.items()}
    metadata = {
        **geometry.meta,
        "x_range": (float(coords["x"][0, 0]), float(coords["x"][0, -1])),
        "y_range": (float(coords["y"][0, 0]), float(coords["y"][-1, 0])),
        "coordinate_source": {"x": "index" if x is None else "provided", "y": "index" if y is None else "provided"},
    }
    if x is None or y is None:
        metadata["weight_measure"] = "index_coordinate_measure"
        metadata["coordinate_note"] = "Unspecified axes use array indices; no physical units are inferred."
    if height == 1 or width == 1:
        metadata["singleton_index_measure"] = 1.0
    return replace(geometry, coords=coords, weights=weights, meta=metadata)


@dataclass(frozen=True)
class ConditionBatch:
    conditions: np.ndarray
    sample_ids: np.ndarray
    condition_columns: tuple[str, ...]
    metadata: Mapping[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        array = np.asarray(self.conditions)
        if array.ndim != 2:
            raise ValueError("conditions must have shape (N,P).")
        columns = tuple(str(col) for col in self.condition_columns)
        object.__setattr__(self, "conditions", _condition_array(array, len(array), columns))
        object.__setattr__(self, "sample_ids", _ids(self.sample_ids, len(array)))
        object.__setattr__(self, "condition_columns", columns)
        for name, values in self.metadata.items():
            if np.asarray(values).shape != (len(array),):
                raise ValueError(f"metadata {name!r} must have one value per sample.")

    def __len__(self) -> int:
        return len(self.sample_ids)


@dataclass(frozen=True)
class FieldDataset(Sequence[FieldSample]):
    fields: np.ndarray
    geometry: DomainSpec
    sample_ids: np.ndarray
    conditions: np.ndarray | None = None
    condition_columns: tuple[str, ...] = ()
    observation_mask: np.ndarray | None = None
    sample_metadata: Mapping[str, np.ndarray] = field(default_factory=dict)
    revision: str = ""
    manifest: Mapping[str, Any] = field(default_factory=dict)
    components: tuple[str, ...] = ()
    field_kind: str | None = None
    field_units: tuple[str, ...] | None = None

    @classmethod
    def from_arrays(
        cls,
        fields: ArrayLike,
        conditions: ArrayLike | None = None,
        *,
        layout: str = "NHWC",
        geometry: DomainSpec | None = None,
        x: ArrayLike | None = None,
        y: ArrayLike | None = None,
        sample_ids: Sequence[str] | np.ndarray | None = None,
        condition_columns: Sequence[str] | None = None,
        components: Sequence[str] | None = None,
        field_kind: str | None = None,
        field_units: Sequence[str] | None = None,
        observation_mask: ArrayLike | None = None,
        sample_metadata: Mapping[str, ArrayLike] | None = None,
    ) -> Self:
        """Make an in-memory dataset without files or implicit fitting/splitting.

        NHWC is the default batch layout; HWC explicitly denotes one grid field.
        NVC/VC require mesh geometry. Omitted grid coordinates are array indices;
        provide x/y or geometry to define the physical measure. Arrays keep their
        numerical dtype and shared observation masks are not expanded per sample.
        Kind defaults to scalar for one channel and multichannel otherwise;
        physical vector components require an explicit field_kind="vector".
        """
        dimensions = {"NHWC": 4, "HWC": 3, "NVC": 3, "VC": 2}
        values = np.asarray(fields)
        if layout not in dimensions or values.ndim != dimensions[layout]:
            raise ValueError("fields must match the explicit layout NHWC, HWC, NVC, or VC")
        if geometry is not None and (x is not None or y is not None):
            raise ValueError("Use geometry or x/y coordinates, not both")
        if layout in {"NVC", "VC"} and (geometry is None or geometry.name != "mesh"):
            raise ValueError("Mesh layout NVC/VC requires explicit mesh geometry")
        if layout in {"HWC", "VC"}:
            values = values[None]
        kind = ("scalar" if values.shape[-1] == 1 else "multichannel") if field_kind is None else field_kind
        if kind not in {"scalar", "vector", "multichannel"} or (kind == "scalar" and values.shape[-1] != 1):
            raise ValueError("fields.kind must be scalar (C=1), vector, or multichannel.")
        if geometry is None:
            geometry = _array_geometry(values.shape[1:], x, y)
        condition_values = None if conditions is None else np.asarray(conditions)
        columns = (
            tuple(condition_columns)
            if condition_columns is not None
            else tuple(f"condition_{index}" for index in range(condition_values.shape[1]))
            if condition_values is not None and condition_values.ndim == 2
            else ()
        )
        names = (
            tuple(f"component_{index}" for index in range(values.shape[-1]))
            if components is None
            else tuple(components)
        )
        return cls(
            fields=values,
            geometry=geometry,
            sample_ids=np.asarray([f"sample_{index:06d}" for index in range(len(values))])
            if sample_ids is None
            else np.asarray(sample_ids),
            conditions=condition_values,
            condition_columns=columns,
            observation_mask=None if observation_mask is None else np.asarray(observation_mask),
            sample_metadata={name: np.asarray(items) for name, items in (sample_metadata or {}).items()},
            components=names,
            field_kind=kind,
            field_units=_field_units(field_units, values.shape[-1]),
        )

    def __post_init__(self) -> None:
        fields = np.asarray(self.fields)
        expected_ndim = 3 if self.geometry.name == "mesh" else 4
        if fields.ndim != expected_ndim or fields.shape[0] == 0 or fields.shape[-1] < 1:
            raise ValueError("fields must be non-empty NHWC or mesh NVC, with at least one component.")
        if fields.dtype.kind not in "fiu":
            raise ValueError("fields must contain real numeric values.")
        spatial = fields.shape[1:-1]
        expected = self.geometry.grid_shape[:1] if self.geometry.name == "mesh" else self.geometry.grid_shape
        if spatial != expected:
            raise ValueError(f"field spatial shape {spatial} disagrees with geometry {expected}.")
        object.__setattr__(self, "fields", fields)
        object.__setattr__(self, "sample_ids", _ids(self.sample_ids, len(fields)))
        columns = tuple(str(col) for col in self.condition_columns)
        object.__setattr__(self, "condition_columns", columns)
        if self.conditions is not None:
            object.__setattr__(self, "conditions", _condition_array(self.conditions, len(fields), columns))
        elif columns:
            raise ValueError("condition_columns require conditions.")
        mask = self.observation_mask
        if mask is not None:
            mask = np.asarray(mask)
            if mask.shape not in (spatial, fields.shape[:-1]):
                raise ValueError("observation_mask must be spatial or sample-by-spatial.")
            if mask.dtype.kind not in "bui" or not np.isin(mask, (0, 1)).all():
                raise ValueError("observation_mask must contain boolean or 0/1 values.")
            object.__setattr__(self, "observation_mask", mask.astype(bool, copy=False))
        components = tuple(self.components)
        if components and (len(components) != fields.shape[-1] or len(set(components)) != len(components)):
            raise ValueError("components must uniquely name each field channel.")
        object.__setattr__(self, "components", components)
        # Read the former manifest representation only at the construction
        # boundary. Numerical consumers use these regular attributes.
        legacy = self.manifest.get("fields", {})
        kind = self.field_kind if self.field_kind is not None else legacy.get("kind")
        kind = ("scalar" if fields.shape[-1] == 1 else "multichannel") if kind is None else kind
        if kind not in {"scalar", "vector", "multichannel"} or (kind == "scalar" and fields.shape[-1] != 1):
            raise ValueError("fields.kind must be scalar (C=1), vector, or multichannel.")
        units = self.field_units if self.field_units is not None else legacy.get("units")
        object.__setattr__(self, "field_kind", kind)
        object.__setattr__(self, "field_units", _field_units(units, fields.shape[-1]))
        for name, values in self.sample_metadata.items():
            if np.asarray(values).shape != (len(fields),):
                raise ValueError(f"sample_metadata {name!r} must have one value per sample.")
        # Validate mapped arrays one sample at a time, without copying the entire observed dataset.
        for position, sample in enumerate(fields):
            valid = self._sample_mask(position)
            if not valid.any():
                raise ValueError("Every sample needs at least one valid field location.")
            if not np.isfinite(sample).all(where=valid[..., None]):
                raise ValueError("Observed fields must be finite; non-finite values are allowed only outside the mask.")

    def _sample_mask(self, position: int) -> np.ndarray:
        spatial = self.fields.shape[1:-1]
        observation = self.observation_mask
        if observation is None:
            mask = np.ones(spatial, dtype=bool)
        else:
            mask = observation[position] if observation.ndim == self.fields.ndim - 1 else observation
        if self.geometry.mask is not None:
            mask = mask & np.asarray(self.geometry.mask, dtype=bool).reshape(spatial)
        return mask

    @property
    def effective_mask(self) -> np.ndarray:
        spatial = self.fields.shape[1:-1]
        mask = self.observation_mask
        if mask is None:
            mask = np.ones(spatial, dtype=bool)
        domain_mask = self.geometry.mask
        if domain_mask is not None:
            mask = mask & np.asarray(domain_mask, dtype=bool).reshape(spatial)
        return np.broadcast_to(mask, self.fields.shape[:-1])

    @property
    def masks(self) -> np.ndarray:
        return self.effective_mask

    @property
    def domain_spec(self) -> DomainSpec:
        return self.geometry

    def __len__(self) -> int:
        return len(self.sample_ids)

    @overload
    def __getitem__(self, index: int) -> FieldSample: ...

    @overload
    def __getitem__(self, index: slice) -> FieldDataset: ...

    def __getitem__(self, index: int | slice) -> FieldSample | FieldDataset:
        if isinstance(index, slice):
            return self.subset(index)
        position = integer_index(index)
        field_array = self.fields[position]
        mask = self._sample_mask(position)
        if self.geometry.name == "mesh":
            field_array = field_array[:, None, :]
            mask = mask[:, None]
        sample_id = str(self.sample_ids[position])
        metadata = {}
        for name, values in self.sample_metadata.items():
            value = np.asarray(values)[position]
            metadata[name] = value.item() if isinstance(value, np.generic) else value
        return FieldSample(
            None if self.conditions is None else self.conditions[position],
            field_array,
            mask,
            {**metadata, "sample_id": sample_id},
            sample_id,
        )

    def __iter__(self) -> Iterator[FieldSample]:
        for index in range(len(self)):
            yield self[index]

    def subset(self, indices: Any) -> FieldDataset:
        return self._select(indices, validate=True)

    def _subset_validated(self, indices: Any) -> FieldDataset:
        """Select within an already validated, unchanged operation input.

        Slices borrow arrays; arbitrary indices gather only the selected rows.
        This is private so public construction/subset still validates input.
        """
        return self._select(indices, validate=False)

    def _select(self, indices: Any, *, validate: bool) -> FieldDataset:
        if isinstance(indices, slice):
            if not range(*indices.indices(len(self))):
                raise ValueError("A dataset subset requires a non-empty one-dimensional index array.")
        else:
            selected = np.asarray(indices)
            if selected.ndim != 1 or len(selected) == 0 or selected.dtype.kind not in "iu":
                raise ValueError("A dataset subset requires a non-empty one-dimensional index array.")
            indices = selected
            # Internal contiguous partitions can borrow mapped rows. Public
            # integer-array indexing keeps NumPy's gather/copy behavior.
            start = int(selected[0])
            if not validate and 0 <= start and start + len(selected) <= len(self):
                if np.array_equal(selected, np.arange(start, start + len(selected))):
                    indices = slice(start, start + len(selected))
        mask = self.observation_mask
        if mask is not None and mask.ndim == self.fields.ndim - 1:
            mask = mask[indices]
        values = dict(
            fields=self.fields[indices],
            geometry=self.geometry,
            sample_ids=self.sample_ids[indices],
            conditions=None if self.conditions is None else self.conditions[indices],
            condition_columns=self.condition_columns,
            observation_mask=mask,
            sample_metadata={name: np.asarray(values)[indices] for name, values in self.sample_metadata.items()},
            revision=self.revision,
            manifest=self.manifest,
            components=self.components,
            field_kind=self.field_kind,
            field_units=self.field_units,
        )
        if validate:
            return FieldDataset(**values)
        result = object.__new__(FieldDataset)
        for name, value in values.items():
            object.__setattr__(result, name, value)
        return result

    def iter_batches(self, indices: Any = None, *, batch_size: int = 64) -> Iterator[FieldDataset]:
        """Gather only the requested batch; arbitrary index arrays copy that batch.

        ``subset`` retains its ndarray-based public behavior. This iterator lets
        workflows bound temporary input arrays without promising that every
        numerical backend can itself fit out of core.
        """
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")
        if indices is None:
            for start in range(0, len(self), batch_size):
                yield self._subset_validated(slice(start, start + batch_size))
            return
        selected = np.asarray(indices)
        if selected.ndim != 1 or selected.dtype.kind not in "iu":
            raise ValueError("Batch indices must be a one-dimensional integer array.")
        for start in range(0, len(selected), batch_size):
            yield self._subset_validated(selected[start : start + batch_size])
