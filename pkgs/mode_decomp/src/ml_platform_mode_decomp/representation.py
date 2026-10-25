"""Fitted field-to-target composition; independent of runs and file layouts."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Self, cast

import numpy as np

from .catalog import build_coeff_codec, build_coeff_post, build_decomposer, build_preprocess
from .geometry.identity import geometry_fingerprint, snapshot_geometry
from .data.schema import FieldDataset, FieldSpec
from .coefficients.layout import CoefficientLayout
from .config import resolve_representation
from .field_preprocessing import _check_observed_finite


_TRANSFORM_WORK_BYTES = 4 * 1024 * 1024


def _batch_slices(count: int, sample_bytes: int):
    """Bound intermediate transforms, apart from an unavoidable single large field."""
    size = min(256, max(1, _TRANSFORM_WORK_BYTES // max(1, sample_bytes)))
    for start in range(0, count, size):
        yield slice(start, min(start + size, count))


def _observation_mask(mask, shape):
    if mask is None:
        return None
    values = np.asarray(mask)
    if values.shape not in (shape[1:-1], shape[:-1]):
        raise ValueError("Input masks must be spatial or sample-by-spatial")
    if values.dtype.kind not in "bui" or not np.isin(values, (0, 1)).all():
        raise ValueError("Input masks must contain boolean or 0/1 values")
    return values.astype(bool, copy=False)


class FieldRepresentation:
    """A fixed geometry, field transform, decomposer, codec and target transform.

    transform/inverse_transform use Z by default. space='a' explicitly selects
    the real packed coefficients before the fitted coefficient transformation.
    """

    def __init__(self, config: dict[str, Any], *, seed: int = 42):
        self.config = resolve_representation(config, seed=seed)
        self._fitted = False

    def fit(self, dataset: FieldDataset) -> Self:
        self._fit(dataset, coordinates=False)
        return self

    def fit_transform(self, dataset: FieldDataset, *, space: str = "z") -> np.ndarray:
        """Fit and return the already computed training coordinates, without caching them."""
        if space not in {"a", "z"}:
            raise ValueError("space must be 'a' or 'z'")
        packed, targets = self._fit(dataset)
        return packed if space == "a" else targets

    def _fit(self, dataset: FieldDataset, *, coordinates=True) -> tuple[np.ndarray, np.ndarray]:
        self._fitted = False
        if not len(dataset):
            raise ValueError("Representation fit requires field samples")
        self.geometry = snapshot_geometry(dataset.geometry)
        self.geometry_sha256 = geometry_fingerprint(self.geometry)
        self.field_shape = tuple(dataset.fields.shape[1:])
        self.components = tuple(dataset.components)
        self.field_kind = dataset.field_kind
        self.field_units = tuple(dataset.field_units or ())
        self.field_spec = FieldSpec.from_dataset(dataset)
        self.decomposer = build_decomposer(self.config["decomposition"])
        self.decomposer.validate_field_spec(self.field_spec, preprocess_name=self.config["field_preprocess"]["name"])
        self.mesh = len(self.field_shape) == 2
        fields, masks = self._inputs(dataset)
        self.preprocess = build_preprocess(self.config.get("field_preprocess", {"name": "none"}))
        self.preprocess.fit(fields, masks)
        transformed, transformed_masks = self.preprocess.transform(fields, masks)
        self.decomposer.fit_batch(transformed, transformed_masks, self.geometry)
        self.codec = build_coeff_codec(self.config.get("codec", {"name": "auto"}))
        self.post = build_coeff_post(self.config.get("coeff_post", {"name": "none"}))
        # A no-op post needs only one sample to establish packing. Fit callers
        # do not request training coordinates, so avoid encoding and discarding them.
        if not coordinates and self.post.name == "none":
            packed = self._encode(
                transformed[:1], None if transformed_masks is None else transformed_masks[:1], establish_layout=True
            )
        else:
            packed = self._encode(transformed, transformed_masks, establish_layout=True)
        if packed.shape[1] == 0 and self.post.name != "none":
            raise ValueError("A constant field has no coefficient targets; use coeff_post.name: none")
        self.post.fit(packed)
        targets = np.asarray(self.post.transform(packed))
        if targets.ndim != 2 or not np.isfinite(targets).all():
            raise ValueError("Fitted coefficient targets must be finite and two dimensional")
        self.packed_dim, self.target_dim = packed.shape[1], targets.shape[1]
        self.layout = CoefficientLayout.from_meta(self.raw_meta, self.packed_dim, self.decomposer.synthesis_kind)
        self.fit_sample_ids = tuple(str(x) for x in dataset.sample_ids)
        self._fitted = True
        return packed, targets

    def _inputs(self, value, masks=None):
        dataset_mask = None
        if hasattr(value, "geometry") and hasattr(value, "fields"):
            if geometry_fingerprint(value.geometry) != self.geometry_sha256:
                raise ValueError("Input geometry differs from the fitted representation")
            if tuple(value.components) != self.components:
                raise ValueError("Input field components differ from the fitted representation")
            if tuple(value.field_units or ()) != self.field_units:
                raise ValueError("Input field units differ from the fitted representation")
            if value.field_kind != self.field_kind:
                raise ValueError("Input field kind differs from the fitted representation")
            fields = np.asarray(value.fields)
            dataset_mask = value.observation_mask
        else:
            fields = np.asarray(value)
        if fields.ndim != len(self.field_shape) + 1 or tuple(fields.shape[1:]) != self.field_shape:
            raise ValueError(f"Expected a batch with field shape {self.field_shape}; got {fields.shape}")
        if fields.dtype.kind not in "fiu":
            raise ValueError("Input fields must contain real numeric values")
        masks = _observation_mask(masks, fields.shape)
        dataset_mask = _observation_mask(dataset_mask, fields.shape)
        if dataset_mask is not None:
            masks = dataset_mask if masks is None else masks & dataset_mask
        if self.mesh:
            fields = fields[:, :, None, :]
            if masks is not None:
                mask_array = np.asarray(masks)
                masks = mask_array[:, None] if mask_array.ndim == 1 else mask_array[..., None]
        spatial = fields.shape[1:-1]
        domain = self.geometry.mask
        if domain is not None:
            domain = np.asarray(domain, dtype=bool).reshape(spatial)
        if masks is not None:
            masks = np.broadcast_to(np.asarray(masks, dtype=bool), fields.shape[:-1])
            masks = masks if domain is None else masks & domain
        elif domain is not None:
            masks = np.broadcast_to(domain, fields.shape[:-1])
        if masks is not None and masks.all():
            masks = None
        _check_observed_finite(fields, masks)
        return fields, masks

    def _encode(self, fields, masks, *, establish_layout=False, preprocess=False):
        if not len(fields):
            raise ValueError("At least one field sample is required")
        packed = None
        sample_bytes = int(np.prod(fields.shape[1:])) * max(16, fields.dtype.itemsize) * 6
        for section in _batch_slices(len(fields), sample_bytes):
            batch, batch_mask = fields[section], None if masks is None else masks[section]
            if preprocess:
                batch, batch_mask = self.preprocess.transform(batch, batch_mask)
            raw = self.decomposer.transform_batch(batch, batch_mask, self.geometry)
            if establish_layout and section.start == 0:
                self.raw_meta = deepcopy(dict(self.decomposer.coeff_meta()))
                # Sample support belongs to this operation, never to fitted layout.
                self.raw_meta.pop("mask_valid_count", None)
                self.codec = self.codec.fit(raw[0], self.raw_meta)
            vectors = np.asarray(self.codec.encode_batch(raw, self.raw_meta))
            if np.iscomplexobj(vectors) or not np.isfinite(vectors).all():
                raise ValueError("Packed coefficients must be finite real values")
            if packed is None:
                packed = np.empty((len(fields), vectors.shape[1]), dtype=vectors.dtype)
            packed[section] = vectors
        return cast(np.ndarray, packed)

    def transform(
        self, fields: FieldDataset | np.ndarray, masks: np.ndarray | None = None, *, space: str = "z"
    ) -> np.ndarray:
        """Encode observations; an extra mask can only restrict the Dataset's support."""
        self._require_fitted()
        if space not in {"a", "z"}:
            raise ValueError("space must be 'a' or 'z'")
        fields, masks = self._inputs(fields, masks)
        packed = self._encode(fields, masks, preprocess=True)
        return packed if space == "a" else np.asarray(self.post.transform(packed))

    def inverse_transform(self, coefficients: np.ndarray, *, space: str = "z") -> np.ndarray:
        self._require_fitted()
        if space not in {"a", "z"}:
            raise ValueError("space must be 'a' or 'z'")
        values = np.asarray(coefficients)
        dimension = self.packed_dim if space == "a" else self.target_dim
        if (
            values.dtype.kind not in "fiu"
            or values.ndim != 2
            or values.shape[1] != dimension
            or not np.isfinite(values).all()
        ):
            raise ValueError(f"Expected finite {space} coefficients with shape (N,{dimension})")
        if not len(values):
            raise ValueError("At least one coefficient sample is required")
        fields = None
        sample_bytes = max(int(np.prod(self.field_shape)) * 16, self.packed_dim * 16) * 6
        for section in _batch_slices(len(values), sample_bytes):
            coefficients = values[section]
            packed = coefficients if space == "a" else np.asarray(self.post.inverse_transform(coefficients))
            if not np.isfinite(packed).all():
                raise ValueError("Coefficient inverse produced values outside its finite domain")
            raw = self.codec.decode_batch(packed, self.raw_meta)
            batch = self.decomposer.inverse_transform_batch(raw, self.geometry)
            if batch.ndim == 3:
                batch = batch[..., None]
            batch, _ = self.preprocess.inverse_transform(batch, None)
            if not np.isfinite(batch).all():
                raise ValueError("Physical field inverse transformation produced non-finite values")
            batch = batch.reshape((len(batch), *self.field_shape))
            if fields is None:
                fields = np.empty((len(values), *self.field_shape), dtype=batch.dtype)
            fields[section] = batch
        return cast(np.ndarray, fields)

    def reconstruct(self, fields: FieldDataset | np.ndarray, masks: np.ndarray | None = None) -> np.ndarray:
        return self.inverse_transform(self.transform(fields, masks))

    def list_modes(self, indices=None, *, space: str = "a") -> list[dict[str, Any]]:
        """Describe synthesis coordinates; A precedes any fitted coefficient postprocessing."""
        from .modes import list_modes

        return list_modes(self, indices, space=space)

    def mode_fields(self, indices, *, space: str = "a", reference=None, amplitude: float = 1.0) -> np.ndarray:
        """Generate requested baseline-free basis responses or explicit coordinate perturbations."""
        from .modes import mode_fields

        return mode_fields(self, indices, space=space, reference=reference, amplitude=amplitude)

    def reconstruct_modes(self, coefficients: np.ndarray, indices, *, space: str = "a", reference=None) -> np.ndarray:
        """Select coefficient values, using reference (zero by default) for all other coordinates."""
        from .modes import reconstruct_modes

        return reconstruct_modes(self, coefficients, indices, space=space, reference=reference)

    def cumulative_reconstruct(self, coefficients: np.ndarray, order, *, space: str = "a", reference=None):
        """Iterate selected-coordinate reconstructions in the requested order."""
        from .modes import cumulative_reconstruct

        return cumulative_reconstruct(self, coefficients, order, space=space, reference=reference)

    def _require_fitted(self):
        if not self._fitted:
            raise ValueError("Representation must be fitted first")
