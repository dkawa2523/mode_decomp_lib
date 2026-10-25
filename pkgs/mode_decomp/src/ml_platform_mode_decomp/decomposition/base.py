"""Shared decomposer base utilities."""

from __future__ import annotations

from typing import Any, Callable, Dict, Generic, Mapping, NamedTuple, Sequence, TypeVar

import numpy as np

from ml_platform_mode_decomp.config import cfg_get
from ml_platform_mode_decomp.geometry import DomainSpec


def _normalize_field(field: np.ndarray) -> tuple[np.ndarray, bool]:
    field = np.asarray(field)
    if field.ndim == 2:
        return field[..., None], True
    if field.ndim == 3:
        return field, False
    raise ValueError(f"field must be 2D or 3D, got shape {field.shape}")


def _mask_values(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask)
    if mask.dtype.kind not in "bfiu" or (mask.dtype.kind != "b" and not np.isin(mask, (0, 1)).all()):
        raise ValueError("masks must contain boolean or 0/1 values")
    return mask.astype(bool, copy=False)


def _normalize_mask(mask: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray | None:
    if mask is None:
        return None
    mask = _mask_values(mask)
    if mask.shape != shape:
        raise ValueError(f"mask shape {mask.shape} does not match {shape}")
    return mask


def _combine_masks(
    field_mask: np.ndarray | None,
    domain_mask: np.ndarray | None,
) -> np.ndarray | None:
    if field_mask is not None:
        field_mask = _mask_values(field_mask)
    if domain_mask is not None:
        domain_mask = _mask_values(domain_mask)
    if field_mask is None and domain_mask is None:
        return None
    if field_mask is None:
        return domain_mask
    if domain_mask is None:
        return field_mask
    return field_mask & domain_mask


def require_cfg(cfg: Mapping[str, Any], key: str, *, label: str) -> Any:
    value = cfg_get(cfg, key, None)
    if value is None:
        raise ValueError(f"{label}.{key} is required")
    if isinstance(value, str) and not value.strip():
        raise ValueError(f"{label}.{key} must be non-empty")
    return value


def require_cfg_stripped(cfg: Mapping[str, Any], key: str, *, label: str) -> str:
    value = require_cfg(cfg, key, label=label)
    return str(value).strip()


def parse_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
    raise ValueError(f"Invalid boolean value: {value}")


RawCoefficients = TypeVar("RawCoefficients")


class _FieldInput(NamedTuple):
    field: np.ndarray
    mask: np.ndarray | None


class FieldBatch:
    """Borrowed numerical arrays; sample and channel access creates only views."""

    def __init__(self, fields: np.ndarray, masks: np.ndarray | None = None):
        self.fields = np.asarray(fields)
        if masks is not None:
            masks = _mask_values(masks)
        self.masks = masks

    def __len__(self) -> int:
        return len(self.fields)

    def __getitem__(self, index: int) -> _FieldInput:
        return _FieldInput(self.fields[index], None if self.masks is None else self.masks[index])


class _ChannelFields:
    """Borrow one component from a replayable dataset without stacking its fields."""

    def __init__(self, dataset: Any, channel: int, channels: int):
        self.dataset, self.channel, self.channels = dataset, channel, channels

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int) -> _FieldInput:
        sample = self.dataset[index]
        field = np.asarray(sample.field)
        if field.ndim != 3 or field.shape[-1] != self.channels:
            raise ValueError("channelwise adapter requires the same component count in every field")
        return _FieldInput(field[..., self.channel], sample.mask)


def field_arrays(dataset: Any, domain_spec: DomainSpec, *, scalar: bool = False):
    """Materialize legacy sample inputs once, while keeping array batches borrowed."""
    if isinstance(dataset, FieldBatch):
        fields, masks = dataset.fields, dataset.masks
    else:
        samples = [dataset[index] for index in range(len(dataset))]
        if not samples:
            raise ValueError("At least one training field is required")
        fields = np.stack([np.asarray(sample.field) for sample in samples])
        if all(sample.mask is None for sample in samples):
            masks = None
        elif any(sample.mask is None for sample in samples):
            raise ValueError("mask must be present for all samples or none")
        else:
            masks = np.stack([np.asarray(sample.mask) for sample in samples])
    if fields.dtype.kind not in "fiu":
        raise ValueError("Training fields must contain real numeric values")
    if scalar and fields.ndim == 4 and fields.shape[-1] == 1:
        fields = fields[..., 0]
    if masks is not None:
        masks = _mask_values(masks)
    expected_ndim = 3 if scalar else 4
    if not len(fields) or fields.ndim != expected_ndim or tuple(fields.shape[1:3]) != domain_spec.grid_shape:
        raise ValueError(f"Training field batch does not match domain {domain_spec.grid_shape}: {fields.shape}")
    return fields, masks


class BaseDecomposer(Generic[RawCoefficients]):
    """Minimal decomposer interface."""

    name: str
    synthesis_kind = "unknown"

    def __init__(self, *, cfg: Mapping[str, Any]) -> None:
        self._cfg = dict(cfg)
        self._coeff_meta: Dict[str, Any] | None = None
        self._variable_blocks = False

    def fit(self, dataset: Any | None = None, domain_spec: DomainSpec | None = None) -> "BaseDecomposer":
        if domain_spec is not None:
            self._bind_geometry(domain_spec, fitting=True)
        return self

    def validate_field_spec(self, spec, *, preprocess_name="none") -> None:
        """Operators with physical component assumptions validate them here."""

    def _bind_geometry(self, domain_spec: DomainSpec, *, fitting=False):
        from ml_platform_mode_decomp.geometry.identity import geometry_fingerprint

        identity = geometry_fingerprint(domain_spec)
        previous = getattr(self, "_fitted_geometry_sha256", None)
        if not fitting and previous is not None and previous != identity:
            raise ValueError("Geometry differs from the fixed decomposer coordinates, mask or weights")
        self._fitted_geometry_sha256 = identity

    def fit_batch(self, fields: np.ndarray, masks: np.ndarray | None, domain_spec: DomainSpec):
        self._bind_geometry(domain_spec, fitting=True)
        return self.fit(FieldBatch(fields, masks), domain_spec=domain_spec)

    def transform(self, field: np.ndarray, mask: np.ndarray | None, domain_spec: DomainSpec) -> RawCoefficients:
        raise NotImplementedError

    def transform_batch(self, fields: np.ndarray, masks: np.ndarray | None, domain_spec: DomainSpec) -> Any:
        """Transform a bounded batch; structured decomposers retain their sample API."""
        self._bind_geometry(domain_spec)
        coefficients = []
        first_meta = None
        for index, field in enumerate(fields):
            coefficients.append(self.transform(field, None if masks is None else masks[index], domain_spec))
            if first_meta is None:
                first_meta = dict(self.coeff_meta())
        if first_meta is not None:
            self._coeff_meta = first_meta
        return coefficients

    def inverse_transform(self, coeff: RawCoefficients, domain_spec: DomainSpec) -> np.ndarray:
        raise NotImplementedError

    def inverse_transform_batch(self, coefficients, domain_spec: DomainSpec) -> np.ndarray:
        self._bind_geometry(domain_spec)
        return np.stack([self.inverse_transform(coeff, domain_spec) for coeff in coefficients])

    def coeff_meta(self) -> Mapping[str, Any]:
        if self._coeff_meta is None:
            raise ValueError("coeff_meta is not available before transform")
        return self._coeff_meta

    def _prepare_field(
        self,
        field: np.ndarray,
        mask: np.ndarray | None,
        domain_spec: DomainSpec,
        *,
        allow_zero_fill: bool,
    ) -> tuple[np.ndarray, bool]:
        if domain_spec is None:
            raise ValueError("domain_spec is required")
        self._bind_geometry(domain_spec)
        field_3d, was_2d = _normalize_field(field)
        shape = field_3d.shape[:2] if was_2d else field_3d.shape
        if self._coeff_meta is not None and self._coeff_meta.get("field_shape") != list(shape):
            raise ValueError("Field layout differs from the established decomposer layout")
        if field_3d.shape[:2] != domain_spec.grid_shape:
            raise ValueError(f"field shape {field_3d.shape[:2]} does not match domain {domain_spec.grid_shape}")
        field_mask = _normalize_mask(mask, field_3d.shape[:2])
        combined = _combine_masks(field_mask, domain_spec.mask)
        if combined is not None and not combined.all():
            if not allow_zero_fill:
                raise ValueError("mask has invalid entries for FFT/DCT without mask_zero_fill policy")
            # CONTRACT: only zero-fill when policy explicitly allows it.
            field_3d = field_3d.copy()
            field_3d[~combined] = 0.0
        return field_3d, was_2d

    @staticmethod
    def _reshape_coeff(coeff: np.ndarray, shape: Sequence[int], *, name: str | None = None) -> np.ndarray:
        coeff = np.asarray(coeff).reshape(-1)
        expected = int(np.prod(shape))
        if coeff.size != expected:
            label = name or "coeff"
            raise ValueError(f"{label} size {coeff.size} does not match expected {expected}")
        return coeff.reshape(tuple(int(x) for x in shape), order="C")

    def _coeff_meta_base(
        self,
        *,
        field_shape: Sequence[int],
        field_ndim: int,
        field_layout: str,
        channels: int,
        coeff_shape: Sequence[int],
        coeff_layout: str,
        complex_format: str,
        flatten_order: str = "C",
    ) -> Dict[str, Any]:
        if self._coeff_meta is not None and self._coeff_meta.get("field_shape") != list(field_shape):
            self._field_ndim = self._coeff_meta["field_ndim"]
            raise ValueError("Field layout differs from the established decomposer layout")
        return {
            "method": self.name,
            "field_shape": [int(x) for x in field_shape],
            "field_ndim": int(field_ndim),
            "field_layout": str(field_layout),
            "channels": int(channels),
            "coeff_shape": [int(x) for x in coeff_shape],
            "coeff_layout": str(coeff_layout),
            "flatten_order": str(flatten_order),
            "complex_format": str(complex_format),
            "keep": "all",
        }


class GridDecomposerBase(BaseDecomposer[np.ndarray]):
    """Baseclass for grid decomposers with per-channel transforms."""

    def __init__(self, *, cfg: Mapping[str, Any]) -> None:
        super().__init__(cfg=cfg)
        self._coeff_shape: tuple[int, ...] | None = None
        self._field_ndim: int | None = None

    def _set_grid_layout(self, field_shape, was_2d, coeff_shape, *, coeff_layout, complex_format, extra_meta=None):
        shape = field_shape[:2] if was_2d else field_shape
        if self._coeff_meta is not None and self._coeff_meta.get("field_shape") != list(shape):
            raise ValueError("Field layout differs from the established decomposer layout")
        if (
            self._coeff_meta is not None
            and self._coeff_meta.get("field_shape") == list(shape)
            and self._coeff_shape == tuple(coeff_shape)
        ):
            return
        self._coeff_shape = tuple(coeff_shape)
        self._field_ndim = 2 if was_2d else 3
        meta = self._coeff_meta_base(
            field_shape=shape,
            field_ndim=self._field_ndim,
            field_layout="HW" if was_2d else "HWC",
            channels=int(field_shape[-1]),
            coeff_shape=coeff_shape,
            coeff_layout=coeff_layout,
            complex_format=complex_format,
        )
        if extra_meta:
            meta.update(extra_meta)
        self._coeff_meta = meta

    def _grid_transform_batch(
        self,
        fields,
        masks,
        domain_spec,
        *,
        allow_zero_fill,
        forward_fn,
        coeff_layout,
        complex_format,
        extra_meta=None,
    ):
        from ml_platform_mode_decomp.geometry import validate_decomposer_compatibility

        validate_decomposer_compatibility(domain_spec, self._cfg)
        self._bind_geometry(domain_spec)
        fields = np.asarray(fields)
        was_2d = fields.ndim == 3
        if was_2d:
            fields = fields[..., None]
        if fields.ndim != 4 or tuple(fields.shape[1:3]) != domain_spec.grid_shape:
            raise ValueError(f"Field batch does not match domain {domain_spec.grid_shape}: {fields.shape}")
        combined = domain_spec.mask
        if masks is not None:
            masks = _mask_values(masks)
            if masks.shape not in (fields.shape[1:3], fields.shape[:3]):
                raise ValueError("Batch masks must be spatial or sample-by-spatial")
            combined = masks if combined is None else masks & combined
        if combined is not None and not combined.all():
            if not allow_zero_fill:
                raise ValueError("mask has invalid entries for FFT/DCT without mask_zero_fill policy")
            fields = np.where(combined[..., None], fields, 0.0)
        coefficients = forward_fn(np.ascontiguousarray(np.moveaxis(fields, -1, 1)))
        self._set_grid_layout(
            fields.shape[1:],
            was_2d,
            coefficients.shape[1:],
            coeff_layout=coeff_layout,
            complex_format=complex_format,
            extra_meta=extra_meta,
        )
        return coefficients

    def _grid_inverse_batch(self, coefficients, domain_spec, *, inverse_fn):
        from ml_platform_mode_decomp.geometry import validate_decomposer_compatibility

        validate_decomposer_compatibility(domain_spec, self._cfg)
        self._bind_geometry(domain_spec)
        if self._coeff_shape is None or self._field_ndim is None:
            raise ValueError("transform must be called before inverse_transform")
        coefficients = np.asarray(coefficients)
        if coefficients.ndim != len(self._coeff_shape) + 1 or coefficients.shape[1:] != self._coeff_shape:
            raise ValueError("Coefficient batch does not match the fitted grid layout")
        fields = np.moveaxis(inverse_fn(coefficients), 1, -1)
        return fields[..., 0] if self._field_ndim == 2 else fields

    def _grid_transform(
        self,
        field: np.ndarray,
        mask: np.ndarray | None,
        domain_spec: DomainSpec,
        *,
        allow_zero_fill: bool,
        forward_fn: Callable[[np.ndarray], np.ndarray],
        coeff_layout: str,
        complex_format: str,
        extra_meta: Mapping[str, Any] | None = None,
    ) -> np.ndarray:
        from ml_platform_mode_decomp.geometry import validate_decomposer_compatibility

        validate_decomposer_compatibility(domain_spec, self._cfg)
        field_3d, was_2d = self._prepare_field(field, mask, domain_spec, allow_zero_fill=allow_zero_fill)
        coeffs = [forward_fn(field_3d[..., ch]) for ch in range(field_3d.shape[-1])]
        coeff_tensor = np.stack(coeffs, axis=0)
        self._set_grid_layout(
            field_3d.shape,
            was_2d,
            coeff_tensor.shape,
            coeff_layout=coeff_layout,
            complex_format=complex_format,
            extra_meta=extra_meta,
        )
        return coeff_tensor

    def _grid_inverse(
        self,
        coeff: np.ndarray,
        domain_spec: DomainSpec,
        *,
        inverse_fn: Callable[[np.ndarray], np.ndarray],
    ) -> np.ndarray:
        from ml_platform_mode_decomp.geometry import validate_decomposer_compatibility

        validate_decomposer_compatibility(domain_spec, self._cfg)
        self._bind_geometry(domain_spec)
        if self._coeff_shape is None or self._field_ndim is None:
            raise ValueError("transform must be called before inverse_transform")
        coeff_tensor = self._reshape_coeff(coeff, self._coeff_shape, name=self.name)
        field_channels = [inverse_fn(coeff_tensor[ch]) for ch in range(coeff_tensor.shape[0])]
        field_hat = np.stack(field_channels, axis=-1)
        if self._field_ndim == 2 and field_hat.shape[-1] == 1:
            return field_hat[..., 0]
        return field_hat


class WeightedBasisDecomposerBase(BaseDecomposer[np.ndarray]):
    """Weighted projection with a shared fixed-support solve across batch samples."""

    def __init__(self, *, cfg: Mapping[str, Any]) -> None:
        super().__init__(cfg=cfg)
        self._coeff_shape: tuple[int, ...] | None = None
        self._field_ndim: int | None = None

    def _get_basis(self, domain_spec: DomainSpec) -> np.ndarray:
        raise NotImplementedError

    def _projection_layout(self, field_3d, was_2d, coeff_tensor, domain_spec):
        raise NotImplementedError

    def transform_batch(self, fields: np.ndarray, masks: np.ndarray | None, domain_spec: DomainSpec):
        fields = np.asarray(fields)
        if not len(fields):
            raise ValueError("At least one projection field is required")
        masks = None if masks is None else np.broadcast_to(masks, fields.shape[:3])
        if masks is not None and any(not np.array_equal(mask, masks[0]) for mask in masks[1:]):
            return super().transform_batch(fields, masks, domain_spec)
        sample, was_2d, weights, _ = self._prepare_zernike_inputs(
            fields[0], None if masks is None else masks[0], domain_spec
        )
        values = fields[..., None] if fields.ndim == 3 else fields
        # One bounded RHS per (sample, component), with the same design matrix.
        rhs = values.transpose(1, 2, 0, 3).reshape(*values.shape[1:3], -1)
        coefficients = self._solve_weighted_least_squares(self._get_basis(domain_spec), rhs, weights)
        coefficients = coefficients.reshape(len(values), values.shape[-1], -1)
        self._projection_layout(sample, was_2d, coefficients[0], domain_spec)
        return coefficients

    def _prepare_zernike_inputs(
        self,
        field: np.ndarray,
        mask: np.ndarray | None,
        domain_spec: DomainSpec,
    ) -> tuple[np.ndarray, bool, np.ndarray, np.ndarray | None]:
        from ml_platform_mode_decomp.geometry import validate_decomposer_compatibility

        validate_decomposer_compatibility(domain_spec, self._cfg)
        self._bind_geometry(domain_spec)
        field_3d, was_2d = _normalize_field(field)
        shape = field_3d.shape[:2] if was_2d else field_3d.shape
        if self._coeff_meta is not None and self._coeff_meta.get("field_shape") != list(shape):
            raise ValueError("Field layout differs from the established decomposer layout")
        if field_3d.shape[:2] != domain_spec.grid_shape:
            raise ValueError(f"field shape {field_3d.shape[:2]} does not match domain {domain_spec.grid_shape}")
        field_mask = _normalize_mask(mask, field_3d.shape[:2])
        combined_mask = _combine_masks(field_mask, domain_spec.mask)
        weights = domain_spec.weights
        if weights is None:
            weights = np.ones(field_3d.shape[:2], dtype=np.float64)
        else:
            weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != field_3d.shape[:2]:
            raise ValueError(f"weights shape {weights.shape} does not match {field_3d.shape[:2]}")
        if combined_mask is not None:
            # REVIEW: masked points are ignored via zeroed weights to avoid silent fill.
            weights = np.where(combined_mask, weights, 0.0)
        return field_3d, was_2d, weights, combined_mask

    @staticmethod
    def _solve_weighted_least_squares(
        basis: np.ndarray,
        field_3d: np.ndarray,
        weights: np.ndarray,
    ) -> np.ndarray:
        weights_flat = weights.reshape(-1)
        valid = weights_flat > 0
        if not np.any(valid):
            raise ValueError("Projection weights are empty after masking")
        basis_flat = basis.reshape(basis.shape[0], -1).T
        design = basis_flat[valid]
        if design.shape[0] < design.shape[1]:
            raise ValueError("Projection basis has more modes than valid samples")
        field_flat = field_3d.reshape(-1, field_3d.shape[-1])
        if not np.isfinite(field_flat[valid]).all():
            raise ValueError("field has non-finite values within valid mask")
        sqrt_w = np.sqrt(weights_flat[valid])
        design_w = design * sqrt_w[:, None]
        field_w = field_flat[valid] * sqrt_w[:, None]
        coeffs, _, rank, _ = np.linalg.lstsq(design_w, field_w, rcond=None)
        if rank < coeffs.shape[0]:
            raise ValueError("Projection basis is rank-deficient; reduce modes or check mask coverage")
        return coeffs.T

    @staticmethod
    def _reconstruct_from_basis(coeff_tensor: np.ndarray, basis: np.ndarray) -> np.ndarray:
        field_channels = []
        for idx in range(coeff_tensor.shape[0]):
            field_c = np.tensordot(coeff_tensor[idx], basis, axes=(0, 0))
            field_channels.append(field_c)
        return np.stack(field_channels, axis=-1)


# Keep the import and serialized class path used by earlier versions.
ZernikeFamilyBase = WeightedBasisDecomposerBase


class EigenBasisDecomposerBase(BaseDecomposer[np.ndarray]):
    """Baseclass for eigenbasis decomposers with sign-fix handling."""

    def __init__(self, *, cfg: Mapping[str, Any]) -> None:
        super().__init__(cfg=cfg)
        self._basis: np.ndarray | None = None
        self._eigenvalues: np.ndarray | None = None
        self._sign_fix_rule = "max_abs_component"

    def _fix_eigenvector_signs(self, eigvecs: np.ndarray) -> np.ndarray:
        fixed = np.asarray(eigvecs, dtype=np.float64).copy()
        if fixed.ndim != 2:
            raise ValueError("eigenvectors must be 2D (nodes x modes)")
        for idx in range(fixed.shape[1]):
            vec = fixed[:, idx]
            if vec.size == 0:
                continue
            anchor = int(np.argmax(np.abs(vec)))
            if vec[anchor] < 0:
                fixed[:, idx] = -vec
        return fixed

    def _set_basis(self, eigvecs: np.ndarray, eigvals: np.ndarray) -> None:
        self._basis = self._fix_eigenvector_signs(eigvecs)
        self._eigenvalues = np.asarray(eigvals, dtype=np.float64)

    def _eigen_meta(self, *, projection: str, mode_order: str = "ascending_eigenvalue") -> Dict[str, Any]:
        if self._eigenvalues is None:
            raise ValueError("eigenvalues are not available")
        return {
            "mode_order": mode_order,
            "eigenvalues": [float(val) for val in self._eigenvalues],
            "sign_fix_rule": self._sign_fix_rule,
            "projection": str(projection),
        }


class ChannelwiseAdapter(BaseDecomposer[np.ndarray]):
    """Apply a scalar decomposer channel-by-channel for vector fields."""

    def __init__(
        self,
        *,
        cfg: Mapping[str, Any],
        decomposer_factory: Callable[..., BaseDecomposer[np.ndarray]],
        label: str | None = None,
    ) -> None:
        super().__init__(cfg=cfg)
        self.name = label or "channelwise"
        self._factory = decomposer_factory
        self._decomposers: list[BaseDecomposer] = []
        self._channels: int | None = None
        self._coeff_shape: tuple[int, ...] | None = None
        self._field_ndim: int | None = None
        self._channel_shapes: list[tuple[int, ...]] | None = None

    def _ensure_decomposers(self, channels: int) -> None:
        if self._decomposers:
            if len(self._decomposers) != channels:
                raise ValueError("channelwise adapter channel count mismatch")
            return
        if self._factory is None:
            raise ValueError(
                "channelwise adapter is missing decomposer_factory (likely loaded from state); refit is required"
            )
        self._decomposers = [self._factory(cfg=dict(self._cfg)) for _ in range(channels)]
        self._channels = channels

    def fit(self, dataset: Any | None = None, domain_spec: DomainSpec | None = None) -> "ChannelwiseAdapter":
        if dataset is None:
            return self
        if domain_spec is None:
            raise ValueError("channelwise adapter requires a domain")
        if not len(dataset):
            raise ValueError("At least one training field is required")
        field = np.asarray(dataset[0].field)
        if field.ndim != 3 or tuple(field.shape[:2]) != domain_spec.grid_shape:
            raise ValueError(f"Training field does not match domain {domain_spec.grid_shape}: {field.shape}")
        self._ensure_decomposers(field.shape[-1])
        for index, decomposer in enumerate(self._decomposers):
            channel = (
                FieldBatch(dataset.fields[..., index], dataset.masks)
                if isinstance(dataset, FieldBatch)
                else _ChannelFields(dataset, index, field.shape[-1])
            )
            try:
                decomposer.fit(channel, domain_spec=domain_spec)
            except ValueError as exc:
                raise ValueError(f"{self.name} component {index}: {exc}") from exc
        return self

    def transform(
        self,
        field: np.ndarray,
        mask: np.ndarray | None,
        domain_spec: DomainSpec,
    ) -> np.ndarray:
        field_arr = np.asarray(field)
        if field_arr.ndim == 2:
            field_arr = field_arr[..., None]
            was_2d = True
        elif field_arr.ndim == 3:
            was_2d = False
        else:
            raise ValueError(f"channelwise adapter expects 2D or 3D field, got {field_arr.shape}")
        channels = int(field_arr.shape[-1])
        self._ensure_decomposers(channels)
        coeffs: list[np.ndarray] = []
        channel_metas = []
        base_layout: str | None = None
        channel_shapes: list[tuple[int, ...]] = []
        for idx, decomposer in enumerate(self._decomposers):
            raw_coeff = decomposer.transform(field_arr[..., idx], mask=mask, domain_spec=domain_spec)
            meta = decomposer.coeff_meta()
            channel_metas.append(meta)
            raw_arr = np.asarray(raw_coeff)
            coeff_shape = meta.get("coeff_shape")
            if isinstance(coeff_shape, (list, tuple)) and coeff_shape:
                expected_shape = tuple(int(x) for x in coeff_shape)
                if raw_arr.shape != expected_shape:
                    order = meta.get("flatten_order", "C")
                    if order not in ("C", "F"):
                        raise ValueError("flatten_order must be C or F")
                    raw_arr = raw_arr.reshape(expected_shape, order="F" if order == "F" else "C")
            layout = str(meta.get("coeff_layout", "")).strip()
            if (
                int(meta.get("channels", 1)) == 1
                and raw_arr.ndim >= 1
                and raw_arr.shape[0] == 1
                and isinstance(coeff_shape, (list, tuple))
                and coeff_shape
                and int(coeff_shape[0]) == 1
            ):
                raw_arr = raw_arr[0]
                if layout.startswith("C"):
                    layout = layout[1:]
            if base_layout is None:
                base_layout = layout
            elif base_layout != layout:
                raise ValueError("channelwise adapter requires consistent coeff_layout across channels")
            coeffs.append(raw_arr)
            channel_shapes.append(raw_arr.shape)
        variable = getattr(self, "_variable_blocks", False)
        coeff_tensor = np.concatenate([value.reshape(-1) for value in coeffs]) if variable else np.stack(coeffs, axis=0)
        self._coeff_shape = coeff_tensor.shape
        self._field_ndim = 2 if was_2d else 3
        self._channel_shapes = channel_shapes
        if base_layout is None:
            base_layout = "K"
        meta = self._coeff_meta_base(
            field_shape=field_arr.shape[:2] if was_2d else field_arr.shape,
            field_ndim=self._field_ndim,
            field_layout="HW" if was_2d else "HWC",
            channels=channels,
            coeff_shape=coeff_tensor.shape,
            coeff_layout=f"C{base_layout}",
            complex_format="channelwise",
        )
        meta.update({"base_method": channel_metas[0]["method"] if channel_metas else None})
        if variable:
            from .pod_utils import component_blocks_meta

            meta = component_blocks_meta(meta, [value.size for value in coeffs])
        self._coeff_meta = meta
        return coeff_tensor

    def inverse_transform(self, coeff: np.ndarray, domain_spec: DomainSpec) -> np.ndarray:
        if self._coeff_shape is None or self._field_ndim is None or self._channel_shapes is None:
            raise ValueError("transform must be called before inverse_transform")
        coeff_arr = np.asarray(coeff)
        expected = int(np.prod(self._coeff_shape))
        if coeff_arr.shape != self._coeff_shape:
            coeff_flat = coeff_arr.reshape(-1)
            if coeff_flat.size > expected:
                raise ValueError(f"coeff size {coeff_flat.size} exceeds expected {expected}")
            if coeff_flat.size < expected:
                raise ValueError("Coefficient size is smaller than the fitted layout")
            coeff_arr = coeff_flat.reshape(self._coeff_shape, order="C")
        fields = []
        offsets = np.cumsum([0, *(int(np.prod(shape)) for shape in self._channel_shapes)])
        for idx, (decomposer, shape) in enumerate(zip(self._decomposers, self._channel_shapes, strict=True)):
            chunk = (
                coeff_arr[offsets[idx] : offsets[idx + 1]]
                if getattr(self, "_variable_blocks", False)
                else coeff_arr[idx]
            )
            if shape and chunk.shape != shape:
                chunk = chunk.reshape(shape, order="C")
            fields.append(decomposer.inverse_transform(chunk, domain_spec=domain_spec))
        field_hat = np.stack(fields, axis=-1)
        if self._field_ndim == 2 and field_hat.shape[-1] == 1:
            return field_hat[..., 0]
        return field_hat


__all__ = [
    "BaseDecomposer",
    "GridDecomposerBase",
    "ZernikeFamilyBase",
    "EigenBasisDecomposerBase",
    "ChannelwiseAdapter",
]
