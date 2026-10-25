"""Fixed-domain input validation and stable solves for learned field decoders."""

from __future__ import annotations

import numpy as np

from .base import FieldBatch, _normalize_field


def _boolean_mask(values, shape, *, label):
    mask = np.asarray(values)
    if mask.shape != shape or mask.dtype.kind not in "bfiu" or not np.isin(mask, (0, 1)).all():
        raise ValueError(f"{label} must contain boolean or 0/1 values with shape {shape}")
    return mask.astype(bool, copy=False)


def complete_training(dataset, domain_spec, inner_product, *, method):
    """Return owned NHWC float64 fields, fixed domain and spatial weights.

    Mesh fields use the internal (V,1,C) layout. Training observations must
    cover every domain point; unknown values outside that domain become zero.
    Positive domain weights make whitening and physical synthesis invertible.
    """
    if dataset is None or domain_spec is None or not len(dataset):
        raise ValueError(f"{method} needs nonempty training fields and a domain")
    shape = domain_spec.grid_shape
    if isinstance(dataset, FieldBatch):
        fields, masks = np.asarray(dataset.fields), dataset.masks
    else:
        samples = [dataset[index] for index in range(len(dataset))]
        fields = np.stack([np.asarray(sample.field) for sample in samples])
        available = [sample.mask is not None for sample in samples]
        if any(available) and not all(available):
            raise ValueError(f"{method} training masks must be present for all samples or none")
        masks = np.stack([sample.mask for sample in samples]) if all(available) else None
    if fields.ndim == 3:
        fields = fields[:, :, None, :] if domain_spec.name == "mesh" else fields[..., None]
    if fields.ndim != 4 or tuple(fields.shape[1:3]) != shape or fields.shape[-1] < 1:
        raise ValueError(f"{method} training field layout does not match domain {shape}")
    if fields.dtype.kind not in "fiu":
        raise ValueError(f"{method} requires real numeric training fields")
    domain = np.ones(shape, dtype=bool)
    if domain_spec.mask is not None:
        mask = np.asarray(domain_spec.mask)
        if domain_spec.name == "mesh" and mask.shape == (shape[0],):
            mask = mask[:, None]
        domain &= _boolean_mask(mask, shape, label=f"{method} domain mask")
    if not domain.any():
        raise ValueError(f"{method} requires a nonempty fixed domain")
    if masks is not None:
        masks = np.asarray(masks)
        if domain_spec.name == "mesh" and masks.shape in ((shape[0],), (len(fields), shape[0])):
            masks = masks[..., None]
        expected = shape if masks.shape == shape else fields.shape[:3]
        observed = np.broadcast_to(_boolean_mask(masks, expected, label=f"{method} training mask"), fields.shape[:3])
        if not observed[:, domain].all():
            raise ValueError(f"{method} requires complete training observations on the fixed domain")
    if not np.isfinite(fields[:, domain, :]).all():
        raise ValueError(f"{method} requires finite training values on the fixed domain")
    weights = np.ones(shape, dtype=np.float64)
    if inner_product == "domain_weights":
        supplied = domain_spec.integration_weights()
        if supplied is None or np.asarray(supplied).shape not in (shape, (shape[0],) if shape[1] == 1 else shape):
            raise ValueError(f"{method} needs spatial domain integration weights")
        if np.asarray(supplied).dtype.kind not in "fiu":
            raise ValueError(f"{method} requires real domain integration weights")
        weights = np.array(supplied, dtype=np.float64, copy=True).reshape(shape)
        if not np.isfinite(weights[domain]).all() or np.any(weights[domain] <= 0):
            raise ValueError(f"{method} requires strictly positive finite integration weights on the domain")
    elif inner_product != "euclidean":
        raise ValueError(f"{method} inner_product must be euclidean or domain_weights")
    weights[~domain] = 0.0
    clean = np.zeros(fields.shape, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        clean[:, domain, :] = fields[:, domain, :]
    if not np.isfinite(clean).all():
        raise ValueError(f"{method} training values must be finite in float64")
    return clean, domain, weights


def observed_field(field, mask, domain, weights, *, method, field_shape=None):
    """Validate one query without modifying borrowed field, mask or weights."""
    values, was_2d = _normalize_field(field)
    if values.dtype.kind not in "fiu" or values.shape[-1] < 1 or tuple(values.shape[:2]) != tuple(domain.shape):
        raise ValueError(f"{method} query must be a real field matching the fixed domain")
    if field_shape is not None and tuple(values.shape) != tuple(field_shape):
        raise ValueError(f"{method} field layout differs from training")
    observed = np.array(domain, copy=True)
    if mask is not None:
        candidate = np.asarray(mask)
        if domain.shape[1] == 1 and candidate.shape == (domain.shape[0],):
            candidate = candidate[:, None]
        observed &= _boolean_mask(candidate, domain.shape, label=f"{method} observation mask")
    weights = np.asarray(weights)
    if not observed.any() or weights.shape != domain.shape:
        raise ValueError(f"{method} needs observed points on the fixed domain")
    if weights.dtype.kind not in "fiu" or not np.isfinite(weights[observed]).all() or np.any(weights[observed] <= 0):
        raise ValueError(f"{method} needs strictly positive finite weights at observed domain points")
    if not np.isfinite(values[observed]).all():
        raise ValueError(f"{method} requires finite values at observed domain points")
    return values, was_2d, observed


def stable_ridge(design, targets, alpha, *, method):
    """Solve weighted-ready least squares; unregularized queries need full rank."""
    if isinstance(alpha, (bool, np.bool_)) or not isinstance(alpha, (int, float, np.integer, np.floating)):
        raise ValueError(f"{method} ridge_alpha must be finite and nonnegative")
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError(f"{method} ridge_alpha must be finite and nonnegative")
    matrix, rhs = np.asarray(design), np.asarray(targets)
    if matrix.dtype.kind not in "fiu" or rhs.dtype.kind not in "fiu":
        raise ValueError(f"{method} least-squares inputs must be real numeric arrays")
    if matrix.ndim != 2 or rhs.ndim not in (1, 2) or matrix.shape[0] != rhs.shape[0]:
        raise ValueError(f"{method} least-squares design and target shapes disagree")
    if not np.isfinite(matrix).all() or not np.isfinite(rhs).all():
        raise ValueError(f"{method} least-squares inputs must be finite")
    matrix, rhs = np.asarray(matrix, dtype=np.float64), np.asarray(rhs, dtype=np.float64)
    if alpha > 0:
        matrix = np.vstack((matrix, np.sqrt(alpha) * np.eye(matrix.shape[1])))
        rhs = np.concatenate((rhs, np.zeros((matrix.shape[1], *rhs.shape[1:]))))
    coefficients, _, rank, _ = np.linalg.lstsq(matrix, rhs, rcond=None)
    if rank < matrix.shape[1]:
        raise ValueError(f"{method} observed basis is rank-deficient; add observations or regularization")
    return coefficients
