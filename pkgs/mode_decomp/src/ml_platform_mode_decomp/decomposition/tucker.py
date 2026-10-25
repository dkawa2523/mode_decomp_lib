"""Shared Tucker factors for steady rectangular fields, without compressing samples."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ml_platform_mode_decomp.config import constructor_config
from ml_platform_mode_decomp.geometry import DomainSpec

from .base import BaseDecomposer
from .learned_support import complete_training, observed_field, stable_ridge


def _mode_product(values: np.ndarray, matrix: np.ndarray, axis: int) -> np.ndarray:
    """Multiply one tensor axis, retaining the positions of all other axes."""
    return np.moveaxis(np.tensordot(matrix, values, axes=(1, axis)), 0, axis)


def _leading_vectors(values: np.ndarray, axis: int, rank: int) -> np.ndarray:
    unfolding = np.moveaxis(values, axis, 0).reshape(values.shape[axis], -1)
    left, _, _ = np.linalg.svd(unfolding, full_matrices=False)
    vectors = np.array(left[:, :rank], copy=True)
    if vectors.shape[1] < rank:
        # Reduced HOOI unfoldings can have fewer columns than the requested
        # axis rank. Complete only the required columns, never a dense d-by-d Q.
        completed = np.empty((len(vectors), rank), dtype=np.float64)
        count = vectors.shape[1]
        completed[:, :count] = vectors
        while count < rank:
            coordinate = int(np.argmin(np.sum(completed[:, :count] ** 2, axis=1)))
            candidate = np.zeros(len(vectors))
            candidate[coordinate] = 1.0
            for _ in range(2):
                candidate -= completed[:, :count] @ (completed[:, :count].T @ candidate)
            norm = np.linalg.norm(candidate)
            if norm <= 32 * np.finfo(float).eps:
                raise np.linalg.LinAlgError("Tucker could not complete an orthonormal factor")
            completed[:, count] = candidate / norm
            count += 1
        vectors = completed
    # Fix the otherwise arbitrary singular-vector sign for saved coordinates.
    pivots = np.argmax(np.abs(vectors), axis=0)
    signs = np.sign(vectors[pivots, np.arange(rank)])
    vectors *= signs
    return vectors


def _project(values: np.ndarray, factors: tuple[np.ndarray, ...]) -> np.ndarray:
    for axis, factor in enumerate(factors, start=values.ndim - 3):
        values = _mode_product(values, factor.T, axis)
    return values


def _nonnegative(value: Any, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"tucker {label} must be a finite nonnegative number")
    result = float(value)
    if not np.isfinite(result) or result < 0:
        raise ValueError(f"tucker {label} must be a finite nonnegative number")
    return result


class TuckerDecomposer(BaseDecomposer[np.ndarray]):
    """HOSVD/HOOI on y, x and component axes with one core per sample.

    Factors are orthonormal in whitened coordinates. Training must be complete;
    a partially observed query uses a stable weighted least-squares core solve.
    """

    synthesis_kind = "affine"

    def __init__(
        self,
        *,
        cfg: Mapping[str, Any] | None = None,
        ranks: Any = None,
        center: Any = True,
        inner_product: Any = "euclidean",
        n_iter: Any = 20,
        tol: Any = 1e-8,
        ridge_alpha: Any = 0.0,
    ) -> None:
        cfg = constructor_config(cfg, locals(), name="tucker")
        super().__init__(cfg=cfg)
        self.name = "tucker"
        ranks = cfg["ranks"]
        if ranks is not None:
            if (
                not isinstance(ranks, (list, tuple, np.ndarray))
                or np.asarray(ranks).ndim != 1
                or len(ranks) != 3
                or any(
                    isinstance(rank, (bool, np.bool_)) or not isinstance(rank, (int, np.integer)) or rank < 1
                    for rank in ranks
                )
            ):
                raise ValueError("tucker ranks must contain three positive integers in y/x/component order")
            ranks = tuple(int(rank) for rank in ranks)
        self._requested_ranks = ranks
        if not isinstance(cfg["center"], (bool, np.bool_)):
            raise ValueError("tucker center must be boolean")
        self._center = bool(cfg["center"])
        self._inner_product = cfg["inner_product"]
        if self._inner_product not in {"euclidean", "domain_weights"}:
            raise ValueError("tucker inner_product must be euclidean or domain_weights")
        iterations = cfg["n_iter"]
        if isinstance(iterations, (bool, np.bool_)) or not isinstance(iterations, (int, np.integer)) or iterations < 0:
            raise ValueError("tucker n_iter must be a nonnegative integer (zero selects HOSVD only)")
        self._n_iter = int(iterations)
        self._tol = _nonnegative(cfg["tol"], "tol")
        self._ridge_alpha = _nonnegative(cfg["ridge_alpha"], "ridge_alpha")
        self._fitted = False

    @staticmethod
    def _rectangle(domain_spec: DomainSpec) -> None:
        if domain_spec.name != "rectangle":
            raise ValueError("tucker requires fixed rectangular geometry; mesh and irregular domains are unsupported")
        shape = domain_spec.grid_shape
        if domain_spec.mask is not None and not np.asarray(domain_spec.mask).all():
            raise ValueError("tucker requires a complete rectangle without an irregular domain mask")
        for name, axis in (("y", 0), ("x", 1)):
            coordinates = np.asarray(domain_spec.coords.get(name))
            if coordinates.shape != shape or coordinates.dtype.kind not in "fiu" or not np.isfinite(coordinates).all():
                raise ValueError("tucker requires finite rectilinear x/y coordinate grids")
            line = coordinates[:, 0] if axis == 0 else coordinates[0, :]
            expected = line[:, None] if axis == 0 else line[None, :]
            if not np.array_equal(coordinates, np.broadcast_to(expected, shape)) or np.any(np.diff(line) <= 0):
                raise ValueError("tucker requires strictly increasing rectilinear x/y coordinates")

    @staticmethod
    def _separable_weights(weights: np.ndarray) -> None:
        if not np.isfinite(weights).all() or np.any(weights <= 0):
            raise ValueError("tucker domain_weights requires strictly positive finite spatial weights")
        # Log ratios avoid overflow/underflow when physical units make weights
        # very large or small. No approximate nearest separable measure is used.
        logs = np.log(weights)
        residual = logs - logs[:, :1] - logs[:1, :] + logs[0, 0]
        if not np.all(np.abs(residual) <= 1e-10):
            raise ValueError("tucker domain_weights requires separable y-by-x spatial weights")

    def fit(self, dataset: Any | None = None, domain_spec: DomainSpec | None = None) -> "TuckerDecomposer":
        self._fitted = False
        self._coeff_meta = None
        if dataset is None or domain_spec is None or not len(dataset):
            raise ValueError("tucker fit requires nonempty training data and domain_spec")
        self._rectangle(domain_spec)
        self._field_ndim = np.asarray(dataset[0].field).ndim
        values, self._domain, self._weights = complete_training(
            dataset, domain_spec, self._inner_product, method=self.name
        )
        self._field_shape = tuple(values.shape[1:])
        self._ranks = self._requested_ranks or tuple(min(8, size) for size in self._field_shape)
        if any(rank > size for rank, size in zip(self._ranks, self._field_shape, strict=True)):
            raise ValueError("tucker ranks cannot exceed the y/x/component field dimensions")
        if self._inner_product == "domain_weights":
            self._separable_weights(self._weights)
        self._sqrt_weights = np.sqrt(self._weights)
        with np.errstate(over="ignore", invalid="ignore"):
            self._mean = values.mean(axis=0) if self._center else np.zeros(self._field_shape)
            values -= self._mean
            values *= self._sqrt_weights[None, ..., None]
        if not np.isfinite(values).all():
            raise ValueError("tucker centered/weighted training values must be finite; rescale the fields")
        # Factors and relative HOOI convergence are invariant to global scale.
        # Normalize the owned work tensor so norm reductions do not overflow.
        magnitude = float(np.max(np.abs(values)))
        if magnitude:
            values /= magnitude
        factors = tuple(_leading_vectors(values, axis, rank) for axis, rank in enumerate(self._ranks, start=1))
        total_norm = float(np.linalg.norm(values))
        core_norm = float(np.linalg.norm(_project(values, factors)))
        self._iterations = 0
        self._converged = total_norm == 0
        if total_norm:
            for iteration in range(self._n_iter):
                updated = list(factors)
                for axis, rank in enumerate(self._ranks, start=1):
                    reduced = values
                    for other in range(1, 4):
                        if other != axis:
                            reduced = _mode_product(reduced, updated[other - 1].T, other)
                    updated[axis - 1] = _leading_vectors(reduced, axis, rank)
                factors = tuple(updated)
                next_norm = float(np.linalg.norm(_project(values, factors)))
                self._iterations = iteration + 1
                self._converged = abs(next_norm - core_norm) <= self._tol * max(next_norm, core_norm)
                core_norm = next_norm
                if self._converged:
                    break
        self._factors = factors
        if not all(np.isfinite(factor).all() for factor in self._factors):
            raise ValueError("tucker factors must be finite")
        self._bind_geometry(domain_spec, fitting=True)
        shape = self._field_shape[:2] if self._field_ndim == 2 else self._field_shape
        self._coeff_meta = self._coeff_meta_base(
            field_shape=shape,
            field_ndim=self._field_ndim,
            field_layout="HW" if self._field_ndim == 2 else "HWC",
            channels=self._field_shape[-1],
            coeff_shape=self._ranks,
            coeff_layout="RYRXRC",
            complex_format="real",
        )
        self._coeff_meta.update(
            coeff_format="tucker_core_v1",
            core_axes=["y", "x", "component"],
            ranks=list(self._ranks),
            ranks_config=None if self._requested_ranks is None else list(self._requested_ranks),
            sample_axis_compressed=False,
            mean_centered=self._center,
            inner_product=self._inner_product,
            projection="orthogonal_complete_or_weighted_observed_least_squares",
            algorithm="hosvd" if self._n_iter == 0 else "hooi",
            training_samples=len(values),
            diagnostics={
                "iterations": self._iterations,
                "converged": self._converged,
                "relative_training_residual": float(np.sqrt(max(0.0, 1 - (core_norm / total_norm) ** 2)))
                if total_norm
                else 0.0,
            },
        )
        self._fitted = True
        return self

    def _require_fitted(self, domain_spec: DomainSpec) -> None:
        if not self._fitted:
            raise ValueError("tucker fit must be called before transform or inverse_transform")
        self._bind_geometry(domain_spec)

    def transform(self, field: np.ndarray, mask: np.ndarray | None, domain_spec: DomainSpec) -> np.ndarray:
        self._require_fitted(domain_spec)
        values, was_2d, observed = observed_field(
            field, mask, self._domain, self._weights, method=self.name, field_shape=self._field_shape
        )
        if (2 if was_2d else 3) != self._field_ndim:
            raise ValueError("tucker query field layout differs from its fitted HW/HWC layout")
        if observed.all():
            with np.errstate(over="ignore", invalid="ignore"):
                weighted = (values - self._mean) * self._sqrt_weights[..., None]
            if not np.isfinite(weighted).all():
                raise ValueError("tucker weighted query must be finite; rescale the fields")
            core = _project(weighted, self._factors)
        else:
            rows, columns = np.nonzero(observed)
            factor_y, factor_x, factor_c = self._factors
            design = np.einsum("pa,pb,cd->pcabd", factor_y[rows], factor_x[columns], factor_c, optimize=True).reshape(
                len(rows) * self._field_shape[-1], -1
            )
            with np.errstate(over="ignore", invalid="ignore"):
                targets = ((values[observed] - self._mean[observed]) * self._sqrt_weights[observed][:, None]).reshape(
                    -1
                )
            core = stable_ridge(design, targets, self._ridge_alpha, method=self.name).reshape(self._ranks)
        if not np.isfinite(core).all():
            raise ValueError("tucker query core must be finite")
        return core

    def transform_batch(self, fields: np.ndarray, masks: np.ndarray | None, domain_spec: DomainSpec):
        self._require_fitted(domain_spec)
        values = np.asarray(fields)
        if values.ndim not in (3, 4) or not len(values):
            raise ValueError("tucker requires a nonempty field batch")
        if masks is None:
            return np.stack([self.transform(field, None, domain_spec) for field in values])
        mask_array = np.asarray(masks)
        if mask_array.shape not in (self._domain.shape, (len(values), *self._domain.shape)):
            raise ValueError("tucker batch masks must be spatial or sample-by-spatial")
        observed = np.broadcast_to(mask_array, (len(values), *self._domain.shape))
        return np.stack(
            [self.transform(field, mask, domain_spec) for field, mask in zip(values, observed, strict=True)]
        )

    def inverse_transform(self, coeff: np.ndarray, domain_spec: DomainSpec) -> np.ndarray:
        self._require_fitted(domain_spec)
        coefficients = np.asarray(coeff)
        if coefficients.dtype.kind not in "fiu" or not np.isfinite(coefficients).all():
            raise ValueError("tucker core coefficients must be finite real numbers")
        core = self._reshape_coeff(coefficients, self._ranks, name=self.name)
        with np.errstate(over="ignore", invalid="ignore"):
            weighted = np.einsum("abc,ya,xb,dc->yxd", core, *self._factors, optimize=True)
            restored = weighted / self._sqrt_weights[..., None] + self._mean
        if not np.isfinite(restored).all():
            raise ValueError("tucker synthesized field must be finite; rescale the coefficients")
        return restored[..., 0] if self._field_ndim == 2 else restored


__all__ = ["TuckerDecomposer"]
