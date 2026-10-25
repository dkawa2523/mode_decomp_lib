"""A fixed POD encoder with an orthogonal quadratic field decoder."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from scipy.optimize import least_squares

from ml_platform_mode_decomp.config import constructor_config
from ml_platform_mode_decomp.geometry import DomainSpec

from .base import BaseDecomposer
from .learned_support import complete_training, observed_field, stable_ridge
from .pod_utils import singular_value_rank


def _nonnegative(value, name):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not np.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"quadratic_manifold {name} must be finite and nonnegative")
    return float(value)


class QuadraticManifoldDecomposer(BaseDecomposer[np.ndarray]):
    """Learn joint field coordinates and a residual quadratic correction.

    Complete fields use the fixed weighted linear projection. The correction
    is orthogonal to that projection, so encoding any decoded coordinate
    recovers that coordinate. Partial observations use local nonlinear least
    squares; local Jacobian rank does not establish global identifiability.
    """

    synthesis_kind = "nonlinear"

    def __init__(
        self,
        *,
        cfg: Mapping[str, Any] | None = None,
        n_modes: Any = None,
        center: Any = True,
        inner_product: Any = "euclidean",
        quadratic_alpha: Any = 1e-6,
        ridge_alpha: Any = 0.0,
        max_nfev: Any = 200,
        tol: Any = 1e-8,
    ) -> None:
        cfg = constructor_config(cfg, locals(), name="quadratic_manifold")
        super().__init__(cfg=cfg)
        self.name = "quadratic_manifold"
        count = cfg["n_modes"]
        if count is not None and (
            isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer)) or count < 1
        ):
            raise ValueError("quadratic_manifold n_modes must be a positive integer or null")
        if not isinstance(cfg["center"], (bool, np.bool_)):
            raise ValueError("quadratic_manifold center must be boolean")
        if not isinstance(cfg["inner_product"], str) or cfg["inner_product"] not in {"euclidean", "domain_weights"}:
            raise ValueError("quadratic_manifold inner_product must be euclidean or domain_weights")
        limit = cfg["max_nfev"]
        if isinstance(limit, (bool, np.bool_)) or not isinstance(limit, (int, np.integer)) or limit < 1:
            raise ValueError("quadratic_manifold max_nfev must be a positive integer")
        self._tol = _nonnegative(cfg["tol"], "tol")
        if self._tol <= np.finfo(float).eps:
            raise ValueError("quadratic_manifold tol must exceed float64 machine precision")
        self._n_modes = None if count is None else int(count)
        self._center = bool(cfg["center"])
        self._inner_product = cfg["inner_product"]
        self._quadratic_alpha = _nonnegative(cfg["quadratic_alpha"], "quadratic_alpha")
        self._ridge_alpha = _nonnegative(cfg["ridge_alpha"], "ridge_alpha")
        self._max_nfev = int(limit)
        self._mean: np.ndarray | None = None
        self._field_ndim: int | None = None

    def _quadratic_features(self, coordinates):
        with np.errstate(over="ignore", invalid="ignore"):
            values = coordinates[..., self._pairs[0]] * coordinates[..., self._pairs[1]]
        if not np.isfinite(values).all():
            raise ValueError("quadratic_manifold quadratic features overflow; rescale the input fields")
        return values

    def _quadratic_jacobian(self, coordinates):
        result = np.zeros((len(self._pairs[0]), self._rank))
        rows = np.arange(len(result))
        np.add.at(result, (rows, self._pairs[0]), coordinates[self._pairs[1]])
        np.add.at(result, (rows, self._pairs[1]), coordinates[self._pairs[0]])
        return result

    def fit(self, dataset: Any | None = None, domain_spec: DomainSpec | None = None):
        self._mean = None
        self._coeff_meta = None
        self._field_ndim = None
        fields, domain, weights = complete_training(dataset, domain_spec, self._inner_product, method=self.name)
        assert domain_spec is not None
        self._bind_geometry(domain_spec, fitting=True)
        self._field_shape = tuple(fields.shape[1:])
        self._domain, self._weights = domain, weights
        channels = fields.shape[-1]
        sample_count = len(fields)
        self._indices = np.flatnonzero(np.repeat(domain[..., None], channels, axis=-1).reshape(-1))
        self._sqrt_weights = np.sqrt(np.repeat(weights[..., None], channels, axis=-1).reshape(-1)[self._indices])
        snapshots = fields.reshape(len(fields), -1)[:, self._indices]
        del fields
        # Differences from a reference preserve exact rank zero for constant
        # snapshots, including values whose repeated sum rounds differently.
        with np.errstate(over="ignore", invalid="ignore"):
            mean = (
                snapshots[0] + np.mean(snapshots - snapshots[0], axis=0)
                if self._center
                else np.zeros(len(self._indices))
            )
            snapshots -= mean
            snapshots *= self._sqrt_weights
        centered = snapshots
        if not np.isfinite(centered).all():
            raise ValueError("quadratic_manifold centering or weighting overflow; rescale training fields")
        u, singular, vh = np.linalg.svd(centered, full_matrices=False)
        del u
        available = singular_value_rank(singular, centered.shape)
        count = min(16, available) if self._n_modes is None else self._n_modes
        if available == 0:
            count = 0
        elif count > available:
            raise ValueError("quadratic_manifold n_modes exceeds the numerical training rank")
        basis = vh[:count].T.copy()
        del vh
        coordinates = centered @ basis
        self._rank = count
        self._pairs = np.triu_indices(count)
        quadratic = self._quadratic_features(coordinates)
        self._quadratic_mean = quadratic.mean(axis=0) if self._center else np.zeros(quadratic.shape[1])
        quadratic -= self._quadratic_mean
        centered -= coordinates @ basis.T
        residual = centered
        linear_error = float(np.einsum("ij,ij->", residual, residual))
        if count:
            correction = stable_ridge(
                quadratic, residual, self._quadratic_alpha, method="quadratic_manifold training correction"
            ).T
            # Remove numerical projection leakage as well as imposing the
            # scientific constraint V.T W H = 0 on the decoder correction.
            correction -= basis @ (basis.T @ correction)
        else:
            correction = np.empty((len(self._indices), 0))
        residual -= quadratic @ correction.T
        quadratic_error = float(np.einsum("ij,ij->", residual, residual))
        physical_basis = basis / self._sqrt_weights[:, None]
        physical_correction = correction / self._sqrt_weights[:, None]
        encoder = basis * self._sqrt_weights[:, None]
        if (
            not all(np.isfinite(value).all() for value in (mean, physical_basis, physical_correction, encoder))
            or not np.isfinite([linear_error, quadratic_error]).all()
        ):
            raise ValueError("quadratic_manifold fitted state is not finite; rescale training fields or weights")
        self._basis = physical_basis
        self._correction = physical_correction
        self._encoder = encoder
        self._diagnostics = {
            "effective_rank": available,
            "selected_modes": count,
            "quadratic_features": quadratic.shape[1],
            "training_samples": sample_count,
            "centered": self._center,
            "energy_metric": self._inner_product,
            "linear_training_squared_error": linear_error,
            "quadratic_training_squared_error": quadratic_error,
        }
        self._mean = mean.copy()
        return self

    def _require_fitted(self, domain_spec):
        if self._mean is None:
            raise ValueError("quadratic_manifold fit must be called first")
        if domain_spec is None:
            raise ValueError("quadratic_manifold requires a domain")
        self._bind_geometry(domain_spec)

    def _set_metadata(self, was_2d):
        shape = self._field_shape[:2] if was_2d else self._field_shape
        meta = self._coeff_meta_base(
            field_shape=shape,
            field_ndim=2 if was_2d else 3,
            field_layout="HW" if was_2d else "HWC",
            channels=self._field_shape[-1],
            coeff_shape=(self._rank,),
            coeff_layout="D",
            complex_format="real",
        )
        if self._coeff_meta is None:
            meta.update(
                projection="joint_weighted_linear_pod",
                partial_observation_encoder="local_nonlinear_least_squares",
                quadratic_products="unique_i_le_j",
                quadratic_feature_centering="training_mean" if self._center else "none",
                inner_product=self._inner_product,
                diagnostics=dict(self._diagnostics),
            )
            self._coeff_meta = meta
            self._field_ndim = 2 if was_2d else 3

    def _partial_coordinates(self, vector, observed):
        assert self._mean is not None
        selected = np.repeat(observed[..., None], self._field_shape[-1], axis=-1).reshape(-1)[self._indices]
        scale = self._sqrt_weights[selected]
        linear = self._basis[selected] * scale[:, None]
        correction = self._correction[selected] * scale[:, None]
        target = (vector[selected] - self._mean[selected]) * scale
        if self._rank == 0:
            return np.empty(0)
        if len(target) < self._rank and self._ridge_alpha == 0:
            raise ValueError(
                "quadratic_manifold observed coordinates are not identifiable; add observations or ridge_alpha"
            )
        initial_target = target + correction @ self._quadratic_mean
        if self._ridge_alpha:
            initial = stable_ridge(linear, initial_target, self._ridge_alpha, method=self.name)
        else:
            initial = np.linalg.lstsq(linear, initial_target, rcond=None)[0]
        prior = np.sqrt(self._ridge_alpha)

        def residual(coordinates):
            result = (
                linear @ coordinates
                + correction @ (self._quadratic_features(coordinates) - self._quadratic_mean)
                - target
            )
            return np.concatenate((result, prior * coordinates)) if prior else result

        def jacobian(coordinates):
            result = linear + correction @ self._quadratic_jacobian(coordinates)
            return np.vstack((result, prior * np.eye(self._rank))) if prior else result

        solution = least_squares(
            residual,
            initial,
            jac=jacobian,
            max_nfev=self._max_nfev,
            ftol=self._tol,
            xtol=self._tol,
            gtol=self._tol,
        )
        if not solution.success or not np.isfinite(solution.x).all() or not np.isfinite(solution.fun).all():
            raise RuntimeError(
                f"quadratic_manifold partial-observation optimization did not converge: {solution.message}"
            )
        if self._ridge_alpha == 0 and np.linalg.matrix_rank(jacobian(solution.x)) < self._rank:
            raise ValueError("quadratic_manifold observed Jacobian is rank-deficient; add observations or ridge_alpha")
        return solution.x

    def transform(self, field: np.ndarray, mask: np.ndarray | None, domain_spec: DomainSpec):
        self._require_fitted(domain_spec)
        assert self._mean is not None
        values, was_2d, observed = observed_field(
            field, mask, self._domain, self._weights, method=self.name, field_shape=self._field_shape
        )
        self._set_metadata(was_2d)
        vector = np.asarray(values, dtype=np.float64).reshape(-1)[self._indices]
        if np.array_equal(observed, self._domain):
            with np.errstate(over="ignore", invalid="ignore"):
                coordinates = (vector - self._mean) @ self._encoder
        else:
            coordinates = self._partial_coordinates(vector, observed)
        if not np.isfinite(coordinates).all():
            raise ValueError("quadratic_manifold encoded coordinates are not finite; rescale query fields")
        return coordinates

    def transform_batch(self, fields: np.ndarray, masks: np.ndarray | None, domain_spec: DomainSpec):
        self._require_fitted(domain_spec)
        values = np.asarray(fields)
        if values.ndim not in (3, 4) or not len(values):
            raise ValueError("quadratic_manifold requires a nonempty field batch")
        if masks is None:
            return np.stack([self.transform(field, None, domain_spec) for field in values])
        mask_array = np.asarray(masks)
        if mask_array.shape not in (self._domain.shape, (len(values), *self._domain.shape)):
            raise ValueError("quadratic_manifold batch masks must be spatial or sample-by-spatial")
        observed = np.broadcast_to(mask_array, (len(values), *self._domain.shape))
        return np.stack(
            [self.transform(field, mask, domain_spec) for field, mask in zip(values, observed, strict=True)]
        )

    def inverse_transform(self, coeff: np.ndarray, domain_spec: DomainSpec):
        self._require_fitted(domain_spec)
        assert self._mean is not None
        coefficients = np.asarray(coeff)
        if coefficients.dtype.kind not in "fiu" or not np.isfinite(coefficients).all():
            raise ValueError("quadratic_manifold coefficients must be finite real values")
        coordinates = self._reshape_coeff(coefficients, (self._rank,), name=self.name)
        with np.errstate(over="ignore", invalid="ignore"):
            values = (
                self._mean
                + self._basis @ coordinates
                + self._correction @ (self._quadratic_features(coordinates) - self._quadratic_mean)
            )
        if not np.isfinite(values).all():
            raise ValueError("quadratic_manifold decoded field is not finite; rescale coefficients")
        output = np.zeros(self._field_shape)
        output.reshape(-1)[self._indices] = values
        return output[..., 0] if self._field_ndim == 2 else output


__all__ = ["QuadraticManifoldDecomposer"]
