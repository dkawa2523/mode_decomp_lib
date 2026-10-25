"""Dictionary learning decomposer for data-driven sparse coding."""

from __future__ import annotations

from .base import field_arrays

from ml_platform_mode_decomp.config import constructor_config, integer_parameter

from typing import Any, Literal, Mapping, cast

import numpy as np
from sklearn.decomposition import DictionaryLearning

from ml_platform_mode_decomp.geometry import DomainSpec, validate_decomposer_compatibility

from .base import BaseDecomposer, _combine_masks, _normalize_field, _normalize_mask, require_cfg

_MASK_POLICIES = {"error", "ignore_masked_points"}
_FIT_ALGORITHMS = {"lars", "cd"}
_TRANSFORM_ALGORITHMS = {"omp", "lasso_lars", "lasso_cd", "threshold"}
FitAlgorithm = Literal["lars", "cd"]
TransformAlgorithm = Literal["omp", "lasso_lars", "lasso_cd", "threshold"]


class DictLearningDecomposer(BaseDecomposer[np.ndarray]):
    """Dictionary learning decomposer fitted on training snapshots."""

    synthesis_kind = "affine"

    def __init__(
        self,
        *,
        cfg: Mapping[str, Any] | None = None,
        n_components: Any = 64,
        transform_n_nonzero_coefs: Any = None,
        transform_alpha: Any = None,
        seed: Any = None,
        mask_policy: Any = "error",
        alpha: Any = 1.0,
        max_iter: Any = 200,
        tol: Any = 1e-06,
        fit_algorithm: Any = "lars",
        transform_algorithm: Any = "lasso_cd",
        positive_code: Any = False,
        positive_dict: Any = False,
    ) -> None:
        cfg = constructor_config(cfg, locals(), name="dict_learning")
        super().__init__(cfg=cfg)
        self.name = "dict_learning"
        self._mask_policy = str(require_cfg(cfg, "mask_policy", label="decompose"))
        if self._mask_policy not in _MASK_POLICIES:
            raise ValueError(f"decompose.mask_policy must be one of {_MASK_POLICIES}, got {self._mask_policy}")
        n_components = cfg["n_components"]
        if n_components is None:
            self._n_components = None
        else:
            self._n_components = integer_parameter(n_components, "n_components")
            if self._n_components <= 0:
                raise ValueError("decompose.n_components must be > 0 for dict_learning")
        self._alpha = float(cfg["alpha"])
        if self._alpha <= 0.0:
            raise ValueError("decompose.alpha must be > 0 for dict_learning")
        self._max_iter = integer_parameter(cfg["max_iter"], "max_iter")
        if self._max_iter <= 0:
            raise ValueError("decompose.max_iter must be > 0 for dict_learning")
        self._tol = float(cfg["tol"])
        if self._tol <= 0.0:
            raise ValueError("decompose.tol must be > 0 for dict_learning")
        fit_algorithm = str(cfg["fit_algorithm"])
        if fit_algorithm not in _FIT_ALGORITHMS:
            raise ValueError(f"decompose.fit_algorithm must be one of {_FIT_ALGORITHMS}, got {fit_algorithm}")
        self._fit_algorithm = cast(FitAlgorithm, fit_algorithm)
        transform_algorithm = str(cfg["transform_algorithm"])
        if transform_algorithm not in _TRANSFORM_ALGORITHMS:
            raise ValueError(
                f"decompose.transform_algorithm must be one of {_TRANSFORM_ALGORITHMS}, got {transform_algorithm}"
            )
        self._transform_algorithm = cast(TransformAlgorithm, transform_algorithm)
        transform_n_nonzero = cfg["transform_n_nonzero_coefs"]
        if transform_n_nonzero is None:
            self._transform_n_nonzero = None
        else:
            self._transform_n_nonzero = integer_parameter(transform_n_nonzero, "transform_n_nonzero")
            if self._transform_n_nonzero <= 0:
                raise ValueError("decompose.transform_n_nonzero_coefs must be > 0 for dict_learning")
        transform_alpha = cfg["transform_alpha"]
        if transform_alpha is None:
            self._transform_alpha = None
        else:
            self._transform_alpha = float(transform_alpha)
            if self._transform_alpha <= 0.0:
                raise ValueError("decompose.transform_alpha must be > 0 for dict_learning")
        self._positive_code = bool(cfg["positive_code"])
        self._positive_dict = bool(cfg["positive_dict"])
        seed = cfg["seed"]
        self._seed = integer_parameter(seed, "seed") if seed is not None else None

        self._models: list[DictionaryLearning] | None = None
        self._mask: np.ndarray | None = None
        self._mask_indices: np.ndarray | None = None
        self._grid_shape: tuple[int, int] | None = None
        self._channels: int | None = None
        self._n_samples: int | None = None
        self._n_components_effective: int | None = None
        self._coeff_shape: tuple[int, ...] | None = None
        self._field_ndim: int | None = None
        self._transform_algorithm_effective: str | None = None

    def _resolve_mask(
        self,
        domain_spec: DomainSpec,
        mask_arr: np.ndarray | None,
    ) -> np.ndarray | None:
        domain_mask = domain_spec.mask
        if self._mask_policy == "error":
            if domain_mask is not None or mask_arr is not None:
                raise ValueError("dict_learning mask_policy=error does not allow masks")
            return None
        if self._mask_policy != "ignore_masked_points":
            raise ValueError(f"Unsupported mask_policy for dict_learning: {self._mask_policy}")
        # CONTRACT: ignore_masked_points requires a fixed mask across samples.
        mask = domain_mask
        if mask_arr is not None:
            sample_mask = np.asarray(mask_arr[0]).astype(bool)
            if not np.all(mask_arr == sample_mask):
                raise ValueError("dict_learning requires a fixed mask across samples")
            mask = sample_mask if mask is None else (mask & sample_mask)
        if mask is not None and mask.shape != domain_spec.grid_shape:
            raise ValueError(f"mask shape {mask.shape} does not match {domain_spec.grid_shape}")
        return mask

    def fit(
        self,
        dataset: Any | None = None,
        domain_spec: DomainSpec | None = None,
    ) -> "DictLearningDecomposer":
        if dataset is None:
            raise ValueError("dict_learning requires dataset for fit")
        if domain_spec is None:
            raise ValueError("dict_learning requires domain_spec for fit")
        field_arr, mask_arr = field_arrays(dataset, domain_spec, scalar=False)

        mask = self._resolve_mask(domain_spec, mask_arr)
        if mask is not None:
            mask_flat = mask.reshape(-1, order="C")
            mask_indices = np.flatnonzero(mask_flat)
            if mask_indices.size == 0:
                raise ValueError("dict_learning mask has no valid entries")
        else:
            mask_indices = None

        n_samples = field_arr.shape[0]
        channels = field_arr.shape[-1]
        models: list[DictionaryLearning] = []
        resolved_components: int | None = None
        for ch in range(channels):
            snapshots = field_arr[..., ch].reshape(n_samples, -1, order="C")
            if mask_indices is not None:
                snapshots = snapshots[:, mask_indices]
            if not np.isfinite(snapshots).all():
                raise ValueError("dict_learning requires finite snapshots")
            transform_algorithm = self._transform_algorithm
            model = DictionaryLearning(
                n_components=self._n_components,
                alpha=self._alpha,
                max_iter=self._max_iter,
                tol=self._tol,
                fit_algorithm=self._fit_algorithm,
                transform_algorithm=transform_algorithm,
                transform_n_nonzero_coefs=self._transform_n_nonzero,
                transform_alpha=self._transform_alpha,
                positive_code=self._positive_code,
                positive_dict=self._positive_dict,
                random_state=self._seed,
            )
            model.fit(snapshots)
            if resolved_components is None:
                resolved_components = int(model.components_.shape[0])
            elif resolved_components != int(model.components_.shape[0]):
                raise ValueError("dict_learning component count mismatch across channels")
            models.append(model)

        self._models = models
        self._transform_algorithm_effective = transform_algorithm
        self._mask = mask
        self._mask_indices = mask_indices
        self._grid_shape = domain_spec.grid_shape
        self._channels = channels
        self._n_samples = n_samples
        self._n_components_effective = resolved_components
        return self

    def _check_mask_consistency(
        self,
        field_mask: np.ndarray | None,
        domain_spec: DomainSpec,
    ) -> np.ndarray | None:
        combined = _combine_masks(field_mask, domain_spec.mask)
        if self._mask_policy == "error":
            if combined is not None:
                raise ValueError("dict_learning mask_policy=error does not allow masks")
            if self._mask is not None:
                raise ValueError("dict_learning was fit without mask but mask is present")
            return None
        if combined is None:
            if self._mask is not None:
                raise ValueError("dict_learning requires mask but none was provided")
            return None
        if self._mask is None:
            raise ValueError("dict_learning was fit without mask but mask is present")
        if combined.shape != self._mask.shape or not np.array_equal(combined, self._mask):
            raise ValueError("dict_learning requires the same mask used during fit")
        return combined

    def transform(
        self,
        field: np.ndarray,
        mask: np.ndarray | None,
        domain_spec: DomainSpec,
    ) -> np.ndarray:
        validate_decomposer_compatibility(domain_spec, self._cfg)
        if self._models is None or self._grid_shape is None or self._channels is None:
            raise ValueError("dict_learning fit must be called before transform")
        field_3d, was_2d = _normalize_field(field)
        if field_3d.shape[:2] != domain_spec.grid_shape:
            raise ValueError(f"field shape {field_3d.shape[:2]} does not match domain {domain_spec.grid_shape}")
        if field_3d.shape[:2] != self._grid_shape:
            raise ValueError("dict_learning domain grid does not match fit")
        if field_3d.shape[-1] != self._channels:
            raise ValueError("dict_learning field channels do not match fit")
        field_mask = _normalize_mask(mask, field_3d.shape[:2])
        self._check_mask_consistency(field_mask, domain_spec)
        mask_indices = self._mask_indices

        coeffs = []
        for ch in range(field_3d.shape[-1]):
            vec = field_3d[..., ch].reshape(-1, order="C")
            if mask_indices is not None:
                vec = vec[mask_indices]
            code = self._models[ch].transform(vec[None, :])[0]
            coeffs.append(code)
        coeff_tensor = np.stack(coeffs, axis=0)
        self._coeff_shape = coeff_tensor.shape
        self._field_ndim = 2 if was_2d else 3
        valid_count = (
            int(self._mask_indices.size)
            if self._mask_indices is not None
            else int(field_3d.shape[0] * field_3d.shape[1])
        )
        # REVIEW: channel-first coeff layout keeps dict_learning consistent with other decomposers.
        self._coeff_meta = {
            "method": self.name,
            "field_shape": [int(field_3d.shape[0]), int(field_3d.shape[1])]
            if was_2d
            else [int(field_3d.shape[0]), int(field_3d.shape[1]), int(field_3d.shape[2])],
            "field_ndim": self._field_ndim,
            "field_layout": "HW" if was_2d else "HWC",
            "channels": int(field_3d.shape[-1]),
            "coeff_shape": [int(x) for x in coeff_tensor.shape],
            "coeff_layout": "CK",
            "flatten_order": "C",
            "complex_format": "real",
            "keep": "all",
            "n_samples": int(self._n_samples) if self._n_samples is not None else None,
            "n_components": int(self._n_components_effective) if self._n_components_effective is not None else None,
            "projection": "dictionary_learning",
            "mask_policy": self._mask_policy,
            "mask_valid_count": valid_count,
            "mean_centered": False,
            "fit_algorithm": self._fit_algorithm,
            "transform_algorithm": self._transform_algorithm_effective,
            "transform_n_nonzero_coefs": self._transform_n_nonzero,
            "transform_alpha": self._transform_alpha,
            "alpha": self._alpha,
            "positive_code": self._positive_code,
            "positive_dict": self._positive_dict,
            "seed": self._seed,
        }
        return coeff_tensor

    def inverse_transform(self, coeff: np.ndarray, domain_spec: DomainSpec) -> np.ndarray:
        validate_decomposer_compatibility(domain_spec, self._cfg)
        if self._models is None or self._grid_shape is None or self._field_ndim is None:
            raise ValueError("dict_learning transform must be called before inverse_transform")
        if domain_spec.grid_shape != self._grid_shape:
            raise ValueError("dict_learning domain grid does not match fit")
        if self._coeff_shape is None:
            raise ValueError("dict_learning coeff_shape is not available")
        coeff = np.asarray(coeff).reshape(-1)
        expected = int(np.prod(self._coeff_shape))
        if coeff.size > expected:
            raise ValueError(f"coeff size {coeff.size} exceeds expected {expected}")
        if coeff.size < expected:
            raise ValueError("Coefficient size is smaller than the fitted layout")
        coeff_tensor = coeff.reshape(self._coeff_shape, order="C")
        height, width = self._grid_shape
        mask_indices = self._mask_indices

        field_channels = []
        for ch in range(coeff_tensor.shape[0]):
            # Sparse coding's decoder is linear; components_ is public across
            # supported sklearn versions, unlike inverse_transform's stubs.
            vec = coeff_tensor[ch] @ self._models[ch].components_
            if mask_indices is not None:
                full = np.zeros(height * width, dtype=vec.dtype)
                full[mask_indices] = vec
                field_c = full.reshape(height, width, order="C")
            else:
                field_c = vec.reshape(height, width, order="C")
            field_channels.append(field_c)
        field_hat = np.stack(field_channels, axis=-1)
        if self._field_ndim == 2 and field_hat.shape[-1] == 1:
            return field_hat[..., 0]
        return field_hat
