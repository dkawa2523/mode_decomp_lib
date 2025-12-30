"""Dictionary learning decomposer for data-driven sparse coding."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from sklearn.decomposition import DictionaryLearning

from mode_decomp_ml.domain import DomainSpec, validate_decomposer_compatibility

from . import BaseDecomposer, register_decomposer

_MASK_POLICIES = {"error", "ignore_masked_points"}
_FIT_ALGORITHMS = {"lars", "cd"}
_TRANSFORM_ALGORITHMS = {"omp", "lasso_lars", "lasso_cd", "threshold"}


def _cfg_get(cfg: Mapping[str, Any] | None, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        try:
            return cfg.get(key, default)
        except TypeError:
            pass
    return getattr(cfg, key, default)


def _require_cfg(cfg: Mapping[str, Any], key: str) -> Any:
    value = _cfg_get(cfg, key, None)
    if value is None:
        raise ValueError(f"decompose.{key} is required for dict_learning")
    if isinstance(value, str) and not value.strip():
        raise ValueError(f"decompose.{key} must be non-empty for dict_learning")
    return value


def _normalize_field(field: np.ndarray) -> tuple[np.ndarray, bool]:
    field = np.asarray(field)
    if field.ndim == 2:
        return field[..., None], True
    if field.ndim == 3:
        return field, False
    raise ValueError(f"field must be 2D or 3D, got shape {field.shape}")


def _normalize_mask(mask: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray | None:
    if mask is None:
        return None
    mask = np.asarray(mask)
    if mask.shape != shape:
        raise ValueError(f"mask shape {mask.shape} does not match {shape}")
    return mask.astype(bool)


def _combine_masks(
    field_mask: np.ndarray | None,
    domain_mask: np.ndarray | None,
) -> np.ndarray | None:
    if field_mask is None and domain_mask is None:
        return None
    if field_mask is None:
        return domain_mask
    if domain_mask is None:
        return field_mask
    return field_mask & domain_mask


@register_decomposer("dict_learning")
class DictLearningDecomposer(BaseDecomposer):
    """Dictionary learning decomposer fitted on training snapshots."""

    def __init__(self, *, cfg: Mapping[str, Any]) -> None:
        super().__init__(cfg=cfg)
        self.name = "dict_learning"
        self._mask_policy = str(_require_cfg(cfg, "mask_policy"))
        if self._mask_policy not in _MASK_POLICIES:
            raise ValueError(
                f"decompose.mask_policy must be one of {_MASK_POLICIES}, got {self._mask_policy}"
            )
        n_components = _cfg_get(cfg, "n_components", None)
        if n_components is None:
            self._n_components = None
        else:
            self._n_components = int(n_components)
            if self._n_components <= 0:
                raise ValueError("decompose.n_components must be > 0 for dict_learning")
        self._alpha = float(_cfg_get(cfg, "alpha", 1.0))
        if self._alpha <= 0.0:
            raise ValueError("decompose.alpha must be > 0 for dict_learning")
        self._max_iter = int(_cfg_get(cfg, "max_iter", 200))
        if self._max_iter <= 0:
            raise ValueError("decompose.max_iter must be > 0 for dict_learning")
        self._tol = float(_cfg_get(cfg, "tol", 1.0e-6))
        if self._tol <= 0.0:
            raise ValueError("decompose.tol must be > 0 for dict_learning")
        self._fit_algorithm = str(_cfg_get(cfg, "fit_algorithm", "lars"))
        if self._fit_algorithm not in _FIT_ALGORITHMS:
            raise ValueError(
                f"decompose.fit_algorithm must be one of {_FIT_ALGORITHMS}, got {self._fit_algorithm}"
            )
        self._transform_algorithm = str(_cfg_get(cfg, "transform_algorithm", "omp"))
        if self._transform_algorithm not in _TRANSFORM_ALGORITHMS:
            raise ValueError(
                "decompose.transform_algorithm must be one of "
                f"{_TRANSFORM_ALGORITHMS}, got {self._transform_algorithm}"
            )
        transform_n_nonzero = _cfg_get(cfg, "transform_n_nonzero_coefs", None)
        if transform_n_nonzero is None:
            self._transform_n_nonzero = None
        else:
            self._transform_n_nonzero = int(transform_n_nonzero)
            if self._transform_n_nonzero <= 0:
                raise ValueError("decompose.transform_n_nonzero_coefs must be > 0 for dict_learning")
        transform_alpha = _cfg_get(cfg, "transform_alpha", None)
        if transform_alpha is None:
            self._transform_alpha = None
        else:
            self._transform_alpha = float(transform_alpha)
            if self._transform_alpha <= 0.0:
                raise ValueError("decompose.transform_alpha must be > 0 for dict_learning")
        self._positive_code = bool(_cfg_get(cfg, "positive_code", False))
        self._positive_dict = bool(_cfg_get(cfg, "positive_dict", False))
        seed = _cfg_get(cfg, "seed", None)
        self._seed = int(seed) if seed is not None else None

        self._models: list[DictionaryLearning] | None = None
        self._mask: np.ndarray | None = None
        self._mask_indices: np.ndarray | None = None
        self._grid_shape: tuple[int, int] | None = None
        self._channels: int | None = None
        self._n_samples: int | None = None
        self._n_components_effective: int | None = None
        self._coeff_shape: tuple[int, ...] | None = None
        self._field_ndim: int | None = None

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
        fields: list[np.ndarray] = []
        masks: list[np.ndarray | None] = []
        for idx in range(len(dataset)):
            sample = dataset[idx]
            field = np.asarray(sample.field)
            if field.ndim != 3:
                raise ValueError(f"field must be 3D per sample, got {field.shape}")
            if field.shape[:2] != domain_spec.grid_shape:
                raise ValueError(
                    f"field shape {field.shape[:2]} does not match domain {domain_spec.grid_shape}"
                )
            fields.append(field)
            masks.append(None if sample.mask is None else np.asarray(sample.mask))
        if not fields:
            raise ValueError("dict_learning requires at least one training sample")
        field_arr = np.stack(fields, axis=0)
        if all(mask is None for mask in masks):
            mask_arr = None
        elif any(mask is None for mask in masks):
            raise ValueError("mask must be present for all samples or none")
        else:
            mask_arr = np.stack([mask for mask in masks if mask is not None], axis=0)

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
            model = DictionaryLearning(
                n_components=self._n_components,
                alpha=self._alpha,
                max_iter=self._max_iter,
                tol=self._tol,
                fit_algorithm=self._fit_algorithm,
                transform_algorithm=self._transform_algorithm,
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
            raise ValueError(
                f"field shape {field_3d.shape[:2]} does not match domain {domain_spec.grid_shape}"
            )
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
        flat = coeff_tensor.reshape(-1, order="C")
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
            "n_components": int(self._n_components_effective)
            if self._n_components_effective is not None
            else None,
            "projection": "dictionary_learning",
            "mask_policy": self._mask_policy,
            "mask_valid_count": valid_count,
            "mean_centered": False,
            "fit_algorithm": self._fit_algorithm,
            "transform_algorithm": self._transform_algorithm,
            "transform_n_nonzero_coefs": self._transform_n_nonzero,
            "transform_alpha": self._transform_alpha,
            "alpha": self._alpha,
            "positive_code": self._positive_code,
            "positive_dict": self._positive_dict,
            "seed": self._seed,
        }
        return flat

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
            padded = np.zeros(expected, dtype=coeff.dtype)
            padded[: coeff.size] = coeff
            coeff = padded
        coeff_tensor = coeff.reshape(self._coeff_shape, order="C")
        height, width = self._grid_shape
        mask_indices = self._mask_indices

        field_channels = []
        for ch in range(coeff_tensor.shape[0]):
            vec = self._models[ch].inverse_transform(coeff_tensor[ch][None, :])[0]
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
