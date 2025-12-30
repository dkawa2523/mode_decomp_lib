"""Decomposer registry and baseline FFT2/DCT2 implementations."""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Callable, Dict, Mapping

import numpy as np
from scipy.fft import dctn, idctn

from mode_decomp_ml.domain import DomainSpec, validate_decomposer_compatibility

_FFT_NORM = "ortho"
_FFT_SHIFT = False
_DCT_TYPE = 2
_DCT_NORM = "ortho"

_DECOMPOSER_REGISTRY: Dict[str, Callable[..., "BaseDecomposer"]] = {}


def _cfg_get(cfg: Mapping[str, Any] | None, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        try:
            return cfg.get(key, default)
        except TypeError:
            pass
    return getattr(cfg, key, default)


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


def register_decomposer(name: str) -> Callable[[Callable[..., "BaseDecomposer"]], Callable[..., "BaseDecomposer"]]:
    def _wrapper(cls: Callable[..., "BaseDecomposer"]) -> Callable[..., "BaseDecomposer"]:
        if name in _DECOMPOSER_REGISTRY:
            raise KeyError(f"Decomposer already registered: {name}")
        _DECOMPOSER_REGISTRY[name] = cls
        return cls

    return _wrapper


def list_decomposers() -> tuple[str, ...]:
    return tuple(sorted(_DECOMPOSER_REGISTRY.keys()))


def build_decomposer(cfg: Mapping[str, Any]) -> "BaseDecomposer":
    name = str(_cfg_get(cfg, "name", _cfg_get(cfg, "method", ""))).strip()
    if not name:
        raise ValueError("decompose.name is required")
    cls = _DECOMPOSER_REGISTRY.get(name)
    if cls is None:
        raise KeyError(f"Unknown decomposer: {name}. Available: {list_decomposers()}")
    return cls(cfg=cfg)


class BaseDecomposer:
    """Minimal decomposer interface."""

    name: str

    def __init__(self, *, cfg: Mapping[str, Any]) -> None:
        self._cfg = dict(cfg)
        self._coeff_meta: Dict[str, Any] | None = None

    def fit(self, dataset: Any | None = None, domain_spec: DomainSpec | None = None) -> "BaseDecomposer":
        return self

    def coeff_meta(self) -> Mapping[str, Any]:
        if self._coeff_meta is None:
            raise ValueError("coeff_meta is not available before transform")
        return self._coeff_meta

    def save_coeff_meta(self, run_dir: str | Path) -> Path:
        meta = self.coeff_meta()
        out_dir = Path(run_dir) / "artifacts" / "decomposer"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "coeff_meta.json"
        # CONTRACT: coeff_meta must be persisted for comparability.
        with path.open("w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)
        return path

    def save_state(self, run_dir: str | Path) -> Path:
        out_dir = Path(run_dir) / "artifacts" / "decomposer"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "state.pkl"
        with path.open("wb") as fh:
            pickle.dump(self, fh)
        return path

    @staticmethod
    def load_state(path: str | Path) -> "BaseDecomposer":
        with Path(path).open("rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, BaseDecomposer):
            raise TypeError("Loaded decomposer state is not a BaseDecomposer")
        return obj

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
        field_3d, was_2d = _normalize_field(field)
        if field_3d.shape[:2] != domain_spec.grid_shape:
            raise ValueError(
                f"field shape {field_3d.shape[:2]} does not match domain {domain_spec.grid_shape}"
            )
        field_mask = _normalize_mask(mask, field_3d.shape[:2])
        combined = _combine_masks(field_mask, domain_spec.mask)
        if combined is not None and not combined.all():
            if not allow_zero_fill:
                raise ValueError("mask has invalid entries for FFT/DCT without mask_zero_fill policy")
            # CONTRACT: only zero-fill when policy explicitly allows it.
            field_3d = field_3d.copy()
            field_3d[~combined] = 0.0
        return field_3d, was_2d


@register_decomposer("fft2")
class FFT2Decomposer(BaseDecomposer):
    """FFT2 decomposer (complex split into real/imag)."""

    def __init__(self, *, cfg: Mapping[str, Any]) -> None:
        super().__init__(cfg=cfg)
        self.name = "fft2"
        self._disk_policy = str(_cfg_get(cfg, "disk_policy", "")).strip()
        if not self._disk_policy:
            raise ValueError("decompose.disk_policy is required for fft2")
        self._coeff_shape: tuple[int, ...] | None = None
        self._field_ndim: int | None = None

    def transform(
        self,
        field: np.ndarray,
        mask: np.ndarray | None,
        domain_spec: DomainSpec,
    ) -> np.ndarray:
        validate_decomposer_compatibility(domain_spec, self._cfg)
        allow_zero_fill = domain_spec.name == "disk" and self._disk_policy == "mask_zero_fill"
        field_3d, was_2d = self._prepare_field(
            field, mask, domain_spec, allow_zero_fill=allow_zero_fill
        )
        height, width, channels = field_3d.shape

        coeffs = []
        for idx in range(channels):
            coeff = np.fft.fft2(field_3d[..., idx], norm=_FFT_NORM)
            if _FFT_SHIFT:
                coeff = np.fft.fftshift(coeff, axes=(0, 1))
            coeffs.append(np.stack((coeff.real, coeff.imag), axis=-1))

        coeff_tensor = np.stack(coeffs, axis=0)
        flat = coeff_tensor.reshape(-1, order="C")
        self._coeff_shape = coeff_tensor.shape
        self._field_ndim = 2 if was_2d else 3
        # REVIEW: flatten order and complex split define coeff index semantics.
        self._coeff_meta = {
            "method": self.name,
            "field_shape": [int(height), int(width)] if was_2d else [int(height), int(width), int(channels)],
            "field_ndim": self._field_ndim,
            "field_layout": "HW" if was_2d else "HWC",
            "channels": int(channels),
            "coeff_shape": [int(x) for x in coeff_tensor.shape],
            "coeff_layout": "CHWRI",
            "flatten_order": "C",
            "complex_format": "real_imag",
            "keep": "all",
            "fft_norm": _FFT_NORM,
            "fft_shift": _FFT_SHIFT,
            "disk_policy": self._disk_policy,
        }
        return flat

    def inverse_transform(self, coeff: np.ndarray, domain_spec: DomainSpec) -> np.ndarray:
        validate_decomposer_compatibility(domain_spec, self._cfg)
        if self._coeff_shape is None or self._field_ndim is None:
            raise ValueError("transform must be called before inverse_transform")
        coeff = np.asarray(coeff).reshape(-1)
        expected = int(np.prod(self._coeff_shape))
        if coeff.size > expected:
            raise ValueError(f"coeff size {coeff.size} exceeds expected {expected}")
        if coeff.size < expected:
            padded = np.zeros(expected, dtype=coeff.dtype)
            padded[: coeff.size] = coeff
            coeff = padded
        coeff_tensor = coeff.reshape(self._coeff_shape, order="C")
        field_channels = []
        for idx in range(coeff_tensor.shape[0]):
            complex_coeff = coeff_tensor[idx, ..., 0] + 1j * coeff_tensor[idx, ..., 1]
            if _FFT_SHIFT:
                complex_coeff = np.fft.ifftshift(complex_coeff, axes=(0, 1))
            field_c = np.fft.ifft2(complex_coeff, norm=_FFT_NORM).real
            field_channels.append(field_c)
        field_hat = np.stack(field_channels, axis=-1)
        if self._field_ndim == 2 and field_hat.shape[-1] == 1:
            return field_hat[..., 0]
        return field_hat


@register_decomposer("dct2")
class DCT2Decomposer(BaseDecomposer):
    """DCT2 decomposer (real coefficients)."""

    def __init__(self, *, cfg: Mapping[str, Any]) -> None:
        super().__init__(cfg=cfg)
        self.name = "dct2"
        self._disk_policy = str(_cfg_get(cfg, "disk_policy", "")).strip()
        if not self._disk_policy:
            raise ValueError("decompose.disk_policy is required for dct2")
        self._coeff_shape: tuple[int, ...] | None = None
        self._field_ndim: int | None = None

    def transform(
        self,
        field: np.ndarray,
        mask: np.ndarray | None,
        domain_spec: DomainSpec,
    ) -> np.ndarray:
        validate_decomposer_compatibility(domain_spec, self._cfg)
        allow_zero_fill = domain_spec.name == "disk" and self._disk_policy == "mask_zero_fill"
        field_3d, was_2d = self._prepare_field(
            field, mask, domain_spec, allow_zero_fill=allow_zero_fill
        )
        height, width, channels = field_3d.shape

        coeffs = []
        for idx in range(channels):
            coeff = dctn(field_3d[..., idx], type=_DCT_TYPE, norm=_DCT_NORM, axes=(0, 1))
            coeffs.append(coeff)

        coeff_tensor = np.stack(coeffs, axis=0)
        flat = coeff_tensor.reshape(-1, order="C")
        self._coeff_shape = coeff_tensor.shape
        self._field_ndim = 2 if was_2d else 3
        # REVIEW: flatten order defines coeff index semantics.
        self._coeff_meta = {
            "method": self.name,
            "field_shape": [int(height), int(width)] if was_2d else [int(height), int(width), int(channels)],
            "field_ndim": self._field_ndim,
            "field_layout": "HW" if was_2d else "HWC",
            "channels": int(channels),
            "coeff_shape": [int(x) for x in coeff_tensor.shape],
            "coeff_layout": "CHW",
            "flatten_order": "C",
            "complex_format": "real",
            "keep": "all",
            "dct_type": _DCT_TYPE,
            "dct_norm": _DCT_NORM,
            "disk_policy": self._disk_policy,
        }
        return flat

    def inverse_transform(self, coeff: np.ndarray, domain_spec: DomainSpec) -> np.ndarray:
        validate_decomposer_compatibility(domain_spec, self._cfg)
        if self._coeff_shape is None or self._field_ndim is None:
            raise ValueError("transform must be called before inverse_transform")
        coeff = np.asarray(coeff).reshape(-1)
        expected = int(np.prod(self._coeff_shape))
        if coeff.size > expected:
            raise ValueError(f"coeff size {coeff.size} exceeds expected {expected}")
        if coeff.size < expected:
            padded = np.zeros(expected, dtype=coeff.dtype)
            padded[: coeff.size] = coeff
            coeff = padded
        coeff_tensor = coeff.reshape(self._coeff_shape, order="C")
        field_channels = []
        for idx in range(coeff_tensor.shape[0]):
            field_c = idctn(coeff_tensor[idx], type=_DCT_TYPE, norm=_DCT_NORM, axes=(0, 1))
            field_channels.append(field_c)
        field_hat = np.stack(field_channels, axis=-1)
        if self._field_ndim == 2 and field_hat.shape[-1] == 1:
            return field_hat[..., 0]
        return field_hat


from .autoencoder import AutoencoderDecomposer
from .dict_learning import DictLearningDecomposer
from .fourier_bessel import FourierBesselDecomposer
from .graph_fourier import GraphFourierDecomposer
from .helmholtz import HelmholtzDecomposer
from .laplace_beltrami import LaplaceBeltramiDecomposer
from .pod_svd import PODSVDDecomposer
from .zernike import ZernikeDecomposer

__all__ = [
    "BaseDecomposer",
    "FFT2Decomposer",
    "DCT2Decomposer",
    "AutoencoderDecomposer",
    "DictLearningDecomposer",
    "FourierBesselDecomposer",
    "GraphFourierDecomposer",
    "HelmholtzDecomposer",
    "LaplaceBeltramiDecomposer",
    "PODSVDDecomposer",
    "ZernikeDecomposer",
    "build_decomposer",
    "list_decomposers",
    "register_decomposer",
]
