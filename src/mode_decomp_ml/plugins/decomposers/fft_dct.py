"""Grid-based FFT/DCT decomposers."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from mode_decomp_ml.config import cfg_get
from scipy.fft import dctn, idctn

from mode_decomp_ml.domain import DomainSpec
from mode_decomp_ml.plugins.decomposers.base import GridDecomposerBase
from mode_decomp_ml.plugins.registry import register_decomposer

_FFT_NORM = "ortho"
_FFT_SHIFT = False
_DCT_TYPE = 2
_DCT_NORM = "ortho"



@register_decomposer("fft2")
class FFT2Decomposer(GridDecomposerBase):
    """FFT2 decomposer (complex coefficients)."""

    def __init__(self, *, cfg: Mapping[str, Any]) -> None:
        super().__init__(cfg=cfg)
        self.name = "fft2"
        self._disk_policy = str(cfg_get(cfg, "disk_policy", "")).strip()
        if not self._disk_policy:
            raise ValueError("decompose.disk_policy is required for fft2")
        self._coeff_shape: tuple[int, ...] | None = None
        self._field_ndim: int | None = None

    def _forward_fft(self, field_2d: np.ndarray) -> np.ndarray:
        coeff = np.fft.fft2(field_2d, norm=_FFT_NORM)
        if _FFT_SHIFT:
            coeff = np.fft.fftshift(coeff, axes=(0, 1))
        return coeff

    def _inverse_fft(self, coeff_2d: np.ndarray) -> np.ndarray:
        coeff = np.asarray(coeff_2d)
        if _FFT_SHIFT:
            coeff = np.fft.ifftshift(coeff, axes=(0, 1))
        return np.fft.ifft2(coeff, norm=_FFT_NORM).real

    def transform(
        self,
        field: np.ndarray,
        mask: np.ndarray | None,
        domain_spec: DomainSpec,
    ) -> np.ndarray:
        allow_zero_fill = domain_spec.name == "disk" and self._disk_policy == "mask_zero_fill"
        return self._grid_transform(
            field,
            mask,
            domain_spec,
            allow_zero_fill=allow_zero_fill,
            forward_fn=self._forward_fft,
            coeff_layout="CHW",
            complex_format="complex",
            extra_meta={
                "fft_norm": _FFT_NORM,
                "fft_shift": _FFT_SHIFT,
                "disk_policy": self._disk_policy,
            },
        )

    def inverse_transform(self, coeff: np.ndarray, domain_spec: DomainSpec) -> np.ndarray:
        return self._grid_inverse(coeff, domain_spec, inverse_fn=self._inverse_fft)


@register_decomposer("dct2")
class DCT2Decomposer(GridDecomposerBase):
    """DCT2 decomposer (real coefficients)."""

    def __init__(self, *, cfg: Mapping[str, Any]) -> None:
        super().__init__(cfg=cfg)
        self.name = "dct2"
        self._disk_policy = str(cfg_get(cfg, "disk_policy", "")).strip()
        if not self._disk_policy:
            raise ValueError("decompose.disk_policy is required for dct2")
        self._coeff_shape: tuple[int, ...] | None = None
        self._field_ndim: int | None = None

    def _forward_dct(self, field_2d: np.ndarray) -> np.ndarray:
        return dctn(field_2d, type=_DCT_TYPE, norm=_DCT_NORM)

    def _inverse_dct(self, coeff_2d: np.ndarray) -> np.ndarray:
        return idctn(coeff_2d, type=_DCT_TYPE, norm=_DCT_NORM)

    def transform(
        self,
        field: np.ndarray,
        mask: np.ndarray | None,
        domain_spec: DomainSpec,
    ) -> np.ndarray:
        allow_zero_fill = domain_spec.name == "disk" and self._disk_policy == "mask_zero_fill"
        return self._grid_transform(
            field,
            mask,
            domain_spec,
            allow_zero_fill=allow_zero_fill,
            forward_fn=self._forward_dct,
            coeff_layout="CHW",
            complex_format="real",
            extra_meta={
                "dct_type": _DCT_TYPE,
                "dct_norm": _DCT_NORM,
                "disk_policy": self._disk_policy,
            },
        )

    def inverse_transform(self, coeff: np.ndarray, domain_spec: DomainSpec) -> np.ndarray:
        return self._grid_inverse(coeff, domain_spec, inverse_fn=self._inverse_dct)


__all__ = ["FFT2Decomposer", "DCT2Decomposer"]
