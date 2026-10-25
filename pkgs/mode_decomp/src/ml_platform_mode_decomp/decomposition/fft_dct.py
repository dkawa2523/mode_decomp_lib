"""Grid-based FFT/DCT decomposers."""

from __future__ import annotations


from ml_platform_mode_decomp.config import constructor_config

from typing import Any, Mapping

import numpy as np
from scipy.fft import dctn, idctn

from ml_platform_mode_decomp.geometry import DomainSpec
from ml_platform_mode_decomp.decomposition.base import GridDecomposerBase

_FFT_NORM = "ortho"
_FFT_SHIFT = False
_DCT_TYPE = 2
_DCT_NORM = "ortho"


class FFT2Decomposer(GridDecomposerBase):
    """FFT2 decomposer (complex coefficients)."""

    synthesis_kind = "affine"

    def __init__(self, *, cfg: Mapping[str, Any] | None = None, disk_policy: Any = "error") -> None:
        cfg = constructor_config(cfg, locals(), name="fft2")
        super().__init__(cfg=cfg)
        self.name = "fft2"
        self._disk_policy = str(cfg["disk_policy"]).strip()
        if not self._disk_policy:
            raise ValueError("decompose.disk_policy is required for fft2")
        self._coeff_shape: tuple[int, ...] | None = None
        self._field_ndim: int | None = None

    def _forward_fft(self, fields: np.ndarray) -> np.ndarray:
        coeff = np.fft.fft2(fields, axes=(-2, -1), norm=_FFT_NORM)
        if _FFT_SHIFT:
            coeff = np.fft.fftshift(coeff, axes=(-2, -1))
        return coeff

    def _inverse_fft(self, coeff_2d: np.ndarray) -> np.ndarray:
        coeff = np.asarray(coeff_2d)
        if _FFT_SHIFT:
            coeff = np.fft.ifftshift(coeff, axes=(-2, -1))
        return np.fft.ifft2(coeff, axes=(-2, -1), norm=_FFT_NORM).real

    def transform(
        self,
        field: np.ndarray,
        mask: np.ndarray | None,
        domain_spec: DomainSpec,
    ) -> np.ndarray:
        return self.transform_batch(
            np.asarray(field)[None], None if mask is None else np.asarray(mask)[None], domain_spec
        )[0]

    def transform_batch(self, fields, masks, domain_spec):
        allow_zero_fill = domain_spec.name == "disk" and self._disk_policy == "mask_zero_fill"
        return self._grid_transform_batch(
            fields,
            masks,
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
        if self._coeff_shape is None:
            raise ValueError("transform must be called before inverse_transform")
        coefficients = self._reshape_coeff(coeff, self._coeff_shape, name=self.name)
        return self.inverse_transform_batch(coefficients[None], domain_spec)[0]

    def inverse_transform_batch(self, coefficients, domain_spec):
        return self._grid_inverse_batch(coefficients, domain_spec, inverse_fn=self._inverse_fft)


class DCT2Decomposer(GridDecomposerBase):
    """DCT2 decomposer (real coefficients)."""

    synthesis_kind = "affine"

    def __init__(self, *, cfg: Mapping[str, Any] | None = None, disk_policy: Any = "error") -> None:
        cfg = constructor_config(cfg, locals(), name="dct2")
        super().__init__(cfg=cfg)
        self.name = "dct2"
        self._disk_policy = str(cfg["disk_policy"]).strip()
        if not self._disk_policy:
            raise ValueError("decompose.disk_policy is required for dct2")
        self._coeff_shape: tuple[int, ...] | None = None
        self._field_ndim: int | None = None

    def _forward_dct(self, fields: np.ndarray) -> np.ndarray:
        return dctn(fields, axes=(-2, -1), type=_DCT_TYPE, norm=_DCT_NORM)

    def _inverse_dct(self, coeff_2d: np.ndarray) -> np.ndarray:
        return idctn(coeff_2d, axes=(-2, -1), type=_DCT_TYPE, norm=_DCT_NORM)

    def transform(
        self,
        field: np.ndarray,
        mask: np.ndarray | None,
        domain_spec: DomainSpec,
    ) -> np.ndarray:
        return self.transform_batch(
            np.asarray(field)[None], None if mask is None else np.asarray(mask)[None], domain_spec
        )[0]

    def transform_batch(self, fields, masks, domain_spec):
        allow_zero_fill = domain_spec.name == "disk" and self._disk_policy == "mask_zero_fill"
        return self._grid_transform_batch(
            fields,
            masks,
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
        if self._coeff_shape is None:
            raise ValueError("transform must be called before inverse_transform")
        coefficients = self._reshape_coeff(coeff, self._coeff_shape, name=self.name)
        return self.inverse_transform_batch(coefficients[None], domain_spec)[0]

    def inverse_transform_batch(self, coefficients, domain_spec):
        return self._grid_inverse_batch(coefficients, domain_spec, inverse_fn=self._inverse_dct)


__all__ = ["FFT2Decomposer", "DCT2Decomposer"]
