"""Grid-based FFT2 low-pass decomposer.

This is a frequency-discretized variant of `fft2` that keeps only a centered
low-frequency block in the (fftshifted) frequency domain. It reduces the
coefficient dimensionality from O(H*W) to O(block_size^2).

Notes:
- Coefficients are complex (stored as complex arrays; typically encoded via
  `fft_complex_codec_v1` through `auto_codec_v1`).
- For disk domains, this decomposer can be used with `disk_policy=mask_zero_fill`
  to zero-fill outside the disk mask (approximate baseline; not domain-orthogonal).
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from mode_decomp_ml.config import cfg_get
from mode_decomp_ml.domain import DomainSpec
from mode_decomp_ml.plugins.decomposers.base import GridDecomposerBase
from mode_decomp_ml.plugins.registry import register_decomposer

_FFT_NORM = "ortho"


def _center_block_slices(size: int, block: int) -> slice:
    if block > size:
        raise ValueError("block_size exceeds coefficient grid size")
    start = int(size // 2 - block // 2)
    start = max(0, min(start, int(size - block)))
    return slice(start, int(start + block))


@register_decomposer("fft2_lowpass")
class FFT2LowpassDecomposer(GridDecomposerBase):
    """FFT2 decomposer that keeps only a centered low-frequency block."""

    def __init__(self, *, cfg: Mapping[str, Any]) -> None:
        super().__init__(cfg=cfg)
        self.name = "fft2_lowpass"
        self._disk_policy = str(cfg_get(cfg, "disk_policy", "")).strip()
        if not self._disk_policy:
            raise ValueError("decompose.disk_policy is required for fft2_lowpass")
        block_size = cfg_get(cfg, "block_size", None)
        if block_size is None:
            raise ValueError("decompose.block_size is required for fft2_lowpass")
        self._block_size = int(block_size)
        if self._block_size <= 0:
            raise ValueError("decompose.block_size must be >= 1 for fft2_lowpass")

        # Needed for inverse padding.
        self._orig_hw: tuple[int, int] | None = None

    def _forward_fft_lowpass(self, field_2d: np.ndarray) -> np.ndarray:
        coeff = np.fft.fft2(np.asarray(field_2d), norm=_FFT_NORM)
        coeff = np.fft.fftshift(coeff, axes=(0, 1))
        h, w = int(coeff.shape[0]), int(coeff.shape[1])
        if self._block_size > min(h, w):
            raise ValueError(
                f"decompose.block_size={self._block_size} exceeds min(H,W)={min(h,w)}"
            )
        sy = _center_block_slices(h, self._block_size)
        sx = _center_block_slices(w, self._block_size)
        return coeff[sy, sx]

    def _inverse_fft_lowpass(self, coeff_2d: np.ndarray) -> np.ndarray:
        if self._orig_hw is None:
            raise ValueError("transform must be called before inverse_transform")
        h, w = int(self._orig_hw[0]), int(self._orig_hw[1])
        block = np.asarray(coeff_2d)
        if block.ndim != 2:
            raise ValueError("fft2_lowpass inverse expects a 2D coefficient block per channel")
        bh, bw = int(block.shape[0]), int(block.shape[1])
        if bh > h or bw > w:
            raise ValueError("fft2_lowpass coefficient block exceeds original grid size")
        out_shifted = np.zeros((h, w), dtype=np.complex64 if block.dtype == np.complex64 else np.complex128)
        sy = _center_block_slices(h, bh)
        sx = _center_block_slices(w, bw)
        out_shifted[sy, sx] = block
        out = np.fft.ifftshift(out_shifted, axes=(0, 1))
        return np.fft.ifft2(out, norm=_FFT_NORM).real

    def transform(
        self,
        field: np.ndarray,
        mask: np.ndarray | None,
        domain_spec: DomainSpec,
    ) -> np.ndarray:
        allow_zero_fill = domain_spec.name == "disk" and self._disk_policy == "mask_zero_fill"
        self._orig_hw = tuple(int(x) for x in domain_spec.grid_shape)
        return self._grid_transform(
            field,
            mask,
            domain_spec,
            allow_zero_fill=allow_zero_fill,
            forward_fn=self._forward_fft_lowpass,
            coeff_layout="CHW",
            complex_format="complex",
            extra_meta={
                "fft_norm": _FFT_NORM,
                "fft_shift": True,  # coefficients are stored in shifted coordinates
                "disk_policy": self._disk_policy,
                "block_size": int(self._block_size),
                "orig_hw": [int(self._orig_hw[0]), int(self._orig_hw[1])],
            },
        )

    def inverse_transform(self, coeff: np.ndarray, domain_spec: DomainSpec) -> np.ndarray:
        return self._grid_inverse(coeff, domain_spec, inverse_fn=self._inverse_fft_lowpass)


__all__ = ["FFT2LowpassDecomposer"]

