"""Grid-based FFT2 low-pass decomposer.

This is a frequency-discretized variant of `fft2` that keeps only a centered
low-frequency block in the (fftshifted) frequency domain. It reduces the
coefficient dimensionality from O(H*W) to O(block_size^2).

Notes:
- Raw coefficients are complex. The default auto codec packs them into the
  independent real coordinates of `real_fft` for learning.
- For disk domains, this decomposer can be used with `disk_policy=mask_zero_fill`
  to zero-fill outside the disk mask (approximate baseline; not domain-orthogonal).
"""

from __future__ import annotations

from ml_platform_mode_decomp.config import constructor_config

from typing import Any, Mapping

import numpy as np

from ml_platform_mode_decomp.geometry import DomainSpec
from ml_platform_mode_decomp.decomposition.base import GridDecomposerBase

_FFT_NORM = "ortho"


def _center_block_slices(size: int, block: int) -> slice:
    if block > size:
        raise ValueError("block_size exceeds coefficient grid size")
    start = int(size // 2 - block // 2)
    start = max(0, min(start, int(size - block)))
    return slice(start, int(start + block))


def _conjugate_closed_size(size: int, requested: int) -> int:
    """Include both cutoff frequencies, except a full-grid Nyquist bin."""
    if requested > size:
        raise ValueError("block_size exceeds coefficient grid size")
    return requested + 1 if requested % 2 == 0 and requested < size else requested


class FFT2LowpassDecomposer(GridDecomposerBase):
    """FFT2 decomposer that keeps only a centered low-frequency block."""

    synthesis_kind = "affine"

    def __init__(
        self, *, cfg: Mapping[str, Any] | None = None, block_size: Any = 8, disk_policy: Any = "error"
    ) -> None:
        cfg = constructor_config(cfg, locals(), name="fft2_lowpass")
        super().__init__(cfg=cfg)
        self.name = "fft2_lowpass"
        self._disk_policy = str(cfg["disk_policy"]).strip()
        if not self._disk_policy:
            raise ValueError("decompose.disk_policy is required for fft2_lowpass")
        block_size = cfg["block_size"]
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
            raise ValueError(f"decompose.block_size={self._block_size} exceeds min(H,W)={min(h, w)}")
        sy = _center_block_slices(h, _conjugate_closed_size(h, self._block_size))
        sx = _center_block_slices(w, _conjugate_closed_size(w, self._block_size))
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
        height, width = domain_spec.grid_shape
        original_shape = (int(height), int(width))
        self._orig_hw = original_shape
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
                "retained_hw": [
                    _conjugate_closed_size(original_shape[0], self._block_size),
                    _conjugate_closed_size(original_shape[1], self._block_size),
                ],
                "frequency_selection": "conjugate_closed_center_block",
                "orig_hw": [original_shape[0], original_shape[1]],
            },
        )

    def inverse_transform(self, coeff: np.ndarray, domain_spec: DomainSpec) -> np.ndarray:
        return self._grid_inverse(coeff, domain_spec, inverse_fn=self._inverse_fft_lowpass)


__all__ = ["FFT2LowpassDecomposer"]
