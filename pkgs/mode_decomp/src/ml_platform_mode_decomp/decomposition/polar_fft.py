"""Polar-grid FFT decomposer for disk domain (approximate; interpolation-based)."""

from __future__ import annotations

from ml_platform_mode_decomp.config import constructor_config

from typing import Any, Mapping

import numpy as np
from scipy.fft import dct, idct

from ml_platform_mode_decomp.geometry import DomainSpec, validate_decomposer_compatibility
from ml_platform_mode_decomp.geometry.identity import geometry_fingerprint

from .base import BaseDecomposer, _combine_masks, _normalize_field, _normalize_mask, parse_bool, require_cfg

_RADIAL_TRANSFORMS = {"dct"}
_ANGULAR_TRANSFORMS = {"fft"}
_INTERPOLATIONS = {"bilinear"}
_MASK_POLICIES = {"ignore_masked_points"}

_DCT_TYPE = 2
_DCT_NORM = "ortho"
_FFT_NORM = "ortho"


def _cartesian_axes(domain_spec: DomainSpec) -> tuple[np.ndarray, np.ndarray]:
    x = domain_spec.coords.get("x")
    y = domain_spec.coords.get("y")
    if x is None or y is None:
        raise ValueError("disk domain must provide x/y coords for polar_fft")
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    height, width = domain_spec.grid_shape
    if width < 2 or height < 2:
        raise ValueError("polar_fft requires grid with height/width >= 2")
    if x.shape != (height, width) or y.shape != x.shape:
        raise ValueError("polar_fft Cartesian coordinates must match the field grid")
    x_axis, y_axis = x[0], y[:, 0]
    if not np.allclose(x, x_axis[None, :]) or not np.allclose(y, y_axis[:, None]):
        raise ValueError("polar_fft requires a rectilinear Cartesian grid")
    if (
        not np.isfinite(x).all()
        or not np.isfinite(y).all()
        or not (np.diff(x_axis) > 0).all()
        or not (np.diff(y_axis) > 0).all()
    ):
        raise ValueError("polar_fft requires finite, strictly increasing Cartesian axes")
    return x_axis, y_axis


def _bilinear_sample_cart(
    image: np.ndarray,
    xq: np.ndarray,
    yq: np.ndarray,
    *,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    img = np.asarray(image)
    if img.ndim != 2:
        raise ValueError("bilinear sampler expects 2D image")
    height, width = img.shape
    xq, yq = np.broadcast_arrays(np.asarray(xq, dtype=np.float64), np.asarray(yq, dtype=np.float64))
    j0 = np.searchsorted(x_axis, xq, side="right") - 1
    i0 = np.searchsorted(y_axis, yq, side="right") - 1
    j0 = np.clip(j0, 0, width - 2)
    i0 = np.clip(i0, 0, height - 2)
    j1 = j0 + 1
    i1 = i0 + 1
    a = (yq - y_axis[i0]) / (y_axis[i1] - y_axis[i0])
    b = (xq - x_axis[j0]) / (x_axis[j1] - x_axis[j0])
    a = np.clip(a, 0.0, 1.0)
    b = np.clip(b, 0.0, 1.0)

    support = (xq >= x_axis[0]) & (xq <= x_axis[-1]) & (yq >= y_axis[0]) & (yq <= y_axis[-1])
    vm = np.ones(img.shape, dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    if vm.shape != img.shape:
        raise ValueError("valid_mask shape mismatch for bilinear sampler")
    # Remove unobserved NaN before multiplying by interpolation weights.
    safe = np.where(vm, img, 0.0)
    f00 = safe[i0, j0]
    f01 = safe[i0, j1]
    f10 = safe[i1, j0]
    f11 = safe[i1, j1]
    w00 = (1 - a) * (1 - b)
    w01 = (1 - a) * b
    w10 = a * (1 - b)
    w11 = a * b
    if valid_mask is not None:
        m00 = vm[i0, j0]
        m01 = vm[i0, j1]
        m10 = vm[i1, j0]
        m11 = vm[i1, j1]
        w00 = np.where(m00, w00, 0.0)
        w01 = np.where(m01, w01, 0.0)
        w10 = np.where(m10, w10, 0.0)
        w11 = np.where(m11, w11, 0.0)
        denom = w00 + w01 + w10 + w11
        num = w00 * f00 + w01 * f01 + w10 * f10 + w11 * f11
        return np.divide(num, denom, out=np.zeros_like(num, dtype=np.float64), where=(denom > 0) & support)
    return np.where(support, w00 * f00 + w01 * f01 + w10 * f10 + w11 * f11, 0.0)


def _bilinear_sample_polar(polar: np.ndarray, r_idx: np.ndarray, t_idx: np.ndarray) -> np.ndarray:
    pol = np.asarray(polar)
    if pol.ndim != 2:
        raise ValueError("polar sampler expects 2D polar grid")
    n_r, n_t = pol.shape
    rr = np.asarray(r_idx, dtype=np.float64)
    tt = np.asarray(t_idx, dtype=np.float64)

    i0 = np.floor(rr).astype(int)
    i0 = np.clip(i0, 0, n_r - 2)
    i1 = i0 + 1
    a = rr - i0
    a = np.clip(a, 0.0, 1.0)

    j0f = np.floor(tt)
    b = tt - j0f
    b = np.clip(b, 0.0, 1.0)
    j0 = (j0f.astype(int)) % n_t
    j1 = (j0 + 1) % n_t

    f00 = pol[i0, j0]
    f01 = pol[i0, j1]
    f10 = pol[i1, j0]
    f11 = pol[i1, j1]
    return (1 - a) * (1 - b) * f00 + (1 - a) * b * f01 + a * (1 - b) * f10 + a * b * f11


class PolarFFTDecomposer(BaseDecomposer[np.ndarray]):
    """Polar FFT decomposer (approximate) for disk domain.

    Forward:
      Cartesian -> polar (bilinear) -> rFFT(theta) -> DCT(r)
    Inverse:
      iDCT(r) -> iRFFT(theta) -> polar -> Cartesian (bilinear)
    """

    synthesis_kind = "affine"

    def __init__(
        self,
        *,
        cfg: Mapping[str, Any] | None = None,
        n_r: Any = 32,
        n_theta: Any = 64,
        boundary_condition: Any = "dirichlet",
        mask_policy: Any = "ignore_masked_points",
        mask_aware_sampling: Any = None,
        radial_transform: Any = "dct",
        angular_transform: Any = "fft",
        interpolation: Any = "bilinear",
    ) -> None:
        cfg = constructor_config(cfg, locals(), name="polar_fft")
        super().__init__(cfg=cfg)
        self.name = "polar_fft"
        n_r = require_cfg(cfg, "n_r", label="decompose")
        n_theta = require_cfg(cfg, "n_theta", label="decompose")
        self._n_r = int(n_r)
        self._n_theta = int(n_theta)
        if self._n_r < 8:
            raise ValueError("decompose.n_r must be >= 8 for polar_fft")
        if self._n_theta < 16:
            raise ValueError("decompose.n_theta must be >= 16 for polar_fft")

        self._radial_transform = str(require_cfg(cfg, "radial_transform", label="decompose")).strip().lower()
        self._angular_transform = str(require_cfg(cfg, "angular_transform", label="decompose")).strip().lower()
        self._interpolation = str(require_cfg(cfg, "interpolation", label="decompose")).strip().lower()
        if self._radial_transform not in _RADIAL_TRANSFORMS:
            raise ValueError(
                f"decompose.radial_transform must be one of {_RADIAL_TRANSFORMS}, got {self._radial_transform}"
            )
        if self._angular_transform not in _ANGULAR_TRANSFORMS:
            raise ValueError(
                f"decompose.angular_transform must be one of {_ANGULAR_TRANSFORMS}, got {self._angular_transform}"
            )
        if self._interpolation not in _INTERPOLATIONS:
            raise ValueError(f"decompose.interpolation must be one of {_INTERPOLATIONS}, got {self._interpolation}")

        # CONTRACT: boundary_condition is required for comparability across disk bases.
        self._boundary_condition = str(require_cfg(cfg, "boundary_condition", label="decompose"))
        self._mask_policy = str(require_cfg(cfg, "mask_policy", label="decompose"))
        if self._mask_policy not in _MASK_POLICIES:
            raise ValueError(f"decompose.mask_policy must be one of {_MASK_POLICIES}, got {self._mask_policy}")
        self._mask_aware_sampling = parse_bool(cfg["mask_aware_sampling"], default=True)

        self._coeff_shape: tuple[int, ...] | None = None
        self._field_ndim: int | None = None
        self._grid_shape: tuple[int, int] | None = None
        self._geometry_key: str | None = None

    def transform(self, field: np.ndarray, mask: np.ndarray | None, domain_spec: DomainSpec) -> np.ndarray:
        validate_decomposer_compatibility(domain_spec, self._cfg)
        field_3d, was_2d = _normalize_field(field)
        if field_3d.shape[:2] != domain_spec.grid_shape:
            raise ValueError(f"field shape {field_3d.shape[:2]} does not match domain {domain_spec.grid_shape}")

        field_mask = _normalize_mask(mask, field_3d.shape[:2])
        combined_mask = _combine_masks(field_mask, domain_spec.mask)
        valid = np.ones(field_3d.shape[:2], dtype=bool) if combined_mask is None else combined_mask
        if not np.isfinite(field_3d[valid]).all():
            raise ValueError("polar_fft fields must be finite at observed locations")
        if not self._mask_aware_sampling:
            field_3d = np.where(valid[..., None], field_3d, 0.0)

        meta = domain_spec.meta or {}
        center = meta.get("center", None)
        if center is None or not hasattr(center, "__len__") or len(center) != 2:
            raise ValueError("disk domain must provide meta.center for polar_fft")
        cx = float(center[0])
        cy = float(center[1])
        # Disk uses meta.radius; annulus uses meta.r_outer.
        radius_value = meta.get("radius", meta.get("r_outer", 0.0))
        radius = float(radius_value) if radius_value is not None else 0.0
        if radius <= 0.0:
            raise ValueError("domain must provide meta.radius (disk) or meta.r_outer (annulus) > 0 for polar_fft")
        r_inner_norm = meta.get("r_inner_norm", None) if domain_spec.name == "annulus" else None
        r0 = float(r_inner_norm) if r_inner_norm is not None else 0.0
        if r0 < 0.0 or r0 >= 1.0:
            raise ValueError("annulus meta.r_inner_norm must satisfy 0 <= r_inner_norm < 1 for polar_fft")

        x_axis, y_axis = _cartesian_axes(domain_spec)

        r_vals = np.linspace(r0, 1.0, self._n_r, dtype=np.float64)
        theta_vals = np.linspace(-np.pi, np.pi, self._n_theta, endpoint=False, dtype=np.float64)
        rr = r_vals[:, None]
        tt = theta_vals[None, :]
        xq = cx + radius * rr * np.cos(tt)
        yq = cy + radius * rr * np.sin(tt)

        coeffs = []
        for ch in range(field_3d.shape[-1]):
            img = np.asarray(field_3d[..., ch], dtype=np.float64)
            polar = _bilinear_sample_cart(
                img,
                xq,
                yq,
                x_axis=x_axis,
                y_axis=y_axis,
                valid_mask=combined_mask if self._mask_aware_sampling else None,
            )
            fft_theta = np.fft.rfft(polar, axis=1, norm=_FFT_NORM)
            # dct does not reliably preserve complex dtype across scipy versions; apply to real/imag separately.
            dct_real = dct(fft_theta.real, type=_DCT_TYPE, norm=_DCT_NORM, axis=0)
            dct_imag = dct(fft_theta.imag, type=_DCT_TYPE, norm=_DCT_NORM, axis=0)
            coeffs.append(dct_real + 1j * dct_imag)
        coeff_tensor = np.stack(coeffs, axis=0)

        self._coeff_shape = coeff_tensor.shape
        self._field_ndim = 2 if was_2d else 3
        self._grid_shape = domain_spec.grid_shape
        self._geometry_key = geometry_fingerprint(domain_spec)
        meta_out = self._coeff_meta_base(
            field_shape=field_3d.shape[:2] if was_2d else field_3d.shape,
            field_ndim=self._field_ndim,
            field_layout="HW" if was_2d else "HWC",
            channels=int(field_3d.shape[-1]),
            coeff_shape=coeff_tensor.shape,
            coeff_layout="CRT",
            complex_format="complex",
        )
        meta_out.update(
            {
                "n_r": int(self._n_r),
                "n_theta": int(self._n_theta),
                "radial_transform": self._radial_transform,
                "angular_transform": self._angular_transform,
                "interpolation": self._interpolation,
                "boundary_condition": self._boundary_condition,
                "fft_norm": _FFT_NORM,
                "dct_type": _DCT_TYPE,
                "dct_norm": _DCT_NORM,
                "projection": "resample_then_separable_transform",
                "mask_policy": self._mask_policy,
                "mask_aware_sampling": bool(self._mask_aware_sampling),
                "outside_cartesian_support": "zero",
                "approximate": True,
                "r_inner_norm": float(r0) if domain_spec.name == "annulus" else None,
            }
        )
        self._coeff_meta = meta_out
        return coeff_tensor

    def inverse_transform(self, coeff: np.ndarray, domain_spec: DomainSpec) -> np.ndarray:
        validate_decomposer_compatibility(domain_spec, self._cfg)
        if self._coeff_shape is None or self._field_ndim is None or self._grid_shape is None:
            raise ValueError("transform must be called before inverse_transform")
        if domain_spec.grid_shape != self._grid_shape:
            raise ValueError("polar_fft domain grid does not match cached shape")
        if geometry_fingerprint(domain_spec) != self._geometry_key:
            raise ValueError("polar_fft geometry differs from the coefficient geometry")
        coeff_tensor = self._reshape_coeff(coeff, self._coeff_shape, name=self.name)

        meta = domain_spec.meta or {}
        center = meta.get("center", None)
        if center is None or not hasattr(center, "__len__") or len(center) != 2:
            raise ValueError("disk domain must provide meta.center for polar_fft")
        cx = float(center[0])
        cy = float(center[1])
        radius_value = meta.get("radius", meta.get("r_outer", 0.0))
        radius = float(radius_value) if radius_value is not None else 0.0
        if radius <= 0.0:
            raise ValueError("domain must provide meta.radius (disk) or meta.r_outer (annulus) > 0 for polar_fft")
        r_inner_norm = meta.get("r_inner_norm", None) if domain_spec.name == "annulus" else None
        r0 = float(r_inner_norm) if r_inner_norm is not None else 0.0
        if r0 < 0.0 or r0 >= 1.0:
            raise ValueError("annulus meta.r_inner_norm must satisfy 0 <= r_inner_norm < 1 for polar_fft")

        # Reconstruct polar grids per channel.
        polar_fields = []
        for ch in range(coeff_tensor.shape[0]):
            coeff_ch = np.asarray(coeff_tensor[ch])
            idct_real = idct(coeff_ch.real, type=_DCT_TYPE, norm=_DCT_NORM, axis=0)
            idct_imag = idct(coeff_ch.imag, type=_DCT_TYPE, norm=_DCT_NORM, axis=0)
            fft_theta = idct_real + 1j * idct_imag
            polar = np.fft.irfft(fft_theta, n=self._n_theta, axis=1, norm=_FFT_NORM)
            polar_fields.append(np.asarray(polar, dtype=np.float64))

        # Map polar -> cartesian by evaluating at each grid point.
        x_in = domain_spec.coords.get("x")
        y_in = domain_spec.coords.get("y")
        if x_in is None or y_in is None:
            raise ValueError("disk domain must provide x/y coords for polar_fft inverse")
        x = np.asarray(x_in, dtype=np.float64)
        y = np.asarray(y_in, dtype=np.float64)
        x_shift = x - cx
        y_shift = y - cy
        r = np.sqrt(x_shift * x_shift + y_shift * y_shift) / radius
        theta = np.arctan2(y_shift, x_shift)  # [-pi, pi]

        # Map r in [r0, 1] to [0, n_r-1]. Values outside are clipped and later masked by domain mask.
        if r0 > 0.0:
            r_scaled = (r - r0) / (1.0 - r0)
        else:
            r_scaled = r
        r_idx = np.clip(r_scaled, 0.0, 1.0) * float(self._n_r - 1)
        t_idx = (theta + np.pi) / (2.0 * np.pi) * float(self._n_theta)

        field_channels = []
        for polar in polar_fields:
            field_c = _bilinear_sample_polar(polar, r_idx, t_idx)
            field_channels.append(field_c)
        field_hat = np.stack(field_channels, axis=-1)
        if domain_spec.mask is not None:
            field_hat = field_hat.copy()
            field_hat[~domain_spec.mask] = 0.0
        if self._field_ndim == 2 and field_hat.shape[-1] == 1:
            return field_hat[..., 0]
        return field_hat


__all__ = ["PolarFFTDecomposer"]
