"""Coeff codec plugins."""
from __future__ import annotations

from mode_decomp_ml.plugins.registry import build_coeff_codec, list_coeff_codecs, register_coeff_codec

from .basic import BaseCoeffCodec, NoOpCoeffCodec
from .auto import AutoCodecV1
from .fft_complex import FFTComplexCodecV1
from .offset_residual_pack import OffsetResidualPackCodecV1
from .sh_pack import SHPackCodecV1
from .slepian_pack import SlepianPackCodecV1
from .tensor_pack import TensorPackCodecV1
from .wavelet_pack import WaveletPackCodecV1
from .zernike_pack import ZernikePackCodecV1

__all__ = [
    "BaseCoeffCodec",
    "NoOpCoeffCodec",
    "AutoCodecV1",
    "FFTComplexCodecV1",
    "OffsetResidualPackCodecV1",
    "SHPackCodecV1",
    "SlepianPackCodecV1",
    "TensorPackCodecV1",
    "WaveletPackCodecV1",
    "ZernikePackCodecV1",
    "register_coeff_codec",
    "list_coeff_codecs",
    "build_coeff_codec",
]
