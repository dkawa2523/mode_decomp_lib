"""Real-vector coefficient representations, selected through the lazy catalog."""

from ml_platform_mode_decomp.catalog import build_coeff_codec, list_coeff_codecs
from .basic import BaseCoeffCodec

__all__ = ["BaseCoeffCodec", "build_coeff_codec", "list_coeff_codecs"]
