"""Decomposer plugins grouped by domain/type."""
from __future__ import annotations

from mode_decomp_ml.plugins.registry import build_decomposer, list_decomposers, register_decomposer

from .base import (
    BaseDecomposer,
    ChannelwiseAdapter,
    EigenBasisDecomposerBase,
    GridDecomposerBase,
    ZernikeFamilyBase,
)
from .annular_zernike import AnnularZernikeDecomposer
from .autoencoder import AutoencoderDecomposer
from .dict_learning import DictLearningDecomposer
from .fft_dct import DCT2Decomposer, FFT2Decomposer
from .fourier_bessel import FourierBesselDecomposer
from .gappy_pod import GappyPODDecomposer
from .graph_fourier import GraphFourierDecomposer
from .helmholtz import HelmholtzDecomposer
from .laplace_beltrami import LaplaceBeltramiDecomposer
from .pod import PODDecomposer
from .pod_svd import PODSVDDecomposer
from .pswf2d_tensor import PSWF2DTensorDecomposer
from .spherical_harmonics import SphericalHarmonicsDecomposer
from .spherical_slepian import SphericalSlepianDecomposer
from .wavelet2d import Wavelet2DDecomposer
from .zernike_decomposer import ZernikeDecomposer

__all__ = [
    "BaseDecomposer",
    "GridDecomposerBase",
    "ZernikeFamilyBase",
    "EigenBasisDecomposerBase",
    "ChannelwiseAdapter",
    "register_decomposer",
    "list_decomposers",
    "build_decomposer",
    "FFT2Decomposer",
    "DCT2Decomposer",
    "PSWF2DTensorDecomposer",
    "Wavelet2DDecomposer",
    "SphericalHarmonicsDecomposer",
    "SphericalSlepianDecomposer",
    "ZernikeDecomposer",
    "AnnularZernikeDecomposer",
    "FourierBesselDecomposer",
    "GraphFourierDecomposer",
    "LaplaceBeltramiDecomposer",
    "HelmholtzDecomposer",
    "PODDecomposer",
    "PODSVDDecomposer",
    "GappyPODDecomposer",
    "DictLearningDecomposer",
    "AutoencoderDecomposer",
]
