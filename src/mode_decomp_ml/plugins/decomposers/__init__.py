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
from .disk_slepian import DiskSlepianDecomposer
from .fft_dct import DCT2Decomposer, FFT2Decomposer
from .fft2_lowpass import FFT2LowpassDecomposer
from .fourier_bessel import FourierBesselDecomposer
from .fourier_jacobi import FourierJacobiDecomposer
from .rbf_expansion import RBFExpansionDecomposer
from .gappy_pod import GappyPODDecomposer
from .gappy_graph_fourier import GappyGraphFourierDecomposer
from .graph_fourier import GraphFourierDecomposer
from .offset_residual import OffsetResidualDecomposer
from .helmholtz import HelmholtzDecomposer
from .helmholtz_poisson import HelmholtzPoissonDecomposer
from .laplace_beltrami import LaplaceBeltramiDecomposer
from .pod import PODDecomposer
from .pod_joint import JointPODDecomposer
from .pod_joint_em import JointPODEMDecomposer
from .pod_em import PODEMDecomposer
from .pod_svd import PODSVDDecomposer
from .pswf2d_tensor import PSWF2DTensorDecomposer
from .pseudo_zernike import PseudoZernikeDecomposer
from .polar_fft import PolarFFTDecomposer
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
    "FFT2LowpassDecomposer",
    "DCT2Decomposer",
    "PSWF2DTensorDecomposer",
    "Wavelet2DDecomposer",
    "SphericalHarmonicsDecomposer",
    "SphericalSlepianDecomposer",
    "ZernikeDecomposer",
    "PseudoZernikeDecomposer",
    "PolarFFTDecomposer",
    "AnnularZernikeDecomposer",
    "FourierBesselDecomposer",
    "FourierJacobiDecomposer",
    "GraphFourierDecomposer",
    "DiskSlepianDecomposer",
    "LaplaceBeltramiDecomposer",
    "HelmholtzDecomposer",
    "HelmholtzPoissonDecomposer",
    "PODDecomposer",
    "JointPODDecomposer",
    "PODEMDecomposer",
    "JointPODEMDecomposer",
    "PODSVDDecomposer",
    "GappyPODDecomposer",
    "RBFExpansionDecomposer",
    "GappyGraphFourierDecomposer",
    "DictLearningDecomposer",
    "AutoencoderDecomposer",
    "OffsetResidualDecomposer",
]
