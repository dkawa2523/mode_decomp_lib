"""Explicit, lazy catalog of supported numerical components.

Listing methods loads no backend; construction imports the explicitly selected class.
"""

from __future__ import annotations

from importlib import import_module
from inspect import Parameter, signature
from copy import deepcopy
from typing import Any, Mapping

from .config import component_config

_DECOMPOSERS: dict[str, str] = {
    "annular_zernike": "ml_platform_mode_decomp.decomposition.annular_zernike:AnnularZernikeDecomposer",
    "autoencoder": "ml_platform_mode_decomp.decomposition.autoencoder:AutoencoderDecomposer",
    "dct2": "ml_platform_mode_decomp.decomposition.fft_dct:DCT2Decomposer",
    "dict_learning": "ml_platform_mode_decomp.decomposition.dict_learning:DictLearningDecomposer",
    "disk_slepian": "ml_platform_mode_decomp.decomposition.disk_slepian:DiskSlepianDecomposer",
    "fft2": "ml_platform_mode_decomp.decomposition.fft_dct:FFT2Decomposer",
    "fft2_lowpass": "ml_platform_mode_decomp.decomposition.fft2_lowpass:FFT2LowpassDecomposer",
    "fourier_bessel": "ml_platform_mode_decomp.decomposition.fourier_bessel:FourierBesselDecomposer",
    "fourier_jacobi": "ml_platform_mode_decomp.decomposition.fourier_jacobi:FourierJacobiDecomposer",
    "gappy_graph_fourier": "ml_platform_mode_decomp.decomposition.gappy_graph_fourier:GappyGraphFourierDecomposer",
    "gappy_pod": "ml_platform_mode_decomp.decomposition.gappy_pod:GappyPODDecomposer",
    "graph_fourier": "ml_platform_mode_decomp.decomposition.graph_fourier:GraphFourierDecomposer",
    "helmholtz": "ml_platform_mode_decomp.decomposition.helmholtz:HelmholtzDecomposer",
    "hodge": "ml_platform_mode_decomp.decomposition.hodge:HodgeDecomposer",
    "laplace_beltrami": "ml_platform_mode_decomp.decomposition.laplace_beltrami:LaplaceBeltramiDecomposer",
    "offset_residual": "ml_platform_mode_decomp.decomposition.offset_residual:OffsetResidualDecomposer",
    "pod": "ml_platform_mode_decomp.decomposition.pod:PODDecomposer",
    "pod_em": "ml_platform_mode_decomp.decomposition.pod_em:PODEMDecomposer",
    "pod_joint": "ml_platform_mode_decomp.decomposition.pod_joint:JointPODDecomposer",
    "pod_joint_em": "ml_platform_mode_decomp.decomposition.pod_joint_em:JointPODEMDecomposer",
    "pod_svd": "ml_platform_mode_decomp.decomposition.pod_svd:PODSVDDecomposer",
    "polar_fft": "ml_platform_mode_decomp.decomposition.polar_fft:PolarFFTDecomposer",
    "pseudo_zernike": "ml_platform_mode_decomp.decomposition.pseudo_zernike:PseudoZernikeDecomposer",
    "pswf2d_tensor": "ml_platform_mode_decomp.decomposition.pswf2d_tensor:PSWF2DTensorDecomposer",
    "rbf_expansion": "ml_platform_mode_decomp.decomposition.rbf_expansion:RBFExpansionDecomposer",
    "spherical_harmonics": "ml_platform_mode_decomp.decomposition.spherical_harmonics:SphericalHarmonicsDecomposer",
    "spherical_slepian": "ml_platform_mode_decomp.decomposition.spherical_slepian:SphericalSlepianDecomposer",
    "wavelet2d": "ml_platform_mode_decomp.decomposition.wavelet2d:Wavelet2DDecomposer",
    "zernike": "ml_platform_mode_decomp.decomposition.zernike_decomposer:ZernikeDecomposer",
}

_CODECS: dict[str, str] = {
    "real_fft": "ml_platform_mode_decomp.coefficients.codecs.real_fft:RealFFTCodec",
    "auto_codec_v1": "ml_platform_mode_decomp.coefficients.codecs.auto:AutoCodecV1",
    "fft_complex_codec_v1": "ml_platform_mode_decomp.coefficients.codecs.fft_complex:FFTComplexCodecV1",
    "none": "ml_platform_mode_decomp.coefficients.codecs.basic:NoOpCoeffCodec",
    "offset_residual_pack_v1": "ml_platform_mode_decomp.coefficients.codecs.offset_residual_pack:OffsetResidualPackCodecV1",
    "wavelet_pack_v1": "ml_platform_mode_decomp.coefficients.codecs.wavelet_pack:WaveletPackCodecV1",
}

_POSTPROCESSORS: dict[str, str] = {
    "dict_learning": "ml_platform_mode_decomp.coefficients.post.basic:DictLearningCoeffPost",
    "none": "ml_platform_mode_decomp.coefficients.post.basic:NoOpCoeffPost",
    "pca": "ml_platform_mode_decomp.coefficients.post.basic:PCACoeffPost",
    "power_yeojohnson": "ml_platform_mode_decomp.coefficients.post.basic:PowerYeoJohnsonCoeffPost",
    "quantile": "ml_platform_mode_decomp.coefficients.post.basic:QuantileCoeffPost",
    "standardize": "ml_platform_mode_decomp.coefficients.post.basic:StandardizeCoeffPost",
}

_REGRESSORS: dict[str, str] = {
    "catboost": "ml_platform_mode_decomp.regressors.gbdt:CatBoostRegressor",
    "elasticnet": "ml_platform_mode_decomp.regressors.sklearn:ElasticNetRegressor",
    "gpr": "ml_platform_mode_decomp.regressors.sklearn:GPRRegressor",
    "lgbm": "ml_platform_mode_decomp.regressors.gbdt:LGBMRegressor",
    "mtgp": "ml_platform_mode_decomp.regressors.mtgp:MTGPRegressor",
    "multitask_elasticnet": "ml_platform_mode_decomp.regressors.sklearn:MultiTaskElasticNetRegressor",
    "multitask_lasso": "ml_platform_mode_decomp.regressors.sklearn:MultiTaskLassoRegressor",
    "ridge": "ml_platform_mode_decomp.regressors.sklearn:RidgeRegressor",
    "xgb": "ml_platform_mode_decomp.regressors.gbdt:XGBRegressor",
}

_PREPROCESSORS: dict[str, str] = {
    "basic": "ml_platform_mode_decomp.field_preprocessing:BasicPreprocess",
    "none": "ml_platform_mode_decomp.field_preprocessing:NoOpPreprocess",
}


def component_defaults(name: str, kind: str) -> dict[str, Any]:
    """Read accepted parameters and defaults from the selected constructor."""
    catalog = {
        "decomposition": _DECOMPOSERS,
        "codec": _CODECS,
        "coeff_post": _POSTPROCESSORS,
        "field_preprocess": _PREPROCESSORS,
        "regression": _REGRESSORS,
    }[kind]
    target = catalog.get("auto_codec_v1" if kind == "codec" and name == "auto" else name)
    if target is None:
        raise ValueError(f"Unknown {kind}: {name}. Available: {tuple(sorted(catalog))}")
    module_name, class_name = target.split(":", 1)
    implementation = getattr(import_module(module_name), class_name)
    return {
        key: deepcopy(parameter.default)
        for key, parameter in signature(implementation).parameters.items()
        if key != "cfg" and parameter.default is not Parameter.empty
    }


def _build(cfg: Mapping[str, Any], catalog: Mapping[str, str], *, label: str) -> Any:
    normalized = component_config(cfg)
    name = normalized["name"]
    if catalog is _CODECS and name == "auto":
        name = "auto_codec_v1"
        normalized["name"] = name
    target = catalog.get(name)
    if target is None:
        raise ValueError(f"Unknown {label}: {name}. Available: {tuple(sorted(catalog))}")
    module_name, class_name = target.split(":", 1)
    implementation = getattr(import_module(module_name), class_name)
    return implementation(cfg=normalized)


def build_decomposer(cfg: Mapping[str, Any]) -> Any:
    return _build(cfg, _DECOMPOSERS, label="decomposer")


def build_coeff_codec(cfg: Mapping[str, Any]) -> Any:
    return _build(cfg, _CODECS, label="coefficient codec")


def build_coeff_post(cfg: Mapping[str, Any]) -> Any:
    return _build(cfg, _POSTPROCESSORS, label="coefficient postprocess")


def build_regressor(cfg: Mapping[str, Any]) -> Any:
    return _build(cfg, _REGRESSORS, label="regressor")


def build_preprocess(cfg: Mapping[str, Any]) -> Any:
    return _build(cfg, _PREPROCESSORS, label="field preprocess")


def list_decomposers() -> tuple[str, ...]:
    return tuple(sorted(_DECOMPOSERS))


def decomposition_role(name: str) -> str:
    """Describe the scientific use; this does not promise orthogonality or losslessness."""
    if name not in _DECOMPOSERS:
        raise ValueError(f"Unknown decomposition: {name}")
    if name in {"helmholtz", "hodge"}:
        return "field_component_separation"
    if name in {"gappy_pod", "gappy_graph_fourier", "pod_em", "pod_joint_em"}:
        return "partial_observation_representation"
    if name in {"pod", "pod_svd", "pod_joint", "autoencoder", "dict_learning"}:
        return "learned_representation"
    if name in {"fft2", "dct2", "wavelet2d"}:
        return "coordinate_transform"
    return "basis_projection"


def list_coeff_codecs() -> tuple[str, ...]:
    return tuple(sorted(_CODECS))


def list_coeff_posts() -> tuple[str, ...]:
    return tuple(sorted(_POSTPROCESSORS))


def list_regressors() -> tuple[str, ...]:
    return tuple(sorted(_REGRESSORS))


def list_preprocess() -> tuple[str, ...]:
    return tuple(sorted(_PREPROCESSORS))
