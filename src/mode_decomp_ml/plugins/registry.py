"""Central plugin registry for decomposers, coeff_post, and models."""
from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, TYPE_CHECKING, TypeVar

from mode_decomp_ml.config import cfg_get

if TYPE_CHECKING:  # pragma: no cover - typing only
    from mode_decomp_ml.plugins.coeff_post.basic import BaseCoeffPost
    from mode_decomp_ml.plugins.codecs.basic import BaseCoeffCodec
    from mode_decomp_ml.plugins.decomposers.base import BaseDecomposer
    from mode_decomp_ml.plugins.models.base import BaseRegressor

TPlugin = TypeVar("TPlugin")

_DECOMPOSER_REGISTRY: Dict[str, Callable[..., "BaseDecomposer"]] = {}
_COEFF_POST_REGISTRY: Dict[str, Callable[..., "BaseCoeffPost"]] = {}
_REGRESSOR_REGISTRY: Dict[str, Callable[..., "BaseRegressor"]] = {}
_COEFF_CODEC_REGISTRY: Dict[str, Callable[..., "BaseCoeffCodec"]] = {}
_PREPROCESS_REGISTRY: Dict[str, Callable[..., Any]] = {}


def _register(
    name: str,
    registry: Dict[str, Callable[..., TPlugin]],
    *,
    label: str,
) -> Callable[[Callable[..., TPlugin]], Callable[..., TPlugin]]:
    def _wrapper(cls: Callable[..., TPlugin]) -> Callable[..., TPlugin]:
        if name in registry:
            raise KeyError(f"{label} already registered: {name}")
        registry[name] = cls
        return cls

    return _wrapper


def _build_from_registry(
    cfg: Mapping[str, Any],
    registry: Dict[str, Callable[..., TPlugin]],
    *,
    label: str,
    list_fn: Callable[[], tuple[str, ...]],
) -> TPlugin:
    name = str(cfg_get(cfg, "name", cfg_get(cfg, "method", ""))).strip()
    if not name:
        raise ValueError(f"{label}.name is required")
    cls = registry.get(name)
    if cls is None:
        raise KeyError(f"Unknown {label}: {name}. Available: {list_fn()}")
    return cls(cfg=cfg)


def register_decomposer(name: str) -> Callable[[Callable[..., "BaseDecomposer"]], Callable[..., "BaseDecomposer"]]:
    return _register(name, _DECOMPOSER_REGISTRY, label="Decomposer")


def list_decomposers() -> tuple[str, ...]:
    return tuple(sorted(_DECOMPOSER_REGISTRY.keys()))


def build_decomposer(cfg: Mapping[str, Any]) -> "BaseDecomposer":
    return _build_from_registry(
        cfg,
        _DECOMPOSER_REGISTRY,
        label="decomposer",
        list_fn=list_decomposers,
    )


def register_coeff_post(name: str) -> Callable[[Callable[..., "BaseCoeffPost"]], Callable[..., "BaseCoeffPost"]]:
    return _register(name, _COEFF_POST_REGISTRY, label="CoeffPost")


def register_coeff_codec(name: str) -> Callable[[Callable[..., "BaseCoeffCodec"]], Callable[..., "BaseCoeffCodec"]]:
    return _register(name, _COEFF_CODEC_REGISTRY, label="CoeffCodec")


def list_coeff_codecs() -> tuple[str, ...]:
    return tuple(sorted(_COEFF_CODEC_REGISTRY.keys()))


def build_coeff_codec(cfg: Mapping[str, Any]) -> "BaseCoeffCodec":
    return _build_from_registry(
        cfg,
        _COEFF_CODEC_REGISTRY,
        label="codec",
        list_fn=list_coeff_codecs,
    )


def list_coeff_posts() -> tuple[str, ...]:
    return tuple(sorted(_COEFF_POST_REGISTRY.keys()))


def build_coeff_post(cfg: Mapping[str, Any]) -> "BaseCoeffPost":
    return _build_from_registry(
        cfg,
        _COEFF_POST_REGISTRY,
        label="coeff_post",
        list_fn=list_coeff_posts,
    )


def register_regressor(name: str) -> Callable[[Callable[..., "BaseRegressor"]], Callable[..., "BaseRegressor"]]:
    return _register(name, _REGRESSOR_REGISTRY, label="Regressor")


def list_regressors() -> tuple[str, ...]:
    return tuple(sorted(_REGRESSOR_REGISTRY.keys()))


def build_regressor(cfg: Mapping[str, Any]) -> "BaseRegressor":
    return _build_from_registry(
        cfg,
        _REGRESSOR_REGISTRY,
        label="model",
        list_fn=list_regressors,
    )


def register_preprocess(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    # Preprocess classes live in `mode_decomp_ml.preprocess` to preserve pickle module paths.
    return _register(name, _PREPROCESS_REGISTRY, label="Preprocess")


def list_preprocess() -> tuple[str, ...]:
    return tuple(sorted(_PREPROCESS_REGISTRY.keys()))


def build_preprocess(cfg: Mapping[str, Any]) -> Any:
    return _build_from_registry(
        cfg,
        _PREPROCESS_REGISTRY,
        label="preprocess",
        list_fn=list_preprocess,
    )


__all__ = [
    "register_decomposer",
    "list_decomposers",
    "build_decomposer",
    "register_coeff_codec",
    "list_coeff_codecs",
    "build_coeff_codec",
    "register_coeff_post",
    "list_coeff_posts",
    "build_coeff_post",
    "register_regressor",
    "list_regressors",
    "build_regressor",
    "register_preprocess",
    "list_preprocess",
    "build_preprocess",
]
