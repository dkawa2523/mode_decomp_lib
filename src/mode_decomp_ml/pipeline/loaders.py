"""Artifact loaders for process entrypoints."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from mode_decomp_ml.pipeline.utils import resolve_path
from mode_decomp_ml.plugins.coeff_post import BaseCoeffPost
from mode_decomp_ml.plugins.codecs import BaseCoeffCodec
from mode_decomp_ml.plugins.decomposers import BaseDecomposer
from mode_decomp_ml.plugins.models import BaseRegressor
from mode_decomp_ml.preprocess import load_preprocess_state


def _resolve_existing(primary: Path, fallback: Path | None, label: str) -> Path:
    if primary.exists():
        return primary
    if fallback is not None and fallback.exists():
        return fallback
    fallback_msg = f" or {fallback}" if fallback is not None else ""
    raise FileNotFoundError(f"{label} not found: {primary}{fallback_msg}")


def _resolve_component_state(run_root: Path, component: str) -> Path:
    return _resolve_existing(
        run_root / "states" / component / "state.pkl",
        run_root / "artifacts" / component / "state.pkl",
        f"{component} state",
    )


def load_train_artifacts(
    train_run_dir: str | Path,
) -> tuple[BaseDecomposer, BaseCoeffPost, BaseCoeffCodec, BaseRegressor]:
    """Load decomposer/coeff_post/codec/model states from a training run."""
    run_root = resolve_path(str(train_run_dir))
    decomposer = BaseDecomposer.load_state(_resolve_component_state(run_root, "decomposer"))
    coeff_post = BaseCoeffPost.load_state(_resolve_component_state(run_root, "coeff_post"))
    codec = BaseCoeffCodec.load_state(_resolve_component_state(run_root, "coeff_codec"))
    model = BaseRegressor.load_state(
        _resolve_existing(
            run_root / "model" / "model.pkl",
            run_root / "artifacts" / "model" / "model.pkl",
            "model state",
        )
    )
    return decomposer, coeff_post, codec, model


def load_preprocess_state_from_run(train_run_dir: str | Path) -> Any:
    """Load preprocess state from a training run."""
    run_root = resolve_path(str(train_run_dir))
    path = _resolve_component_state(run_root, "preprocess")
    return load_preprocess_state(path)


def load_model_state(train_run_dir: str | Path) -> BaseRegressor:
    """Load model state from a training run."""
    run_root = resolve_path(str(train_run_dir))
    path = _resolve_existing(
        run_root / "model" / "model.pkl",
        run_root / "artifacts" / "model" / "model.pkl",
        "model state",
    )
    return BaseRegressor.load_state(path)


__all__ = [
    "load_model_state",
    "load_preprocess_state_from_run",
    "load_train_artifacts",
]
