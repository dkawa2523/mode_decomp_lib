"""Preprocess registry and implementations (field/mask only)."""
from __future__ import annotations

from pathlib import Path
import pickle
from typing import Any, Callable, Dict, Mapping

import numpy as np

from mode_decomp_ml.config import cfg_get
from mode_decomp_ml.plugins.registry import (
    build_preprocess as _plugins_build_preprocess,
    list_preprocess as _plugins_list_preprocess,
    register_preprocess as _plugins_register_preprocess,
)


def _validate_train_split(split: str) -> None:
    if not split:
        raise ValueError("split is required for preprocess.fit")
    if split != "train":
        raise ValueError("preprocess.fit requires split='train' to avoid train/serve skew")


def _ensure_field_batch(fields: np.ndarray) -> np.ndarray:
    fields = np.asarray(fields)
    if fields.ndim == 3:
        return fields[None, ...]
    if fields.ndim == 4:
        return fields
    raise ValueError(f"fields must be 3D or 4D, got shape {fields.shape}")


def _ensure_mask_batch(masks: np.ndarray | None, expected_shape: tuple[int, int, int]) -> np.ndarray | None:
    if masks is None:
        return None
    masks = np.asarray(masks)
    if masks.ndim == 2:
        masks = masks[None, ...]
    if masks.ndim != 3:
        raise ValueError(f"masks must be 2D or 3D, got shape {masks.shape}")
    if masks.shape != expected_shape:
        raise ValueError(f"masks shape {masks.shape} does not match expected {expected_shape}")
    return masks.astype(bool)


def register_preprocess(name: str) -> Callable[[Callable[..., "BasePreprocess"]], Callable[..., "BasePreprocess"]]:
    # Delegate registration to the central plugins registry. Keep classes in this module for pickle compatibility.
    return _plugins_register_preprocess(name)


def list_preprocess() -> tuple[str, ...]:
    return _plugins_list_preprocess()


def build_preprocess(cfg: Mapping[str, Any]) -> "BasePreprocess":
    return _plugins_build_preprocess(cfg)


def load_preprocess_state(path: str | Path) -> "BasePreprocess":
    path = Path(path)
    if not path.exists():
        return NoOpPreprocess(cfg={"name": "none"})
    return BasePreprocess.load_state(path)


class BasePreprocess:
    """Base preprocess protocol (field/mask only)."""

    name: str

    def __init__(self, *, cfg: Mapping[str, Any]) -> None:
        self._cfg = dict(cfg)
        self._fitted = False
        self._state: Dict[str, Any] | None = None

    def fit(self, fields: np.ndarray, masks: np.ndarray | None, *, split: str) -> "BasePreprocess":
        raise NotImplementedError

    def transform(
        self, fields: np.ndarray, masks: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        raise NotImplementedError

    def inverse_transform(
        self, fields: np.ndarray, masks: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        raise NotImplementedError

    def inverse_transform_std(self, field_std: np.ndarray) -> np.ndarray:
        return np.asarray(field_std)

    def state(self) -> Mapping[str, Any]:
        if self._state is None:
            raise ValueError("state is not available before fit")
        return self._state

    def save_state(self, run_dir: str | Path) -> Path:
        out_dir = Path(run_dir) / "outputs" / "states" / "preprocess"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "state.pkl"
        with path.open("wb") as fh:
            pickle.dump(self, fh)
        return path

    @staticmethod
    def load_state(path: str | Path) -> "BasePreprocess":
        with Path(path).open("rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, BasePreprocess):
            raise TypeError("Loaded preprocess state is not a BasePreprocess")
        return obj

    def _mark_fitted(self) -> None:
        self._fitted = True

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise ValueError("preprocess must be fit before transform/inverse")


@register_preprocess("none")
class NoOpPreprocess(BasePreprocess):
    """No-op preprocess."""

    def __init__(self, *, cfg: Mapping[str, Any]) -> None:
        super().__init__(cfg=cfg)
        self.name = "none"
        self._fitted = True
        self._state = {"method": self.name}

    def fit(self, fields: np.ndarray, masks: np.ndarray | None, *, split: str) -> "NoOpPreprocess":
        _validate_train_split(split)
        self._mark_fitted()
        return self

    def transform(
        self, fields: np.ndarray, masks: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        return _ensure_field_batch(fields), masks

    def inverse_transform(
        self, fields: np.ndarray, masks: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        return _ensure_field_batch(fields), masks


@register_preprocess("basic")
class BasicPreprocess(BasePreprocess):
    """Basic preprocess pipeline with optional ops."""

    def __init__(self, *, cfg: Mapping[str, Any]) -> None:
        super().__init__(cfg=cfg)
        self.name = "basic"
        self._ops = _normalize_ops(cfg_get(cfg, "ops", []))
        self._ops_state: Dict[str, Any] = {}

    def fit(self, fields: np.ndarray, masks: np.ndarray | None, *, split: str) -> "BasicPreprocess":
        _validate_train_split(split)
        fields = _ensure_field_batch(fields)
        masks = _ensure_mask_batch(masks, fields.shape[:3]) if masks is not None else None

        for op in self._ops:
            name = op["name"]
            if name == "field_standardize":
                per_channel = bool(op.get("per_channel", True))
                eps = float(op.get("eps", 1e-8))
                if per_channel:
                    mean = fields.mean(axis=(0, 1, 2), keepdims=True)
                    std = fields.std(axis=(0, 1, 2), keepdims=True)
                else:
                    mean = fields.mean()
                    std = fields.std()
                scale = np.where(std == 0.0, 1.0, std)
                scale = scale + eps
                self._ops_state[name] = {
                    "mean": mean,
                    "scale": scale,
                    "per_channel": per_channel,
                    "eps": eps,
                }
            else:
                raise ValueError(f"Unknown preprocess op: {name}")

        self._state = {"method": self.name, "ops": list(self._ops), "ops_state": self._ops_state}
        self._mark_fitted()
        return self

    def transform(
        self, fields: np.ndarray, masks: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        self._require_fitted()
        fields = _ensure_field_batch(fields)
        masks = _ensure_mask_batch(masks, fields.shape[:3]) if masks is not None else None
        out_fields = fields
        out_masks = masks
        for op in self._ops:
            name = op["name"]
            if name == "field_standardize":
                state = self._ops_state.get(name)
                if not state:
                    raise ValueError("field_standardize state is missing")
                out_fields = (out_fields - state["mean"]) / state["scale"]
            else:
                raise ValueError(f"Unknown preprocess op: {name}")
        return out_fields, out_masks

    def inverse_transform(
        self, fields: np.ndarray, masks: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        self._require_fitted()
        fields = _ensure_field_batch(fields)
        masks = _ensure_mask_batch(masks, fields.shape[:3]) if masks is not None else None
        out_fields = fields
        out_masks = masks
        for op in reversed(self._ops):
            name = op["name"]
            if name == "field_standardize":
                state = self._ops_state.get(name)
                if not state:
                    raise ValueError("field_standardize state is missing")
                out_fields = out_fields * state["scale"] + state["mean"]
            else:
                raise ValueError(f"Unknown preprocess op: {name}")
        return out_fields, out_masks

    def inverse_transform_std(self, field_std: np.ndarray) -> np.ndarray:
        self._require_fitted()
        out_std = np.asarray(field_std)
        for op in reversed(self._ops):
            name = op["name"]
            if name == "field_standardize":
                state = self._ops_state.get(name)
                if not state:
                    raise ValueError("field_standardize state is missing")
                out_std = out_std * state["scale"]
            else:
                raise ValueError(f"Unknown preprocess op: {name}")
        return out_std


def _normalize_ops(ops: Any) -> list[dict[str, Any]]:
    if ops is None:
        return []
    if isinstance(ops, Mapping):
        ops = [ops]
    out: list[dict[str, Any]] = []
    for op in ops:
        if isinstance(op, str):
            name = op.strip()
            if not name:
                continue
            out.append({"name": name})
            continue
        if isinstance(op, Mapping):
            name = str(op.get("name", "")).strip()
            if not name:
                raise ValueError("preprocess op requires name")
            out.append(dict(op))
            continue
        raise ValueError(f"Unsupported preprocess op type: {type(op)}")
    return out


__all__ = [
    "BasePreprocess",
    "BasicPreprocess",
    "NoOpPreprocess",
    "build_preprocess",
    "list_preprocess",
    "load_preprocess_state",
    "register_preprocess",
]
