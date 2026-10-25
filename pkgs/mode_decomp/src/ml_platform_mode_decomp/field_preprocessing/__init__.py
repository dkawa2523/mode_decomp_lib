"""Mask-aware field transformations with no artifact or run lifecycle."""

from __future__ import annotations


from ml_platform_mode_decomp.config import constructor_config

from typing import Any, Mapping

import numpy as np

from ml_platform_mode_decomp.catalog import build_preprocess, list_preprocess


_VALUE_BLOCK_ITEMS = 64 * 1024


def _observed_blocks(values: np.ndarray, valid: np.ndarray | None = None):
    """Visit even strided/memory-mapped arrays without flattening or gathering all observations."""
    operands = [values] if valid is None else [values, valid]
    iterator = np.nditer(
        operands,
        flags=["external_loop", "buffered", "zerosize_ok"],
        op_flags=["readonly"],
        order="C",
        buffersize=_VALUE_BLOCK_ITEMS,
    )
    for block in iterator:
        if valid is None:
            yield block
        else:
            block_values, observed = block
            if observed.all():
                yield block_values
            elif observed.any():
                yield block_values[observed]


def _check_observed_finite(fields: np.ndarray, masks: np.ndarray | None):
    valid = None if masks is None else masks[..., None]
    for values in _observed_blocks(fields, valid):
        if not np.isfinite(values).all():
            raise ValueError("observed field values must be finite")


def _moments(values: np.ndarray, valid: np.ndarray | None):
    """Population moments from stable, bounded, shifted block statistics."""
    count = 0
    mean = 0.0
    m2 = 0.0
    origin = 0.0
    for observed in _observed_blocks(values, valid):
        if not count:
            origin = float(observed[0])
        centered = np.asarray(observed, dtype=np.float64) - origin
        block_mean = float(centered.mean())
        block_m2 = float(np.sum((centered - block_mean) ** 2))
        combined = count + len(centered)
        delta = block_mean - mean
        m2 += block_m2 + delta * delta * count * (len(centered) / combined)
        mean += delta * (len(centered) / combined)
        count = combined
    if not count:
        raise ValueError("field preprocessing requires observed training points")
    return np.float64(origin + mean), np.sqrt(m2 / count)


def _ensure_field_batch(fields: np.ndarray) -> np.ndarray:
    fields = np.asarray(fields)
    if fields.ndim == 3:
        fields = fields[None, ...]
    if fields.ndim != 4:
        raise ValueError(f"fields must be HWC or NHWC, got {fields.shape}")
    if any(size == 0 for size in fields.shape):
        raise ValueError("fields must have non-empty sample, spatial, and channel axes")
    if not np.issubdtype(fields.dtype, np.number) or np.iscomplexobj(fields):
        raise ValueError("fields must be real numerical arrays")
    return fields


def _ensure_mask_batch(masks: np.ndarray | None, expected_shape: tuple[int, int, int]) -> np.ndarray | None:
    if masks is None:
        return None
    masks = np.asarray(masks)
    if masks.ndim == 4 and masks.shape[-1] == 1:
        masks = masks[..., 0]
    if masks.ndim == 2:
        masks = masks[None, ...]
    if masks.ndim != 3 or masks.shape[1:] != expected_shape[1:]:
        raise ValueError(f"masks shape {masks.shape} does not match {expected_shape}")
    if masks.shape[0] not in {1, expected_shape[0]}:
        raise ValueError(f"mask sample count {masks.shape[0]} does not match {expected_shape[0]}")
    return np.broadcast_to(masks.astype(bool, copy=False), expected_shape)


def _inputs(fields: np.ndarray, masks: np.ndarray | None) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    fields = _ensure_field_batch(fields)
    masks = _ensure_mask_batch(masks, fields.shape[:3])
    _check_observed_finite(fields, masks)
    return fields, masks, masks


class BasePreprocess:
    name: str

    def __init__(self, *, cfg: Mapping[str, Any]) -> None:
        self._cfg = dict(cfg)
        self._fitted = False
        self._state: dict[str, Any] | None = None

    def fit(self, fields: np.ndarray, masks: np.ndarray | None = None) -> "BasePreprocess":
        raise NotImplementedError

    def transform(self, fields: np.ndarray, masks: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
        raise NotImplementedError

    def inverse_transform(
        self, fields: np.ndarray, masks: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        raise NotImplementedError

    def state(self) -> Mapping[str, Any]:
        if self._state is None:
            raise ValueError("state is not available before fit")
        return self._state

    def _mark_fitted(self) -> None:
        self._fitted = True

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise ValueError("preprocess must be fit before transform/inverse")


class NoOpPreprocess(BasePreprocess):
    def __init__(self, *, cfg: Mapping[str, Any] | None = None) -> None:
        cfg = constructor_config(cfg, locals(), name="none")
        super().__init__(cfg=cfg)
        self.name = "none"
        self._fitted = True
        self._state = {"method": self.name}

    def fit(self, fields: np.ndarray, masks: np.ndarray | None = None) -> "NoOpPreprocess":
        _inputs(fields, masks)
        return self

    def transform(self, fields: np.ndarray, masks: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
        fields, masks, valid = _inputs(fields, masks)
        return (fields if valid is None or valid.all() else np.where(valid[..., None], fields, 0.0)), masks

    def inverse_transform(
        self, fields: np.ndarray, masks: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        return self.transform(fields, masks)


class BasicPreprocess(BasePreprocess):
    def __init__(self, *, cfg: Mapping[str, Any] | None = None, ops: Any = ()) -> None:
        cfg = constructor_config(cfg, locals(), name="basic")
        super().__init__(cfg=cfg)
        self.name = "basic"
        self._ops = _normalize_ops(cfg["ops"])
        self._ops_state: dict[str, Any] = {}
        self._channels: int | None = None

    def fit(self, fields: np.ndarray, masks: np.ndarray | None = None) -> "BasicPreprocess":
        fields, _, valid = _inputs(fields, masks)
        if valid is not None and not valid.any():
            raise ValueError("field preprocessing requires observed training points")
        self._channels = int(fields.shape[-1])
        self._ops_state = {}
        for op in self._ops:
            name = op["name"]
            if name != "field_standardize":
                raise ValueError(f"Unknown preprocess op: {name}")
            eps = float(op.get("eps", 1e-8))
            if not np.isfinite(eps) or eps < 0:
                raise ValueError("field_standardize.eps must be finite and non-negative")
            per_channel = bool(op.get("per_channel", True))
            if per_channel:
                statistics = [_moments(fields[..., channel], valid) for channel in range(self._channels)]
                mean, std = np.asarray(statistics).T
            else:
                mean, std = _moments(fields, None if valid is None else valid[..., None])
            scale = np.where(std == 0.0, 1.0, std) + eps
            if per_channel:
                mean = np.asarray(mean).reshape(1, 1, 1, -1)
                scale = scale.reshape(1, 1, 1, -1)
            self._ops_state[name] = {"mean": mean, "scale": scale, "per_channel": per_channel, "eps": eps}
        self._state = {"method": self.name, "ops": list(self._ops), "ops_state": self._ops_state}
        self._mark_fitted()
        return self

    def _apply(
        self, fields: np.ndarray, masks: np.ndarray | None, *, inverse: bool
    ) -> tuple[np.ndarray, np.ndarray | None]:
        self._require_fitted()
        fields, masks, valid = _inputs(fields, masks)
        if fields.shape[-1] != self._channels:
            raise ValueError("field channel count does not match fitted preprocessing")
        result = fields if valid is None else np.where(valid[..., None], fields, 0.0)
        ops = reversed(self._ops) if inverse else iter(self._ops)
        for op in ops:
            state = self._ops_state[op["name"]]
            if inverse:
                result = result * state["scale"] + state["mean"]
            else:
                result = (result - state["mean"]) / state["scale"]
            if valid is not None:
                result = np.where(valid[..., None], result, 0.0)
        return result, masks

    def transform(self, fields: np.ndarray, masks: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
        return self._apply(fields, masks, inverse=False)

    def inverse_transform(
        self, fields: np.ndarray, masks: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        return self._apply(fields, masks, inverse=True)


def _normalize_ops(ops: Any) -> list[dict[str, Any]]:
    if ops is None:
        return []
    if isinstance(ops, (Mapping, str)):
        ops = [ops]
    result = []
    names = set()
    for op in ops:
        if isinstance(op, str):
            item = {"name": op.strip()}
        elif isinstance(op, Mapping):
            item = dict(op)
        else:
            raise ValueError("preprocess ops must be names or mappings")
        name = item.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("preprocess op requires name")
        if name in names:
            raise ValueError(f"duplicate preprocess op: {name}")
        names.add(name)
        result.append(item)
    return result


__all__ = ["BasePreprocess", "BasicPreprocess", "NoOpPreprocess", "build_preprocess", "list_preprocess"]
