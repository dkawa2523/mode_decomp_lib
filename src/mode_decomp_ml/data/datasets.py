"""Dataset implementations and registry."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping

import numpy as np


@dataclass(frozen=True)
class FieldSample:
    """Container for a single field sample."""

    cond: np.ndarray
    field: np.ndarray
    mask: np.ndarray | None
    meta: Dict[str, Any]


class BaseDataset:
    """Minimal dataset protocol."""

    name: str

    def __len__(self) -> int:  # pragma: no cover - interface only
        raise NotImplementedError

    def __getitem__(self, index: int) -> FieldSample:  # pragma: no cover - interface only
        raise NotImplementedError


_DATASET_REGISTRY: Dict[str, Callable[..., BaseDataset]] = {}
_VALID_MASK_POLICIES = {"require", "allow_none", "forbid"}


def _cfg_get(cfg: Mapping[str, Any] | None, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        try:
            return cfg.get(key, default)
        except TypeError:
            pass
    return getattr(cfg, key, default)


def _normalize_index(index: int, size: int) -> int:
    if isinstance(index, np.integer):
        index = int(index)
    if not isinstance(index, int):
        raise TypeError(f"Index must be int, got {type(index)}")
    if index < 0 or index >= size:
        raise IndexError(f"Index out of range: {index}")
    return index


def _validate_mask_policy(cfg: Mapping[str, Any] | None) -> str:
    policy = _cfg_get(cfg, "mask_policy", None)
    if policy is None:
        raise ValueError("mask_policy is required in dataset config")
    policy = str(policy)
    if policy not in _VALID_MASK_POLICIES:
        raise ValueError(f"mask_policy must be one of {_VALID_MASK_POLICIES}, got {policy}")
    return policy


def _ensure_cond_batch(cond: np.ndarray) -> np.ndarray:
    cond = np.asarray(cond)
    if cond.ndim == 1:
        return cond[None, :]
    if cond.ndim == 2:
        return cond
    raise ValueError(f"cond must be 1D or 2D, got shape {cond.shape}")


def _ensure_field_batch(field: np.ndarray) -> np.ndarray:
    field = np.asarray(field)
    if field.ndim == 2:
        field = field[..., None]
    if field.ndim == 3:
        return field[None, ...]
    if field.ndim == 4:
        return field
    raise ValueError(f"field must be 2D, 3D, or 4D, got shape {field.shape}")


def _ensure_mask_batch(mask: np.ndarray | None, expected: tuple[int, int, int]) -> np.ndarray | None:
    if mask is None:
        return None
    mask = np.asarray(mask)
    if mask.ndim == 2:
        mask = mask[None, ...]
    elif mask.ndim == 4 and mask.shape[-1] == 1:
        mask = mask[..., 0]
    if mask.ndim != 3:
        raise ValueError(f"mask must be 2D or 3D, got shape {mask.shape}")
    if mask.shape != expected:
        raise ValueError(f"mask shape {mask.shape} does not match expected {expected}")
    return mask.astype(bool)


def _validate_sample(sample: FieldSample) -> None:
    # CONTRACT: dataset samples always follow cond/field/mask/meta schema.
    if sample.cond.ndim != 1:
        raise ValueError(f"cond must be 1D per sample, got {sample.cond.shape}")
    if sample.field.ndim != 3:
        raise ValueError(f"field must be 3D per sample, got {sample.field.shape}")
    if sample.mask is not None:
        if sample.mask.ndim != 2:
            raise ValueError(f"mask must be 2D per sample, got {sample.mask.shape}")
        if sample.mask.shape != sample.field.shape[:2]:
            raise ValueError("mask spatial shape must match field")


def register_dataset(name: str) -> Callable[[Callable[..., BaseDataset]], Callable[..., BaseDataset]]:
    def _wrapper(cls: Callable[..., BaseDataset]) -> Callable[..., BaseDataset]:
        if name in _DATASET_REGISTRY:
            raise KeyError(f"Dataset already registered: {name}")
        _DATASET_REGISTRY[name] = cls
        return cls

    return _wrapper


def list_datasets() -> tuple[str, ...]:
    return tuple(sorted(_DATASET_REGISTRY.keys()))


def build_dataset(
    cfg: Mapping[str, Any],
    *,
    domain_cfg: Mapping[str, Any] | None = None,
    seed: int | None = None,
) -> BaseDataset:
    name = str(_cfg_get(cfg, "name", ""))
    if not name:
        raise ValueError("dataset.name is required")
    cls = _DATASET_REGISTRY.get(name)
    if cls is None:
        raise KeyError(f"Unknown dataset: {name}. Available: {list_datasets()}")
    return cls(cfg=cfg, domain_cfg=domain_cfg, seed=seed)


@register_dataset("synthetic")
class SyntheticDataset(BaseDataset):
    """Synthetic dataset for smoke checks."""

    def __init__(
        self,
        *,
        cfg: Mapping[str, Any],
        domain_cfg: Mapping[str, Any] | None = None,
        seed: int | None = None,
    ) -> None:
        self.name = "synthetic"
        self.domain_name = str(_cfg_get(domain_cfg, "name", "unknown"))
        self.num_samples = int(_cfg_get(cfg, "num_samples", 1))
        self.cond_dim = int(_cfg_get(cfg, "cond_dim", 3))
        self.height = int(_cfg_get(cfg, "height", 32))
        self.width = int(_cfg_get(cfg, "width", 32))
        self.channels = int(_cfg_get(cfg, "channels", 1))
        self.mask_policy = _validate_mask_policy(cfg)
        self.mask_mode = str(_cfg_get(cfg, "mask_mode", "none"))
        self.mask_radius = float(_cfg_get(cfg, "mask_radius", 0.45))

        if self.num_samples <= 0:
            raise ValueError("num_samples must be > 0")
        if self.cond_dim <= 0:
            raise ValueError("cond_dim must be > 0")
        if self.height <= 0 or self.width <= 0:
            raise ValueError("height/width must be > 0")
        if self.channels <= 0:
            raise ValueError("channels must be > 0")

        if self.mask_policy == "require" and self.mask_mode == "none":
            raise ValueError("mask_policy=require but mask_mode=none")
        if self.mask_policy == "forbid" and self.mask_mode != "none":
            raise ValueError("mask_policy=forbid but mask_mode is not none")

        rng = np.random.default_rng(seed)
        self._conds = rng.normal(size=(self.num_samples, self.cond_dim)).astype(np.float32)
        self._weights = rng.normal(size=(self.cond_dim, self.channels)).astype(np.float32)

        x = np.linspace(0.0, 1.0, self.width, dtype=np.float32)
        y = np.linspace(0.0, 1.0, self.height, dtype=np.float32)
        xx, yy = np.meshgrid(x, y, indexing="xy")
        base = np.sin(2 * np.pi * xx) + np.cos(2 * np.pi * yy)
        self._base = base.astype(np.float32)[..., None]
        self._mask = self._build_mask()

    def _build_mask(self) -> np.ndarray | None:
        if self.mask_mode == "none":
            return None
        if self.mask_mode == "full":
            return np.ones((self.height, self.width), dtype=bool)
        if self.mask_mode == "disk":
            x = np.linspace(0.0, 1.0, self.width, dtype=np.float32)
            y = np.linspace(0.0, 1.0, self.height, dtype=np.float32)
            xx, yy = np.meshgrid(x, y, indexing="xy")
            r2 = (xx - 0.5) ** 2 + (yy - 0.5) ** 2
            return r2 <= (self.mask_radius ** 2)
        raise ValueError(f"Unknown mask_mode: {self.mask_mode}")

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> FieldSample:
        index = _normalize_index(index, self.num_samples)
        cond = self._conds[index]
        amps = cond @ self._weights
        field = self._base * amps[None, None, :]
        mask = self._mask
        meta = {
            "sample_id": f"synthetic_{index:04d}",
            "dataset": self.name,
            "domain": self.domain_name,
            "mask_policy": self.mask_policy,
            "mask_mode": self.mask_mode,
        }
        sample = FieldSample(cond=cond, field=field, mask=mask, meta=meta)
        _validate_sample(sample)
        return sample


@register_dataset("npy_dir")
class NpyDirDataset(BaseDataset):
    """Dataset loader for .npy files."""

    def __init__(
        self,
        *,
        cfg: Mapping[str, Any],
        domain_cfg: Mapping[str, Any] | None = None,
        seed: int | None = None,
    ) -> None:
        self.name = "npy_dir"
        self.domain_name = str(_cfg_get(domain_cfg, "name", "unknown"))
        self.mask_policy = _validate_mask_policy(cfg)

        root = Path(str(_cfg_get(cfg, "root", "data")))
        cond_path = root / str(_cfg_get(cfg, "cond_file", "cond.npy"))
        field_path = root / str(_cfg_get(cfg, "field_file", "field.npy"))
        mask_path = root / str(_cfg_get(cfg, "mask_file", "mask.npy"))

        if not cond_path.exists():
            raise FileNotFoundError(f"cond file not found: {cond_path}")
        if not field_path.exists():
            raise FileNotFoundError(f"field file not found: {field_path}")

        cond = np.load(cond_path)
        field = np.load(field_path)
        mask = None
        if mask_path.exists():
            if self.mask_policy == "forbid":
                raise ValueError("mask_policy=forbid but mask.npy exists")
            mask = np.load(mask_path)
        elif self.mask_policy == "require":
            raise FileNotFoundError("mask_policy=require but mask.npy not found")

        # REVIEW: normalize to batch-first arrays to keep the schema fixed.
        cond = _ensure_cond_batch(cond)
        field = _ensure_field_batch(field)
        mask = _ensure_mask_batch(mask, field.shape[:3])

        if cond.shape[0] != field.shape[0]:
            raise ValueError("cond and field sample counts must match")
        if mask is not None and mask.shape[0] != field.shape[0]:
            raise ValueError("mask and field sample counts must match")

        self._cond = cond
        self._field = field
        self._mask = mask
        self._size = field.shape[0]
        self._root = root

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, index: int) -> FieldSample:
        index = _normalize_index(index, self._size)
        cond = self._cond[index]
        field = self._field[index]
        mask = self._mask[index] if self._mask is not None else None
        meta = {
            "sample_id": f"npy_{index:04d}",
            "dataset": self.name,
            "domain": self.domain_name,
            "mask_policy": self.mask_policy,
            "root": str(self._root),
        }
        sample = FieldSample(cond=cond, field=field, mask=mask, meta=meta)
        _validate_sample(sample)
        return sample
