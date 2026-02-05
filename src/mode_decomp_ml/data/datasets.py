"""Dataset implementations and registry."""
from __future__ import annotations

from dataclasses import dataclass
import csv
from pathlib import Path
from typing import Any, Callable, Dict, Mapping

import numpy as np
from .manifest import load_manifest, resolve_root, validate_field_against_manifest
from mode_decomp_ml.config import cfg_get

PROJECT_ROOT = Path(__file__).resolve().parents[3]

try:  # optional dependency for interpolation
    from scipy.interpolate import griddata as _griddata
except Exception:  # pragma: no cover - fallback when scipy is missing
    _griddata = None

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


def _normalize_index(index: int, size: int) -> int:
    if isinstance(index, np.integer):
        index = int(index)
    if not isinstance(index, int):
        raise TypeError(f"Index must be int, got {type(index)}")
    if index < 0 or index >= size:
        raise IndexError(f"Index out of range: {index}")
    return index


def _validate_mask_policy(cfg: Mapping[str, Any] | None) -> str:
    policy = cfg_get(cfg, "mask_policy", None)
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
    name = str(cfg_get(cfg, "name", "")).strip()
    root = cfg_get(cfg, "root", None)
    if (not name or name == "synthetic") and root is not None:
        if load_manifest(Path(str(root))) is not None:
            name = "npy_dir"
    if not name:
        if cfg_get(cfg, "conditions_csv", None) is not None or cfg_get(cfg, "fields_dir", None) is not None:
            name = "csv_fields"
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
        self.domain_name = str(cfg_get(domain_cfg, "name", "unknown"))
        self.num_samples = int(cfg_get(cfg, "num_samples", 1))
        self.cond_dim = int(cfg_get(cfg, "cond_dim", 3))
        self.height = int(cfg_get(cfg, "height", 32))
        self.width = int(cfg_get(cfg, "width", 32))
        self.channels = int(cfg_get(cfg, "channels", 1))
        self.mask_policy = _validate_mask_policy(cfg)
        self.mask_mode = str(cfg_get(cfg, "mask_mode", "none"))
        self.mask_radius = float(cfg_get(cfg, "mask_radius", 0.45))

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
        self.domain_name = str(cfg_get(domain_cfg, "name", "unknown"))
        self.mask_policy = _validate_mask_policy(cfg)
        self.manifest: Dict[str, Any] | None = None
        self.field_kind: str | None = None
        self.grid: Dict[str, Any] | None = None
        self.mask_generated = False

        root = resolve_root(cfg_get(cfg, "root", "data"))
        cond_path = root / str(cfg_get(cfg, "cond_file", "cond.npy"))
        field_path = root / str(cfg_get(cfg, "field_file", "field.npy"))
        mask_path = root / str(cfg_get(cfg, "mask_file", "mask.npy"))

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

        manifest = load_manifest(root)
        if manifest is not None:
            validate_field_against_manifest(field, mask, manifest)
            self.manifest = dict(manifest)
            self.field_kind = str(manifest.get("field_kind"))
            grid = manifest.get("grid")
            if isinstance(grid, Mapping):
                self.grid = dict(grid)
            if mask is None and self.mask_policy != "forbid":
                self.mask_generated = True

        # REVIEW: normalize to batch-first arrays to keep the schema fixed.
        cond = _ensure_cond_batch(cond)
        field = _ensure_field_batch(field)
        if mask is None and self.mask_generated:
            mask = np.ones(field.shape[:3], dtype=bool)
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


def _parse_range(value: Any, label: str) -> tuple[float, float]:
    if value is None:
        raise ValueError(f"{label} is required")
    if hasattr(value, "__len__") and len(value) == 2:
        return float(value[0]), float(value[1])
    raise ValueError(f"{label} must be a pair, got {value}")


@register_dataset("csv_fields")
class CsvFieldsDataset(BaseDataset):
    """Dataset loader for per-sample CSV fields."""

    def __init__(
        self,
        *,
        cfg: Mapping[str, Any],
        domain_cfg: Mapping[str, Any] | None = None,
        seed: int | None = None,
    ) -> None:
        _ = seed
        self.name = "csv_fields"
        self.domain_name = str(cfg_get(domain_cfg, "name", "unknown"))
        self.mask_policy = _validate_mask_policy(cfg)
        self.manifest: Dict[str, Any] | None = None
        self.field_kind: str | None = "scalar"
        self.grid: Dict[str, Any] | None = None
        self.mask_generated = False

        cond_path = Path(str(cfg_get(cfg, "conditions_csv", ""))).expanduser()
        if not cond_path.is_absolute() and not cond_path.exists():
            alt = PROJECT_ROOT / cond_path
            if alt.exists():
                cond_path = alt
        fields_dir = Path(str(cfg_get(cfg, "fields_dir", ""))).expanduser()
        if not fields_dir.is_absolute() and not fields_dir.exists():
            alt = PROJECT_ROOT / fields_dir
            if alt.exists():
                fields_dir = alt
        if not cond_path.exists():
            raise FileNotFoundError(f"conditions_csv not found: {cond_path}")
        if not fields_dir.exists():
            raise FileNotFoundError(f"fields_dir not found: {fields_dir}")

        id_column = str(cfg_get(cfg, "id_column", "id")).strip()
        if not id_column:
            raise ValueError("dataset.id_column is required")
        target_column = cfg_get(cfg, "target_column", None)
        if target_column is not None and str(target_column).strip() == "":
            target_column = None

        feature_columns = cfg_get(cfg, "feature_columns", None)
        if isinstance(feature_columns, str):
            feature_columns = [feature_columns]
        if feature_columns is not None:
            feature_columns = [str(col) for col in feature_columns]

        field_components = cfg_get(cfg, "field_components", None)
        if isinstance(field_components, str):
            field_components = [field_components]
        if field_components is not None:
            field_components = [str(item).strip() for item in field_components if str(item).strip()]

        grid_cfg = cfg_get(cfg, "grid", None)
        if not isinstance(grid_cfg, Mapping):
            raise ValueError("dataset.grid is required and must be a mapping")
        height = cfg_get(grid_cfg, "H", None)
        width = cfg_get(grid_cfg, "W", None)
        if height is None or width is None:
            raise ValueError("dataset.grid.H and dataset.grid.W are required")
        height = int(height)
        width = int(width)
        if height <= 0 or width <= 0:
            raise ValueError("dataset.grid.H and dataset.grid.W must be > 0")

        domain_cfg = dict(domain_cfg or {})
        if self.domain_name == "sphere_grid":
            x_range = cfg_get(grid_cfg, "lon_range", None) or cfg_get(grid_cfg, "x_range", None)
            y_range = cfg_get(grid_cfg, "lat_range", None) or cfg_get(grid_cfg, "y_range", None)
            if x_range is None:
                x_range = cfg_get(domain_cfg, "lon_range", None) or cfg_get(domain_cfg, "x_range", None)
            if y_range is None:
                y_range = cfg_get(domain_cfg, "lat_range", None) or cfg_get(domain_cfg, "y_range", None)
            x_min, x_max = _parse_range(x_range, "dataset.grid.lon_range")
            y_min, y_max = _parse_range(y_range, "dataset.grid.lat_range")
        else:
            x_range = cfg_get(grid_cfg, "x_range", None) or cfg_get(domain_cfg, "x_range", None)
            y_range = cfg_get(grid_cfg, "y_range", None) or cfg_get(domain_cfg, "y_range", None)
            x_min, x_max = _parse_range(x_range, "dataset.grid.x_range")
            y_min, y_max = _parse_range(y_range, "dataset.grid.y_range")

        self._x_vals = np.linspace(x_min, x_max, width, dtype=np.float32)
        self._y_vals = np.linspace(y_min, y_max, height, dtype=np.float32)
        self.grid = {
            "H": int(height),
            "W": int(width),
            "x_range": [float(x_min), float(x_max)],
            "y_range": [float(y_min), float(y_max)],
        }

        self._sample_ids: list[str] = []
        self._conds: list[np.ndarray] = []
        self._field_paths: list[list[Path]] = []
        self._field_cache: dict[int, tuple[np.ndarray, np.ndarray | None]] = {}
        self._mask_stack: np.ndarray | None = None
        self._field_components: list[str] | None = field_components

        with cond_path.open("r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None:
                raise ValueError("conditions_csv must include a header row")
            header = [str(name) for name in reader.fieldnames]
            if id_column not in header:
                raise ValueError(f"conditions_csv missing id_column={id_column}")
            if feature_columns is None:
                feature_columns = [col for col in header if col != id_column and col != target_column]
            if not feature_columns:
                raise ValueError("No feature columns available for conditions_csv")
            self.cond_columns = list(feature_columns)
            for row_idx, row in enumerate(reader):
                sample_id = str(row.get(id_column, "")).strip()
                if not sample_id:
                    raise ValueError(f"conditions_csv missing id at row {row_idx + 2}")
                if sample_id in self._sample_ids:
                    raise ValueError(f"Duplicate id in conditions_csv: {sample_id}")
                cond_values = []
                for col in feature_columns:
                    if col not in row:
                        raise ValueError(f"conditions_csv missing column {col}")
                    value = row.get(col, "")
                    try:
                        cond_values.append(float(value))
                    except Exception as exc:
                        raise ValueError(f"conditions_csv invalid value for {col}: {value}") from exc
                cond = np.asarray(cond_values, dtype=np.float32)
                field_paths = self._resolve_field_paths(fields_dir, sample_id)
                for field_path in field_paths:
                    if not field_path.exists():
                        raise FileNotFoundError(f"field csv not found: {field_path}")
                self._sample_ids.append(sample_id)
                self._conds.append(cond)
                self._field_paths.append(field_paths)

        if not self._sample_ids:
            raise ValueError("conditions_csv contains no rows")

        mask_file = cfg_get(cfg, "mask_file", None)
        if mask_file is not None and str(mask_file).strip() != "":
            mask_path = Path(str(mask_file)).expanduser()
            if not mask_path.is_absolute() and not mask_path.exists():
                alt = PROJECT_ROOT / mask_path
                if alt.exists():
                    mask_path = alt
            if not mask_path.is_absolute() and not mask_path.exists():
                mask_path = cond_path.parent / mask_path
            if not mask_path.exists():
                raise FileNotFoundError(f"mask_file not found: {mask_path}")
            mask_arr = np.asarray(np.load(mask_path)).astype(bool)
            if mask_arr.ndim == 2:
                if mask_arr.shape != (height, width):
                    raise ValueError("mask_file shape does not match grid shape")
            elif mask_arr.ndim == 3:
                if mask_arr.shape[0] != len(self._sample_ids):
                    raise ValueError("mask_file sample count does not match conditions_csv")
                if mask_arr.shape[1:] != (height, width):
                    raise ValueError("mask_file spatial shape does not match grid shape")
            else:
                raise ValueError("mask_file must be 2D or 3D")
            self._mask_stack = mask_arr

    def __len__(self) -> int:
        return len(self._sample_ids)

    def _resolve_field_paths(self, fields_dir: Path, sample_id: str) -> list[Path]:
        if self._field_components is None:
            legacy_path = fields_dir / f"{sample_id}.csv"
            if legacy_path.exists():
                self._field_components = ["f"]
                self.field_kind = "scalar"
                return [legacy_path]
            fx_path = fields_dir / f"{sample_id}_fx.csv"
            fy_path = fields_dir / f"{sample_id}_fy.csv"
            if fx_path.exists() and fy_path.exists():
                self._field_components = ["fx", "fy"]
                self.field_kind = "vector"
                return [fx_path, fy_path]
            raise FileNotFoundError(f"field csv not found for sample: {sample_id}")

        components = self._field_components
        if len(components) == 1 and components[0] in {"f", ""}:
            self.field_kind = "scalar"
            return [fields_dir / f"{sample_id}.csv"]
        if len(components) == 2:
            self.field_kind = "vector"
        return [fields_dir / f"{sample_id}_{suffix}.csv" for suffix in components]

    def _load_component(self, field_path: Path) -> tuple[np.ndarray, np.ndarray]:
        x_vals: list[float] = []
        y_vals: list[float] = []
        f_vals: list[float] = []
        with field_path.open("r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None:
                raise ValueError(f"field csv missing header: {field_path}")
            header = [str(name) for name in reader.fieldnames]
            for col in ("x", "y", "f"):
                if col not in header:
                    raise ValueError(f"field csv missing column {col}: {field_path}")
            for row_idx, row in enumerate(reader):
                try:
                    x = float(row.get("x", ""))
                    y = float(row.get("y", ""))
                    f = float(row.get("f", ""))
                except Exception as exc:
                    raise ValueError(f"field csv invalid numeric at row {row_idx + 2}: {field_path}") from exc
                x_vals.append(x)
                y_vals.append(y)
                f_vals.append(f)

        if not x_vals:
            raise ValueError(f"field csv has no data rows: {field_path}")
        points = np.column_stack([np.asarray(x_vals), np.asarray(y_vals)])
        values = np.asarray(f_vals)
        valid = np.isfinite(points).all(axis=1) & np.isfinite(values)
        points = points[valid]
        values = values[valid]
        if points.size == 0:
            raise ValueError(f"field csv contains no finite values: {field_path}")

        if _griddata is None:
            raise RuntimeError("scipy is required for CSV field interpolation")

        xx, yy = np.meshgrid(self._x_vals, self._y_vals, indexing="xy")
        grid_f = _griddata(points, values, (xx, yy), method="linear")
        if np.any(~np.isfinite(grid_f)):
            grid_nearest = _griddata(points, values, (xx, yy), method="nearest")
            grid_f = np.where(np.isfinite(grid_f), grid_f, grid_nearest)

        mask = np.isfinite(grid_f)
        if not mask.all():
            grid_f = np.where(mask, grid_f, 0.0)
        return np.asarray(grid_f, dtype=np.float32), mask.astype(bool)

    def _load_field(self, index: int) -> tuple[np.ndarray, np.ndarray | None]:
        cached = self._field_cache.get(index)
        if cached is not None:
            return cached

        field_paths = self._field_paths[index]
        grids: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        for field_path in field_paths:
            grid_f, mask = self._load_component(field_path)
            grids.append(grid_f)
            masks.append(mask)

        mask = masks[0]
        for mask_item in masks[1:]:
            mask = mask & mask_item

        if self.mask_policy == "forbid" and not mask.all():
            raise ValueError("mask_policy=forbid but missing values detected in field")

        if self._mask_stack is not None:
            if self._mask_stack.ndim == 2:
                dataset_mask = self._mask_stack.astype(bool)
            else:
                dataset_mask = self._mask_stack[index].astype(bool)
            mask = mask & dataset_mask
        else:
            dataset_mask = None

        if self.mask_policy == "require":
            mask_out = mask.astype(bool)
        else:
            mask_out = None if mask.all() else mask.astype(bool)

        field = np.stack(grids, axis=-1)
        if mask_out is not None:
            field = np.where(mask_out[..., None], field, 0.0)
        cached = (field, mask_out)
        self._field_cache[index] = cached
        return cached

    def __getitem__(self, index: int) -> FieldSample:
        index = _normalize_index(index, len(self._sample_ids))
        cond = self._conds[index]
        field, mask = self._load_field(index)
        meta = {
            "sample_id": self._sample_ids[index],
            "dataset": self.name,
            "domain": self.domain_name,
            "field_path": str(self._field_paths[index]),
            "field_components": list(self._field_components or []),
            "field_kind": self.field_kind,
        }
        sample = FieldSample(cond=cond, field=field, mask=mask, meta=meta)
        _validate_sample(sample)
        return sample
