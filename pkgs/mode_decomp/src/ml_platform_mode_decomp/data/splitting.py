"""Deterministic sample holdouts; no fitted representation crosses a split."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .schema import FieldDataset


def validate_split_settings(
    config: Mapping[str, Any],
    *,
    cross_validation: bool = False,
    nested: bool | None = None,
    calibration: Mapping[str, Any] | None = None,
) -> float:
    """Validate the small split vocabulary and return the time gap in seconds."""
    allowed = {"method", "seed", "group_column", "time_column", "gap_seconds"}
    allowed |= (
        {"n_splits"}
        if cross_validation
        else {"valid_size", "selection_size", "selection_method", "valid_filter_column", "valid_filter_value"}
    )
    unknown = set(config) - allowed
    if unknown:
        raise ValueError(f"Unsupported {'CV' if cross_validation else 'split'} settings: {sorted(unknown)}")
    method = config.get("method", "random")
    methods = {"random", "group", "time"} | (set() if cross_validation else {"fixed"})
    if not isinstance(method, str) or method not in methods:
        raise ValueError(f"Split method must be one of {sorted(methods)}")
    if not cross_validation:
        if method == "fixed" and "valid_size" in config:
            raise ValueError(
                "split.valid_size is not used by a fixed split; membership comes from valid_filter_column/value"
            )
        if method != "fixed" and {"valid_filter_column", "valid_filter_value"}.intersection(config):
            raise ValueError("split.valid_filter_column/value require method: fixed")
        if "selection_method" in config and (
            not isinstance(config["selection_method"], str)
            or config["selection_method"] not in {"random", "group", "time"}
        ):
            raise ValueError("split.selection_method must be random, group, or time")
        if nested is False and {"selection_size", "selection_method"}.intersection(config):
            raise ValueError("split.selection_size/selection_method are not used without inner holdout selection")
    if cross_validation or nested is not None:
        used_methods = {method}
        if nested:
            used_methods.add(config.get("selection_method", method))
        if calibration and calibration.get("enabled", False):
            used_methods.add(calibration["calibration_method"])
        for selection in ("group", "time"):
            if selection not in used_methods and f"{selection}_column" in config:
                raise ValueError(f"split.{selection}_column is not used by the configured split methods")
    if "seed" in config:
        _seed(config["seed"], "split seed")
    if cross_validation:
        count = config.get("n_splits")
        if isinstance(count, (bool, np.bool_)) or not isinstance(count, (int, np.integer)) or count < 2:
            raise ValueError("CV n_splits must be an integer of at least two")
    if "gap_seconds" not in config:
        return 0.0
    if config.get("method", "random") != "time":
        raise ValueError("gap_seconds is supported only by time splits")
    value = config["gap_seconds"]
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError("gap_seconds must be a finite nonnegative number of seconds")
    try:
        gap = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError("gap_seconds must be a finite nonnegative number of seconds") from exc
    if not math.isfinite(gap) or gap < 0:
        raise ValueError("gap_seconds must be a finite nonnegative number of seconds")
    return gap


def _seed(value: Any, label: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not 0 <= value < 2**32:
        raise ValueError(f"{label} must be an integer between zero and 4294967295")
    return int(value)


def _fraction(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value < 1:
        raise ValueError(f"{name} must be a number strictly between zero and one.")
    return float(value)


def _column(dataset: FieldDataset, name: Any) -> np.ndarray:
    if not isinstance(name, str) or name not in dataset.sample_metadata:
        raise ValueError(f"Split metadata column is required and must exist: {name!r}.")
    values = np.asarray(dataset.sample_metadata[name])
    if np.asarray(pd.isna(values)).any() or any(str(value).strip() == "" for value in values):
        raise ValueError(f"Split metadata column {name!r} contains missing values.")
    return values


def _timestamps(values: np.ndarray) -> np.ndarray:
    timestamps = []
    for value in values:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"Invalid split timestamp: {value!r}.") from exc
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        timestamps.append(parsed.timestamp())
    return np.asarray(timestamps)


def _purge_time_fit(times: np.ndarray, fit: np.ndarray, validation: np.ndarray, gap: float) -> np.ndarray:
    """Keep past samples at least gap seconds before the first validation time."""
    first = float(times[validation].min())
    kept = fit[(times[fit] < first) & (times[fit] <= first - gap)]
    if not len(kept):
        raise ValueError("Time gap leaves an empty training partition; reduce gap_seconds or use more history")
    return kept


def _partition(
    dataset: FieldDataset,
    indices: np.ndarray,
    config: Mapping[str, Any],
    method: str,
    fraction: float | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if fraction is None:
        if method != "fixed":
            raise ValueError("A generated holdout requires a split fraction.")
        fraction = 0.0  # Fixed membership does not use a split fraction.
    rng = np.random.default_rng(seed)
    if method == "random":
        shuffled = rng.permutation(indices)
        count = min(max(1, round(len(indices) * fraction)), len(indices) - 1)
        train, test = shuffled[count:], shuffled[:count]
    elif method == "group":
        groups = _column(dataset, config.get("group_column"))[indices].astype(str)
        unique = np.unique(groups)
        if len(unique) < 2:
            raise ValueError("group split requires at least two distinct groups in each split level.")
        unique = rng.permutation(unique)
        count = min(max(1, round(len(unique) * fraction)), len(unique) - 1)
        mask = np.isin(groups, unique[:count])
        train, test = indices[~mask], indices[mask]
    elif method == "time":
        timestamps = _timestamps(_column(dataset, config.get("time_column"))[indices])
        unique = np.unique(timestamps)
        if len(unique) < 2:
            raise ValueError("time split requires at least two distinct timestamps in each split level.")
        target = len(indices) * fraction
        cutoff = min(unique[1:], key=lambda value: (abs(int((timestamps >= value).sum()) - target), value))
        mask = timestamps >= cutoff
        test = indices[mask]
        fit_local = _purge_time_fit(
            timestamps, np.flatnonzero(~mask), np.flatnonzero(mask), float(config.get("gap_seconds", 0))
        )
        train = indices[fit_local]
    elif method == "fixed":
        values = _column(dataset, config.get("valid_filter_column"))[indices]
        filter_value = config.get("valid_filter_value")
        if filter_value is None or str(filter_value) == "":
            raise ValueError("fixed split requires valid_filter_value.")
        mask = values.astype(str) == str(filter_value)
        train, test = indices[~mask], indices[mask]
    else:
        raise ValueError("split method must be random, group, time, or fixed.")
    if len(train) == 0 or len(test) == 0:
        raise ValueError(f"{method} split produced an empty partition.")
    return np.sort(train), np.sort(test)


def split_dataset(
    dataset: FieldDataset,
    cfg: Mapping[str, Any],
    seed: int = 42,
    *,
    nested: bool = True,
    calibration: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    config = cfg.get("split", cfg)
    if not isinstance(config, Mapping):
        raise ValueError("split must be a mapping.")
    gap = validate_split_settings(config)
    seed = _seed(config.get("seed", seed), "seed")
    if not isinstance(nested, bool):
        raise ValueError("nested must be boolean.")
    if len(dataset) < (3 if nested else 2):
        raise ValueError(
            "Nested holdout requires at least three samples." if nested else "Holdout requires at least two samples."
        )
    method = config.get("method", "random")
    selection_method = config.get("selection_method", method if method != "fixed" else "")
    if nested and selection_method not in {"random", "group", "time"}:
        raise ValueError("split.selection_method must explicitly be random/group/time for a fixed outer split.")
    if nested and method in {"group", "time"} and selection_method != method:
        raise ValueError("Selection must preserve the outer group/time split method")
    validate_split_settings(config, nested=nested, calibration=calibration)
    outer_fraction = None if method == "fixed" else _fraction(config.get("valid_size", 0.2), "split.valid_size")
    train, test = _partition(dataset, np.arange(len(dataset)), config, method, outer_fraction, int(seed))
    result: dict[str, Any] = {
        "schema_version": 1,
        "method": method,
        "nested": nested,
        "seed": int(seed),
        "dataset_revision": dataset.revision,
        "train_indices": train.tolist(),
        "test_indices": test.tolist(),
        "train_ids": dataset.sample_ids[train].tolist(),
        "test_ids": dataset.sample_ids[test].tolist(),
        "valid_size": outer_fraction,
        "actual_valid_size": len(test) / len(dataset),
    }
    if gap:
        purged = np.setdiff1d(np.arange(len(dataset)), np.concatenate((train, test)))
        result.update(
            gap_seconds=gap, outer_purged_indices=purged.tolist(), outer_purged_ids=dataset.sample_ids[purged].tolist()
        )
    if calibration and calibration.get("enabled", False):
        outer_train = train
        calibration_method = calibration["calibration_method"]
        if method in {"group", "time"} and calibration_method != method:
            raise ValueError("Calibration must preserve the outer group/time split method")
        proper, heldout = _partition(
            dataset,
            outer_train,
            config,
            calibration_method,
            _fraction(calibration["calibration_size"], "uncertainty.calibration_size"),
            _seed(calibration["calibration_seed"], "calibration_seed"),
        )
        purged = np.setdiff1d(outer_train, np.concatenate((proper, heldout)))
        train = proper
        result.update(
            outer_train_indices=outer_train.tolist(),
            outer_train_ids=dataset.sample_ids[outer_train].tolist(),
            train_indices=train.tolist(),
            train_ids=dataset.sample_ids[train].tolist(),
            calibration_indices=heldout.tolist(),
            calibration_ids=dataset.sample_ids[heldout].tolist(),
            calibration_method=calibration_method,
            calibration_size=calibration["calibration_size"],
            calibration_seed=int(calibration["calibration_seed"]),
        )
        if len(purged):
            result.update(
                calibration_purged_indices=purged.tolist(), calibration_purged_ids=dataset.sample_ids[purged].tolist()
            )
    if nested:
        inner_fraction = _fraction(config.get("selection_size", 0.2), "split.selection_size")
        fit, selection = _partition(dataset, train, config, selection_method, inner_fraction, int(seed) + 1)
        result.update(
            selection_method=selection_method,
            selection_size=inner_fraction,
            fit_indices=fit.tolist(),
            selection_indices=selection.tolist(),
            fit_ids=dataset.sample_ids[fit].tolist(),
            selection_ids=dataset.sample_ids[selection].tolist(),
            actual_selection_size=len(selection) / len(train),
        )
        if gap:
            purged = np.setdiff1d(train, np.concatenate((fit, selection)))
            result.update(inner_purged_indices=purged.tolist(), inner_purged_ids=dataset.sample_ids[purged].tolist())
    canonical = json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    result["fingerprint"] = hashlib.sha256(canonical.encode()).hexdigest()
    return result


def cross_validation_folds(dataset: FieldDataset, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Split only the supplied training Dataset; indices are local to that Dataset.

    Time folds use expanding past-only training and nonoverlapping future blocks.
    gap_seconds purges only adjacent training history; no future-data embargo is needed.
    """
    count = config["n_splits"]
    method = config["method"]
    gap = validate_split_settings(config, cross_validation=True)
    rng = np.random.default_rng(config["seed"])
    all_indices = np.arange(len(dataset))
    purged_partitions: list[np.ndarray] = []
    if method == "random":
        if len(dataset) < count:
            raise ValueError("Cross-validation needs at least n_splits training samples")
        blocks = np.array_split(rng.permutation(all_indices), count)
        partitions = [(np.setdiff1d(all_indices, block), block) for block in blocks]
    elif method == "group":
        groups = _column(dataset, config.get("group_column")).astype(str)
        unique = np.unique(groups)
        if len(unique) < count:
            raise ValueError("Group CV needs at least n_splits distinct training groups")
        blocks = np.array_split(rng.permutation(unique), count)
        partitions = [(all_indices[~np.isin(groups, block)], all_indices[np.isin(groups, block)]) for block in blocks]
    elif method == "time":
        times = _timestamps(_column(dataset, config.get("time_column")))
        unique = np.unique(times)
        if len(unique) < count + 1:
            raise ValueError("Time CV needs at least n_splits+1 distinct training timestamps")
        blocks = np.array_split(unique, count + 1)
        partitions = []
        for block in blocks[1:]:
            validation = all_indices[np.isin(times, block)]
            past = all_indices[times < block[0]]
            fit = _purge_time_fit(times, past, validation, gap)
            partitions.append((fit, validation))
            purged_partitions.append(np.setdiff1d(past, fit))
    else:
        raise ValueError("Cross-validation method must be random, group, or time")
    folds = [
        {
            "fold": index,
            "fit_indices": np.sort(fit).tolist(),
            "validation_indices": np.sort(validation).tolist(),
            "fit_ids": dataset.sample_ids[np.sort(fit)].tolist(),
            "validation_ids": dataset.sample_ids[np.sort(validation)].tolist(),
        }
        for index, (fit, validation) in enumerate(partitions)
    ]
    if gap:
        for fold, purged in zip(folds, purged_partitions, strict=True):
            fold.update(gap_seconds=gap, purged_indices=purged.tolist(), purged_ids=dataset.sample_ids[purged].tolist())
    return folds
