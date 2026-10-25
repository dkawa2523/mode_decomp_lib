"""Purge adjacent training history without changing held-out time blocks."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from ml_platform_mode_decomp.data import FieldDataset, split_dataset
from ml_platform_mode_decomp.data.splitting import cross_validation_folds
from ml_platform_mode_decomp.geometry import DomainSpec
from ml_platform_mode_decomp.policy import resolve_config
from ml_platform_mode_decomp.surrogate import FieldSurrogate
from ml_platform_mode_decomp.training.candidate_training import selection_options
from ml_platform_mode_decomp.training.search import search_candidate


def timed_data(minutes=None):
    values = np.repeat([0, 10, 30, 60, 90, 120, 150, 180, 240, 300, 360, 420], 2)
    if minutes is not None:
        values = np.asarray(minutes)
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return FieldDataset(
        fields=(1 + values / 100).reshape(-1, 1, 1, 1),
        conditions=values[:, None],
        condition_columns=("elapsed_minutes",),
        sample_ids=np.asarray([f"s{i:02d}" for i in range(len(values))]),
        geometry=DomainSpec("rectangle", {"x": np.zeros((1, 1)), "y": np.zeros((1, 1))}, None, np.ones((1, 1)), {}),
        sample_metadata={
            "time": np.asarray([(start + timedelta(minutes=int(value))).isoformat() for value in values]),
            "group": np.asarray([f"g{i // 2}" for i in range(len(values))]),
        },
        revision="purged_time_fixture",
    )


def elapsed(dataset, indices):
    return dataset.conditions[np.asarray(indices, dtype=int), 0]


def test_time_gap_purges_outer_and_inner_history_and_preserves_timestamp_groups():
    dataset = timed_data()
    config = {"method": "time", "time_column": "time", "valid_size": 0.25, "selection_size": 0.25}
    original = split_dataset(dataset, config)
    split = split_dataset(dataset, {**config, "gap_seconds": 90 * 60})
    assert split["test_ids"] == original["test_ids"]
    assert elapsed(dataset, split["outer_purged_indices"]).tolist() == [240, 240]
    assert elapsed(dataset, split["inner_purged_indices"]).tolist() == [90, 90, 120, 120]
    for fit_name, validation_name in (("train", "test"), ("fit", "selection")):
        fit, validation = split[f"{fit_name}_indices"], split[f"{validation_name}_indices"]
        assert min(elapsed(dataset, validation)) - max(elapsed(dataset, fit)) >= 90
        assert set(elapsed(dataset, fit)).isdisjoint(elapsed(dataset, validation))
    assert 60 in elapsed(dataset, split["fit_indices"])  # Exactly 90 minutes before selection is retained.
    accounted = [
        split[key]
        for key in ("fit_indices", "selection_indices", "test_indices", "outer_purged_indices", "inner_purged_indices")
    ]
    assert sorted(index for part in accounted for index in part) == list(range(len(dataset)))


def test_gap_is_elapsed_seconds_with_an_inclusive_minimum_not_a_row_count():
    dataset = timed_data([0, 0, 30, 30, 60, 60, 90, 90])
    config = {"method": "time", "time_column": "time", "valid_size": 0.25}
    exact = split_dataset(dataset, {**config, "gap_seconds": 1800}, nested=False)
    extra = split_dataset(dataset, {**config, "gap_seconds": 1800.1}, nested=False)
    assert elapsed(dataset, exact["train_indices"]).tolist()[-2:] == [60, 60]
    assert elapsed(dataset, extra["outer_purged_indices"]).tolist() == [60, 60]
    assert exact["test_ids"] == extra["test_ids"]
    assert split_dataset(dataset, {**config, "gap_seconds": 0}, nested=False) == split_dataset(
        dataset, config, nested=False
    )


def test_time_cv_uses_the_same_purge_rule_without_moving_validation_blocks():
    dataset = timed_data()
    config = {"method": "time", "time_column": "time", "n_splits": 3, "seed": 5}
    original = cross_validation_folds(dataset, config)
    folds = cross_validation_folds(dataset, {**config, "gap_seconds": 45 * 60})
    assert [f["validation_ids"] for f in folds] == [f["validation_ids"] for f in original]
    assert elapsed(dataset, folds[0]["purged_indices"]).tolist() == [30, 30]
    assert elapsed(dataset, folds[1]["purged_indices"]).tolist() == [120, 120]
    for fold in folds:
        assert min(elapsed(dataset, fold["validation_indices"])) - max(elapsed(dataset, fold["fit_indices"])) >= 45
        assert set(fold["purged_ids"]).isdisjoint(fold["fit_ids"] + fold["validation_ids"])


def test_cv_inherits_outer_gap_and_every_trial_fits_only_the_purged_fold(monkeypatch):
    dataset = timed_data()
    cfg = resolve_config(
        {
            "task": "mode_decomp_pipeline",
            "split": {"method": "time", "time_column": "time", "valid_size": 0.25, "gap_seconds": 2700},
            "model": {
                "candidates": [
                    {
                        "id": "pod",
                        "representation": {"decomposition": {"name": "pod_svd", "params": {"n_modes": 1}}},
                        "regression": {"name": "ridge", "params": {"alpha": 0}},
                    }
                ]
            },
            "tuning": {"enabled": True, "cv": {"n_splits": 3}},
        }
    )
    assert cfg["tuning"]["cv"]["gap_seconds"] == 2700
    split = split_dataset(dataset, cfg["split"], nested=False)
    training = dataset.subset(split["train_indices"])
    calls = []
    original = FieldSurrogate.fit

    def fit(self, fold_training):
        calls.append(fold_training.sample_ids.tolist())
        return original(self, fold_training)

    monkeypatch.setattr(FieldSurrogate, "fit", fit)
    outcome = search_candidate(cfg, cfg["model"]["candidates"][0], training, selection_options(cfg))
    assert outcome.report is not None
    assert calls == [fold["fit_ids"] for fold in outcome.folds]
    for fold in outcome.folds:
        assert set(fold["fit_ids"]).isdisjoint(fold["purged_ids"] + split["test_ids"])


def test_large_gap_fails_instead_of_falling_back_to_adjacent_history():
    dataset = timed_data()
    with pytest.raises(ValueError, match="empty training"):
        split_dataset(dataset, {"method": "time", "time_column": "time", "gap_seconds": 1e9}, nested=False)
    with pytest.raises(ValueError, match="empty training"):
        cross_validation_folds(
            dataset, {"method": "time", "time_column": "time", "n_splits": 3, "seed": 1, "gap_seconds": 1e9}
        )


def test_group_partition_is_unchanged_and_rejects_a_time_gap():
    dataset = timed_data()
    config = {"method": "group", "group_column": "group"}
    first = split_dataset(dataset, config, seed=15)
    assert first == split_dataset(dataset, config, seed=15)
    groups = dataset.sample_metadata["group"]
    assert set(groups[first["train_indices"]]).isdisjoint(groups[first["test_indices"]])
    with pytest.raises(ValueError, match="only by time"):
        split_dataset(dataset, {**config, "gap_seconds": 0})


@pytest.mark.parametrize("gap", [True, -1, "1h", np.inf, np.nan])
def test_gap_settings_reject_invalid_units_and_nonfinite_values(gap):
    with pytest.raises(ValueError, match="finite nonnegative"):
        split_dataset(timed_data(), {"method": "time", "time_column": "time", "gap_seconds": gap})


def test_temporal_settings_do_not_silently_ignore_unsupported_options_or_weaken_cv():
    dataset = timed_data()
    with pytest.raises(ValueError, match="Unsupported split"):
        split_dataset(dataset, {"method": "time", "time_column": "time", "embargo_seconds": 30})
    with pytest.raises(ValueError, match="Unsupported CV"):
        cross_validation_folds(
            dataset, {"method": "time", "time_column": "time", "n_splits": 3, "seed": 1, "gap_count": 1}
        )
    with pytest.raises(ValueError, match="Selection must preserve"):
        split_dataset(
            dataset, {"method": "time", "time_column": "time", "gap_seconds": 1, "selection_method": "random"}
        )
    with pytest.raises(ValueError, match="smaller than"):
        resolve_config(
            {
                "task": "mode_decomp_pipeline",
                "split": {"method": "time", "time_column": "time", "gap_seconds": 60},
                "tuning": {"enabled": True, "cv": {"gap_seconds": 30}},
            }
        )


def test_equal_instants_with_different_offsets_are_purged_together():
    dataset = timed_data([0, 0, 30, 30, 60, 60, 90, 90])
    values = dataset.sample_metadata["time"].copy()
    values[4] = "2026-01-01T10:00:00+09:00"  # Same instant as 01:00 UTC.
    dataset = replace(dataset, sample_metadata={**dataset.sample_metadata, "time": values})
    split = split_dataset(dataset, {"method": "time", "time_column": "time", "gap_seconds": 1801}, nested=False)
    assert split["outer_purged_ids"] == ["s04", "s05"]
