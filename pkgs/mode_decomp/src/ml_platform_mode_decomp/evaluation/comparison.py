"""Paired field-loss comparisons on one fixed test split at a time."""

from itertools import combinations

import numpy as np
import pandas as pd


_BOOTSTRAP_VALUES = 262_144


def paired_comparisons(
    samples: pd.DataFrame, *, draws=1000, confidence=0.95, seed=42, independent_samples=True
) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(seed)
    grouped = samples.groupby(["split_seed", "model_seed"], sort=True) if not samples.empty else []
    for (split_seed, model_seed), repeated in grouped:
        for first, second in combinations(sorted(repeated.candidate_id.unique()), 2):
            left = repeated[repeated.candidate_id == first].set_index("sample_id").sort_index()
            right = repeated[repeated.candidate_id == second].set_index("sample_id").sort_index()
            if not left.index.equals(right.index):
                raise ValueError("Paired comparisons require identical reference sample IDs")
            for name in ("sample_weight", "valid_points", "coverage"):
                if not np.allclose(left[name], right[name]):
                    raise ValueError("Paired comparisons require the same evaluation support and measure")
            active = left.sample_weight.to_numpy(dtype=float) > 0
            left, right = left[active], right[active]
            weights = left.sample_weight.to_numpy(dtype=float)
            loss_a, loss_b = left.mse.to_numpy(dtype=float), right.mse.to_numpy(dtype=float)
            difference = float(np.sqrt(weights @ loss_a) - np.sqrt(weights @ loss_b))
            row = {
                "split_seed": split_seed,
                "model_seed": model_seed,
                "candidate_a": first,
                "candidate_b": second,
                "rmse_a_minus_b": difference,
                "interval_scope": "conditional_on_fitted_models_and_this_test_split",
                "resampling_unit": "group"
                if "group" in left
                else "sample"
                if independent_samples
                else "not_assumed_independent",
            }
            units = left["group"].astype(str).to_numpy() if "group" in left else left.index.to_numpy()
            unique, inverse = np.unique(units, return_inverse=True)
            row["independent_units"] = len(unique)
            if draws and len(unique) >= 2 and (independent_samples or "group" in left):
                # Resample complete groups together; never count pixels or coefficients as observations.
                unit_mse_a = np.bincount(inverse, weights=weights * loss_a, minlength=len(unique))
                unit_mse_b = np.bincount(inverse, weights=weights * loss_b, minlength=len(unique))
                unit_weight = np.bincount(inverse, weights=weights, minlength=len(unique))
                differences = np.empty(draws)
                # Keep the same RNG draw order without allocating draws × units at once.
                batch_size = max(1, _BOOTSTRAP_VALUES // len(unique))
                for start in range(0, draws, batch_size):
                    indices = rng.integers(0, len(unique), (min(batch_size, draws - start), len(unique)))
                    total = unit_weight[indices].sum(axis=1)
                    block = np.sqrt(unit_mse_a[indices].sum(axis=1) / total)
                    block -= np.sqrt(unit_mse_b[indices].sum(axis=1) / total)
                    differences[start : start + len(block)] = block
                if len(differences):
                    alpha = (1 - confidence) / 2
                    row.update(
                        ci_low=float(np.quantile(differences, alpha)),
                        ci_high=float(np.quantile(differences, 1 - alpha)),
                        confidence_level=confidence,
                    )
            rows.append(row)
    return pd.DataFrame(
        rows,
        columns=[
            "split_seed",
            "model_seed",
            "candidate_a",
            "candidate_b",
            "rmse_a_minus_b",
            "ci_low",
            "ci_high",
            "confidence_level",
            "independent_units",
            "resampling_unit",
            "interval_scope",
        ],
    )
