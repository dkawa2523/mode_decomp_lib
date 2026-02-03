# Special Function Suite Notes

## Summary CSV skip_code

The `tools/special_function_suite.py` summary includes `skip_code` to keep skip reasons compact:

- `missing_dependency`: optional backend/import not available.
- `rank_deficient`: least-squares basis is rank-deficient (reduce `l_max`, check grid/mask).
- `missing_domain_mask`: decomposer requires a domain mask and none is present.
- `missing_dataset_mask`: decomposer requires a dataset mask and none is present.
- `empty_weights`: computed weights are all zero after masking.
- `insufficient_samples`: more modes than valid samples for least squares.
- `build_error`: error while constructing the decomposer.
- `fit_error`: error during `fit`.
- `transform_error`: error during `transform`.
- `transform_batch_error`: error while transforming multiple samples.

These codes are designed for quick filtering; `skip_reason` preserves the full exception text.

## sphere_grid range utility

The suite uses `mode_decomp_ml.domain.sphere_grid` helpers to standardize ranges:

- `sphere_grid_lon_range(n_lon, angle_unit)` returns a non-overlapping longitude range.
- `sphere_grid_lat_range(n_lat, angle_unit)` returns a full latitude range (poles included).
- `sphere_grid_domain_cfg(...)` provides a canonical domain config for `sphere_grid`.
