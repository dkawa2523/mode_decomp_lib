# coeff_meta contract

This document defines the minimum keys expected in `coeff_meta` to keep decomposers comparable and debuggable.

## Required keys
- `method`: decomposer name
- `field_shape`: list of input field shape
- `field_ndim`: original field ndim
- `field_layout`: e.g. `HW`, `HWC`, `NC`, `N1C`
- `channels`: number of channels
- `coeff_shape`: list of coefficient tensor shape
- `coeff_layout`: layout string, e.g. `CK`, `CHW`, `PHWC`
- `flatten_order`: `C` or `F`
- `complex_format`: `real`, `complex`, `real_imag`, `mag_phase`, `logmag_phase`
- `keep`: retention policy (usually `all`)

## Optional keys (when applicable)
- `coeff_dtype`: dtype string
- `projection`: method detail (e.g. `graph_laplacian_eigs`)
- `mask_policy` / `mask_valid_count`
- `n_modes`, `n_samples`, `eigvals`
- codec or coeff_post metadata

## Notes
- If a decomposer cannot provide a key, it should set the value to `None` instead of omitting it.
- Changes to key meaning or ordering are breaking changes and require an RFC.
