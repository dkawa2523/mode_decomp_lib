# Decomposer recommendations

This document collects practical defaults for stable operation. Adjust based on your dataset size and budget.

## offset_split (offset vs residual)
When the dataset has a strong per-sample DC/offset component (e.g. `field = offset + small fluct + noise`),
training on full coefficients can waste capacity on predicting the offset while underfitting residual structure.

Enable `offset_split` to separate:
- `offset`: per-channel weighted mean on the valid mask
- `residual`: `field - offset` (then decomposed as usual)

Recommended starting point:
- `offset_split.enabled`: `auto`
- `offset_split.f_offset`: `5.0` (enable when offset RMS is >= 5x residual RMS, median across samples)
- `codec`: `auto_codec_v1` (required when split is enabled)

Notes:
- `energy_cumsum` / `n_components_required` are computed on the **residual slice** when split is enabled, so the
  reported "required components" stays comparable between offset-dominant and non-offset datasets.

## graph_fourier
Recommended starting point for grid sizes up to ~128x128.

- `n_modes`: 32 to 128 for small grids; 128 to 256 for medium grids. Use `auto` to pick a safe default.
- `connectivity`: 4 for stability, 8 for smoother bases (slower).
- `laplacian_type`: `combinatorial` for most use cases.
- `mask_policy`: `allow_full` unless a fixed mask is required.
- `solver`: `auto`.
- `dense_threshold`: set to `H*W` upper bound where dense eigendecomposition is still acceptable.
- `eigsh_tol`: `1.0e-6`.
- `eigsh_maxiter`: `1000` to `2000`.

## laplace_beltrami
Mesh eigenbasis (costly). Start with a small mode count.

- `n_modes`: 16 to 64 for initial runs. Use `auto` to pick a safe default.
- `laplacian_type`: `cotangent` only.
- `mass_type`: `lumped` only.
- `boundary_condition`: `neumann`.
- `mask_policy`: `allow` unless you must enforce a mask.
- `solver`: `auto`.
- `dense_threshold`: 2000 to 5000 vertices for dense; otherwise use eigsh.
- `eigsh_tol`: `1.0e-6`.
- `eigsh_maxiter`: `2000` or higher for large meshes.

## fourier_bessel
Disk-domain decomposer.

- `m_max`: 4 to 8 for quick experiments; 12+ for higher detail.
- `n_max`: 4 to 8 for quick experiments; 12+ for higher detail.
- `ordering`: `m_then_n`.
- `normalization`: `orthonormal` for stable coefficients.
- `boundary_condition`: `dirichlet` for zero boundary; `neumann` for derivative zero.
- `mask_policy`: `ignore_masked_points` when a disk mask exists.

## pseudo_zernike
Disk-domain decomposer (pseudo-zernike family). Less strict index parity than standard Zernike.

- `n_max`: 6 to 10 for quick experiments; 12+ for higher detail (watch rank/LS stability).
- `m_max`: default `n_max`. Reduce (e.g. 4 to 8) to cap angular complexity.
- `ordering`: `n_then_m` only.
- `normalization`: `orthonormal` recommended for stable scales.
- `mask_policy`: fixed behavior via weighted LS (masked points ignored via zero weights).

## fourier_jacobi
Disk-domain decomposer (Fourier in angle + Jacobi in radius).

- `m_max`: 4 to 8 for quick experiments; 12+ for higher angular detail.
- `k_max`: 4 to 8 for quick experiments; 12+ for higher radial detail.
- `ordering`: `m_then_k` only.
- `normalization`: `orthonormal` recommended for stable coefficients.
- `mask_policy`: `ignore_masked_points` (masked points ignored via zeroed weights).

## polar_fft
Approximate disk/annulus-domain decomposer (polar resample + rFFT(theta) + DCT(r)).

- `n_r`: 32 to 64 for 64x64 grids; start with `n_r=32`.
- `n_theta`: 64 to 128; choose a power-of-two for FFT speed.
- `interpolation`: bilinear only (v1).
- `codec`: use `fft_complex_codec_v1` to vectorize complex coefficients.
- Note: interpolation mixes boundary values; results depend on how the dataset fills values outside the disk.
- Annulus: uses `r_inner_norm` to skip the inner hole; inner region is masked to 0 on inverse.

## disk_slepian
Disk-domain bandlimited eigenbasis (Slepian/PSWF-like on a discrete grid).

- `n_modes`: 8 to 32 for initial runs (cost grows with mask size).
- `freq_radius`: 3 to 8 (smaller is cheaper, more concentrated; larger captures more detail).
- `solver`: `eigsh` for medium/large grids; `dense` only for small grids.
- `dense_threshold`: keep small (e.g. 256 to 1024) to avoid O(N^3) costs.
- Note: this basis optimizes concentration under a bandlimit, not exact reconstruction of simple fields with few modes.

## dict_learning
Sparse coding. Sensitive to dictionary size and solver choices.

- `n_components`: start at 2x to 4x the number of modes you expect.
- `alpha`: 0.5 to 2.0 for most cases.
- `max_iter`: 200 to 500.
- `fit_algorithm`: `lars` (default) or `cd` for large data.
- `transform_algorithm`: `omp` (fast), `lasso_cd` (stable).
- If OMP warnings appear, reduce `n_components` or increase `alpha`.
- Practical starting point for small grids: `n_components=64`, `alpha=1.0`, `transform_algorithm=lasso_cd`.
- `warn_on_omp_fallback`: when true (default), the decomposer will switch from `omp` to `lasso_cd` if `n_components`
  exceeds the sample/feature count to avoid unstable sparse codes.

## autoencoder
See `docs/25_AUTOENCODER_GUIDE.md` for detailed guidance.

## rbf_expansion
General-purpose basis for grid domains with arbitrary or varying masks.

- `stride`: 3 to 8. Larger stride reduces mode count and improves conditioning.
- `kernel`: `gaussian` for smooth fields; `thin_plate` for broader support (can be less stable).
- `length_scale` (gaussian): start at ~0.1 to 0.3 in normalized coordinates (depends on `x_range/y_range`).
- `ridge_alpha`: `1.0e-6` to `1.0e-3` to stabilize near-collinear centers and heavy masking.
- `mask_policy`: `ignore_masked_points` (fixed behavior via weighted ridge; per-sample masks supported).

## pod_joint
Joint POD across channels (vector-friendly). Captures cross-channel correlation by sharing spatial modes.

- `n_modes`: 8 to 64 for small grids; use lower values when masks vary and rank drops.
- `mask_policy`: use `zero_fill` if sample masks vary; use `error` when masks must be absent.
- `inner_product`: `euclidean` as default; `domain_weights` when comparing across non-uniform domains (requires weights).

## gappy_graph_fourier
Fixed graph Fourier basis (built on domain mask) with per-sample coefficient estimation on observed points.

- `n_modes`: 16 to 128 for small grids; must satisfy `valid_points >= n_modes` per sample.
- `ridge_alpha`: `1.0e-6` to `1.0e-3` (increase when masks are sparse or ill-conditioned).
- `mask_policy`: `allow_full` (basis ignores dataset masks; transform uses sample mask).
- Other parameters follow `graph_fourier`.

## helmholtz_poisson
Vector-field Helmholtz decomposition on rectangle domains.

- `boundary_condition`: `periodic` (FFT, exact discrete); `dirichlet`/`neumann` (Poisson solves, approximate under FD div/curl).
- `mask_policy`: `error` (v1 does not support masked Poisson/Hodge).
- Use smoother fields or higher resolution for `dirichlet/neumann` to reduce boundary artifacts.

## pod_em
POD with varying masks via iterative imputation (EM/ALS style).

- `n_modes`: start small (8 to 32). Must satisfy `observed_entries >= n_modes` per sample.
- `n_iter`: 5 to 20. More iterations help when masks are sparse.
- `ridge_alpha`: `1.0e-6` to `1.0e-3` for stability under heavy masking.
- `init`: `mean_fill` is usually more stable than `zero_fill`.
- `inner_product`: `euclidean` as default; `domain_weights` when weights are meaningful/comparable.

## pod_joint_em
Joint POD (across channels) with varying masks via iterative imputation (EM/ALS style).

- Same knobs as `pod_em`.
- Prefer `pod_joint_em` for vector fields when cross-channel correlation matters.
