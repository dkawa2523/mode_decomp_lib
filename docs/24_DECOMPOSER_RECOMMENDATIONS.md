# Decomposer recommendations

This document collects practical defaults for stable operation. Adjust based on your dataset size and budget.

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
