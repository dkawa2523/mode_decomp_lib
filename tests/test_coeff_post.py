from __future__ import annotations

import numpy as np
import pytest

from mode_decomp_ml.plugins.coeff_post import build_coeff_post


def test_standardize_roundtrip() -> None:
    rng = np.random.default_rng(0)
    A = rng.normal(size=(16, 5)).astype(np.float32)

    coeff_post = build_coeff_post({"name": "standardize"})
    coeff_post.fit(A, split="train")
    Z = coeff_post.transform(A)
    A_hat = coeff_post.inverse_transform(Z)

    assert A_hat.shape == A.shape
    assert np.allclose(A_hat, A, atol=1e-6)


def test_pca_inverse_and_latent_dim() -> None:
    rng = np.random.default_rng(1)
    A = rng.normal(size=(32, 6)).astype(np.float32)

    coeff_post = build_coeff_post({"name": "pca", "n_components": 3, "whiten": False})
    coeff_post.fit(A, split="train")
    Z = coeff_post.transform(A)
    A_hat = coeff_post.inverse_transform(Z)

    assert Z.shape[1] == 3
    assert coeff_post.latent_dim == 3
    assert A_hat.shape == A.shape
    assert np.isfinite(A_hat).all()


def test_quantile_transform_roundtrip() -> None:
    rng = np.random.default_rng(4)
    A = rng.normal(size=(128, 5)).astype(np.float32)

    coeff_post = build_coeff_post(
        {
            "name": "quantile",
            "n_quantiles": 64,
            "output_distribution": "uniform",
            "subsample": 128,
            "seed": 0,
        }
    )
    coeff_post.fit(A, split="train")
    Z = coeff_post.transform(A)
    A_hat = coeff_post.inverse_transform(Z)

    assert Z.shape == A.shape
    assert A_hat.shape == A.shape
    assert np.isfinite(Z).all()
    assert np.isfinite(A_hat).all()
    assert Z.min() >= -1.0e-6
    assert Z.max() <= 1.0 + 1.0e-6


def test_power_yeojohnson_roundtrip_and_stability() -> None:
    rng = np.random.default_rng(7)
    A = rng.normal(size=(128, 4)).astype(np.float64)
    A[:, 0] = rng.exponential(scale=2.0, size=128)
    A[:8, 0] = 0.0
    A[:, 1] = -rng.exponential(scale=1.5, size=128)
    A[:, 2] = rng.normal(loc=0.0, scale=5.0, size=128)
    A[:, 3] = rng.normal(loc=0.0, scale=0.1, size=128)

    coeff_post = build_coeff_post({"name": "power_yeojohnson", "standardize": True})
    coeff_post.fit(A, split="train")
    Z = coeff_post.transform(A)
    A_hat = coeff_post.inverse_transform(Z)

    assert Z.shape == A.shape
    assert A_hat.shape == A.shape
    assert np.isfinite(Z).all()
    assert np.isfinite(A_hat).all()
    assert np.allclose(A_hat, A, rtol=1e-5, atol=1e-6)


def test_power_yeojohnson_no_standardize_roundtrip() -> None:
    rng = np.random.default_rng(9)
    A = rng.normal(size=(64, 3)).astype(np.float64)

    coeff_post = build_coeff_post({"name": "power_yeojohnson", "standardize": False})
    coeff_post.fit(A, split="train")
    Z = coeff_post.transform(A)
    A_hat = coeff_post.inverse_transform(Z)

    assert Z.shape == A.shape
    assert np.allclose(A_hat, A, rtol=1e-5, atol=1e-6)


def test_dict_learning_sparse_codes() -> None:
    rng = np.random.default_rng(3)
    A = rng.normal(size=(24, 6)).astype(np.float32)

    coeff_post = build_coeff_post(
        {
            "name": "dict_learning",
            "n_components": 4,
            "alpha": 1.0,
            "max_iter": 50,
            "tol": 1.0e-6,
            "fit_algorithm": "lars",
            "transform_algorithm": "omp",
            "transform_n_nonzero_coefs": 2,
            "seed": 0,
        }
    )
    coeff_post.fit(A, split="train")
    Z = coeff_post.transform(A)
    A_hat = coeff_post.inverse_transform(Z)

    assert Z.shape[1] == 4
    assert A_hat.shape == A.shape
    assert np.isfinite(A_hat).all()
    nonzero = np.count_nonzero(np.abs(Z) > 1.0e-8, axis=1)
    assert np.all(nonzero <= 2)


def test_coeff_post_fit_requires_train_split() -> None:
    rng = np.random.default_rng(2)
    A = rng.normal(size=(8, 4)).astype(np.float32)

    coeff_post = build_coeff_post({"name": "standardize"})
    with pytest.raises(ValueError):
        coeff_post.fit(A, split="val")
