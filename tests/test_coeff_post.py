from __future__ import annotations

import numpy as np
import pytest

from mode_decomp_ml.coeff_post import build_coeff_post


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
