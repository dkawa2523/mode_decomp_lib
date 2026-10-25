"""CoeffPost implementations."""

from __future__ import annotations


from ml_platform_mode_decomp.config import constructor_config, integer_parameter

from typing import Any, Dict, Literal, Mapping, cast

import numpy as np
from sklearn.decomposition import DictionaryLearning, PCA
from sklearn.preprocessing import PowerTransformer, QuantileTransformer


def _ensure_2d(array: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {array.shape}")
    if array.dtype.kind not in "fiu" or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite real numeric coefficients")
    return array


def _random_seed(cfg: Mapping[str, Any]) -> int | None:
    seed, random_state = cfg["seed"], cfg["random_state"]
    seed = None if seed is None else integer_parameter(seed, "seed")
    random_state = None if random_state is None else integer_parameter(random_state, "random_state")
    if seed is not None and random_state is not None and seed != random_state:
        raise ValueError("seed and random_state must match when both are specified")
    return seed if seed is not None else random_state


class BaseCoeffPost:
    """Minimal coeff post interface."""

    name: str

    def __init__(self, *, cfg: Mapping[str, Any]) -> None:
        self._cfg = dict(cfg)
        self._fitted = False
        self._state: Dict[str, Any] | None = None

    def fit(self, A: np.ndarray) -> "BaseCoeffPost":
        raise NotImplementedError

    def transform(self, A: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def state(self) -> Mapping[str, Any]:
        if self._state is None:
            raise ValueError("state is not available before fit")
        return self._state

    def _mark_fitted(self) -> None:
        self._fitted = True

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise ValueError("coeff_post must be fit before transform/inverse")


class NoOpCoeffPost(BaseCoeffPost):
    """No-op coeff post (pass-through)."""

    def __init__(self, *, cfg: Mapping[str, Any] | None = None) -> None:
        cfg = constructor_config(cfg, locals(), name="none")
        super().__init__(cfg=cfg)
        self.name = "none"
        self._fitted = True
        self._state = {"method": self.name}

    def fit(self, A: np.ndarray) -> "NoOpCoeffPost":
        _ensure_2d(A, "A")
        self._mark_fitted()
        return self

    def transform(self, A: np.ndarray) -> np.ndarray:
        return _ensure_2d(A, "A")

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        return _ensure_2d(Z, "Z")


class StandardizeCoeffPost(BaseCoeffPost):
    """Per-dimension standardization."""

    def __init__(self, *, cfg: Mapping[str, Any] | None = None) -> None:
        cfg = constructor_config(cfg, locals(), name="standardize")
        super().__init__(cfg=cfg)
        self.name = "standardize"
        self._mean: np.ndarray | None = None
        self._scale: np.ndarray | None = None

    def fit(self, A: np.ndarray) -> "StandardizeCoeffPost":
        A = _ensure_2d(A, "A")
        mean = A.mean(axis=0)
        std = A.std(axis=0)
        scale = np.where(std == 0.0, 1.0, std)
        if not np.isfinite(mean).all() or not np.isfinite(scale).all():
            raise ValueError("Coefficient standardization statistics overflow; rescale the coefficients")
        self._mean = mean
        self._scale = scale
        self._state = {"method": self.name, "mean": mean, "scale": scale}
        self._mark_fitted()
        return self

    def transform(self, A: np.ndarray) -> np.ndarray:
        self._require_fitted()
        if self._mean is None or self._scale is None:
            raise ValueError("standardize state is missing")
        A = _ensure_2d(A, "A")
        if A.shape[1] != self._mean.shape[0]:
            raise ValueError("A feature dimension does not match standardize state")
        return _ensure_2d((A - self._mean) / self._scale, "Z")

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        self._require_fitted()
        if self._mean is None or self._scale is None:
            raise ValueError("standardize state is missing")
        Z = _ensure_2d(Z, "Z")
        if Z.shape[1] != self._mean.shape[0]:
            raise ValueError("Z feature dimension does not match standardize state")
        return _ensure_2d(Z * self._scale + self._mean, "A")


class QuantileCoeffPost(BaseCoeffPost):
    """QuantileTransformer coeff post (robust to skew/outliers)."""

    def __init__(
        self,
        *,
        cfg: Mapping[str, Any] | None = None,
        n_quantiles: Any = 1000,
        subsample: Any = 10000,
        seed: Any = None,
        ignore_implicit_zeros: Any = None,
        random_state: Any = None,
        copy: Any = True,
        output_distribution: Any = "uniform",
    ) -> None:
        cfg = constructor_config(cfg, locals(), name="quantile")
        super().__init__(cfg=cfg)
        self.name = "quantile"
        n_quantiles = cfg["n_quantiles"]
        if n_quantiles is None:
            self._n_quantiles = 1000
        else:
            self._n_quantiles = integer_parameter(n_quantiles, "n_quantiles")
            if self._n_quantiles <= 0:
                raise ValueError("n_quantiles must be > 0 for quantile")
        self._output_distribution = str(cfg["output_distribution"]).strip().lower()
        if self._output_distribution not in {"uniform", "normal"}:
            raise ValueError("output_distribution must be 'uniform' or 'normal' for quantile")
        subsample = cfg["subsample"]
        if subsample is None:
            self._subsample = None
        else:
            self._subsample = integer_parameter(subsample, "subsample")
            if self._subsample <= 0:
                raise ValueError("subsample must be > 0 for quantile")
        self._random_state = _random_seed(cfg)
        self._copy = bool(cfg["copy"])
        ignore_implicit = cfg["ignore_implicit_zeros"]
        if ignore_implicit is not None and not isinstance(ignore_implicit, (bool, np.bool_)):
            raise ValueError("ignore_implicit_zeros must be boolean or None")
        self._ignore_implicit_zeros = bool(ignore_implicit) if ignore_implicit is not None else None
        self._transformer: QuantileTransformer | None = None

    def fit(self, A: np.ndarray) -> "QuantileCoeffPost":
        A = _ensure_2d(A, "A")
        subsample = self._subsample if self._subsample is not None else A.shape[0]
        kwargs: dict[str, Any] = {
            "n_quantiles": self._n_quantiles,
            "output_distribution": self._output_distribution,
            "subsample": subsample,
            "random_state": self._random_state,
            "copy": self._copy,
        }
        if self._ignore_implicit_zeros is not None:
            kwargs["ignore_implicit_zeros"] = self._ignore_implicit_zeros
        transformer = QuantileTransformer(**kwargs)
        transformer.fit(A if self._copy else A.copy())
        self._transformer = transformer
        self._state = {
            "method": self.name,
            "n_quantiles": self._n_quantiles,
            "n_quantiles_": int(transformer.n_quantiles_),
            "output_distribution": self._output_distribution,
            "subsample": subsample,
            "random_state": self._random_state,
            "copy": self._copy,
            "ignore_implicit_zeros": self._ignore_implicit_zeros,
            "inverse_approx": True,
            "transformer": transformer,
        }
        self._mark_fitted()
        return self

    def transform(self, A: np.ndarray) -> np.ndarray:
        self._require_fitted()
        if self._transformer is None:
            raise ValueError("quantile state is missing")
        A = _ensure_2d(A, "A")
        if A.shape[1] != self._transformer.n_features_in_:
            raise ValueError("A feature dimension does not match quantile state")
        return self._transformer.transform(A if self._copy else A.copy())

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        self._require_fitted()
        if self._transformer is None:
            raise ValueError("quantile state is missing")
        Z = _ensure_2d(Z, "Z")
        if Z.shape[1] != self._transformer.n_features_in_:
            raise ValueError("Z feature dimension does not match quantile state")
        return self._transformer.inverse_transform(Z if self._copy else Z.copy())


class PowerYeoJohnsonCoeffPost(BaseCoeffPost):
    """PowerTransformer coeff post (Yeo-Johnson)."""

    def __init__(self, *, cfg: Mapping[str, Any] | None = None, standardize: Any = True, copy: Any = True) -> None:
        cfg = constructor_config(cfg, locals(), name="power_yeojohnson")
        super().__init__(cfg=cfg)
        self.name = "power_yeojohnson"
        self._method: Literal["yeo-johnson"] = "yeo-johnson"
        self._standardize = bool(cfg["standardize"])
        self._copy = bool(cfg["copy"])
        self._transformer: PowerTransformer | None = None

    def fit(self, A: np.ndarray) -> "PowerYeoJohnsonCoeffPost":
        A = _ensure_2d(A, "A")
        transformer = PowerTransformer(method=self._method, standardize=self._standardize, copy=self._copy)
        transformer.fit(A if self._copy else A.copy())
        self._transformer = transformer
        self._state = {
            "method": self.name,
            "power_method": self._method,
            "standardize": self._standardize,
            "copy": self._copy,
            "lambdas_": transformer.lambdas_.copy(),
            "transformer": transformer,
        }
        self._mark_fitted()
        return self

    def transform(self, A: np.ndarray) -> np.ndarray:
        self._require_fitted()
        if self._transformer is None:
            raise ValueError("power_yeojohnson state is missing")
        A = _ensure_2d(A, "A")
        if A.shape[1] != self._transformer.n_features_in_:
            raise ValueError("A feature dimension does not match power_yeojohnson state")
        return self._transformer.transform(A if self._copy else A.copy())

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        self._require_fitted()
        if self._transformer is None:
            raise ValueError("power_yeojohnson state is missing")
        Z = _ensure_2d(Z, "Z")
        if Z.shape[1] != self._transformer.n_features_in_:
            raise ValueError("Z feature dimension does not match power_yeojohnson state")
        result = self._transformer.inverse_transform(Z if self._copy else Z.copy())
        if not np.isfinite(result).all():
            raise ValueError("power_yeojohnson prediction is outside the inverse transformation domain")
        return result


class PCACoeffPost(BaseCoeffPost):
    """PCA-based coeff post."""

    def __init__(
        self,
        *,
        cfg: Mapping[str, Any] | None = None,
        n_components: Any = None,
        energy_threshold: Any = None,
        seed: Any = None,
        whiten: Any = False,
        svd_solver: Any = "full",
        random_state: Any = None,
    ) -> None:
        cfg = constructor_config(cfg, locals(), name="pca")
        super().__init__(cfg=cfg)
        self.name = "pca"
        self._n_components = cfg["n_components"]
        self._energy_threshold = cfg["energy_threshold"]
        self._whiten = bool(cfg["whiten"])
        solver = str(cfg["svd_solver"])
        if solver not in {"auto", "full", "arpack", "randomized"}:
            raise ValueError(f"Unsupported PCA svd_solver: {solver}")
        self._svd_solver = cast(Literal["auto", "full", "arpack", "randomized"], solver)
        self._random_state = _random_seed(cfg)
        self._resolved_components = self._resolve_components()
        self._pca: PCA | None = None
        self.latent_dim: int | None = None

    def _resolve_components(self) -> int | float | None:
        if self._n_components is not None and self._energy_threshold is not None:
            raise ValueError("Specify only one of n_components or energy_threshold")
        if self._energy_threshold is not None:
            threshold = float(self._energy_threshold)
            if not (0.0 < threshold <= 1.0):
                raise ValueError("energy_threshold must be in (0, 1]")
            if threshold == 1.0 and self._svd_solver == "arpack":
                raise ValueError(
                    "PCA energy_threshold=1.0 requires a complete spectrum; arpack always omits a component"
                )
            return None if threshold == 1.0 else threshold
        if self._n_components is not None:
            n_components = integer_parameter(self._n_components, "n_components")
            if n_components <= 0:
                raise ValueError("n_components must be > 0")
            return n_components
        return None

    def fit(self, A: np.ndarray) -> "PCACoeffPost":
        A = _ensure_2d(A, "A")
        pca = PCA(
            n_components=self._resolved_components,
            whiten=self._whiten,
            svd_solver=self._svd_solver,
            random_state=self._random_state,
        )
        pca.fit(A)
        self._pca = pca
        self.latent_dim = int(pca.n_components_)
        # REVIEW: latent_dim is derived from fitted PCA to keep metadata stable.
        self._state = {
            "method": self.name,
            "latent_dim": self.latent_dim,
            "whiten": self._whiten,
            "n_components": self._n_components,
            "energy_threshold": self._energy_threshold,
            "svd_solver": self._svd_solver,
            "random_state": self._random_state,
            "pca": pca,
        }
        self._mark_fitted()
        return self

    def transform(self, A: np.ndarray) -> np.ndarray:
        self._require_fitted()
        if self._pca is None:
            raise ValueError("PCA state is missing")
        A = _ensure_2d(A, "A")
        if A.shape[1] != self._pca.n_features_in_:
            raise ValueError("A feature dimension does not match PCA state")
        return self._pca.transform(A)

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        self._require_fitted()
        if self._pca is None:
            raise ValueError("PCA state is missing")
        Z = _ensure_2d(Z, "Z")
        if self.latent_dim is not None and Z.shape[1] != self.latent_dim:
            raise ValueError("Z feature dimension does not match PCA latent_dim")
        return self._pca.inverse_transform(Z)


class DictLearningCoeffPost(BaseCoeffPost):
    """Dictionary learning coeff post."""

    def __init__(
        self,
        *,
        cfg: Mapping[str, Any] | None = None,
        n_components: Any = 64,
        transform_n_nonzero_coefs: Any = None,
        transform_alpha: Any = None,
        seed: Any = None,
        alpha: Any = 1.0,
        max_iter: Any = 200,
        tol: Any = 1e-06,
        fit_algorithm: Any = "lars",
        transform_algorithm: Any = "lasso_cd",
        positive_code: Any = False,
        positive_dict: Any = False,
    ) -> None:
        cfg = constructor_config(cfg, locals(), name="dict_learning")
        super().__init__(cfg=cfg)
        self.name = "dict_learning"
        n_components = cfg["n_components"]
        if n_components is None:
            self._n_components = None
        else:
            self._n_components = integer_parameter(n_components, "n_components")
            if self._n_components <= 0:
                raise ValueError("n_components must be > 0 for dict_learning")
        self._alpha = float(cfg["alpha"])
        if self._alpha <= 0.0:
            raise ValueError("alpha must be > 0 for dict_learning")
        self._max_iter = integer_parameter(cfg["max_iter"], "max_iter")
        if self._max_iter <= 0:
            raise ValueError("max_iter must be > 0 for dict_learning")
        self._tol = float(cfg["tol"])
        if self._tol <= 0.0:
            raise ValueError("tol must be > 0 for dict_learning")
        fit_algorithm = str(cfg["fit_algorithm"])
        if fit_algorithm not in {"lars", "cd"}:
            raise ValueError("fit_algorithm must be one of {'lars', 'cd'} for dict_learning")
        transform_algorithm = str(cfg["transform_algorithm"])
        if transform_algorithm not in {"omp", "lasso_lars", "lasso_cd", "threshold"}:
            raise ValueError(
                "transform_algorithm must be one of {'omp', 'lasso_lars', 'lasso_cd', 'threshold'} for dict_learning"
            )
        self._fit_algorithm = cast(Literal["lars", "cd"], fit_algorithm)
        self._transform_algorithm = cast(Literal["omp", "lasso_lars", "lasso_cd", "threshold"], transform_algorithm)
        transform_n_nonzero = cfg["transform_n_nonzero_coefs"]
        if transform_n_nonzero is None:
            self._transform_n_nonzero = None
        else:
            self._transform_n_nonzero = integer_parameter(transform_n_nonzero, "transform_n_nonzero")
            if self._transform_n_nonzero <= 0:
                raise ValueError("transform_n_nonzero_coefs must be > 0 for dict_learning")
        transform_alpha = cfg["transform_alpha"]
        if transform_alpha is None:
            self._transform_alpha = None
        else:
            self._transform_alpha = float(transform_alpha)
            if self._transform_alpha <= 0.0:
                raise ValueError("transform_alpha must be > 0 for dict_learning")
        self._positive_code = bool(cfg["positive_code"])
        self._positive_dict = bool(cfg["positive_dict"])
        seed = cfg["seed"]
        self._seed = integer_parameter(seed, "seed") if seed is not None else None
        self._dict: DictionaryLearning | None = None
        self.latent_dim: int | None = None

    def fit(self, A: np.ndarray) -> "DictLearningCoeffPost":
        A = _ensure_2d(A, "A")
        model = DictionaryLearning(
            n_components=self._n_components,
            alpha=self._alpha,
            max_iter=self._max_iter,
            tol=self._tol,
            fit_algorithm=self._fit_algorithm,
            transform_algorithm=self._transform_algorithm,
            transform_n_nonzero_coefs=self._transform_n_nonzero,
            transform_alpha=self._transform_alpha,
            positive_code=self._positive_code,
            positive_dict=self._positive_dict,
            random_state=self._seed,
        )
        model.fit(A)
        self._dict = model
        self.latent_dim = int(model.components_.shape[0])
        self._state = {
            "method": self.name,
            "latent_dim": self.latent_dim,
            "alpha": self._alpha,
            "max_iter": self._max_iter,
            "tol": self._tol,
            "fit_algorithm": self._fit_algorithm,
            "transform_algorithm": self._transform_algorithm,
            "transform_n_nonzero_coefs": self._transform_n_nonzero,
            "transform_alpha": self._transform_alpha,
            "positive_code": self._positive_code,
            "positive_dict": self._positive_dict,
            "seed": self._seed,
            "dictionary": model,
        }
        self._mark_fitted()
        return self

    def transform(self, A: np.ndarray) -> np.ndarray:
        self._require_fitted()
        if self._dict is None:
            raise ValueError("dict_learning state is missing")
        A = _ensure_2d(A, "A")
        if A.shape[1] != self._dict.n_features_in_:
            raise ValueError("A feature dimension does not match dict_learning state")
        return self._dict.transform(A)

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        self._require_fitted()
        if self._dict is None:
            raise ValueError("dict_learning state is missing")
        Z = _ensure_2d(Z, "Z")
        if self.latent_dim is not None and Z.shape[1] != self.latent_dim:
            raise ValueError("Z feature dimension does not match dict_learning latent_dim")
        # DictionaryLearning's synthesis is the sparse code times the fitted dictionary.
        # Express it directly so this works across the supported sklearn versions.
        if not np.isfinite(Z).all():
            raise ValueError("Dictionary coefficients must be finite")
        return Z @ self._dict.components_


__all__ = [
    "BaseCoeffPost",
    "NoOpCoeffPost",
    "QuantileCoeffPost",
    "PowerYeoJohnsonCoeffPost",
    "PCACoeffPost",
    "DictLearningCoeffPost",
    "StandardizeCoeffPost",
]
