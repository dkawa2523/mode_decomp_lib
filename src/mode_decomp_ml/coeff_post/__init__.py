"""CoeffPost registry and implementations."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Callable, Dict, Mapping

import numpy as np
from sklearn.decomposition import DictionaryLearning, PCA

_COEFF_POST_REGISTRY: Dict[str, Callable[..., "BaseCoeffPost"]] = {}


def _cfg_get(cfg: Mapping[str, Any] | None, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        try:
            return cfg.get(key, default)
        except TypeError:
            pass
    return getattr(cfg, key, default)


def _ensure_2d(array: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {array.shape}")
    return array


def _validate_train_split(split: str) -> str:
    if split is None:
        raise ValueError("split is required for coeff_post.fit")
    split_value = str(split).strip().lower()
    # CONTRACT: coeff_post.fit must be called with train split only to prevent skew.
    if split_value != "train":
        raise ValueError("coeff_post.fit requires split='train' to avoid train/serve skew")
    return split_value


def register_coeff_post(name: str) -> Callable[[Callable[..., "BaseCoeffPost"]], Callable[..., "BaseCoeffPost"]]:
    def _wrapper(cls: Callable[..., "BaseCoeffPost"]) -> Callable[..., "BaseCoeffPost"]:
        if name in _COEFF_POST_REGISTRY:
            raise KeyError(f"CoeffPost already registered: {name}")
        _COEFF_POST_REGISTRY[name] = cls
        return cls

    return _wrapper


def list_coeff_posts() -> tuple[str, ...]:
    return tuple(sorted(_COEFF_POST_REGISTRY.keys()))


def build_coeff_post(cfg: Mapping[str, Any]) -> "BaseCoeffPost":
    name = str(_cfg_get(cfg, "name", _cfg_get(cfg, "method", ""))).strip()
    if not name:
        raise ValueError("coeff_post.name is required")
    cls = _COEFF_POST_REGISTRY.get(name)
    if cls is None:
        raise KeyError(f"Unknown coeff_post: {name}. Available: {list_coeff_posts()}")
    return cls(cfg=cfg)


class BaseCoeffPost:
    """Minimal coeff post interface."""

    name: str

    def __init__(self, *, cfg: Mapping[str, Any]) -> None:
        self._cfg = dict(cfg)
        self._fitted = False
        self._state: Dict[str, Any] | None = None

    def fit(self, A: np.ndarray, *, split: str) -> "BaseCoeffPost":
        raise NotImplementedError

    def transform(self, A: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def state(self) -> Mapping[str, Any]:
        if self._state is None:
            raise ValueError("state is not available before fit")
        return self._state

    def save_state(self, run_dir: str | Path) -> Path:
        out_dir = Path(run_dir) / "artifacts" / "coeff_post"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "state.pkl"
        with path.open("wb") as fh:
            pickle.dump(self, fh)
        return path

    @staticmethod
    def load_state(path: str | Path) -> "BaseCoeffPost":
        with Path(path).open("rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, BaseCoeffPost):
            raise TypeError("Loaded coeff_post state is not a BaseCoeffPost")
        return obj

    def _mark_fitted(self) -> None:
        self._fitted = True

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise ValueError("coeff_post must be fit before transform/inverse")


@register_coeff_post("none")
class NoOpCoeffPost(BaseCoeffPost):
    """No-op coeff post (pass-through)."""

    def __init__(self, *, cfg: Mapping[str, Any]) -> None:
        super().__init__(cfg=cfg)
        self.name = "none"
        self._fitted = True
        self._state = {"method": self.name}

    def fit(self, A: np.ndarray, *, split: str) -> "NoOpCoeffPost":
        _validate_train_split(split)
        self._mark_fitted()
        return self

    def transform(self, A: np.ndarray) -> np.ndarray:
        return _ensure_2d(A, "A")

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        return _ensure_2d(Z, "Z")


@register_coeff_post("standardize")
class StandardizeCoeffPost(BaseCoeffPost):
    """Per-dimension standardization."""

    def __init__(self, *, cfg: Mapping[str, Any]) -> None:
        super().__init__(cfg=cfg)
        self.name = "standardize"
        self._mean: np.ndarray | None = None
        self._scale: np.ndarray | None = None

    def fit(self, A: np.ndarray, *, split: str) -> "StandardizeCoeffPost":
        _validate_train_split(split)
        A = _ensure_2d(A, "A")
        mean = A.mean(axis=0)
        std = A.std(axis=0)
        scale = np.where(std == 0.0, 1.0, std)
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
        return (A - self._mean) / self._scale

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        self._require_fitted()
        if self._mean is None or self._scale is None:
            raise ValueError("standardize state is missing")
        Z = _ensure_2d(Z, "Z")
        if Z.shape[1] != self._mean.shape[0]:
            raise ValueError("Z feature dimension does not match standardize state")
        return Z * self._scale + self._mean


@register_coeff_post("pca")
class PCACoeffPost(BaseCoeffPost):
    """PCA-based coeff post."""

    def __init__(self, *, cfg: Mapping[str, Any]) -> None:
        super().__init__(cfg=cfg)
        self.name = "pca"
        self._n_components = _cfg_get(cfg, "n_components", None)
        self._energy_threshold = _cfg_get(cfg, "energy_threshold", None)
        self._whiten = bool(_cfg_get(cfg, "whiten", False))
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
            return threshold
        if self._n_components is not None:
            n_components = int(self._n_components)
            if n_components <= 0:
                raise ValueError("n_components must be > 0")
            return n_components
        return None

    def fit(self, A: np.ndarray, *, split: str) -> "PCACoeffPost":
        _validate_train_split(split)
        A = _ensure_2d(A, "A")
        pca = PCA(n_components=self._resolved_components, whiten=self._whiten, svd_solver="full")
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


@register_coeff_post("dict_learning")
class DictLearningCoeffPost(BaseCoeffPost):
    """Dictionary learning coeff post."""

    def __init__(self, *, cfg: Mapping[str, Any]) -> None:
        super().__init__(cfg=cfg)
        self.name = "dict_learning"
        n_components = _cfg_get(cfg, "n_components", None)
        if n_components is None:
            self._n_components = None
        else:
            self._n_components = int(n_components)
            if self._n_components <= 0:
                raise ValueError("n_components must be > 0 for dict_learning")
        self._alpha = float(_cfg_get(cfg, "alpha", 1.0))
        if self._alpha <= 0.0:
            raise ValueError("alpha must be > 0 for dict_learning")
        self._max_iter = int(_cfg_get(cfg, "max_iter", 200))
        if self._max_iter <= 0:
            raise ValueError("max_iter must be > 0 for dict_learning")
        self._tol = float(_cfg_get(cfg, "tol", 1.0e-6))
        if self._tol <= 0.0:
            raise ValueError("tol must be > 0 for dict_learning")
        self._fit_algorithm = str(_cfg_get(cfg, "fit_algorithm", "lars"))
        if self._fit_algorithm not in {"lars", "cd"}:
            raise ValueError("fit_algorithm must be one of {'lars', 'cd'} for dict_learning")
        self._transform_algorithm = str(_cfg_get(cfg, "transform_algorithm", "omp"))
        if self._transform_algorithm not in {"omp", "lasso_lars", "lasso_cd", "threshold"}:
            raise ValueError(
                "transform_algorithm must be one of "
                "{'omp', 'lasso_lars', 'lasso_cd', 'threshold'} for dict_learning"
            )
        transform_n_nonzero = _cfg_get(cfg, "transform_n_nonzero_coefs", None)
        if transform_n_nonzero is None:
            self._transform_n_nonzero = None
        else:
            self._transform_n_nonzero = int(transform_n_nonzero)
            if self._transform_n_nonzero <= 0:
                raise ValueError("transform_n_nonzero_coefs must be > 0 for dict_learning")
        transform_alpha = _cfg_get(cfg, "transform_alpha", None)
        if transform_alpha is None:
            self._transform_alpha = None
        else:
            self._transform_alpha = float(transform_alpha)
            if self._transform_alpha <= 0.0:
                raise ValueError("transform_alpha must be > 0 for dict_learning")
        self._positive_code = bool(_cfg_get(cfg, "positive_code", False))
        self._positive_dict = bool(_cfg_get(cfg, "positive_dict", False))
        seed = _cfg_get(cfg, "seed", None)
        self._seed = int(seed) if seed is not None else None
        self._dict: DictionaryLearning | None = None
        self.latent_dim: int | None = None

    def fit(self, A: np.ndarray, *, split: str) -> "DictLearningCoeffPost":
        _validate_train_split(split)
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
        return self._dict.inverse_transform(Z)


__all__ = [
    "BaseCoeffPost",
    "NoOpCoeffPost",
    "PCACoeffPost",
    "DictLearningCoeffPost",
    "StandardizeCoeffPost",
    "build_coeff_post",
    "list_coeff_posts",
    "register_coeff_post",
]
