"""Self-contained bundles and their field-specific schema."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import tempfile
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Mapping

from ml_platform_core.artifacts import file_hash
from ml_platform_core.io import dump_joblib, load_joblib, read_json, write_json
from ml_platform_core.model_artifact import runtime_identity, validate_artifact_integrity

from .geometry.identity import geometry_fingerprint, geometry_fingerprint_v1, snapshot_geometry
from .representation import FieldRepresentation
from .surrogate import FieldSurrogate
from .coefficients.layout import CoefficientLayout
from .data.schema import FieldSpec

Bundle = FieldRepresentation | FieldSurrogate
_RUNTIME_PACKAGES = (
    "numpy",
    "scipy",
    "scikit-learn",
    "joblib",
    "ml-platform-core",
    "ml-platform-mode-decomp",
)


def fingerprint(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def bundle_schema(bundle: Bundle, *, schema_version: int = 4) -> dict[str, Any]:
    surrogate = isinstance(bundle, FieldSurrogate)
    rep = bundle.representation if surrogate else bundle
    input_representation = getattr(bundle, "input_representation", None) if surrogate else None
    layout_keys = (
        "method",
        "field_shape",
        "field_layout",
        "channels",
        "coeff_shape",
        "coeff_layout",
        "flatten_order",
        "complex_format",
        "coeff_format",
    )
    layout = {key: rep.raw_meta[key] for key in layout_keys if key in rep.raw_meta}
    calibration = getattr(bundle, "calibrator", None) if surrogate else None
    uncertainty = (
        {}
        if calibration is None
        else {
            "field_uncertainty": {
                key: calibration.summary()[key] for key in ("method", "coverage", "scope", "normalization")
            }
        }
    )
    return {
        "schema_version": schema_version,
        "bundle_kind": "surrogate" if surrogate else "representation",
        "field_shape": list(rep.field_shape),
        "components": list(rep.components),
        "field_units": list(rep.field_units),
        "geometry_sha256": (geometry_fingerprint_v1 if schema_version == 1 else geometry_fingerprint)(rep.geometry),
        "packed_dim": rep.packed_dim,
        "target_dim": rep.target_dim,
        "condition_columns": list(bundle.condition_columns) if surrogate else [],
        "coefficient_space": "z",
        "coefficient_layout": layout,
        **({"field_kind": rep.field_kind} if schema_version >= 2 else {}),
        **(
            {
                "fitted_layout": json.loads(json.dumps(asdict(rep.layout))),
                "prediction_kind": getattr(bundle, "prediction_kind", "regression") if surrogate else "representation",
            }
            if schema_version >= 3
            else {}
        ),
        **(
            {
                "input_kind": getattr(bundle, "input_kind", "conditions"),
                "input_dimension": getattr(bundle, "input_dimension", len(bundle.condition_columns)),
                "input_representation": (
                    bundle_schema(input_representation, schema_version=schema_version)
                    if input_representation is not None
                    else None
                ),
                "input_model_sha256": getattr(bundle, "input_model_sha256", None),
            }
            if surrogate and schema_version >= 4
            else {}
        ),
        **uncertainty,
    }


def runtime_packages(config: Mapping[str, Any]) -> set[str]:
    """Selected nested methods matter, including an offset's inner representation."""
    extras = {
        "autoencoder": {"torch"},
        "mtgp": {"gpytorch", "torch", "linear-operator"},
        "wavelet2d": {"PyWavelets"},
        "xgb": {"xgboost"},
        "lgbm": {"lightgbm"},
        "catboost": {"catboost"},
    }
    selected: set[str] = set()

    def visit(value):
        if isinstance(value, dict):
            name = value.get("name")
            if isinstance(name, str):
                selected.update(extras.get(name, set()))
            if value.get("backend") == "modred":
                selected.add("modred")
            for child in value.values():
                visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)

    visit(config)
    return selected


def evaluation_identity(config: Mapping[str, Any]) -> dict[str, Any]:
    """Record released package versions and the selected numerical runtime."""
    optional = runtime_packages(config)
    installed = set()
    for package in optional:
        try:
            version(package)
        except PackageNotFoundError:
            continue
        installed.add(package)
    return {
        "runtime": runtime_identity([*_RUNTIME_PACKAGES, "pandas", "PyYAML", *installed]),
    }


def _info_path(path: Path) -> Path:
    return path.with_name("model_info.json" if path.name == "model.joblib" else f"{path.stem}_info.json")


def save_model(bundle: Bundle, path: str | Path, *, provenance: dict | None = None) -> tuple[Path, Path]:
    path = Path(path)
    info_path = _info_path(path)
    if path.exists() or info_path.exists():
        raise FileExistsError(f"Choose a new model artifact path: {path}")
    rep = bundle.representation if isinstance(bundle, FieldSurrogate) else bundle
    rep._require_fitted()
    if isinstance(bundle, FieldSurrogate):
        if not bundle._fitted:
            raise ValueError("Surrogate must be fitted before saving")
        input_representation = getattr(bundle, "input_representation", None)
        if input_representation is not None:
            input_representation._require_fitted()
    optional, uses_sklearn = _persisted_backends(bundle)
    for package in optional:
        version(package)  # A selected backend must be present, not silently omitted from provenance.
    path.parent.mkdir(parents=True, exist_ok=True)
    runtime = runtime_identity([*_RUNTIME_PACKAGES, *optional])
    required = {"numpy", "scipy", "joblib", *optional}
    # These recipes persist sklearn estimators. Other packages remain provenance.
    if uses_sklearn:
        required.add("scikit-learn")
    with tempfile.TemporaryDirectory(prefix=".field-model-", dir=path.parent) as temporary:
        staged = Path(temporary) / path.name
        dump_joblib(bundle, staged)
        info = {
            "model_artifact_schema_version": 1,
            "model_sha256": file_hash(staged),
            "runtime": runtime,
            "required_packages": sorted(required),
            "input_schema": bundle_schema(bundle),
            "provenance": provenance or {},
        }
        staged_info = write_json(info, _info_path(staged))
        installed = []
        try:
            for source, destination in ((staged, path), (staged_info, info_path)):
                # Exclusive creation prevents replacing an existing model, even
                # if another writer chose the same destination after our check.
                os.link(source, destination)
                installed.append(destination)
        except OSError:
            for destination in installed:
                destination.unlink(missing_ok=True)
            raise
    return path, info_path


def _persisted_backends(bundle: Bundle) -> tuple[set[str], bool]:
    """Runtime requirements follow fitted operators and estimators, not unused recipes."""
    optional, seen = set(), set()
    uses_sklearn = False
    rep = bundle.representation if isinstance(bundle, FieldSurrogate) else bundle

    def visit(value):
        nonlocal uses_sklearn
        if id(value) in seen:
            return
        seen.add(id(value))
        module = type(value).__module__.split(".")[0]
        if module == "sklearn":
            uses_sklearn = True
        if module in {"torch", "gpytorch", "xgboost", "lightgbm", "catboost"}:
            optional.add(module)
        if getattr(value, "name", None) == "wavelet2d":
            optional.add("PyWavelets")
        if getattr(value, "name", None) == "autoencoder":
            optional.add("torch")
        if getattr(value, "name", None) == "mtgp":
            optional.update({"torch", "gpytorch", "linear-operator"})
        if isinstance(value, (list, tuple)):
            for child in value:
                visit(child)
        elif hasattr(value, "__dict__") and module in {"ml_platform_mode_decomp", "sklearn"}:
            for key, child in vars(value).items():
                if key not in {"_cfg", "config", "candidate"}:
                    visit(child)

    visit(rep)
    if isinstance(bundle, FieldSurrogate):
        visit(getattr(bundle, "input_representation", None))
        visit(bundle.regressor)
    return optional, uses_sklearn


def _validate_backend_runtime(info: Mapping[str, Any], schema_version: int) -> None:
    runtime = info.get("runtime")
    if not isinstance(runtime, Mapping) or not isinstance(runtime.get("packages"), Mapping):
        raise ValueError("model_info.runtime is required")
    if str(runtime.get("python", "")).split(".")[:2] != platform.python_version().split(".")[:2]:
        raise ValueError(f"Model requires Python {runtime.get('python')}")
    recorded = runtime["packages"]
    required = (
        info.get("required_packages")
        if schema_version >= 2
        else [name for name in recorded if name not in {"ml-platform-core", "ml-platform-mode-decomp"}]
    )
    if (
        not isinstance(required, list)
        or any(not isinstance(name, str) or name not in recorded for name in required)
        or not {"numpy", "scipy", "joblib"}.issubset(required)
    ):
        raise ValueError("Model backend runtime requirements are incomplete")
    for name in required:
        try:
            current = version(name)
        except PackageNotFoundError as exc:
            raise ValueError(f"Model requires package {name}=={recorded[name]}, but it is not installed") from exc
        if current != recorded[name]:
            raise ValueError(f"Model requires package {name}=={recorded[name]}; current runtime has {current}")


def load_model(
    path: str | Path, info_path: str | Path | None = None, *, expected_context: dict | None = None
) -> Bundle:
    path = Path(path)
    info = read_json(info_path or _info_path(path))
    if info.get("model_artifact_schema_version") != 1:
        raise ValueError("Model sidecar version does not match")
    if not isinstance(info.get("input_schema"), dict) or info["input_schema"].get("schema_version") not in {1, 2, 3, 4}:
        raise ValueError("Unsupported field model schema")
    schema_version = info["input_schema"]["schema_version"]
    validate_artifact_integrity(path, info, expected_context)
    _validate_backend_runtime(info, schema_version)
    provenance = info.get("provenance") or {}
    for name, expected in (expected_context or {}).items():
        if name == "model_sha256" or expected is None or (isinstance(expected, str) and not expected):
            continue
        if name == "execution_release_id":
            actual = provenance.get("source_release_id")
        else:
            actual = provenance.get(name)
        if fingerprint(actual) != fingerprint(expected):
            raise ValueError(f"Model provenance differs: {name}")
    bundle = load_joblib(path)
    if not isinstance(bundle, (FieldRepresentation, FieldSurrogate)):
        raise ValueError("Model bundle is not a field representation or surrogate")
    if bundle_schema(bundle, schema_version=schema_version) != info["input_schema"]:
        raise ValueError("Bundle schema differs from model_info")
    if schema_version == 1:
        rep = bundle.representation if isinstance(bundle, FieldSurrogate) else bundle
        previous_geometry = geometry_fingerprint_v1(rep.geometry)
        rep.geometry = snapshot_geometry(rep.geometry)
        rep.geometry_sha256 = geometry_fingerprint(rep.geometry)
        _upgrade_v1_geometry_state(rep.decomposer, previous_geometry, rep.geometry_sha256)
        rep.field_kind = (
            "vector"
            if rep.config["decomposition"]["name"] in {"hodge", "helmholtz"}
            else "scalar"
            if rep.field_shape[-1] == 1
            else "multichannel"
        )
        if isinstance(bundle, FieldSurrogate) and bundle.calibrator is not None:
            bundle.calibrator.geometry_sha256 = rep.geometry_sha256
    if schema_version < 3:
        rep = bundle.representation if isinstance(bundle, FieldSurrogate) else bundle
        rep.field_spec = FieldSpec(rep.field_shape, str(rep.field_kind), rep.components, rep.field_units)
        rep.layout = CoefficientLayout.from_meta(rep.raw_meta, rep.packed_dim, rep.decomposer.synthesis_kind)
        if isinstance(bundle, FieldSurrogate):
            bundle.prediction_kind = "regression"
    if isinstance(bundle, FieldSurrogate) and schema_version < 4:
        bundle.input_representation = None
        bundle.input_kind = "conditions"
        bundle.input_dimension = len(bundle.condition_columns)
        bundle.input_model_sha256 = None
    if isinstance(bundle, FieldSurrogate) and bundle.calibrator is not None:
        bundle.calibrator.field_kind = bundle.representation.field_kind
    return bundle


def _upgrade_v1_geometry_state(decomposer: Any, previous: str, current: str) -> None:
    """Rebase known schema-1 fitted identities after validating the original schema.

    These fields identify the same fixed operator; they do not rebuild or refit it.
    An unexpected identity is deliberately left intact so its own check still fails.
    """
    for name in ("_geometry_sha256", "_geometry_key", "_mesh_hash"):
        if getattr(decomposer, name, None) == previous:
            setattr(decomposer, name, current)
    if getattr(decomposer, "_domain_key", None) == (previous,):
        decomposer._domain_key = (current,)
    cache_key = getattr(decomposer, "_basis_cache_key", None)
    if isinstance(cache_key, tuple) and cache_key and cache_key[0] == (previous,):
        decomposer._basis_cache_key = None
        decomposer._basis_cache = None
    for child in getattr(decomposer, "_decomposers", ()):
        _upgrade_v1_geometry_state(child, previous, current)
    inner = getattr(decomposer, "_inner", None)
    if inner is not None:
        _upgrade_v1_geometry_state(inner, previous, current)
