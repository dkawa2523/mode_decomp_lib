"""Load one run YAML with core helpers and normalize numerical component options."""

from __future__ import annotations

from copy import deepcopy
from inspect import signature
from numbers import Integral
from pathlib import Path
from typing import Any, Mapping

import yaml
from ml_platform_core.config import OverrideInput, apply_overrides, deep_merge, parse_overrides
from ml_platform_core.config_validation import validate_run_config


def load_config(
    config_path: str | Path, profile_path: str | Path | None = None, *, overrides: OverrideInput = None
) -> dict[str, Any]:
    """Load profile (optional), run YAML, then overrides; leave paths relative to cwd.

    Business defaults and path resolution use the same ``run_task`` policy as a
    directly supplied dictionary. Inline data descriptions stay in that dictionary.
    """

    def read_mapping(path):
        value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"Run/profile YAML must contain a mapping: {path}")
        return value

    if overrides is not None and not isinstance(overrides, (dict, list)):
        raise TypeError("overrides must be a mapping, a list of key=value strings, or None")
    cfg = deep_merge({} if profile_path is None else read_mapping(profile_path), read_mapping(config_path))
    cfg = apply_overrides(cfg, overrides)
    cfg["_meta"] = {
        "task_config": str(config_path),
        "profile_config": None if profile_path is None else str(profile_path),
        "overrides": parse_overrides(overrides) if isinstance(overrides, list) else deepcopy(overrides or {}),
    }
    validate_run_config(cfg)
    return cfg


def cfg_get(cfg: Mapping[str, Any] | None, key: str, default: Any = None) -> Any:
    return default if cfg is None else cfg.get(key, default)


def integer_parameter(value: Any, label: str) -> int:
    """Validate an actual integer without truncating a float or parsing text."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{label} must be an integer")
    return int(value)


def component_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Expand ``{name, params}`` without letting parameters change identity.

    Flat options are also accepted for composition inside numerical methods.
    An option supplied both flat and under ``params`` is ambiguous and rejected.
    """
    if not isinstance(cfg, Mapping):
        raise TypeError("component config must be a mapping")
    name = cfg.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("component.name must be a non-empty string")
    if "id" in cfg:
        raise ValueError("component id belongs to the candidate, not its numerical parameters")
    params = cfg.get("params", {})
    if not isinstance(params, Mapping):
        raise TypeError("component.params must be a mapping")
    reserved = {"name", "id", "params", "method"}.intersection(params)
    if reserved:
        raise ValueError(f"component.params cannot override identity: {sorted(reserved)}")
    options = {key: value for key, value in cfg.items() if key not in {"name", "params"}}
    duplicates = set(options).intersection(params)
    if duplicates:
        raise ValueError(f"component options supplied twice: {sorted(duplicates)}")
    return {"name": name.strip(), **deepcopy(options), **deepcopy(dict(params))}


def constructor_config(cfg: Mapping[str, Any] | None, parameters: dict[str, Any], *, name: str) -> dict[str, Any]:
    """Normalize the retained ``cfg=`` interface against explicit constructor parameters.

    Defaults live in the selected implementation's signature. This helper owns no
    method defaults or parameter registry, and never changes a caller's mapping.
    Explicit cfg options override keywords, except contradictory seed aliases.
    """
    values = {key: value for key, value in parameters.items() if key not in {"self", "cfg", "__class__"}}
    if cfg is not None and (not isinstance(cfg, Mapping) or ("name" in cfg and cfg["name"] != name)):
        raise ValueError(f"{name} constructor cfg must be a mapping with the same component name")
    options = {} if cfg is None else component_config({"name": name, **cfg})
    options.pop("name", None)
    unknown = set(options) - set(values)
    if unknown:
        raise ValueError(f"Unknown {name} parameters: {sorted(unknown)}")
    if "seed" in values and "seed" not in options and options.get("random_state") is not None:
        if values["seed"] is not None and values["seed"] != options["random_state"]:
            raise ValueError("seed and random_state must match when both are provided")
        options["seed"] = options["random_state"]
    resolved = {"name": name, **deepcopy(values), **options}
    defaults = {
        key: parameter.default for key, parameter in signature(type(parameters["self"]).__init__).parameters.items()
    }
    _boolean_parameters(resolved, defaults, name)
    return resolved


def _boolean_parameters(options: Mapping[str, Any], defaults: Mapping[str, Any], name: str) -> None:
    for key, default in defaults.items():
        if isinstance(default, bool) and key in options and not isinstance(options[key], bool):
            raise ValueError(f"{name}.{key} must be boolean, not a string or number")


def resolve_component(value: Mapping[str, Any] | None, kind: str, seed: int = 42) -> dict[str, Any]:
    """Resolve selectors and implementation-owned defaults for every public entry."""
    from .catalog import component_defaults

    default_names = {
        "decomposition": "pod_svd",
        "codec": "auto",
        "coeff_post": "none",
        "field_preprocess": "none",
        "regression": "ridge",
    }
    options = component_config({"name": default_names[kind]} if value is None else value)
    name = options.pop("name")
    defaults = component_defaults(name, kind)
    unknown = set(options) - set(defaults)
    if unknown:
        raise ValueError(f"Unknown {name} parameters: {sorted(unknown)}")
    _boolean_parameters(options, defaults, name)
    if "seed" in defaults and "seed" not in options and "random_state" not in options:
        defaults["seed"] = seed
    elif "seed" in defaults and "seed" not in options and options.get("random_state") is not None:
        defaults["seed"] = options["random_state"]
    return {"name": name, "params": {**defaults, **options}}


def resolve_representation(config: Mapping[str, Any], *, seed: int = 42) -> dict[str, Any]:
    kinds = ("field_preprocess", "decomposition", "codec", "coeff_post")
    if not isinstance(config, Mapping):
        raise TypeError("representation must be a mapping")
    unknown = set(config) - set(kinds)
    if unknown:
        raise ValueError(f"Unknown representation keys: {sorted(unknown)}")
    if any(kind in config and config[kind] is None for kind in kinds):
        raise TypeError("Explicit representation components must be mappings, not null; omit them to use defaults")
    return {kind: resolve_component(config.get(kind), kind, seed) for kind in kinds}


__all__ = ["load_config", "cfg_get", "component_config", "resolve_component", "resolve_representation"]
