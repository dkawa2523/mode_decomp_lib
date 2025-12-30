"""Unified Hydra entrypoint for process tasks."""
from __future__ import annotations

import datetime as dt
import importlib
import inspect
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence

try:
    import hydra
    _HYDRA_AVAILABLE = True
except Exception:  # pragma: no cover - fallback path when hydra-core is missing
    hydra = None
    _HYDRA_AVAILABLE = False
try:  # optional for interpolation in fallback mode
    from omegaconf import OmegaConf
except Exception:  # pragma: no cover - fallback when OmegaConf is missing
    OmegaConf = None

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs"


def _task_name(cfg: Mapping[str, Any]) -> str:
    task_cfg = cfg.get("task") if cfg is not None else None
    if task_cfg is None:
        raise ValueError("task config is missing")
    if isinstance(task_cfg, str):
        return task_cfg
    name = getattr(task_cfg, "name", None)
    if not name and hasattr(task_cfg, "get"):
        name = task_cfg.get("name")
    if not name:
        raise ValueError("task.name is missing")
    return str(name)


def _load_task_module(task_name: str):
    try:
        return importlib.import_module(f"processes.{task_name}")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(f"Unknown task module: processes.{task_name}") from exc


def _call_main(main_fn: Callable[..., int], cfg: Mapping[str, Any]) -> int:
    params = list(inspect.signature(main_fn).parameters.values())
    if not params:
        return main_fn()
    first = params[0]
    if first.kind in (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.VAR_POSITIONAL,
    ):
        return main_fn(cfg)
    return main_fn(cfg=cfg)


def _load_yaml(path: Path) -> MutableMapping[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data or {}


_SIMPLE_INTERP_RE = re.compile(r"^\$\{([A-Za-z0-9_.]+)\}$")


def _lookup_dotted(cfg: Mapping[str, Any], key: str) -> Any:
    cur: Any = cfg
    for part in key.split("."):
        if isinstance(cur, Mapping) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def _resolve_simple_interpolations(cfg: Mapping[str, Any]) -> MutableMapping[str, Any]:
    def _resolve(obj: Any, depth: int) -> Any:
        if depth > 10:
            return obj
        if isinstance(obj, str):
            match = _SIMPLE_INTERP_RE.match(obj)
            if not match:
                return obj
            ref = match.group(1)
            if ":" in ref:
                return obj
            value = _lookup_dotted(cfg, ref)
            if value is None:
                return obj
            return _resolve(value, depth + 1)
        if isinstance(obj, Mapping):
            return {k: _resolve(v, depth) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_resolve(v, depth) for v in obj]
        return obj

    resolved = _resolve(cfg, 0)
    return resolved if isinstance(resolved, MutableMapping) else dict(cfg)


def _resolve_interpolations(cfg: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    if OmegaConf is None:
        return _resolve_simple_interpolations(cfg)
    cfg_copy: MutableMapping[str, Any] = dict(cfg)
    hydra_cfg = cfg_copy.pop("hydra", None)
    run_dir_value = cfg_copy.get("run_dir")
    remove_run_dir = isinstance(run_dir_value, str) and (
        "${hydra:" in run_dir_value or "${now:" in run_dir_value
    )
    if remove_run_dir:
        run_dir_value = cfg_copy.pop("run_dir", None)
    resolved = OmegaConf.to_container(OmegaConf.create(cfg_copy), resolve=True)
    if not isinstance(resolved, MutableMapping):
        return _resolve_simple_interpolations(cfg)
    if hydra_cfg is not None:
        resolved["hydra"] = hydra_cfg
    if remove_run_dir and run_dir_value is not None:
        resolved["run_dir"] = run_dir_value
    return _resolve_simple_interpolations(resolved)


def _parse_defaults(defaults: Sequence[Any]) -> Iterable[tuple[str, str]]:
    for item in defaults:
        if isinstance(item, str) and item.strip() == "_self_":
            continue
        if isinstance(item, Mapping):
            group, name = next(iter(item.items()))
            yield str(group), str(name)


def _set_dotted(cfg: MutableMapping[str, Any], key: str, value: Any) -> None:
    parts = key.split(".")
    cur: MutableMapping[str, Any] = cfg
    for part in parts[:-1]:
        nxt = cur.get(part)
        if not isinstance(nxt, MutableMapping):
            nxt = {}
            cur[part] = nxt
        cur = nxt
    cur[parts[-1]] = value


def _parse_value(raw: str) -> Any:
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if raw.isdigit():
        return int(raw)
    try:
        return float(raw)
    except ValueError:
        return raw


def _apply_overrides(
    cfg: MutableMapping[str, Any],
    overrides: Sequence[str],
    group_names: Iterable[str],
) -> None:
    group_set = set(group_names)
    for raw in overrides:
        if "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        if key in group_set:
            cfg[key] = _load_yaml(CONFIG_DIR / key / f"{value}.yaml")
            continue
        _set_dotted(cfg, key, _parse_value(value))


def _compose_config(overrides: Sequence[str]) -> MutableMapping[str, Any]:
    base = _load_yaml(CONFIG_DIR / "config.yaml")
    defaults = base.get("defaults", [])
    cfg: MutableMapping[str, Any] = {k: v for k, v in base.items() if k != "defaults"}
    group_names = []
    for group, name in _parse_defaults(defaults):
        group_names.append(group)
        cfg[group] = _load_yaml(CONFIG_DIR / group / f"{name}.yaml")
    _apply_overrides(cfg, overrides, group_names)
    return _resolve_interpolations(cfg)


def _materialize_run_dir(cfg: MutableMapping[str, Any]) -> Path:
    output_dir = str(cfg.get("output_dir", "outputs"))
    tag = str(cfg.get("tag", ""))
    task = _task_name(cfg)
    timestamp = dt.datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
    run_dir = f"{output_dir}/{task}/{timestamp}{tag}"
    cfg["run_dir"] = run_dir
    hydra_cfg = cfg.get("hydra")
    if isinstance(hydra_cfg, MutableMapping):
        hydra_cfg.setdefault("run", {})
        if isinstance(hydra_cfg.get("run"), MutableMapping):
            hydra_cfg["run"]["dir"] = run_dir
    return Path(run_dir)


def _write_hydra_snapshot(run_dir: Path, cfg: Mapping[str, Any]) -> None:
    hydra_dir = run_dir / ".hydra"
    hydra_dir.mkdir(parents=True, exist_ok=True)
    with (hydra_dir / "config.yaml").open("w", encoding="utf-8") as fh:
        yaml.safe_dump(dict(cfg), fh, sort_keys=False)


def _run_fallback(argv: Sequence[str]) -> int:
    cfg = _compose_config(argv)
    run_dir = _materialize_run_dir(cfg)
    _write_hydra_snapshot(run_dir, cfg)
    os.makedirs(run_dir, exist_ok=True)
    os.chdir(run_dir)
    task_name = _task_name(cfg)
    module = _load_task_module(task_name)
    if not hasattr(module, "main"):
        raise AttributeError(f"processes.{task_name} has no main()")
    return int(_call_main(module.main, cfg))


if _HYDRA_AVAILABLE:
    # CONTRACT: Hydra config is the single source of truth and defines task routing.
    @hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="config")
    def main(cfg: Mapping[str, Any]) -> int:
        task_name = _task_name(cfg)
        module = _load_task_module(task_name)
        if not hasattr(module, "main"):
            raise AttributeError(f"processes.{task_name} has no main()")
        return int(_call_main(module.main, cfg))
else:
    def main(cfg: Mapping[str, Any] | None = None) -> int:
        if cfg is not None:
            task_name = _task_name(cfg)
            module = _load_task_module(task_name)
            if not hasattr(module, "main"):
                raise AttributeError(f"processes.{task_name} has no main()")
            return int(_call_main(module.main, cfg))
        return _run_fallback(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
