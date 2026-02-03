#!/usr/bin/env python3
"""Run benchmark sweeps based on scripts/bench/matrix.yaml."""
from __future__ import annotations

import argparse
import importlib
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = ROOT / "configs"


def _load_yaml(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"matrix not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        return {}
    if not isinstance(data, Mapping):
        raise ValueError("matrix must be a mapping")
    return dict(data)


def _listify(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _profile_allows(entry: Mapping[str, Any], profile: str) -> bool:
    profiles = entry.get("profiles")
    if profiles is None:
        return True
    return profile in _listify(profiles)


def _config_exists(group: str, name: str) -> bool:
    path = CONFIGS_DIR / group / f"{name}.yaml"
    if path.exists():
        return True
    if group == "decompose":
        for subdir in ("analytic", "data_driven"):
            if (CONFIGS_DIR / group / subdir / f"{name}.yaml").exists():
                return True
    return False


def _missing_modules(modules: Iterable[str]) -> list[str]:
    missing: list[str] = []
    for module in modules:
        try:
            importlib.import_module(module)
        except Exception:
            missing.append(module)
    return missing


def _slugify(value: str) -> str:
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
    return "".join(ch if ch in allowed else "-" for ch in value)


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    return str(value)


def _resolve_coeff_posts(profile: str, model_entry: Mapping[str, Any]) -> list[str]:
    if profile == "quick" and "coeff_posts_quick" in model_entry:
        return [str(x) for x in _listify(model_entry.get("coeff_posts_quick"))]
    return [str(x) for x in _listify(model_entry.get("coeff_posts"))]


def _prepare_models(
    matrix: Mapping[str, Any],
    model_names: Iterable[str],
    profile: str,
) -> dict[str, Mapping[str, Any]]:
    models_cfg = matrix.get("models")
    if not isinstance(models_cfg, Mapping):
        raise ValueError("matrix.models must be a mapping")

    model_specs: dict[str, Mapping[str, Any]] = {}
    for model_name in model_names:
        entry = models_cfg.get(model_name)
        if not isinstance(entry, Mapping):
            raise ValueError(f"model entry missing: {model_name}")
        if not _profile_allows(entry, profile):
            continue

        optional = bool(entry.get("optional", False))
        if not _config_exists("model", model_name):
            reason = f"missing config: configs/model/{model_name}.yaml"
            if optional:
                print(f"SKIP model={model_name} reason={reason}")
                continue
            raise FileNotFoundError(reason)

        missing = _missing_modules(_listify(entry.get("requires")))
        if missing:
            reason = f"missing dependency: {', '.join(missing)}"
            if optional:
                print(f"SKIP model={model_name} reason={reason}")
                continue
            raise RuntimeError(reason)

        coeff_posts = _resolve_coeff_posts(profile, entry)
        if not coeff_posts:
            if optional:
                print(f"SKIP model={model_name} reason=empty coeff_posts")
                continue
            raise ValueError(f"model={model_name} requires coeff_posts")

        resolved_coeffs: list[str] = []
        for coeff_name in coeff_posts:
            if _config_exists("coeff_post", coeff_name):
                resolved_coeffs.append(coeff_name)
            else:
                print(
                    f"SKIP model={model_name} coeff_post={coeff_name} reason=missing config"
                )
        if not resolved_coeffs:
            if optional:
                print(f"SKIP model={model_name} reason=no available coeff_post configs")
                continue
            raise FileNotFoundError(f"model={model_name} has no valid coeff_post configs")

        model_specs[model_name] = {
            "name": model_name,
            "coeff_posts": resolved_coeffs,
            "optional": optional,
        }
    return model_specs


def _run_cmd(cmd: list[str], *, env: dict[str, str], dry_run: bool) -> None:
    print("RUN:", " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True, env=env, cwd=str(ROOT))


def _build_env() -> dict[str, str]:
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH", "")
    root_src = str(ROOT / "src")
    if pythonpath:
        env["PYTHONPATH"] = f"{root_src}:{pythonpath}"
    else:
        env["PYTHONPATH"] = root_src
    return env


def _run_leaderboard(tag: str, *, env: dict[str, str], dry_run: bool) -> None:
    pattern = f"runs/{tag}/**/metrics.json"
    cmd = [
        sys.executable,
        "-m",
        "mode_decomp_ml.cli.run",
        "task=leaderboard",
        f"tag={tag}",
        f"task.runs={pattern}",
    ]
    _run_cmd(cmd, env=env, dry_run=dry_run)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run benchmark sweeps from matrix.yaml")
    parser.add_argument(
        "--matrix",
        default=str(Path("scripts/bench/matrix.yaml")),
        help="Path to matrix.yaml",
    )
    parser.add_argument(
        "--profile",
        default="quick",
        choices=["quick", "full"],
        help="Profile name from matrix.yaml",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    matrix_path = Path(args.matrix)
    if not matrix_path.is_absolute():
        matrix_path = (ROOT / matrix_path).resolve()
    matrix = _load_yaml(matrix_path)

    profiles_cfg = matrix.get("profiles")
    if not isinstance(profiles_cfg, Mapping):
        raise ValueError("matrix.profiles must be a mapping")
    profile_cfg = profiles_cfg.get(args.profile)
    if not isinstance(profile_cfg, Mapping):
        raise ValueError(f"profile not found: {args.profile}")

    tag = str(profile_cfg.get("tag", args.profile)).strip() or args.profile
    domains = [str(x) for x in _listify(profile_cfg.get("domains"))]
    models = [str(x) for x in _listify(profile_cfg.get("models"))]
    run_leaderboard = bool(profile_cfg.get("leaderboard", False))

    model_specs = _prepare_models(matrix, models, args.profile)
    if not model_specs:
        raise ValueError("No usable models resolved from matrix")

    domains_cfg = matrix.get("domains")
    if not isinstance(domains_cfg, Mapping):
        raise ValueError("matrix.domains must be a mapping")

    env = _build_env()
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    planned = 0
    executed = 0
    skipped = 0
    run_idx = 0

    for domain_name in domains:
        domain_entry = domains_cfg.get(domain_name)
        if not isinstance(domain_entry, Mapping):
            raise ValueError(f"domain entry missing: {domain_name}")
        domain_optional = bool(domain_entry.get("optional", False))
        domain_config = str(domain_entry.get("config", domain_name)).strip() or domain_name
        if not _config_exists("domain", domain_config):
            reason = f"missing config: configs/domain/{domain_config}.yaml"
            if domain_optional:
                print(f"SKIP domain={domain_name} reason={reason}")
                skipped += 1
                continue
            raise FileNotFoundError(reason)

        dataset_overrides = domain_entry.get("dataset_overrides")
        if dataset_overrides is None:
            dataset_overrides = {}
        if not isinstance(dataset_overrides, Mapping):
            raise ValueError(f"dataset_overrides must be a mapping for domain={domain_name}")

        decomposers = domain_entry.get("decomposers", [])
        if not isinstance(decomposers, list):
            raise ValueError(f"domain.decomposers must be a list for domain={domain_name}")

        for decomp_entry in decomposers:
            if not isinstance(decomp_entry, Mapping):
                raise ValueError(f"decomposer entry invalid for domain={domain_name}")
            if not _profile_allows(decomp_entry, args.profile):
                continue
            decomp_name = str(decomp_entry.get("name", "")).strip()
            if not decomp_name:
                raise ValueError(f"decomposer name missing for domain={domain_name}")
            codec_name = str(decomp_entry.get("codec", "none")).strip() or "none"
            decomp_optional = bool(decomp_entry.get("optional", False))

            if not _config_exists("decompose", decomp_name):
                reason = (
                    f\"missing config: configs/decompose/{decomp_name}.yaml "
                    \"(analytic/data_driven)\"
                )
                if decomp_optional:
                    print(
                        f"SKIP domain={domain_name} decompose={decomp_name} reason={reason}"
                    )
                    skipped += 1
                    continue
                raise FileNotFoundError(reason)

            if not _config_exists("codec", codec_name):
                reason = f"missing config: configs/codec/{codec_name}.yaml"
                if decomp_optional:
                    print(
                        f"SKIP domain={domain_name} decompose={decomp_name} reason={reason}"
                    )
                    skipped += 1
                    continue
                raise FileNotFoundError(reason)

            missing = _missing_modules(_listify(decomp_entry.get("requires")))
            if missing:
                reason = f"missing dependency: {', '.join(missing)}"
                if decomp_optional:
                    print(
                        f"SKIP domain={domain_name} decompose={decomp_name} reason={reason}"
                    )
                    skipped += 1
                    continue
                raise RuntimeError(reason)

            for model_name, model_spec in model_specs.items():
                coeff_posts = list(model_spec["coeff_posts"])
                if not coeff_posts:
                    continue
                run_id = _slugify(
                    f"{args.profile}-{domain_name}-{decomp_name}-{model_name}-{timestamp}-{run_idx:03d}"
                )
                run_idx += 1

                overrides = [
                    "task=benchmark",
                    f"tag={tag}",
                    f"run_id={run_id}",
                    f"domain={domain_config}",
                    f"model={model_name}",
                    f"codec={codec_name}",
                    f"task.decompose_list=[{decomp_name}]",
                    f"task.coeff_post_list=[{','.join(coeff_posts)}]",
                ]
                for key, value in dataset_overrides.items():
                    overrides.append(f"dataset.{key}={_format_value(value)}")

                cmd = [sys.executable, "-m", "mode_decomp_ml.cli.run", *overrides]
                planned += 1
                _run_cmd(cmd, env=env, dry_run=args.dry_run)
                executed += 1

    print(
        f"Summary: planned={planned} executed={executed} skipped={skipped} profile={args.profile} tag={tag}"
    )

    if run_leaderboard:
        _run_leaderboard(tag, env=env, dry_run=args.dry_run)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
