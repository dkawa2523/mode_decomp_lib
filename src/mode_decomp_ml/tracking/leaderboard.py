"""Leaderboard aggregation helpers."""
from __future__ import annotations

import csv
import glob
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

_BASE_COLUMNS = (
    "run_dir",
    "task",
    "dataset",
    "domain",
    "decompose",
    "coeff_post",
    "model",
    "model_target",
    "seed",
    "dataset_hash",
)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _get_nested(cfg: Mapping[str, Any], *keys: str) -> Any:
    cur: Any = cfg
    for key in keys:
        if not isinstance(cur, Mapping) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _format_metric_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (int, float, str, bool)):
        return value
    return json.dumps(value)


def _format_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {str(name): _format_metric_value(value) for name, value in metrics.items()}


def _collect_run_dir(path: Path) -> Path | None:
    if path.is_dir():
        return path
    if path.is_file() and path.name == "metrics.json":
        return path.parent.parent
    return None


def collect_run_dirs(patterns: Sequence[str]) -> list[Path]:
    seen: set[Path] = set()
    run_dirs: list[Path] = []
    for pattern in patterns:
        if not pattern:
            continue
        matches = glob.glob(pattern, recursive=True)
        if not matches:
            candidate = Path(pattern)
            if candidate.exists():
                matches = [str(candidate)]
        for match in matches:
            run_dir = _collect_run_dir(Path(match))
            if run_dir is None:
                continue
            run_dir = run_dir.resolve()
            if run_dir in seen:
                continue
            seen.add(run_dir)
            run_dirs.append(run_dir)
    return run_dirs


def load_run_row(run_dir: Path) -> dict[str, Any] | None:
    # CONTRACT: leaderboard reads metrics from metrics/metrics.json.
    metrics_path = run_dir / "metrics" / "metrics.json"
    if not metrics_path.exists():
        return None

    metrics = _read_json(metrics_path)
    meta = _read_json(run_dir / "meta.json")
    dataset_meta = _read_json(run_dir / "artifacts" / "dataset_meta.json")

    config = {}
    for cfg_path in (run_dir / ".hydra" / "config.yaml", run_dir / "hydra" / "config.yaml"):
        if cfg_path.exists():
            config = _read_yaml(cfg_path)
            break

    row: dict[str, Any] = {
        "run_dir": str(run_dir),
        "task": _get_nested(config, "task", "name") or meta.get("task"),
        "dataset": _get_nested(config, "dataset", "name") or dataset_meta.get("name"),
        "domain": _get_nested(config, "domain", "name") or _get_nested(dataset_meta, "domain_cfg", "name"),
        "decompose": _get_nested(config, "decompose", "name"),
        "coeff_post": _get_nested(config, "coeff_post", "name"),
        "model": _get_nested(config, "model", "name"),
        "model_target": _get_nested(config, "model", "target_space"),
        "seed": config.get("seed") if isinstance(config, Mapping) else meta.get("seed"),
        "dataset_hash": meta.get("dataset_hash") or dataset_meta.get("dataset_hash"),
    }
    row.update(_format_metrics(metrics))
    return row


def collect_rows(patterns: Sequence[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_dir in collect_run_dirs(patterns):
        row = load_run_row(run_dir)
        if row is not None:
            rows.append(row)
    return rows


def build_columns(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    keys = set()
    for row in rows:
        keys.update(row.keys())
    columns = list(_BASE_COLUMNS)
    metrics = sorted(key for key in keys if key not in columns)
    return columns + metrics


def _sort_rows(rows: list[dict[str, Any]], sort_by: str | None, descending: bool) -> None:
    if not sort_by:
        return

    def _key(row: Mapping[str, Any]) -> tuple[int, Any]:
        value = row.get(sort_by)
        if value is None or value == "":
            return (1, 0)
        try:
            return (0, float(value))
        except (TypeError, ValueError):
            return (0, str(value))

    rows.sort(key=_key, reverse=descending)


def write_leaderboard(
    rows: list[dict[str, Any]],
    *,
    output_csv: Path,
    output_md: Path | None = None,
    sort_by: str | None = None,
    descending: bool = False,
) -> list[str]:
    _sort_rows(rows, sort_by, descending)
    columns = build_columns(rows)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})

    if output_md is not None:
        output_md.parent.mkdir(parents=True, exist_ok=True)
        with output_md.open("w", encoding="utf-8") as fh:
            fh.write("| " + " | ".join(columns) + " |\n")
            fh.write("| " + " | ".join(["---"] * len(columns)) + " |\n")
            for row in rows:
                cells = []
                for key in columns:
                    value = row.get(key, "")
                    if isinstance(value, float):
                        cell = f"{value:.6g}"
                    else:
                        cell = "" if value is None else str(value)
                    cells.append(cell.replace("|", "\\|"))
                fh.write("| " + " | ".join(cells) + " |\n")
    return columns
