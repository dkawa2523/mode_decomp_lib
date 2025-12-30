"""Process entrypoint: leaderboard."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from mode_decomp_ml.pipeline import build_meta, cfg_get, ensure_dir, require_cfg_keys, resolve_run_dir, write_json
from mode_decomp_ml.tracking.leaderboard import collect_rows, write_leaderboard


def _require_task_config(task_cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    if task_cfg is None:
        raise ValueError("task config is required for leaderboard")
    return task_cfg


def _resolve_output_path(base_dir: Path, value: str | None, default: str) -> Path:
    raw = value if value is not None and str(value).strip() else default
    path = Path(str(raw))
    return path if path.is_absolute() else base_dir / path


def main(cfg: Mapping[str, Any] | None = None) -> int:
    if cfg is None:
        raise ValueError("leaderboard requires config from the Hydra entrypoint")
    require_cfg_keys(cfg, ["run_dir", "output_dir", "task"])
    task_cfg = _require_task_config(cfg_get(cfg, "task", None))

    run_dir = resolve_run_dir(cfg)
    ensure_dir(run_dir)

    patterns = cfg_get(task_cfg, "runs", None)
    if patterns is None:
        patterns = ["outputs/**/eval"]
    if isinstance(patterns, str):
        patterns = [patterns]

    output_csv = _resolve_output_path(run_dir, cfg_get(task_cfg, "output_csv", None), "leaderboard.csv")
    output_md = cfg_get(task_cfg, "output_md", "leaderboard.md")
    output_md_path = (
        _resolve_output_path(run_dir, output_md, "leaderboard.md") if output_md is not None else None
    )

    sort_by = cfg_get(task_cfg, "sort_by", None)
    descending = bool(cfg_get(task_cfg, "descending", False))

    rows = collect_rows(patterns)
    if not rows:
        raise ValueError("leaderboard found no runs with metrics")
    # CONTRACT: leaderboard aggregates metrics.json into CSV/Markdown tables.
    write_leaderboard(
        rows,
        output_csv=output_csv,
        output_md=output_md_path,
        sort_by=sort_by,
        descending=descending,
    )

    meta = build_meta(cfg)
    write_json(run_dir / "meta.json", meta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
