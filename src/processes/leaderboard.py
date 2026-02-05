"""Process entrypoint: leaderboard."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from mode_decomp_ml.pipeline import (
    ArtifactWriter,
    StepRecorder,
    artifact_ref,
    build_meta,
    cfg_get,
    require_cfg_keys,
    resolve_run_dir,
)
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
    require_cfg_keys(cfg, ["run_dir", "task"])
    task_cfg = _require_task_config(cfg_get(cfg, "task", None))

    run_dir = resolve_run_dir(cfg)
    writer = ArtifactWriter(run_dir)
    steps = StepRecorder(run_dir=run_dir)
    with steps.step(
        "init_run",
        outputs=[artifact_ref("configuration/run.yaml", kind="config")],
    ):
        writer.prepare_layout(clean=True)
        writer.write_run_yaml(cfg)

    patterns = cfg_get(task_cfg, "runs", None)
    if patterns is None:
        patterns = ["runs/**/outputs/metrics.json"]
    if isinstance(patterns, str):
        patterns = [patterns]

    output_csv = _resolve_output_path(
        writer.tables_dir,
        cfg_get(task_cfg, "output_csv", None),
        "leaderboard.csv",
    )
    output_md = cfg_get(task_cfg, "output_md", "leaderboard.md")
    output_md_path = (
        _resolve_output_path(writer.tables_dir, output_md, "leaderboard.md") if output_md is not None else None
    )

    sort_by = cfg_get(task_cfg, "sort_by", None)
    descending = bool(cfg_get(task_cfg, "descending", False))

    output_artifacts = [artifact_ref(output_csv, kind="table")]
    if output_md_path is not None:
        output_artifacts.append(artifact_ref(output_md_path, kind="table"))
    with steps.step(
        "collect_rows",
        outputs=output_artifacts,
    ):
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
    writer.write_manifest(meta=meta, steps=steps.to_list())
    writer.write_steps(steps.to_list())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
