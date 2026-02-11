"""Common helpers for process entrypoints.

These are intentionally small wrappers around `ArtifactWriter` + `StepRecorder` so
individual processes keep their core logic readable while run initialization and
finalization stay consistent across entrypoints.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from mode_decomp_ml.pipeline.artifacts import ArtifactWriter
from mode_decomp_ml.pipeline.steps import StepRecorder, artifact_ref
from mode_decomp_ml.pipeline.utils import build_meta, snapshot_inputs


def init_run(
    *,
    writer: ArtifactWriter,
    steps: StepRecorder,
    cfg: Mapping[str, Any],
    run_dir: str | Path,
    clean: bool = True,
    snapshot: bool = True,
) -> None:
    """Standard run initialization for process entrypoints."""
    with steps.step(
        "init_run",
        outputs=[artifact_ref("configuration/run.yaml", kind="config")],
    ):
        writer.prepare_layout(clean=bool(clean))
        writer.write_run_yaml(cfg)
        if snapshot:
            snapshot_inputs(cfg, run_dir)


def finalize_run(
    *,
    writer: ArtifactWriter,
    steps: StepRecorder,
    cfg: Mapping[str, Any],
    dataset_hash: str | None = None,
    dataset_meta: Mapping[str, Any] | None = None,
    preds_meta: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> None:
    """Write manifest + step log for a run (common across processes)."""
    meta = build_meta(cfg, dataset_hash=dataset_hash)
    writer.write_manifest(
        meta=meta,
        dataset_meta=dataset_meta,
        preds_meta=preds_meta,
        steps=steps.to_list(),
        extra=extra,
    )
    writer.write_steps(steps.to_list())
