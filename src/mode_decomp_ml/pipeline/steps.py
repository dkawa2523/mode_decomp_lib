"""Step tracking helpers for process pipelines."""
from __future__ import annotations

import datetime as dt
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Sequence


def _utcnow() -> str:
    return dt.datetime.utcnow().isoformat() + "Z"


def artifact_ref(path: str | Path, *, kind: str | None = None) -> dict[str, Any]:
    ref = {"path": str(path)}
    if kind:
        ref["kind"] = str(kind)
    return ref


class StepRecorder:
    """Collects ordered step records for run manifests."""

    def __init__(self, *, run_dir: str | Path | None = None) -> None:
        self._steps: list[dict[str, Any]] = []
        self._counter = 0
        self._run_dir = Path(run_dir).resolve() if run_dir is not None else None

    def _normalize_path(self, path: str) -> str:
        path_obj = Path(path)
        if self._run_dir is None:
            return str(path_obj)
        if not path_obj.is_absolute():
            return path_obj.as_posix()
        try:
            return str(path_obj.resolve().relative_to(self._run_dir))
        except Exception:
            return str(path_obj)

    def _normalize_artifacts(self, artifacts: Sequence[Any] | None) -> list[dict[str, Any]]:
        if not artifacts:
            return []
        out: list[dict[str, Any]] = []
        for item in artifacts:
            if isinstance(item, Mapping):
                payload = dict(item)
            else:
                payload = {"path": str(item)}
            if "path" in payload:
                payload["path"] = self._normalize_path(str(payload["path"]))
            out.append(payload)
        return out

    @contextmanager
    def step(
        self,
        name: str,
        *,
        inputs: Sequence[Any] | None = None,
        outputs: Sequence[Any] | None = None,
        meta: Mapping[str, Any] | None = None,
    ) -> Iterable[MutableMapping[str, Any]]:
        record: dict[str, Any] = {
            "index": int(self._counter),
            "name": str(name),
            "status": "running",
            "started_at": _utcnow(),
            "inputs": self._normalize_artifacts(inputs),
            "outputs": self._normalize_artifacts(outputs),
            "meta": dict(meta) if isinstance(meta, Mapping) else {},
        }
        self._counter += 1
        self._steps.append(record)
        try:
            yield record
        except Exception as exc:
            record["status"] = "error"
            record["error"] = f"{type(exc).__name__}: {exc}"
            record["ended_at"] = _utcnow()
            raise
        else:
            record["status"] = "ok"
            record["ended_at"] = _utcnow()

    def add_step(
        self,
        name: str,
        *,
        inputs: Sequence[Any] | None = None,
        outputs: Sequence[Any] | None = None,
        meta: Mapping[str, Any] | None = None,
        status: str = "ok",
    ) -> Mapping[str, Any]:
        record = {
            "index": int(self._counter),
            "name": str(name),
            "status": str(status),
            "started_at": _utcnow(),
            "ended_at": _utcnow(),
            "inputs": self._normalize_artifacts(inputs),
            "outputs": self._normalize_artifacts(outputs),
            "meta": dict(meta) if isinstance(meta, Mapping) else {},
        }
        self._counter += 1
        self._steps.append(record)
        return record

    def to_list(self) -> list[dict[str, Any]]:
        return [dict(step) for step in self._steps]
