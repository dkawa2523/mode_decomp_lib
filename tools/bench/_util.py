from __future__ import annotations

import csv
import json
import math
import os
import re
from pathlib import Path
from typing import Any


def raise_csv_field_size_limit(*, mib: int = 50) -> None:
    """Increase CSV field size limit for benches that store large stringified arrays."""
    try:
        csv.field_size_limit(1024 * 1024 * int(mib))
    except Exception:
        pass


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        return [dict(row) for row in reader]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = sorted({k for r in rows for k in r.keys()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in keys})


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None


def to_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return int(value)
    s = str(value).strip()
    if not s:
        return None
    try:
        return int(float(s))
    except Exception:
        return None


def relpath(path: str | Path, *, base_dir: Path) -> str:
    p = Path(str(path))
    try:
        rel = os.path.relpath(p, start=base_dir)
        return Path(rel).as_posix()
    except Exception:
        return p.as_posix()


def safe_slug(text: str) -> str:
    s = str(text or "").strip()
    if not s:
        return "untitled"
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    s = s.strip("_")
    return s or "untitled"


def md_escape(text: Any) -> str:
    s = "" if text is None else str(text)
    return s.replace("|", "\\|").replace("\n", " ")


def fmt_sci(value: Any, *, digits: int = 3) -> str:
    v = to_float(value)
    if v is None:
        return ""
    return f"{v:.{digits}e}"


def fmt_float(value: Any, *, digits: int = 6) -> str:
    v = to_float(value)
    if v is None:
        return ""
    return f"{v:.{digits}f}"


def fmt_compact(value: float | None, *, digits: int = 2) -> str:
    if value is None:
        return ""
    v = float(value)
    if not math.isfinite(v):
        return ""
    if abs(v - round(v)) < 1e-6:
        return str(int(round(v)))
    return f"{v:.{digits}f}".rstrip("0").rstrip(".")


def fmt_time_sec(value: Any) -> str:
    v = to_float(value)
    if v is None:
        return ""
    if v < 1e-3:
        return f"{v*1e6:.1f}us"
    if v < 1.0:
        return f"{v*1e3:.1f}ms"
    return f"{v:.3f}s"

