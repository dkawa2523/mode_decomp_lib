#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
import sys
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.bench._util import read_csv as _read_csv  # noqa: E402


def _load_yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _resolve_decomposer_from_decompose_cfg_name(cfg_name: str, *, cfg_root: Path) -> str | None:
    p = cfg_root / f"{cfg_name}.yaml"
    if not p.exists():
        return None
    cfg = _load_yaml(p)
    if isinstance(cfg, dict):
        if isinstance(cfg.get("name"), str):
            return str(cfg["name"])
        decompose = cfg.get("decompose")
        if isinstance(decompose, dict) and isinstance(decompose.get("name"), str):
            return str(decompose["name"])
        defaults = cfg.get("defaults")
        if isinstance(defaults, list):
            for item in defaults:
                if not isinstance(item, str):
                    continue
                if item.startswith("/decompose/"):
                    tp = Path("configs/decompose") / (item.split("/decompose/", 1)[1] + ".yaml")
                    if tp.exists():
                        tc = _load_yaml(tp)
                        if isinstance(tc, dict) and isinstance(tc.get("name"), str):
                            return str(tc["name"])
    return None


def _registered_decomposers() -> set[str]:
    # Matches both decorators and direct calls, e.g.:
    #   @register_decomposer("fft2")
    #   register_decomposer('fft2')
    pat = re.compile(r"register_decomposer\(\s*['\"]([^'\"]+)['\"]")
    out: set[str] = set()
    for p in Path("src/mode_decomp_ml/plugins/decomposers").glob("*.py"):
        txt = p.read_text(encoding="utf-8")
        for m in pat.finditer(txt):
            out.add(m.group(1))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1-summary", type=str, default="runs/benchmarks/v1/summary/benchmark_summary_decomposition.csv")
    ap.add_argument(
        "--missing-summary",
        type=str,
        default="runs/benchmarks/v1_missing_methods/summary/benchmark_summary_decomposition.csv",
    )
    ap.add_argument(
        "--gappy-metrics",
        type=str,
        default="runs/benchmarks/v1_missing_methods/gappy_pod_rectangle_scalar/metrics.json",
    )
    args = ap.parse_args()

    cfg_root = Path("configs/decompose")
    executed_cfgs: set[str] = set()
    for p in [Path(args.v1_summary), Path(args.missing_summary)]:
        for r in _read_csv(p):
            name = str(r.get("decompose", "")).strip()
            if name:
                executed_cfgs.add(name)

    executed_decomposers: set[str] = set()
    for cfg_name in executed_cfgs:
        dec = _resolve_decomposer_from_decompose_cfg_name(cfg_name, cfg_root=cfg_root)
        if dec:
            executed_decomposers.add(dec)

    gappy_metrics = Path(args.gappy_metrics)
    if gappy_metrics.exists():
        executed_decomposers.add("gappy_pod")

    registered = _registered_decomposers()
    uncovered = sorted(registered - executed_decomposers)
    print("registered:", len(registered))
    print("executed:", len(executed_decomposers))
    print("uncovered:", len(uncovered))
    for name in uncovered:
        print(" -", name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
