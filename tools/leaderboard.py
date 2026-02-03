#!/usr/bin/env python3
"""CLI for leaderboard aggregation."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mode_decomp_ml.tracking.leaderboard import collect_rows, write_leaderboard


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate run metrics into a leaderboard CSV.")
    parser.add_argument(
        "runs",
        nargs="*",
        default=["runs/**/metrics.json", "outputs/**/eval"],
        help="Run directories or glob patterns (default: runs/**/metrics.json, outputs/**/eval).",
    )
    parser.add_argument("--out", default="leaderboard.csv", help="Output CSV path.")
    parser.add_argument("--md", default="leaderboard.md", help="Output Markdown path.")
    parser.add_argument("--sort", default=None, help="Optional column to sort by.")
    parser.add_argument("--desc", action="store_true", help="Sort descending.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    rows = collect_rows(args.runs)
    if not rows:
        print("leaderboard: no runs found with metrics.json", file=sys.stderr)
        return 1

    output_csv = Path(args.out)
    output_md = Path(args.md) if args.md else None
    write_leaderboard(
        rows,
        output_csv=output_csv,
        output_md=output_md,
        sort_by=args.sort,
        descending=args.desc,
    )
    print(f"leaderboard: wrote {len(rows)} rows -> {output_csv}")
    if output_md is not None:
        print(f"leaderboard: markdown -> {output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
