#!/usr/bin/env python3
"""Migrate legacy run directories to the new layout."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable


LEGACY_MARKERS = (".hydra", "states", "figures", "tables", "metrics.json", "preds.npz")

FILE_MAP = {
    ".hydra/config.yaml": "configuration/run.yaml",
    "hydra/config.yaml": "configuration/run.yaml",
    "run.yaml": "configuration/run.yaml",
    "metrics.json": "outputs/metrics.json",
    "preds.npz": "outputs/preds.npz",
    "coeffs.npz": "outputs/coeffs.npz",
    "manifest_run.json": "outputs/manifest_run.json",
    "meta.json": "outputs/manifest_run.json",
}

DIR_MAP = {
    "states": "outputs/states",
    "figures": "plots",
    "tables": "outputs/tables",
    "artifacts": "outputs/artifacts",
    "preds": "outputs/preds",
}


def _iter_run_dirs(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if not path.is_dir():
            continue
        if any((path / marker).exists() for marker in LEGACY_MARKERS):
            yield path


def _ensure_dir(path: Path, *, dry_run: bool) -> None:
    if dry_run:
        return
    path.mkdir(parents=True, exist_ok=True)


def _merge_dirs(src: Path, dst: Path, *, move: bool, overwrite: bool, dry_run: bool) -> None:
    for child in src.rglob("*"):
        rel = child.relative_to(src)
        target = dst / rel
        if child.is_dir():
            _ensure_dir(target, dry_run=dry_run)
            continue
        if target.exists():
            if not overwrite:
                continue
            if not dry_run:
                target.unlink()
        _ensure_dir(target.parent, dry_run=dry_run)
        if not dry_run:
            if move:
                shutil.move(str(child), str(target))
            else:
                shutil.copy2(child, target)
    if move and not dry_run:
        shutil.rmtree(src)


def _transfer_path(src: Path, dst: Path, *, move: bool, overwrite: bool, dry_run: bool) -> None:
    if src.is_dir():
        _merge_dirs(src, dst, move=move, overwrite=overwrite, dry_run=dry_run)
        return
    if dst.exists():
        if not overwrite:
            return
        if not dry_run:
            dst.unlink()
    _ensure_dir(dst.parent, dry_run=dry_run)
    if not dry_run:
        if move:
            shutil.move(str(src), str(dst))
        else:
            shutil.copy2(src, dst)


def migrate_run_dir(run_dir: Path, *, move: bool, overwrite: bool, dry_run: bool) -> list[str]:
    actions: list[str] = []
    for rel_src, rel_dst in FILE_MAP.items():
        src = run_dir / rel_src
        if not src.exists():
            continue
        dst = run_dir / rel_dst
        actions.append(f"{src} -> {dst}")
        _transfer_path(src, dst, move=move, overwrite=overwrite, dry_run=dry_run)

    for rel_src, rel_dst in DIR_MAP.items():
        src = run_dir / rel_src
        if not src.exists():
            continue
        dst = run_dir / rel_dst
        actions.append(f"{src} -> {dst}")
        _transfer_path(src, dst, move=move, overwrite=overwrite, dry_run=dry_run)

    return actions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Migrate legacy run directories to the new layout.")
    parser.add_argument("--root", default="runs", help="Root directory to scan (default: runs)")
    parser.add_argument("--run", default=None, help="Single run directory to migrate")
    parser.add_argument("--move", action="store_true", help="Move files instead of copying")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing targets")
    parser.add_argument("--dry-run", action="store_true", help="Show actions without modifying files")
    args = parser.parse_args(argv)

    root = Path(args.root).expanduser()
    if args.run:
        run_dirs = [Path(args.run).expanduser()]
    else:
        run_dirs = list(_iter_run_dirs(root))

    if not run_dirs:
        print("No legacy run directories found.")
        return 1

    total = 0
    for run_dir in sorted(set(run_dirs)):
        actions = migrate_run_dir(run_dir, move=args.move, overwrite=args.overwrite, dry_run=args.dry_run)
        if not actions:
            continue
        total += 1
        print(f"[migrate] {run_dir}")
        for action in actions:
            print(f"  - {action}")

    if total == 0:
        print("No migrations applied.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
