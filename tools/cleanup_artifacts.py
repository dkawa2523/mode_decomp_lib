"""Utility to clean generated artifacts in the repo (safe by default)."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]


def _is_within_repo(path: Path) -> bool:
    try:
        path.resolve().relative_to(REPO_ROOT)
        return True
    except ValueError:
        return False


def _iter_targets(include_venv: bool, include_cache: bool) -> Iterable[Path]:
    yield REPO_ROOT / "runs"
    yield REPO_ROOT / "outputs"
    yield REPO_ROOT / "work" / ".autopilot"
    yield REPO_ROOT / "work" / "_logs"
    if include_cache:
        yield REPO_ROOT / ".pytest_cache"
        for root in ("src", "tests", "tools", "mode_decomp_ml"):
            base = REPO_ROOT / root
            if not base.exists():
                continue
            for cache_dir in base.rglob("__pycache__"):
                yield cache_dir
    if include_venv:
        yield REPO_ROOT / ".venv"
        yield REPO_ROOT / ".venv_release"
        yield REPO_ROOT / ".venv_release_copy"


def _remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Clean generated artifacts in the repo.")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete files/directories (default is dry-run).",
    )
    parser.add_argument(
        "--include-venv",
        action="store_true",
        help="Also remove .venv* directories.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Skip removing __pycache__ and .pytest_cache.",
    )
    args = parser.parse_args(argv)

    targets = list(_iter_targets(args.include_venv, not args.no_cache))
    existing = [path for path in targets if path.exists()]

    if not existing:
        print("No artifacts found.")
        return 0

    for path in existing:
        if not _is_within_repo(path):
            raise RuntimeError(f"Refusing to delete outside repo: {path}")
        if args.apply:
            _remove_path(path)
            print(f"deleted: {path.relative_to(REPO_ROOT)}")
        else:
            print(f"would delete: {path.relative_to(REPO_ROOT)}")

    if not args.apply:
        print("Dry-run only. Re-run with --apply to delete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
