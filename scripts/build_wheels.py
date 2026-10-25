"""Build a clean mode sdist and its wheel; include the fixed standalone core wheel."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import shutil
import tempfile
import tomllib


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", help="Output directory; default: dist/<package version>")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pkgs/mode_decomp/pyproject.toml").read_text(encoding="utf-8"))["project"]
    output = (Path(args.out_dir) if args.out_dir else root / "dist" / project["version"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    stem = f"{project['name'].replace('-', '_')}-{project['version']}"
    core = root / "vendor/wheels"
    info = json.loads((core / "core.json").read_text(encoding="utf-8"))
    wheel = core / info["filename"]
    if hashlib.sha256(wheel.read_bytes()).hexdigest() != info["sha256"]:
        raise ValueError("The fixed core artifact differs from vendor/wheels/core.json")
    destination = output / wheel.name
    if destination.resolve() != wheel.resolve():
        shutil.copy2(wheel, destination)
    with tempfile.TemporaryDirectory(prefix="mode-decomp-build-") as temporary:
        environment = dict(os.environ, SOURCE_DATE_EPOCH="1792454400")
        # setuptools can retain deleted modules under a previous build/lib.
        # Stage the current project instead of treating that cache as source.
        mode_project = Path(temporary) / "mode_decomp"
        shutil.copytree(
            root / "pkgs/mode_decomp",
            mode_project,
            ignore=shutil.ignore_patterns(
                "tests", "build", "dist", "*.egg-info", "__pycache__", ".pytest_cache", ".ruff_cache"
            ),
        )
        subprocess.run(
            ["uv", "build", "--sdist", "--out-dir", str(output), str(mode_project)],
            check=True,
            cwd=root,
            env=environment,
        )
        # Build from the delivered source archive, so omitted source/resources
        # cannot be masked by files present only in the development checkout.
        subprocess.run(
            ["uv", "build", "--wheel", "--out-dir", str(output), str(output / f"{stem}.tar.gz")],
            check=True,
            cwd=root,
            env=environment,
        )
    print(f"Standalone core wheel, mode sdist and wheel: {output}")


if __name__ == "__main__":
    main()
