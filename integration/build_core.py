"""Maintainer-only rebuild of the fixed core dependency; normal builds need no Git."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

PLATFORM_REVISION = "91aa29a235527b9ecd6fc7f6d17101beb0cfe9a0"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform-source", default="https://github.com/dkawa2523/ml_platform.git")
    parser.add_argument("--out-dir", default="vendor/wheels")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = Path(args.out_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    patch = root / "integration/ml_platform_core.patch"
    with tempfile.TemporaryDirectory(prefix="mode-core-build-") as temporary:
        checkout = Path(temporary) / "platform"
        subprocess.run(["git", "clone", "--no-checkout", args.platform_source, str(checkout)], check=True)
        subprocess.run(["git", "-C", str(checkout), "checkout", "--detach", PLATFORM_REVISION], check=True)
        subprocess.run(["git", "-C", str(checkout), "apply", "--check", str(patch)], check=True)
        subprocess.run(["git", "-C", str(checkout), "apply", str(patch)], check=True)
        subprocess.run(
            ["uv", "build", "--wheel", "--out-dir", str(output), str(checkout / "pkgs/core")],
            check=True,
            env=dict(os.environ, SOURCE_DATE_EPOCH="1792454400"),
        )
        import tomllib

        version = tomllib.loads((checkout / "pkgs/core/pyproject.toml").read_text(encoding="utf-8"))["project"][
            "version"
        ]
    wheel = output / f"ml_platform_core-{version}-py3-none-any.whl"
    record = {
        "filename": wheel.name,
        "version": version,
        "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
        "platform_revision": PLATFORM_REVISION,
        "patch_sha256": hashlib.sha256(patch.read_bytes()).hexdigest(),
    }
    (output / "core.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
