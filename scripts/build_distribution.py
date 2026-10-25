"""Build a consumer ZIP containing wheels, source archive, examples and user guides."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import tomllib
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


GUIDES = (
    "CONFIGURATION",
    "DATA_FORMAT",
    "METHODS",
    "ANALYSIS",
    "SEARCH",
    "SCATTERED",
    "UNCERTAINTY",
    "HODGE",
    "MODES",
    "FIELD_MAPPING",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, help="Output directory; default: dist/<package version>")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pkgs/mode_decomp/pyproject.toml").read_text(encoding="utf-8"))["project"]
    output = (args.out_dir or root / "dist" / project["version"]).resolve()
    subprocess.run([sys.executable, str(root / "scripts/build_wheels.py"), "--out-dir", str(output)], check=True)
    stem = f"{project['name'].replace('-', '_')}-{project['version']}"
    core = (root / "vendor/wheels/core.json").read_bytes()
    wheels = [output / json.loads(core)["filename"], output / f"{stem}-py3-none-any.whl"]
    source = output / f"{stem}.tar.gz"
    files = {f"wheels/{path.name}": path.read_bytes() for path in wheels}
    files[f"sources/{source.name}"] = source.read_bytes()
    files["core.json"] = core
    files["build-constraints.txt"] = (root / "config/build-constraints.txt").read_bytes()
    readme = (root / "docs/DISTRIBUTION.md").read_text(encoding="utf-8")
    for name in GUIDES:
        filename = f"{name}.md"
        files[f"docs/{filename}"] = (root / "docs" / filename).read_bytes()
        readme = readme.replace(f"]({filename}", f"](docs/{filename}")
    readme = readme.replace("](../config/", "](config/").replace("](../examples/", "](examples/")
    files["README.md"] = readme.encode("utf-8")
    for directory, pattern in (("config/tasks", "*.yaml"), ("examples", "*.py")):
        for path in sorted((root / directory).glob(pattern)):
            files[path.relative_to(root).as_posix()] = path.read_bytes()
    files["SHA256SUMS"] = "".join(
        f"{hashlib.sha256(data).hexdigest()}  {name}\n" for name, data in sorted(files.items())
    ).encode("utf-8")
    archive_path = output / f"{stem}.zip"
    with TemporaryDirectory(prefix=".distribution-", dir=output) as temporary:
        prepared = Path(temporary) / archive_path.name
        with ZipFile(prepared, "w", compression=ZIP_DEFLATED) as archive:
            for name, data in sorted(files.items()):
                item = ZipInfo(f"{stem}/{name}")
                item.compress_type = ZIP_DEFLATED
                item.external_attr = 0o644 << 16
                archive.writestr(item, data)
        prepared.replace(archive_path)
    artifacts = [*wheels, source, archive_path]
    (output / "SHA256SUMS").write_text(
        "".join(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n" for path in artifacts),
        encoding="utf-8",
        newline="\n",
    )
    print(f"Consumer distribution: {archive_path}")


if __name__ == "__main__":
    main()
