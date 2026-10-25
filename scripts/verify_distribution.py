"""Accept a delivered ZIP using stdlib venv and pip, without checkout inputs or uv."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile


_CONSUMER = r"""
import importlib.util
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import ml_platform_mode_decomp as package
from ml_platform_mode_decomp import FieldDataset, FieldRepresentation, load_model, read_field_output

bundle, workspace = (Path(value).resolve() for value in sys.argv[1:])
prefix = Path(sys.prefix).resolve()
installed = Path(package.__file__).resolve()
if not installed.is_relative_to(prefix) or not sys.flags.isolated:
    raise ValueError("Consumer must import its installed wheel in isolated mode")
for name in ("clearml", "ml_platform_tabular", "ml_platform_spdml", "mode_decomp_ml"):
    if importlib.util.find_spec(name) is not None:
        raise ValueError(f"Unexpected execution SDK or sibling package: {name}")

def run(command):
    result = subprocess.run(command, cwd=workspace, check=True, capture_output=True, text=True, encoding="utf-8")
    return result.stdout

def cli(filename, *overrides):
    command = [sys.executable, "-I", "-m", "ml_platform_mode_decomp", str(bundle / "config/tasks" / filename)]
    for override in overrides:
        command.extend(("--set", override))
    return json.loads(run(command))

console = prefix / ("Scripts/ml-platform-mode-decomp.exe" if sys.platform == "win32" else "bin/ml-platform-mode-decomp")
help_text = run([str(console), "--help"])
if "config" not in help_text or "--set" not in help_text:
    raise ValueError("Installed console entry point did not expose the run interface")

rng = np.random.default_rng(5)
fields = rng.normal(size=(3, 5, 6, 2))
data = FieldDataset.from_arrays(fields)
representation = FieldRepresentation({"decomposition": {"name": "dct2"}})
coefficients = representation.fit_transform(data)
np.testing.assert_allclose(representation.inverse_transform(coefficients), fields, rtol=1e-12, atol=1e-12)

single = workspace / "data/single_field"
run([sys.executable, "-I", str(bundle / "examples/create_single_field.py"), "--output", str(single)])
direct = cli("mode_decomp_representation.yaml", "runtime.output_dir=outputs/direct")
summary = json.loads(Path(direct["artifacts"]["analysis_summary"]).read_text(encoding="utf-8"))
if summary["metric_scope"] != "fitted_input_reconstruction":
    raise ValueError("Direct single-YAML example used the wrong evaluation scope")
if not np.isfinite(direct["metrics"]["field_rmse"]):
    raise ValueError("Direct example did not produce a finite reconstruction error")

training = workspace / "data/tiny_fields"
run([sys.executable, "-I", str(bundle / "examples/create_tiny_dataset.py"), "--output", str(training)])
trained = cli("mode_decomp_pipeline.yaml", "runtime.output_dir=outputs/training")
if not np.isfinite(trained["metrics"]["field_rmse"]):
    raise ValueError("Training example did not produce a finite final-test error")
model_path = Path(trained["artifacts"]["model"])
conditions = workspace / "new_conditions.csv"
shutil.copyfile(training / "new_conditions.csv", conditions)
hidden = workspace / "hidden_training_input"
if not training.resolve().is_relative_to(workspace) or not hidden.resolve().is_relative_to(workspace):
    raise ValueError("Training-input move must stay inside this acceptance workspace")
training.rename(hidden)
inferred = cli(
    "mode_decomp_infer.yaml",
    f"data.local_path={conditions}",
    f"model.artifact_path={model_path}",
    "runtime.output_dir=outputs/inference",
    "output.prediction_name=predictions.npy",
)
saved = read_field_output(inferred["artifacts"]["predictions"])
predictions = saved["arrays"]["predictions"]
if predictions.shape != (2, 10, 12, 1) or not np.isfinite(predictions).all():
    raise ValueError("Fresh-process inference returned invalid fields")
if saved["arrays"]["sample_ids"].tolist() != ["new_a", "new_b"]:
    raise ValueError("Fresh-process inference did not preserve condition IDs")
np.testing.assert_array_equal(predictions, load_model(model_path).predict(np.array([[0.2, 0.5], [0.8, -0.4]])))

report = {
    "success": True,
    "python": sys.version.split()[0],
    "installed_package": str(installed),
    "isolated_python": bool(sys.flags.isolated),
    "console_entry_point": str(console),
    "sdk_packages_absent": True,
    "array_api_reconstruction": True,
    "single_yaml_example": direct["run_dir"],
    "pipeline_example": trained["run_dir"],
    "fresh_inference_example": inferred["run_dir"],
    "training_input_hidden_before_inference": not training.exists(),
    "condition_column_reordering_and_ids": True,
    "portable_npy_read": True,
    "bundle_inputs_only": True,
    "runtime": {name: version(name) for name in (
        "ml-platform-mode-decomp", "ml-platform-core", "numpy", "scipy", "scikit-learn", "pandas", "PyYAML", "joblib"
    )},
}
(workspace / "consumer_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, ensure_ascii=False))
"""


def _run(command, workspace, environment, log_name):
    completed = subprocess.run(
        command, cwd=workspace, env=environment, capture_output=True, text=True, encoding="utf-8"
    )
    (workspace / log_name).write_text(completed.stdout + completed.stderr, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(f"Distribution acceptance failed; see {workspace / log_name}")
    return completed.stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True, help="Distribution ZIP to accept")
    parser.add_argument("--work-dir", type=Path, required=True, help="New directory for extraction, venv and results")
    parser.add_argument("--python", default=sys.executable, help="Base interpreter for the single stdlib venv")
    args = parser.parse_args()
    archive = args.bundle.resolve(strict=True)
    workspace = args.work_dir.resolve()
    workspace.mkdir(parents=True, exist_ok=False)
    extracted = workspace / "bundle"
    with ZipFile(archive) as source:
        source.extractall(extracted)
    roots = [path for path in extracted.iterdir() if path.is_dir() and (path / "wheels").is_dir()]
    if len(roots) != 1:
        raise ValueError("Distribution ZIP must contain one package directory with wheels/")
    bundle = roots[0]
    verified = 0
    for line in (bundle / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        expected, reference = line.split(maxsplit=1)
        path = (bundle / reference).resolve()
        if not path.is_relative_to(bundle) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Distribution checksum mismatch: {reference}")
        verified += 1
    if not verified:
        raise ValueError("Distribution SHA256SUMS is empty")
    wheels = []
    for pattern in ("ml_platform_core-*.whl", "ml_platform_mode_decomp-*.whl"):
        matches = list((bundle / "wheels").glob(pattern))
        if len(matches) != 1:
            raise ValueError(f"Distribution must contain exactly one {pattern}")
        wheels.append(str(matches[0]))
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    environment.update(PYTHONUTF8="1", PIP_DISABLE_PIP_VERSION_CHECK="1")
    virtualenv = workspace / "environment"
    _run([str(Path(args.python).resolve()), "-I", "-m", "venv", str(virtualenv)], workspace, environment, "venv.log")
    interpreter = virtualenv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    _run([str(interpreter), "-I", "-m", "pip", "install", *wheels], workspace, environment, "install.log")
    _run([str(interpreter), "-I", "-m", "pip", "check"], workspace, environment, "pip_check.log")
    consumer = workspace / "consumer.py"
    consumer.write_text(_CONSUMER, encoding="utf-8")
    output = _run(
        [str(interpreter), "-I", str(consumer), str(bundle), str(workspace)], workspace, environment, "consumer.log"
    )
    report = json.loads(output)
    report.update(
        bundle=str(archive),
        bundle_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        checksums_verified=verified,
        created_with="python -m venv",
        installed_with="python -m pip install two local wheels",
        pip_check=True,
        uv_required=False,
    )
    (workspace / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
