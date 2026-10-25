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
    result = subprocess.run(command, cwd=workspace, check=True, capture_output=True, text=True, encoding="utf-8", errors="replace")
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
    "output.coefficients_name=target_z.npz",
)
saved = read_field_output(inferred["artifacts"]["predictions"])
predictions = saved["arrays"]["predictions"]
if predictions.shape != (2, 10, 12, 1) or not np.isfinite(predictions).all():
    raise ValueError("Fresh-process inference returned invalid fields")
if saved["arrays"]["sample_ids"].tolist() != ["new_a", "new_b"]:
    raise ValueError("Fresh-process inference did not preserve condition IDs")
np.testing.assert_array_equal(predictions, load_model(model_path).predict(np.array([[0.2, 0.5], [0.8, -0.4]])))
from ml_platform_mode_decomp.inputs import load_surrogate_inputs
coefficient_input = load_surrogate_inputs({"input": {
    "kind": "coefficients", "local_path": inferred["run_dir"],
    "path": "target_z.npz", "model_artifact": str(model_path),
}})
np.testing.assert_array_equal(coefficient_input.sample_ids, saved["arrays"]["sample_ids"])
np.testing.assert_allclose(coefficient_input.representation.inverse_transform(coefficient_input.coefficients), predictions)
np.save(workspace / "reuse_expected.npy", predictions, allow_pickle=False)

mapping_root = workspace / "data/field_mapping"
run([sys.executable, "-I", str(bundle / "examples/create_field_mapping_dataset.py"), "--output", str(mapping_root)])
mapped = cli("mode_decomp_field_mapping.yaml", "runtime.output_dir=outputs/mapping_training")
for name in ("source", "target"):
    original, relocated = mapping_root / name, mapping_root / ("hidden_" + name)
    if not original.resolve().is_relative_to(workspace) or not relocated.resolve().is_relative_to(workspace):
        raise ValueError("Mapping-input move must stay inside this acceptance workspace")
    original.rename(relocated)
mapping_inferred = cli(
    "mode_decomp_field_mapping_infer.yaml",
    f"model.artifact_path={mapped['artifacts']['model']}",
    "runtime.output_dir=outputs/mapping_inference",
)
mapping_output = read_field_output(mapping_inferred["artifacts"]["predictions"])
mapping_fields = mapping_output["arrays"]["predictions"]
if mapping_fields.shape != (2, 7, 5, 1) or mapping_output["metadata"]["field_units"] != ["Pa"]:
    raise ValueError("Source-only mapping did not preserve the target shape and units")
from ml_platform_mode_decomp.data import load_field_dataset
mapping_model = load_model(mapped["artifacts"]["model"])
new_source = load_field_dataset(mapping_root / "new_source")
target_z = mapping_model.predict_coefficients(new_source)
np.testing.assert_allclose(mapping_model.representation.inverse_transform(target_z), mapping_fields, atol=1e-12)
if mapping_output["arrays"]["sample_ids"].tolist() != ["new_0", "new_1"]:
    raise ValueError("Source-only mapping did not preserve query sample IDs")

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
    "reusable_predicted_target_z": True,
    "model": str(model_path),
    "model_requirements": trained["artifacts"]["model_requirements"],
    "bundle_inputs_only": True,
    "field_mapping_source_only_inference": True,
    "field_mapping_target_coefficients_and_units": True,
    "runtime": {name: version(name) for name in (
        "ml-platform-mode-decomp", "ml-platform-core", "numpy", "scipy", "scikit-learn", "pandas", "PyYAML", "joblib"
    )},
}
(workspace / "consumer_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, ensure_ascii=False))
"""

_REUSE_CONSUMER = r"""
import hashlib
from pathlib import Path
import sys
import numpy as np
import ml_platform_mode_decomp as package
from ml_platform_mode_decomp import load_model

model_path, expected_path, expected_hash = sys.argv[1:]
if not sys.flags.isolated or not Path(package.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()):
    raise ValueError("Model reuse must import only the second environment's installed wheel")
if hashlib.sha256(Path(model_path).read_bytes()).hexdigest() != expected_hash:
    raise ValueError("Model reuse must retain the original saved model bytes")
predictions = load_model(model_path).predict(np.array([[0.2, 0.5], [0.8, -0.4]]))
np.testing.assert_array_equal(predictions, np.load(expected_path, allow_pickle=False))
print("Same saved model predictions match in a second environment installed from its runtime pins")
"""


def _run(command, workspace, environment, log_name):
    completed = subprocess.run(
        command, cwd=workspace, env=environment, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    (workspace / log_name).write_text(completed.stdout + completed.stderr, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(f"Distribution acceptance failed; see {workspace / log_name}")
    return completed.stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True, help="Distribution ZIP to accept")
    parser.add_argument("--work-dir", type=Path, required=True, help="New directory for extraction, venv and results")
    parser.add_argument("--python", default=sys.executable, help="Base interpreter for the consumer stdlib venvs")
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
    # Reuse the exact model bytes, rather than retraining in a new environment.
    import shutil

    reuse = workspace / "model-reuse"
    reuse.mkdir()
    for name in ("model.joblib", "model_info.json", "model_requirements.txt"):
        shutil.copyfile(Path(report["model"]).parent / name, reuse / name)
    reuse_environment = workspace / "reuse-environment"
    _run(
        [str(Path(args.python).resolve()), "-I", "-m", "venv", str(reuse_environment)],
        workspace,
        environment,
        "reuse_venv.log",
    )
    reuse_interpreter = reuse_environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    _run(
        [str(reuse_interpreter), "-I", "-m", "pip", "install", *wheels, "-r", str(reuse / "model_requirements.txt")],
        workspace,
        environment,
        "reuse_install.log",
    )
    _run([str(reuse_interpreter), "-I", "-m", "pip", "check"], workspace, environment, "reuse_pip_check.log")
    reuse_consumer = workspace / "reuse.py"
    reuse_consumer.write_text(_REUSE_CONSUMER, encoding="utf-8")
    _run(
        [
            str(reuse_interpreter),
            "-I",
            str(reuse_consumer),
            str(reuse / "model.joblib"),
            str(workspace / "reuse_expected.npy"),
            hashlib.sha256(Path(report["model"]).read_bytes()).hexdigest(),
        ],
        workspace,
        environment,
        "reuse.log",
    )
    report.update(
        bundle=str(archive),
        bundle_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
        checksums_verified=verified,
        created_with="python -m venv",
        installed_with="python -m pip install two local wheels",
        pip_check=True,
        uv_required=False,
        same_bundle_reused_in_second_environment=True,
    )
    (workspace / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
