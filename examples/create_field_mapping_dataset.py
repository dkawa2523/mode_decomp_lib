"""Create paired steady fields on two fixed geometries and a source-only query."""

import argparse
import csv
from pathlib import Path

import numpy as np
import yaml


def _fields(parameters, *, target):
    a, b = parameters.T
    height, width = (7, 5) if target else (6, 8)
    x, y = np.meshgrid(np.linspace(-1, 1, width), np.linspace(-1, 1, height))
    if target:
        basis = np.stack((x, y, 0.12 * (x**2 - y), 0.08 * x * y, 0.1 * (y**2 + x)))
        baseline = 20 + x - y
    else:
        basis = np.stack((x, y, 0.1 * x**2, 0.2 * x * y, 0.1 * y**2))
        baseline = 300 + 0.5 * x
    features = np.column_stack((a, b, a**2, a * b, b**2))
    return (baseline + np.einsum("nk,khw->nhw", features, basis))[..., None]


def _write_dataset(root, parameters, ids, *, target):
    root.mkdir()
    fields = _fields(parameters, target=target)
    np.save(root / "field.npy", fields, allow_pickle=False)
    with (root / "samples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "array_index"])
        writer.writerows((sample_id, index) for index, sample_id in enumerate(ids))
    height, width = fields.shape[1:3]
    x_bounds, y_bounds = ((10.0, 12.0), (0.0, 4.0)) if target else ((0.0, 3.0), (-1.0, 1.0))
    manifest = {
        "schema": "field_dataset_v1",
        "samples": {"path": "samples.csv"},
        "fields": {
            "path": "field.npy",
            "layout": "NHWC",
            "kind": "scalar",
            "components": ["stress" if target else "temperature"],
            "units": ["Pa" if target else "K"],
        },
        "geometry": {
            "kind": "rectangle",
            "axes": {
                "x": {"start": x_bounds[0], "step": (x_bounds[1] - x_bounds[0]) / (width - 1), "count": width},
                "y": {"start": y_bounds[0], "step": (y_bounds[1] - y_bounds[0]) / (height - 1), "count": height},
            },
        },
    }
    (root / "field_dataset.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/field_mapping"))
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    parameters = np.random.default_rng(47).uniform(-1, 1, (48, 2))
    ids = [f"case_{index:04d}" for index in range(len(parameters))]
    _write_dataset(root / "source", parameters, ids, target=False)
    _write_dataset(root / "target", parameters, ids, target=True)
    _write_dataset(root / "new_source", np.asarray([[0.2, -0.3], [-0.4, 0.6]]), ["new_0", "new_1"], target=False)
    print(root)


if __name__ == "__main__":
    main()
