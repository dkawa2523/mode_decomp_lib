"""Create deterministic sample fields and condition-only prediction input."""

import argparse
import csv
from pathlib import Path

import numpy as np
import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/tiny_fields")
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(42)
    conditions = rng.uniform(-1, 1, (40, 2))
    x, y = np.meshgrid(np.arange(12) / 12, np.arange(10) / 10)
    basis = np.stack((np.cos(2 * np.pi * x), np.sin(2 * np.pi * y)), axis=0)
    fields = np.einsum("nk,khw->nhw", conditions, basis)[..., None]
    np.save(root / "field.npy", fields, allow_pickle=False)
    with (root / "samples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "array_index", "x1", "x2"])
        writer.writerows((f"trial_{i:03d}", i, *row) for i, row in enumerate(conditions))
    with (root / "new_conditions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "x2", "x1"])
        writer.writerows([("new_a", 0.5, 0.2), ("new_b", -0.4, 0.8)])
    manifest = {
        "schema": "field_dataset_v1",
        "samples": {"path": "samples.csv", "id_column": "sample_id", "index_column": "array_index"},
        "conditions": {"format": "table", "columns": ["x1", "x2"]},
        "fields": {"path": "field.npy", "layout": "NHWC", "kind": "scalar", "components": ["value"]},
        "geometry": {
            "kind": "rectangle",
            "axes": {
                "x": {"start": 0.0, "step": 1 / 12, "count": 12, "periodic": True},
                "y": {"start": 0.0, "step": 0.1, "count": 10, "periodic": True},
            },
        },
    }
    (root / "field_dataset.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    print(root.resolve())


if __name__ == "__main__":
    main()
