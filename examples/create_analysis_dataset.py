"""A reproducible nonlinear condition-to-field fixture for the analysis examples."""

import argparse
import csv
from pathlib import Path

import numpy as np
import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/nonlinear_fields"))
    parser.add_argument("--samples", type=int, default=120)
    parser.add_argument("--noise", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=81)
    args = parser.parse_args()
    if args.samples < 10 or not np.isfinite(args.noise) or args.noise < 0:
        parser.error("Use at least 10 samples and a finite nonnegative noise standard deviation")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(args.seed)
    conditions = rng.uniform(-1, 1, (args.samples, 2))
    a, b = conditions.T
    coefficients = np.column_stack((a, b, a * b, np.sin(2 * a), b**2, np.cos(2 * b)))
    x, y = np.meshgrid(2 * np.pi * np.arange(24) / 24, 2 * np.pi * np.arange(20) / 20)
    basis = np.stack((np.sin(x), np.cos(y), np.sin(x + y), np.cos(2 * x), np.sin(2 * y), np.cos(x - 2 * y)))
    first = np.einsum("nk,khw->nhw", coefficients, basis)
    second = np.einsum("nk,khw->nhw", coefficients[:, ::-1], basis)
    fields = np.stack((first, second), axis=-1)
    fields += rng.normal(scale=args.noise, size=fields.shape)
    np.save(root / "field.npy", fields)
    with (root / "samples.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "array_index", "control_a", "control_b"])
        writer.writerows((f"case_{i:04d}", i, *values) for i, values in enumerate(conditions))
    manifest = {
        "schema": "field_dataset_v1",
        "samples": {"path": "samples.csv"},
        "conditions": {"format": "table", "columns": ["control_a", "control_b"]},
        "fields": {
            "path": "field.npy",
            "layout": "NHWC",
            "kind": "multichannel",
            "components": ["signal_a", "signal_b"],
            "units": ["arb", "arb"],
        },
        "geometry": {
            "kind": "rectangle",
            "axes": {
                "x": {"start": 0, "step": 2 * np.pi / 24, "count": 24, "periodic": True},
                "y": {"start": 0, "step": 2 * np.pi / 20, "count": 20, "periodic": True},
            },
        },
        "generation": {
            "generator": "examples/create_analysis_dataset.py",
            "seed": args.seed,
            "condition_mapping": ["a", "b", "a*b", "sin(2*a)", "b^2", "cos(2*b)"],
            "noise_std": args.noise,
            "reference": "generated noisy realization; not latent noise-free truth",
        },
    }
    (root / "field_dataset.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    (root / "new_conditions.csv").write_text(
        "sample_id,control_a,control_b\nnew_0,0.2,-0.3\nnew_1,-0.4,0.6\n", encoding="utf-8"
    )
    print(root)


if __name__ == "__main__":
    main()
