"""Create collocated x/y vector observations on an annulus for Hodge analysis."""

import argparse
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/hodge_fields"))
    parser.add_argument("--samples", type=int, default=1)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be at least one")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    axis = -1.0 + (1.0 / 12.0) * np.arange(25, dtype=np.float64)
    x, y = np.meshgrid(axis, axis)
    radius2 = x**2 + y**2
    radius = np.sqrt(radius2).astype(np.float32)
    valid = (radius >= 0.3) & (radius <= 1)
    inverse_radius2 = np.divide(1.0, radius2, out=np.zeros_like(radius2), where=valid)
    fields = np.stack(
        (
            np.stack((2 * x, 2 * y), axis=-1),
            np.stack((-y, x), axis=-1),
            np.stack((-y * inverse_radius2, x * inverse_radius2), axis=-1),
        )
    )
    amplitudes = np.random.default_rng(81).uniform(-1, 1, size=(args.samples, 3))
    samples = np.einsum("nk,khwc->nhwc", amplitudes, fields)
    # A continuous analytic family, not a claim that its node samples are exact cochain components.
    samples[:, ~valid] = np.nan
    np.save(root / "field.npy", samples)
    print(root)


if __name__ == "__main__":
    main()
