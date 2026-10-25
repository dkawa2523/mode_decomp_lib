"""Write one scalar field for the single-YAML direct decomposition example."""

import argparse
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/single_field"))
    args = parser.parse_args()
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    x, y = np.meshgrid(np.arange(12), np.arange(10))
    field = (np.cos(2 * np.pi * x / 12) + 0.4 * np.sin(2 * np.pi * y / 10))[..., None]
    np.save(directory / "field.npy", field, allow_pickle=False)
    print(directory)


if __name__ == "__main__":
    main()
