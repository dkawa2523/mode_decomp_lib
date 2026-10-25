"""Generate irregular observations; --verify executes the single shipped analysis YAML."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def create_dataset(root: Path, *, samples: int = 80, points: int = 64, seed: int = 81) -> Path:
    if samples < 12 or points < 12:
        raise ValueError("Use at least 12 samples and 12 points for the comparison example")
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(seed)
    # A fixed interior hull leaves target boundary points unsupported, explicitly.
    corners = np.array([[0.07, 0.09], [0.93, 0.09], [0.07, 0.91], [0.93, 0.91]])
    xy = np.vstack((corners, rng.uniform([0.08, 0.10], [0.92, 0.90], (points - 4, 2))))
    conditions = rng.uniform(-1, 1, (samples, 2))
    a, b = conditions.T
    coefficients = np.column_stack((a, b, np.sin(2 * a), a * b, b**2, np.cos(3 * b)))
    x, y = xy.T
    basis = np.stack(
        (
            np.sin(2 * np.pi * x),
            np.cos(2 * np.pi * y),
            np.sin(2 * np.pi * (x + y)),
            x**2 - y**2,
            np.cos(3 * np.pi * x) * np.sin(np.pi * y),
            (x - 0.5) * (y - 0.5),
        )
    )
    values = (coefficients @ basis)[..., None]
    np.save(root / "points.npy", xy, allow_pickle=False)
    np.save(root / "values.npy", values, allow_pickle=False)
    with (root / "samples.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "array_index", "control_a", "control_b"])
        writer.writerows((f"measurement_{index:04d}", index, *row) for index, row in enumerate(conditions))
    return root


def verify_example(source: Path) -> Path:
    from ml_platform_mode_decomp import run_task
    from ml_platform_mode_decomp.config import load_config

    repository = Path(__file__).resolve().parents[1]
    tasks = repository / "config" / "tasks"
    output = source.parent / "scattered_runs"
    analysis_config = load_config(tasks / "mode_decomp_scattered_analysis.yaml")
    analysis_config["data"]["local_path"] = str(source)
    analysis_config.setdefault("runtime", {})["output_dir"] = str(output)
    analyzed = run_task(analysis_config)
    summary = json.loads(Path(analyzed.artifacts["analysis_summary"]).read_text(encoding="utf-8"))
    candidates = {candidate["id"] for candidate in analysis_config["model"]["candidates"]}
    details = [row for row in summary["details"] if row["candidate_id"] in candidates]
    if not details or not summary["complete"]:
        raise RuntimeError("The shipped scattered analysis did not complete its declared candidates")
    with Path(analyzed.tables["comparison"]).open(encoding="utf-8", newline="") as handle:
        comparisons = {
            (row["candidate_id"], int(row["split_seed"])): float(row["field_rmse"])
            for row in csv.DictReader(handle)
            if row["candidate_id"] in candidates and row["status"] == "ok"
        }
    rows = [
        {
            "candidate_id": row["candidate_id"],
            "split_seed": row["split_seed"],
            "field_rmse": comparisons[row["candidate_id"], row["split_seed"]],
            "observation_rmse": row["scattered"]["observation"]["rmse"],
            "interpolation_rmse": row["scattered"]["interpolation"]["rmse"],
            "point_coverage": row["scattered"]["observation"]["coverage"],
        }
        for row in details
    ]
    if not all(
        row["observation_rmse"][0] > 0 and row["interpolation_rmse"][0] > 0 and 0 < row["point_coverage"] < 1
        for row in rows
    ):
        raise RuntimeError(
            "Expected nonzero observation/interpolation errors and partial point coverage in this example"
        )
    result = {
        "source": str(source),
        "normalized_dataset": str(analyzed.artifacts["input_dataset_manifest"].parent),
        "analysis_run": str(analyzed.run_dir),
        "candidate_point_scores": rows,
        "note": "These deterministic toy scores demonstrate different measures; no universal error floor is claimed.",
    }
    path = source.parent / "scattered_verification.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/scattered_measurements"))
    parser.add_argument("--samples", type=int, default=80)
    parser.add_argument("--points", type=int, default=64)
    parser.add_argument("--seed", type=int, default=81)
    parser.add_argument("--verify", action="store_true", help="Run the shipped analysis YAML and record point scores")
    args = parser.parse_args()
    if args.samples < 12 or args.points < 12 or not 0 <= args.seed < 2**32:
        parser.error("Use samples/points >= 12 and seed in [0,2**32)")
    source = create_dataset(args.output, samples=args.samples, points=args.points, seed=args.seed)
    print(source)
    if args.verify:
        print(verify_example(source))


if __name__ == "__main__":
    main()
