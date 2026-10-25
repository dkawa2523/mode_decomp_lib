"""Download the official Darcy simulation pairs and preserve the official test split.

Preparation requires PyTorch; the resulting NPY datasets only require the normal
package dependencies. This is a numerical PDE benchmark, not measured user data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import urllib.request

import numpy as np
import yaml


RECORD_URL = "https://zenodo.org/records/12784353"
LOADER_URL = "https://neuraloperator.github.io/dev/_modules/neuralop/data/datasets/darcy.html"
PAPER_URL = "https://arxiv.org/html/2010.08895v3"
ARCHIVE_MD5 = {16: "091575565d8c4ab17f8c3bf2bbb566cb", 32: "24c0033f0c0b75f3df546073111ae154"}


def _digest(path: Path, algorithm: str = "sha256") -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, algorithm).hexdigest()


def _archive(root: Path, resolution: int) -> Path:
    path = root / f"darcy_{resolution}.tgz"
    if not path.exists():
        partial = path.with_suffix(".tgz.partial")
        print(f"Downloading {path.name} from {RECORD_URL}", flush=True)
        with urllib.request.urlopen(f"{RECORD_URL}/files/{path.name}?download=1", timeout=120) as response:
            with partial.open("wb") as handle:
                shutil.copyfileobj(response, handle)
        if _digest(partial, "md5") != ARCHIVE_MD5[resolution]:
            raise ValueError(f"Official MD5 mismatch for {partial}; no input was converted")
        partial.replace(path)
    if _digest(path, "md5") != ARCHIVE_MD5[resolution]:
        raise ValueError(f"Official MD5 mismatch for {path}; no input was converted")
    return path


def _read_pairs(archive: Path, root: Path, resolution: int, counts: dict[str, int]):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Darcy .pt preparation requires PyTorch; the converted NPY files do not") from exc

    raw = root / "raw"
    raw.mkdir(exist_ok=True)
    arrays = {"source": [], "target": []}
    rows = []
    members = {}
    summaries = {}
    all_source_hashes = {}
    with tarfile.open(archive, "r:gz") as bundle:
        for split, count in counts.items():
            filename = f"darcy_{split}_{resolution}.pt"
            matches = [member for member in bundle.getmembers() if Path(member.name).name == filename]
            if len(matches) != 1 or not matches[0].isfile():
                raise ValueError(f"Archive must contain one regular file named {filename}")
            member = matches[0]
            # Copy only the two identified file bodies; never extract archive paths.
            path = raw / filename
            body = bundle.extractfile(member)
            if body is None:
                raise ValueError(f"Cannot read archive member {member.name}")
            with body, path.open("wb") as handle:
                shutil.copyfileobj(body, handle)
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(payload, dict) or not {"x", "y"} <= payload.keys():
                raise ValueError(f"{filename} must contain paired x/y tensors")
            x, y = payload["x"], payload["y"]
            shape = (resolution, resolution)
            if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
                raise ValueError(f"{filename} x/y must be tensors")
            if x.ndim != 3 or y.shape != x.shape or tuple(x.shape[1:]) != shape:
                raise ValueError(f"Expected paired (N,{resolution},{resolution}) tensors; got {x.shape}, {y.shape}")
            if not 1 <= count <= len(x):
                raise ValueError(f"Requested {count} {split} cases but {filename} has {len(x)}")
            if x.dtype not in {torch.bool, torch.float32} or not y.is_floating_point():
                raise ValueError(f"Expected saved bool/float32 x and real floating y; got {x.dtype}, {y.dtype}")
            if not bool(((x == 0) | (x == 1)).all()):
                raise ValueError(f"Expected the saved binary source encoding in {filename}")
            all_source_hashes[split] = {
                hashlib.sha256(case.numpy().tobytes()).hexdigest() for case in x.to(dtype=torch.float32)
            }
            source = x[:count].to(dtype=torch.float32).numpy()[..., None].copy()
            target = y[:count].numpy()[..., None].copy()
            if not np.isfinite(source).all() or not np.isfinite(target).all():
                raise ValueError(f"Selected {split} pairs contain non-finite values")
            arrays["source"].append(source)
            arrays["target"].append(target)
            for index in range(count):
                rows.append(
                    {
                        "sample_id": f"darcy_{resolution}_{split}_{index:06d}",
                        "array_index": len(rows),
                        "official_split": split,
                        "official_index": index,
                        "source_sha256": hashlib.sha256(source[index].tobytes()).hexdigest(),
                        "target_sha256": hashlib.sha256(target[index].tobytes()).hexdigest(),
                    }
                )
            members[split] = {"archive_member": member.name, "sha256": _digest(path)}
            summaries[split] = {
                "available_samples": len(x),
                "selected_samples": count,
                "selected_official_indices": [0, count - 1],
                "selection": "first requested rows, matching the official loader convention",
                "saved_x_dtype": str(x.dtype),
                "saved_y_dtype": str(y.dtype),
                "saved_payload_keys": sorted(payload),
                "converted_source_values": np.unique(source).tolist(),
                "target_range": [float(target.min()), float(target.max())],
                "target_edge_abs_max": [
                    float(np.abs(edge).max())
                    for edge in (target[:, 0], target[:, -1], target[:, :, 0], target[:, :, -1])
                ],
            }
    source_hashes = {
        split: {row["source_sha256"] for row in rows if row["official_split"] == split} for split in counts
    }
    overlap = all_source_hashes["train"] & all_source_hashes["test"]
    if overlap:
        raise ValueError(f"Official full train/test tensors share {len(overlap)} identical source fields")
    duplicate_counts = {split: counts[split] - len(source_hashes[split]) for split in counts}
    if any(duplicate_counts.values()):
        raise ValueError(f"Selected source cases are duplicated: {duplicate_counts}")
    return {key: np.concatenate(values) for key, values in arrays.items()}, rows, members, summaries


def _write_dataset(root: Path, values: np.ndarray, rows: list[dict], *, source: bool) -> None:
    root.mkdir(exist_ok=False)
    np.save(root / "field.npy", values, allow_pickle=False)
    with (root / "samples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    resolution = values.shape[1]
    manifest = {
        "schema": "field_dataset_v1",
        "samples": {"path": "samples.csv"},
        "fields": {
            "path": "field.npy",
            "layout": "NHWC",
            "kind": "scalar",
            "components": ["saved_diffusivity_indicator" if source else "stored_darcy_output"],
            "units": ["stored_benchmark_scale"],
        },
        "geometry": {
            "kind": "rectangle",
            "axes": {
                "x": {"start": 0.0, "step": 1 / (resolution - 1), "count": resolution},
                "y": {"start": 0.0, "step": 1 / (resolution - 1), "count": resolution},
            },
        },
        "imported_from": {"url": RECORD_URL, "provenance": "../provenance.json"},
    }
    (root / "field_dataset.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", type=int, choices=sorted(ARCHIVE_MD5), default=32)
    parser.add_argument("--train-samples", type=int, default=1000)
    parser.add_argument("--test-samples", type=int, default=200)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.train_samples < 3 or args.test_samples < 1:
        parser.error("Use at least three training cases and one official test case")
    root = (args.output or Path(f"data/benchmarks/darcy_{args.resolution}")).resolve()
    if (root / "source").exists() or (root / "target").exists():
        parser.error(f"Converted datasets already exist under {root}; choose a new output directory")
    root.mkdir(parents=True, exist_ok=True)
    archive = _archive(root, args.resolution)
    arrays, rows, members, summaries = _read_pairs(
        archive, root, args.resolution, {"train": args.train_samples, "test": args.test_samples}
    )
    for key, values in arrays.items():
        _write_dataset(root / key, values, rows, source=key == "source")
    provenance = {
        "dataset": "NeuralOperator Darcy Flow Dataset, Zenodo version v2",
        "record_url": RECORD_URL,
        "official_loader_url": LOADER_URL,
        "related_paper_url": PAPER_URL,
        "data_kind": "public steady-state numerical PDE simulation; not laboratory measurements or user application data",
        "archive": {
            "path": archive.name,
            "bytes": archive.stat().st_size,
            "md5": _digest(archive, "md5"),
            "sha256": _digest(archive),
        },
        "members": members,
        "resolution": args.resolution,
        "partitions": summaries,
        "source_conversion": "saved binary x represented exactly as float32 0/1; no inferred 3/12 remapping",
        "target_conversion": "saved y values and floating dtype preserved; no normalisation",
        "target_interpretation": "benchmark stored output; no SI scale or inverse normalisation inferred from the related PDE paper",
        "spatial_convention": "normalised array index axes [0,1], first spatial axis as y and second as x; no transpose; physical sampling coordinates absent from archive",
        "units": "stored_benchmark_scale; the archive does not supply an SI unit or inverse scaling metadata",
        "pairing": "x[i] and y[i] from the same official tensor file; case ID includes official split and index",
        "split_policy": "official test reserved; selection and fitted transforms use official train only",
        "group_information": "no physical group metadata provided; official random-field cases treated as independent",
        "finite_values": True,
        "unique_case_ids": len({row["sample_id"] for row in rows}) == len(rows),
        "duplicate_source_fields": 0,
        "train_test_source_overlap": 0,
        "overlap_check_scope": "all original train and test source tensors, beyond the selected subset",
        "derived": {
            key: {
                "shape": list(values.shape),
                "dtype": str(values.dtype),
                "array_bytes": values.nbytes,
                "file_sha256": _digest(root / key / "field.npy"),
            }
            for key, values in arrays.items()
        },
    }
    (root / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"root": str(root), "shape": list(arrays["source"].shape), "partitions": summaries}, indent=2))


if __name__ == "__main__":
    main()
