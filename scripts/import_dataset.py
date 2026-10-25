"""Explicitly import CSV fields or former NPY data into the canonical Dataset format."""

import argparse

from ml_platform_core.config import load_yaml
from ml_platform_mode_decomp.data import (
    import_csv_field_dataset,
    import_legacy_npy_dataset,
    import_scattered_field_dataset,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="format", required=True)
    csv = commands.add_parser("csv", help="Import explicitly declared fixed-grid or mesh CSV fields")
    csv.add_argument("source")
    csv.add_argument("destination")
    csv.add_argument(
        "--source-manifest", required=True, help="CSV import YAML path relative to the source Dataset root"
    )
    scattered = commands.add_parser("scattered", help="Linearly map explicit scattered observations to a fixed grid")
    scattered.add_argument("source")
    scattered.add_argument("destination")
    scattered.add_argument(
        "--source-manifest", required=True, help="Scattered import YAML path relative to the source Dataset root"
    )
    legacy = commands.add_parser(
        "legacy-npy", help="Convert the former NPY directory format without changing its values"
    )
    legacy.add_argument("source")
    legacy.add_argument("destination")
    legacy.add_argument("--condition-columns", nargs="+")
    legacy.add_argument(
        "--geometry-override", help="YAML geometry mapping; referenced paths use the source Dataset root"
    )
    args = parser.parse_args()
    if args.format == "csv":
        result = import_csv_field_dataset(args.source, args.destination, source_manifest=args.source_manifest)
    elif args.format == "scattered":
        result = import_scattered_field_dataset(args.source, args.destination, source_manifest=args.source_manifest)
    else:
        geometry = load_yaml(args.geometry_override) if args.geometry_override else None
        result = import_legacy_npy_dataset(
            args.source, args.destination, condition_columns=args.condition_columns, geometry_override=geometry
        )
    print(result)


if __name__ == "__main__":
    main()
