"""Persist physical component results; numerical operations live in decomposition."""

from pathlib import Path

import numpy as np
import pandas as pd
from ml_platform_core.io import write_json, write_table

from .data.schema import FieldSpec
from .decomposition.physical import analyze_components


def write_component_report(representation, fields, sample_ids, output_directory, *, method="hodge", params=None):
    representation._require_fitted()
    spec = getattr(representation, "field_spec", None) or FieldSpec(
        representation.field_shape, representation.field_kind, representation.components, representation.field_units
    )
    arrays, rows, summary = analyze_components(
        spec, representation.geometry, fields, sample_ids, method=method, params=params
    )
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    fields_path = directory / "components.npz"
    np.savez_compressed(fields_path, allow_pickle=False, **arrays)
    energy_path = write_table(pd.DataFrame(rows), directory / "component_energy.csv")
    summary_path = write_json(summary, directory / "component_summary.json")
    return {"component_fields": fields_path, "component_energy": energy_path, "component_summary": summary_path}
