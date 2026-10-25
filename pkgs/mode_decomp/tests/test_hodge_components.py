"""Physical export uses the supplied field and native edge energy, not packed labels."""

import json

import numpy as np
import pandas as pd

from ml_platform_mode_decomp.component_reporting import write_component_report
from ml_platform_mode_decomp.data.schema import FieldDataset
from ml_platform_mode_decomp.geometry import build_domain_spec
from ml_platform_mode_decomp.representation import FieldRepresentation


def test_component_export_matches_manufactured_exact_and_sampling_nullspace(tmp_path):
    geometry = build_domain_spec(
        {"name": "rectangle", "x_range": [0, 1], "y_range": [0, 1], "coordinate_units": {"x": "m", "y": "m"}}, (7, 9, 2)
    )
    x, y = geometry.coords["x"], geometry.coords["y"]
    gradient = np.stack((2 * x, 2 * y), axis=-1)
    training = FieldDataset(
        np.stack((gradient, 2 * gradient)),
        geometry,
        np.array(["train_a", "train_b"]),
        components=("u_x", "u_y"),
        field_kind="vector",
        field_units=("m/s", "m/s"),
    )
    representation = FieldRepresentation({"decomposition": {"name": "hodge"}}).fit(training)
    checker = np.zeros_like(gradient)
    checker[..., 0] = (-1.0) ** np.arange(9)
    paths = write_component_report(
        representation, np.stack((gradient, checker)), ["gradient", "unseen_checker"], tmp_path
    )
    energy = pd.read_csv(paths["component_energy"])
    summary = json.loads(paths["component_summary"].read_text(encoding="utf-8"))
    hodge = representation.decomposer.hodge
    assert hodge is not None
    with np.load(paths["component_fields"], allow_pickle=False) as fields:
        assert fields["sample_ids"].tolist() == ["gradient", "unseen_checker"]
        assert fields["components"].tolist() == ["u_x", "u_y"]
        np.testing.assert_allclose(fields["edge_exact"][0], hodge.d0 @ (x**2 + y**2).reshape(-1), atol=1e-9)
        np.testing.assert_allclose(fields["edge_coexact"][0], 0, atol=1e-9)
        np.testing.assert_allclose(fields["edge_harmonic"][0], 0, atol=1e-9)
        np.testing.assert_array_equal(fields["edge_sampled"][1], 0)
        np.testing.assert_array_equal(fields["node_sampling_residual"][1], checker)
        for name in ("exact", "coexact", "harmonic"):
            expected = np.sum(fields["edge_weights"] * fields[f"edge_{name}"] ** 2, axis=1)
            np.testing.assert_allclose(energy[f"{name}_edge_energy"], expected, atol=1e-12)
    np.testing.assert_allclose(energy.input_edge_energy, energy.sum_component_edge_energy, atol=1e-9)
    assert energy.node_sampling_residual_norm.iloc[1] == 1.0
    assert summary["physical_guarantee_scope"] == "native_edge_cochains_resampled_from_supplied_fields"
    assert summary["boundary"] == "natural" and summary["harmonic_dimension"] == 0
