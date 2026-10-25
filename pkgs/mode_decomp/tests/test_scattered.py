import csv
from dataclasses import replace

import numpy as np
import pytest
import yaml
from ml_platform_core.artifacts import file_hash
from scipy.interpolate import LinearNDInterpolator

from ml_platform_mode_decomp.data import (
    BarycentricMap,
    ScatteredGridMap,
    evaluate_scattered_prediction,
    import_scattered_field_dataset,
    load_field_dataset,
    load_scattered_observations,
    point_scores,
)


def test_original_observation_objective_uses_point_sample_and_channel_measure(tmp_path):
    from ml_platform_mode_decomp.evaluation.objectives import prepare_objective

    source, target = tmp_path / "source", tmp_path / "target"
    manifest, _, truth = _write_source(source)
    point_weights = np.array([2.0, 1, 1, 1, 5])
    np.save(source / "weights.npy", point_weights)
    manifest["observations"]["weights"] = "weights.npy"
    (source / "import.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    import_scattered_field_dataset(source, target, source_manifest="import.yaml")
    reference = replace(load_field_dataset(target), sample_metadata={"importance": np.array([1.0, 3.0])})
    options = {
        "reference": "observations",
        "names": ["observation_rmse", "observation_mae"],
        "channel_scales": [2],
        "sample_weight_column": "importance",
    }
    predicted = np.zeros_like(reference.fields)
    predicted[0], predicted[1] = 2.0, 4.0
    baseline = predicted + 1
    prepared = prepare_objective(reference, options, data={"local_path": str(target)}, baseline=baseline)
    report = prepared.score(predicted)
    errors = np.array([2.0, 4.0])[:, None] - truth[[1, 0], :, 0]
    mse = np.average(errors**2, axis=1, weights=point_weights) / 4
    mae = np.average(abs(errors), axis=1, weights=point_weights) / 2
    sample_weights = np.array([0.25, 0.75])
    assert report.metrics["observation_rmse"] == pytest.approx(np.sqrt(sample_weights @ mse))
    assert report.metrics["observation_mae"] == pytest.approx(sample_weights @ mae)
    np.testing.assert_allclose(report.per_sample.mse, mse)
    assert prepared.loss(predicted)[0] == pytest.approx(sample_weights @ mse)
    assert report.summary["evaluation_reference"] == "observations"
    assert report.per_sample.coverage.tolist() == [1, 1]
    damaged = predicted.copy()
    damaged[:, 0, 0] = np.nan
    with pytest.raises(ValueError, match="fixed scattered"):
        prepared.score(damaged)

    # Detailed decomposition/regression identities use the same original points,
    # including their original truth rather than interpolated reference truth.
    from ml_platform_mode_decomp.evaluation import analyze_surrogate
    from ml_platform_mode_decomp.surrogate import FieldSurrogate

    model = FieldSurrogate(
        {"representation": {"decomposition": {"name": "dct2"}}, "regression": {"name": "ridge"}}
    ).fit(reference)
    detailed = analyze_surrogate(model, reference, reference, options, prepared=prepared)
    assert detailed.metrics["observation_rmse"] ** 2 == pytest.approx(
        detailed.metrics["representation_rmse"] ** 2
        + detailed.metrics["regression_rmse"] ** 2
        + detailed.metrics["error_cross_term_mse"]
    )


@pytest.mark.parametrize("tuning", [False, True])
def test_observation_objective_flows_through_pipeline_selection_and_final_scoring(tmp_path, tuning):
    from ml_platform_mode_decomp import run_task
    from ml_platform_core.io import read_json

    source, target = tmp_path / "source", tmp_path / "target"
    _, points, _ = _write_source(source)
    values = (points[:, 0] ** 2 + points[:, 1] ** 2)[None, :, None] + np.arange(12)[:, None, None]
    np.save(source / "values.npy", values)
    _csv(source / "samples.csv", ["sample_id", "array_index", "temperature"], [[f"s{i}", i, i] for i in range(12)])
    import_scattered_field_dataset(source, target, source_manifest="import.yaml")
    result = run_task(
        {
            "task": "mode_decomp_pipeline",
            "runtime": {"output_dir": str(tmp_path / "runs")},
            "data": {"local_path": str(target)},
            "metrics": {"reference": "observations"},
            "model": {
                "candidates": [
                    {
                        "id": "dct",
                        "representation": {"decomposition": {"name": "dct2"}},
                        "regression": {"name": "ridge", "params": {"alpha": 1e-8}},
                    }
                ]
            },
            "tuning": {"enabled": tuning, "n_trials": 1, "cv": {"n_splits": 2}},
        }
    )
    assert np.isfinite(result.metrics["observation_rmse"])
    assert "field_rmse" not in result.metrics
    summary = read_json(result.artifacts["result_summary"])
    assert summary["selection_metric"] == "observation_rmse"
    assert summary["evaluation"]["reference"] == "observations"


def test_point_objective_does_not_require_grid_quadrature_or_score_zero_weight_rows(tmp_path):
    from dataclasses import replace
    from ml_platform_mode_decomp.evaluation.objectives import prepare_objective

    source, target = tmp_path / "source", tmp_path / "target"
    _write_source(source)
    import_scattered_field_dataset(source, target, source_manifest="import.yaml")
    reference = load_field_dataset(target)
    reference = replace(
        reference,
        geometry=replace(reference.geometry, weights=np.zeros(reference.geometry.grid_shape)),
        sample_metadata={"importance": np.array([0.0, 1.0])},
    )
    prepared = prepare_objective(
        reference, {"reference": "observations", "sample_weight_column": "importance"}, data={"local_path": str(target)}
    )
    assert prepared.grid is None
    prediction = reference.fields + 1
    prediction[0] = np.nan
    report = prepared.score(prediction)
    sample = prepared.observations.samples[1]
    interpolated = np.einsum("qv,qvc->qc", sample.weights, prediction[1].reshape(-1, 1)[sample.vertices])
    expected = np.sqrt(np.mean((interpolated[sample.support] - sample.truth[sample.support]) ** 2))
    assert report.metrics["observation_rmse"] == pytest.approx(expected)
    assert report.per_sample.mse.iloc[1] == pytest.approx(expected**2)
    assert np.isnan(report.per_sample.mse.iloc[0])


@pytest.mark.parametrize("point_weight", [1e308, 1e-308])
def test_point_weight_ratios_use_only_scored_support_for_every_entry_point(tmp_path, point_weight):
    from ml_platform_mode_decomp.evaluation.objectives import prepare_objective

    source, target = tmp_path / "source", tmp_path / "target"
    manifest, _, truth = _write_source(source)
    # The unobserved interior point must not determine the numeric scale of the
    # four measured corners. All four corners have exactly equal importance.
    weights = np.array([point_weight] * 4 + [1e308])
    observed = np.array([True] * 4 + [False])
    np.save(source / "weights.npy", weights)
    np.save(source / "observed.npy", observed)
    manifest["observations"].update(weights="weights.npy", mask="observed.npy")
    with np.errstate(over="raise", invalid="raise"):
        import_scattered_field_dataset(source, target, source_manifest=manifest)
        reference = load_field_dataset(target)
        objective = prepare_objective(reference, {"reference": "observations"}, data={"local_path": str(target)})
        prediction = np.ones_like(reference.fields)
        main = objective.score(prediction)
        supplementary = objective.observations.evaluate(prediction)
        public = point_scores(
            truth[0], np.ones_like(truth[0]), observed, observation_mask=observed, point_weights=weights
        )
    # Corner truth is [0,1,1,2] and [1,2,2,3], in reversed sample-table order.
    assert main.metrics["observation_rmse"] == pytest.approx(1.0)
    assert main.metrics["observation_mae"] == pytest.approx(0.75)
    assert supplementary["observation"]["rmse"] == pytest.approx([1.0])
    assert supplementary["observation"]["mae"] == pytest.approx([0.75])
    assert public["rmse"] == pytest.approx([np.sqrt(0.5)])
    assert public["mae"] == pytest.approx([0.5])


def _csv(path, header, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _write_source(root, *, per_sample=False):
    root.mkdir()
    points = np.array([[0.0, 0], [1, 0], [0, 1], [1, 1], [0.3, 0.4]])
    values = (points[:, 0] ** 2 + points[:, 1] ** 2)[None, :, None] + np.arange(2)[:, None, None]
    coordinates = np.stack([points, points]) if per_sample else points
    np.save(root / "points.npy", coordinates)
    np.save(root / "values.npy", values)
    _csv(root / "samples.csv", ["sample_id", "array_index", "temperature"], [["002", 1, 3], ["001", 0, 2]])
    manifest = {
        "schema": "scattered_field_import_v1",
        "samples": {"path": "samples.csv"},
        "conditions": {"format": "table", "columns": ["temperature"]},
        "observations": {"format": "npy", "coordinates": "points.npy", "values": "values.npy"},
        "fields": {"layout": "NHWC", "kind": "scalar", "components": ["value"]},
        "geometry": {
            "kind": "rectangle",
            "axes": {"x": {"start": 0, "step": 0.25, "count": 5}, "y": {"start": 0, "step": 0.25, "count": 5}},
        },
        "interpolation": {"method": "linear", "outside": "mask"},
    }
    (root / "import.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    return manifest, points, values


def test_barycentric_map_matches_affine_solution_and_scipy_without_extrapolation():
    points = np.array([[0, 0], [1, 0], [0, 1], [0.3, 0.3]])
    queries = np.array([[0.2, 0.3], [0, 0], [1, 1]])
    values = np.column_stack([2 * points[:, 0] - points[:, 1] + 3, points[:, 0] + 4 * points[:, 1]])
    mapping = BarycentricMap.build(points, queries)
    predicted, support = mapping.forward(values)
    np.testing.assert_array_equal(support, [True, True, False])
    np.testing.assert_allclose(predicted[:2], [[3.1, 1.4], [3, 0]], atol=1e-14)
    np.testing.assert_allclose(predicted, LinearNDInterpolator(points, values)(queries), equal_nan=True)
    assert np.isnan(predicted[-1]).all()


@pytest.mark.parametrize("points", [[[0, 0], [1, 0], [0, 0]], [[0, 0], [1, 1], [2, 2]]])
def test_duplicate_or_collinear_points_are_rejected_without_jitter(points):
    with pytest.raises(ValueError, match="unique|stable"):
        BarycentricMap.build(points, [[0.1, 0.1]])


def test_allowed_triangles_and_missing_vertices_limit_support():
    points = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    initial = BarycentricMap.build(points, [[0.2, 0.2]])
    queries = points[initial.simplices].mean(axis=1)
    mapping = BarycentricMap.build(points, queries, simplex_support=initial.simplices[:1])
    values, support = mapping.forward(points[:, :1])
    np.testing.assert_array_equal(support, [True, False])
    assert np.isnan(values[1]).all()
    triangle = BarycentricMap.build([[0, 0], [1, 0], [0, 1]], [[0, 0], [0.2, 0.2]])
    values, support = triangle.forward([[3], [4], [np.nan]], [1, 1, 0])
    np.testing.assert_array_equal(support, [True, False])
    assert values[0, 0] == 3


def test_grid_to_point_map_uses_local_cells_and_does_not_bridge_domain_hole():
    points = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [0.3, 0.3]])
    domain = np.ones((3, 3), dtype=bool)
    domain[1, 1] = False
    mapping = ScatteredGridMap.build(points, [0, 0.5, 1], [0, 0.5, 1], domain_mask=domain)
    grid, support = mapping.forward(points[:, :1])
    assert not support[1, 1] and np.isnan(grid[1, 1]).all()
    predicted, supported = mapping.to_points(np.nan_to_num(grid), support)
    assert not supported[-1] and np.isnan(predicted[-1]).all()


def test_point_scores_use_point_mean_or_explicit_weights_and_report_coverage():
    unweighted = point_scores([[0], [0]], [[1], [3]], [1, 1])
    weighted = point_scores([[0], [0]], [[1], [3]], [1, 1], point_weights=[3, 1])
    assert unweighted["rmse"] == pytest.approx([np.sqrt(5)])
    assert weighted["rmse"] == pytest.approx([np.sqrt(3)])
    partial = point_scores([[0], [0]], [[1], [np.nan]], [1, 0])
    assert partial["coverage"] == 0.5 and partial["scored_points"] == 1
    empty = point_scores([[0]], [[np.nan]], [0])
    assert empty["coverage"] == 0 and empty["rmse"] is None


@pytest.mark.parametrize("name", ["points.npy", "values.npy"])
def test_scattered_import_does_not_silently_discard_imaginary_observations(tmp_path, name):
    source, target = tmp_path / "source", tmp_path / "target"
    _write_source(source)
    array = np.load(source / name).astype(complex) + 2j
    np.save(source / name, array)
    with pytest.raises(ValueError, match="real numeric"):
        import_scattered_field_dataset(source, target, source_manifest="import.yaml")
    assert not target.exists()


@pytest.mark.parametrize("per_sample", [False, True])
def test_scattered_import_retains_originals_and_separates_field_and_interpolation_error(tmp_path, per_sample):
    source, target = tmp_path / "source", tmp_path / "target"
    _, points, values = _write_source(source, per_sample=per_sample)
    before = {path.name: file_hash(path) for path in source.iterdir()}
    import_scattered_field_dataset(source, target, source_manifest="import.yaml")
    dataset = load_field_dataset(target)
    original = load_scattered_observations(target)
    assert dataset.sample_ids.tolist() == ["002", "001"]
    assert original["sample_ids"].tolist() == ["001", "002"]
    np.testing.assert_array_equal(original["offsets"], [0, 5, 10])
    np.testing.assert_array_equal(original["points"][:5], points)
    np.testing.assert_array_equal(original["values"], values.reshape(-1, 1))
    assert {path.name: file_hash(path) for path in source.iterdir()} == before
    report = evaluate_scattered_prediction(target, dataset.fields[:1], sample_ids=["002"])
    assert report["field"]["field_rmse"] == pytest.approx(0)
    sample = report["samples"][0]
    assert sample["sample_id"] == "002"
    assert sample["interpolation"]["rmse"][0] > 0.01
    assert sample["observation"]["rmse"] == sample["interpolation"]["rmse"]
    assert sample["observation"]["coverage"] == 1
    assert report["observation"]["rmse"] == sample["observation"]["rmse"]
    assert report["observation"]["scored_samples"] == 1


def test_csv_scattered_import_accepts_ragged_sample_points_and_named_components(tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    manifest, points, _ = _write_source(source)
    _csv(
        source / "samples.csv",
        ["sample_id", "array_index", "point_csv", "temperature"],
        [["002", 1, "b.csv", 3], ["001", 0, "a.csv", 2]],
    )
    for filename, coordinates in (("a.csv", points), ("b.csv", points[:4])):
        _csv(source / filename, ["x", "y", "f", "weight"], [[x, y, x + y, 2] for x, y in coordinates])
    manifest["observations"] = {
        "format": "csv",
        "path_column": "point_csv",
        "coordinates": ["x", "y"],
        "columns": ["f"],
        "weight_column": "weight",
    }
    (source / "import.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    import_scattered_field_dataset(source, target, source_manifest="import.yaml")
    original = load_scattered_observations(target)
    np.testing.assert_array_equal(original["offsets"], [0, 5, 9])
    np.testing.assert_array_equal(original["weights"], np.full(9, 2))
    assert load_field_dataset(target).manifest["scattered"]["weighting"] == "explicit_point_weights"


def test_outside_hull_stays_unavailable_and_unsupported_policies_are_rejected(tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    manifest, points, _ = _write_source(source)
    np.save(source / "points.npy", points[:3])
    np.save(source / "values.npy", np.ones((2, 3, 1)))
    import_scattered_field_dataset(source, target, source_manifest="import.yaml")
    dataset = load_field_dataset(target)
    assert not dataset.observation_mask[:, -1, -1].any()
    assert np.isnan(dataset.fields[:, -1, -1]).all()
    manifest["interpolation"]["outside"] = "nearest"
    (source / "import.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="Supported interpolation"):
        import_scattered_field_dataset(source, tmp_path / "bad", source_manifest="import.yaml")
    assert not (tmp_path / "bad").exists()


def test_arbitrary_domain_requires_explicit_simplex_support(tmp_path):
    source = tmp_path / "source"
    manifest, _, _ = _write_source(source)
    domain = np.ones((5, 5), dtype=np.uint8)
    domain[2, 2] = 0
    np.save(source / "domain.npy", domain)
    manifest["geometry"]["kind"] = "arbitrary_mask"
    manifest["masks"] = {"domain": "domain.npy"}
    (source / "import.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="simplex_support"):
        import_scattered_field_dataset(source, tmp_path / "target", source_manifest="import.yaml")


def test_explicit_arbitrary_support_excludes_cross_gap_triangles_and_retains_coverage(tmp_path):
    source = tmp_path / "source"
    manifest, _, _ = _write_source(source)
    xx, yy = np.meshgrid(np.linspace(0, 1, 5), np.linspace(0, 1, 5))
    points = np.column_stack([xx.ravel(), yy.ravel()])
    points = points[~np.all(points == [0.5, 0.5], axis=1)]
    np.save(source / "points.npy", points)
    np.save(source / "values.npy", np.broadcast_to(points[None, :, :1], (2, len(points), 1)))
    triangles = BarycentricMap.build(points, [[0, 0]]).simplices
    allowed = triangles[np.max(points[triangles, 0], axis=1) <= 0.25]
    np.save(source / "allowed.npy", allowed)
    domain = np.ones((5, 5), dtype=np.uint8)
    domain[2, 2] = 0
    np.save(source / "domain.npy", domain)
    manifest["geometry"]["kind"] = "arbitrary_mask"
    manifest["masks"] = {"domain": "domain.npy"}
    manifest["observations"]["simplex_support"] = "allowed.npy"
    (source / "import.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    target = tmp_path / "target"
    import_scattered_field_dataset(source, target, source_manifest="import.yaml")
    dataset = load_field_dataset(target)
    assert not dataset.effective_mask[:, :, 2:].any()
    assert np.isnan(dataset.fields[:, :, 2:]).all()
    report = evaluate_scattered_prediction(target, dataset.fields)
    assert report["observation"]["coverage"] < 1
    assert report["observation"]["known_points"] == 2 * len(points)
    assert report["observation"]["rmse"] == pytest.approx([0], abs=1e-14)


def test_inverse_rejects_a_thin_declared_wall_even_when_all_coarse_grid_nodes_are_valid():
    points = np.array([(x, y) for x in (0.0, 0.2, 0.4, 0.45, 0.75, 1.0) for y in (0.0, 0.2, 0.5, 0.8, 1.0)])
    triangles = BarycentricMap.build(points, [[0, 0]]).simplices
    allowed = triangles[(points[triangles, 0].max(1) <= 0.4) | (points[triangles, 0].min(1) >= 0.45)]
    mapping = ScatteredGridMap.build(points, [0.0, 0.5, 1.0], [0.0, 0.5, 1.0], simplex_support=allowed)
    truth = (points[:, 0] >= 0.45).astype(float)[:, None]
    grid, grid_support = mapping.forward(truth)
    assert grid_support.all()  # The wall is narrower than, and invisible to, the grid.
    predictions, supported = mapping.to_points(grid, grid_support)
    assert not supported[(points[:, 0] > 0) & (points[:, 0] < 0.5)].any()
    # Exact grid vertices and interpolated points in the neighboring right island remain valid.
    assert supported[(points[:, 0] == 1) | (points[:, 0] == 0.75)].all()
    assert np.isnan(predictions[~supported]).all()
    np.testing.assert_allclose(predictions[supported], truth[supported], atol=1e-14)


def test_inverse_continuous_support_preserves_an_allowed_shared_boundary():
    points = np.array([(x, y) for x in (0.0, 0.25, 0.5, 0.75, 1.0) for y in (0.0, 0.125, 0.25, 0.5, 1.0)])
    triangles = BarycentricMap.build(points, [[0, 0]]).simplices
    # The bottom-left square is supported. Its top boundary can touch excluded triangles.
    allowed = triangles[(points[triangles, 0].max(1) <= 0.5) & (points[triangles, 1].max(1) <= 0.5)]
    mapping = ScatteredGridMap.build(points, [0.0, 0.5, 1.0], [0.0, 0.5, 1.0], simplex_support=allowed)
    truth = (2 * points[:, 0] - points[:, 1])[:, None]
    grid, grid_support = mapping.forward(truth)
    predictions, supported = mapping.to_points(grid, grid_support)
    # Only require stencils whose source-grid vertices were supported by Delaunay boundary choice.
    before, baseline_support = mapping.grid_to_point.forward(grid.reshape(-1, 1), grid_support.reshape(-1))
    expected = baseline_support & (points[:, 0] <= 0.5) & (points[:, 1] <= 0.5)
    assert expected.any()
    assert supported[expected].all()
    np.testing.assert_allclose(predictions[expected], before[expected], atol=1e-14)


def test_inverse_rejects_a_sub_grid_declared_hole_without_masking_valid_neighboring_stencils():
    points = np.array([(x, y) for x in (0.0, 0.2, 0.3, 0.5, 1.0) for y in (0.0, 0.2, 0.3, 0.5, 1.0)])
    triangles = BarycentricMap.build(points, [[0, 0]]).simplices
    in_hole = ((points[triangles] >= 0.2) & (points[triangles] <= 0.3)).all(axis=(1, 2))
    mapping = ScatteredGridMap.build(points, [0.0, 0.5, 1.0], [0.0, 0.5, 1.0], simplex_support=triangles[~in_hole])
    truth = points[:, :1] ** 2 + points[:, 1:] ** 2
    grid, grid_support = mapping.forward(truth)
    assert grid_support.all()
    predictions, supported = mapping.to_points(grid, grid_support)
    assert not supported[((points >= 0.2) & (points <= 0.3)).all(axis=1)].any()
    assert supported[points[:, 0] >= 0.5].all()
    assert np.isfinite(predictions[supported]).all()


def test_imported_observation_mask_blocks_a_sub_grid_missing_strip_during_point_scoring(tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    manifest, _, _ = _write_source(source)
    points = np.array([(x, y) for x in (0.0, 0.4, 0.425, 0.45, 0.5, 1.0) for y in (0.0, 0.5, 1.0)])
    values = (points[:, 0] >= 0.45).astype(float)[None, :, None].repeat(2, axis=0)
    observed = points[:, 0] != 0.425
    values[:, ~observed] = np.nan
    np.save(source / "points.npy", points)
    np.save(source / "values.npy", values)
    np.save(source / "observed.npy", observed)
    manifest["observations"]["mask"] = "observed.npy"
    manifest["geometry"]["axes"] = {
        "x": {"start": 0.0, "step": 0.5, "count": 3},
        "y": {"start": 0.0, "step": 0.5, "count": 3},
    }
    (source / "import.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    import_scattered_field_dataset(source, target, source_manifest="import.yaml")
    dataset = load_field_dataset(target)
    assert dataset.observation_mask.all()
    report = evaluate_scattered_prediction(target, dataset.fields)
    assert report["field"]["field_rmse"] == 0
    assert report["observation"]["coverage"] < 1
    assert report["observation"]["known_points"] == 2 * int(observed.sum())
    assert report["observation"]["rmse"] == pytest.approx([0], abs=1e-14)


def test_npy_scattered_input_needs_no_sample_table_and_prepared_scores_keep_the_same_support(tmp_path):
    from ml_platform_mode_decomp.data.scattered_import import prepare_scattered_evaluation

    source, target = tmp_path / "source", tmp_path / "target"
    manifest, _, _ = _write_source(source)
    manifest.pop("samples")
    manifest.pop("conditions")
    import_scattered_field_dataset(source, target, source_manifest=manifest)
    dataset = load_field_dataset(target)
    assert dataset.sample_ids.tolist() == ["sample_000000", "sample_000001"]
    original = load_scattered_observations(target)
    assert isinstance(original["values"], np.memmap)
    prepared = prepare_scattered_evaluation(target, dataset)
    base = prepared.evaluate(dataset.fields)
    shifted = prepared.evaluate(dataset.fields + 1)
    assert base["observation"]["scored_points"] == shifted["observation"]["scored_points"]
    assert base["interpolation"] == shifted["interpolation"]
    invalid = dataset.fields.copy()
    invalid[:, 0, 0] = np.nan
    with pytest.raises(ValueError, match="fixed scattered scoring support"):
        prepared.evaluate(invalid)


def test_streaming_scattered_failure_does_not_publish_or_leave_open_mapped_outputs(tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    _write_source(source)
    values = np.load(source / "values.npy")
    values[1] = np.nan
    np.save(source / "values.npy", values)
    with pytest.raises(ValueError, match="finite"):
        import_scattered_field_dataset(source, target, source_manifest="import.yaml")
    assert not target.exists()
    assert not list(tmp_path.glob(".scattered_import_*"))


def test_historical_scattered_npz_store_remains_readable(tmp_path):
    source, target = tmp_path / "source", tmp_path / "target"
    _write_source(source)
    import_scattered_field_dataset(source, target, source_manifest="import.yaml")
    expected = load_scattered_observations(target)
    np.savez_compressed(target / "historical.npz", **expected)
    manifest = yaml.safe_load((target / "field_dataset.yaml").read_text(encoding="utf-8"))
    manifest["scattered"]["path"] = "historical.npz"
    actual = load_scattered_observations(target, source_manifest=manifest)
    for name in expected:
        np.testing.assert_array_equal(actual[name], expected[name])
