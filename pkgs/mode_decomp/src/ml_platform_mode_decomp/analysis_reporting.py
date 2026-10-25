"""A portable analysis view of the actual field-space comparison artifacts."""

from html import escape
from dataclasses import replace
from pathlib import Path

import numpy as np


def prepare_preview(reference, report, index):
    """Own only the displayed values, never a view retaining a complete prediction."""
    mask = reference._sample_mask(index)
    geometry = reference.geometry
    if mask.ndim == 2:
        first = np.argwhere(mask)[0]
        rows = np.unique(np.r_[np.arange(0, mask.shape[0], max(1, int(np.ceil(mask.shape[0] / 32)))), first[0]])
        columns = np.unique(np.r_[np.arange(0, mask.shape[1], max(1, int(np.ceil(mask.shape[1] / 32)))), first[1]])
        selection = np.ix_(rows, columns)
    else:
        selection = np.flatnonzero(mask)[:: max(1, int(np.ceil(mask.sum() / 1024)))]
        geometry = replace(geometry, coords={"vertices": np.array(geometry.coords["vertices"][selection], copy=True)})
    item = {
        "reference": np.array(reference.fields[index][selection][..., :3], copy=True),
        "prediction": np.array(report.predictions[index][selection][..., :3], copy=True),
        "mask": np.array(mask[selection], copy=True),
        "sample_id": str(reference.sample_ids[index]),
        "geometry": geometry,
    }
    if "observed_mask" in report.fields:
        item["observed_mask"] = np.array(report.fields["observed_mask"][index][selection], copy=True)
    return item


def _color(value, limit):
    ratio = float(np.clip(value / max(limit, np.finfo(float).tiny), -1, 1))
    pale = int(245 - 180 * abs(ratio))
    return f"rgb(205,{pale},{pale})" if ratio >= 0 else f"rgb({pale},{pale},205)"


def _field_svg(values, mask, limit, geometry):
    """Downsample only the preview; metrics always use every reference location."""
    values, mask = np.asarray(values), np.asarray(mask)
    marks = []
    if values.ndim == 2:
        stride_y, stride_x = max(1, int(np.ceil(values.shape[0] / 32))), max(1, int(np.ceil(values.shape[1] / 32)))
        image, valid = values[::stride_y, ::stride_x], mask[::stride_y, ::stride_x]
        h, w = image.shape
        for i, j in np.argwhere(valid):
            marks.append(
                f'<rect x="{j * 256 / w:.3f}" y="{(h - i - 1) * 192 / h:.3f}" width="{256 / w + 0.1:.3f}" height="{192 / h + 0.1:.3f}" fill="{_color(image[i, j], limit)}"/>'
            )
    else:
        vertices = np.asarray(geometry.coords["vertices"])
        coords = vertices[:, :2]
        lower, extent = coords.min(axis=0), np.ptp(coords, axis=0)
        extent[extent == 0] = 1
        points = (coords - lower) / extent
        indices = np.flatnonzero(mask)[:: max(1, int(np.ceil(mask.sum() / 1024)))]
        for index in indices:
            x, y = points[index]
            marks.append(
                f'<circle cx="{4 + x * 248:.3f}" cy="{188 - y * 184:.3f}" r="2.5" fill="{_color(values[index], limit)}"/>'
            )
    return '<svg viewBox="0 0 256 192" role="img" aria-label="field preview">' + "".join(marks) + "</svg>"


def _capacity_svg(table, reference="grid"):
    rows = table[(table.status == "ok") & (table.method != "baseline")]
    if rows.empty:
        return ""
    metric = {"grid": "field_rmse", "observations": "observation_rmse", "hidden": "hidden_rmse"}[reference]
    dimensions, errors = rows.target_dimension.to_numpy(), rows[metric].to_numpy()
    maximum = max(float(dimensions.max()), 1)
    candidates = list(dict.fromkeys(rows.candidate_id))
    palette = ("#166d91", "#c35a16", "#287c45", "#8b4bab", "#a32d56", "#69651b")
    colors = {name: palette[index % len(palette)] for index, name in enumerate(candidates)}
    top = max(float(errors.max()), np.finfo(float).tiny)
    marks = ['<path d="M55 20 V245 H680" fill="none" stroke="#555"/>']
    for (_, row), x, error in zip(rows.iterrows(), dimensions, errors, strict=True):
        xx, yy = 55 + 600 * np.log1p(x) / np.log1p(maximum), 245 - 210 * error / top
        title = escape(
            f"{row.candidate_id}: dimension {x:g}, RMSE {error:.6g}, split {row.split_seed}, model seed {row.model_seed}"
        )
        marks.append(
            f'<circle cx="{xx:.3f}" cy="{yy:.3f}" r="5" fill="{colors[row.candidate_id]}"><title>{title}</title></circle>'
        )
    for dimension in sorted(set(dimensions)):
        xx = 55 + 600 * np.log1p(dimension) / np.log1p(maximum)
        marks.append(f'<text x="{xx:.3f}" y="265" text-anchor="middle">{dimension:g}</text>')
    marks += [
        '<text x="240" y="290">Z target dimension (log1p scale)</text>',
        f'<text x="5" y="15">{escape(reference)} RMSE</text>',
        f'<text x="5" y="40">{top:.3g}</text>',
        '<text x="25" y="245">0</text>',
    ]
    return (
        '<svg viewBox="0 0 720 300" role="img" aria-label="Field error versus actual target dimension">'
        + "".join(marks)
        + '</svg><p class="legend">'
        + " · ".join(f'<span style="color:{colors[name]}">● {escape(name)}</span>' for name in candidates)
        + "</p>"
    )


def write_analysis_report(summary, table, previews, dataset, path) -> Path:
    kind = summary["analysis_kind"]
    names = {
        "surrogate": "Surrogate field prediction",
        "representation": "Held-out field representation",
        "imputation": "Reconstruction of hidden known values",
    }
    status = "Complete" if summary["complete"] else "Incomplete: some candidates failed"
    columns = [
        x
        for x in (
            "candidate_id",
            "split_seed",
            "model_seed",
            "status",
            "target_dimension",
            "field_rmse",
            "field_skill",
            "representation_rmse",
            "regression_rmse",
            "observed_field_rmse",
            "fit_seconds",
            "model_bytes",
            "error",
        )
        if x in table
    ]
    columns.extend(
        name
        for name in table.columns
        if name.startswith(("interval_", "observation_", "interpolation_", "hidden_")) and name not in columns
    )
    body = [
        f'<h1>{names[kind]}</h1><p class="status">{escape(status)}</p>',
        "<p>All methods use the same training samples, test IDs, known reference support and field measure in each repetition. "
        "These are predeclared fixed configurations. This study does not select or publish a model from test scores.</p>",
        '<nav><a href="comparison.csv">Comparison CSV</a> · <a href="per_sample.csv">Sample losses</a> · '
        '<a href="paired_comparisons.csv">Paired differences</a> · <a href="protocol.json">Protocol</a> · '
        '<a href="analysis_summary.json">Definitions and method diagnostics</a> · <a href="#previews">Field previews</a></nav>',
        "<h2>Field error and retained coordinates</h2>",
        _capacity_svg(table, summary["evaluation"].get("reference", "grid")),
        "<p>Dimensions are the actual fitted Z coordinates. Equal dimensions do not imply equal basis geometry, "
        "training cost or model capacity. Hover over points for the configuration and repetition.</p>",
        '<h2>All evaluations</h2><div class="scroll">',
        table[columns].to_html(index=False, escape=True, float_format=lambda value: f"{value:.6g}", na_rep="—"),
        "</div>",
        "<h2>How to interpret the scores</h2><ul>"
        "<li>RMSE is computed in the original field, after the declared channel scales and spatial weights.</li>"
        "<li>Skill compares with the mean field fitted on training values only; 0 equals that baseline.</li>"
        "<li>Representation and regression errors need a cross term to recover total MSE; they are not generally additive.</li>"
        "<li>Imputation scores use artificially hidden values with known truth, separately from observed points.</li>"
        "<li>Paired intervals condition on fitted models and one test split. They are exploratory and unadjusted; "
        "repeated holdouts are not independent test sets. Time-series intervals require declared independent groups.</li></ul>",
    ]
    if any("field_uncertainty" in item for item in summary["details"]):
        body.append(
            "<h2>Calibrated field prediction bands</h2><p>Coverage targets and empirical coverage are listed above. "
            "Simultaneous bands target a whole future field; pointwise bands target each location/component separately. "
            "Widths are prediction intervals, not standard deviations. Calibration samples are excluded from model fitting and selection. "
            "Finite-sample coverage requires exchangeable calibration and future samples; this is not verified by a time/group split. "
            "The JSON definitions include channel widths in physical units and the declared fixed support.</p>"
        )
    if any("scattered" in item for item in summary["details"]):
        body.append(
            "<h2>Original scattered observations</h2><p>Observation RMSE scores the predicted grid back at retained original points. "
            "Interpolation RMSE is the observation-to-grid-to-observation reference, not a subtractable model-error term or a universal lower bound. "
            "Both use the same supported points and report coverage; unsupported observations are excluded, never replaced by zero. "
            "These point-weighted scores differ from area-weighted field loss.</p>"
        )
    component_studies = [item for item in summary["details"] if "component_analysis" in item]
    if component_studies:
        body.append(
            "<h2>Physical vector components</h2><p>Hodge uses native edge cochains: edge orthogonality and energy are distinct from node sampling residual. Helmholtz uses the declared periodic uniform-grid projection and its Euclidean grid energy; it does not impose nonperiodic wall data.</p><ul>"
        )
        for item in component_studies:
            analysis = item["component_analysis"]
            label = escape(f"{item['candidate_id']} · split {item['split_seed']} · model seed {item['model_seed']}")
            if analysis["status"] == "ok":
                links = " · ".join(
                    f'<a href="{escape(path, quote=True)}">{escape(name)}</a>'
                    for name, path in analysis["outputs"].items()
                )
                body.append(f"<li>{label} ({escape(analysis['source'])}): {links}</li>")
            else:
                body.append(f"<li>{label}: {escape(analysis['reason'])}</li>")
        body.append("</ul>")
    mode_studies = [item for item in summary["details"] if "mode_analysis" in item]
    if mode_studies:
        body.append("<h2>Selected mode and coordinate analysis</h2><ul>")
        for item in mode_studies:
            analysis = item["mode_analysis"]
            label = escape(f"{item['candidate_id']} · split {item['split_seed']} · model seed {item['model_seed']}")
            if analysis["status"] == "ok":
                body.append(f'<li><a href="{escape(analysis["report"], quote=True)}">{label}</a></li>')
            else:
                body.append(f"<li>{label}: unavailable — {escape(analysis['reason'])}</li>")
        body.append("</ul>")
    body.append(
        '<h2 id="previews">Field previews</h2><p>Worst test sample in the first repetition for each method. '
        "The display may be downsampled; every valid location is used in the metrics. "
        "Reference and prediction use a shared color range; residual has its own symmetric range. "
        "Up to three channels are previewed. Grid images use array order; mesh previews use the first two vertex coordinates.</p>"
    )
    for candidate_id, item in previews.items():
        reference, prediction, mask = item["reference"], item["prediction"], item["mask"]
        body.append(f"<h3>{escape(candidate_id)} · {escape(item['sample_id'])}</h3>")
        for channel in range(min(reference.shape[-1], 3)):
            truth, predicted = reference[..., channel], prediction[..., channel]
            error = predicted - truth
            limit = max(float(np.max(np.abs(truth[mask]))), float(np.max(np.abs(predicted[mask]))), 1e-15)
            residual_limit = max(float(np.max(np.abs(error[mask]))), 1e-15)
            component = dataset.components[channel] if dataset.components else f"channel_{channel}"
            body.append(f'<p>{escape(component)}</p><div class="panels">')
            for title, values, scale in (
                ("Reference", truth, limit),
                ("Prediction", predicted, limit),
                ("Residual", error, residual_limit),
            ):
                body.append(
                    f"<figure><figcaption>{title} · ±{scale:.4g}</figcaption>{_field_svg(values, mask, scale, item.get('geometry', dataset.geometry))}</figure>"
                )
            if "observed_mask" in item:
                body.append(
                    "<figure><figcaption>Input observations</figcaption>"
                    + _field_svg(truth, item["observed_mask"], limit, item.get("geometry", dataset.geometry))
                    + "</figure>"
                )
            body.append("</div>")
    document = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
    document += "<title>Field method analysis</title><style>body{font:16px system-ui;margin:32px auto;max-width:1200px;padding:0 20px;color:#20303a;background:#fafbfc}nav a{color:#12688b}h1,h2{line-height:1.25}h2{margin-top:36px}.status{font-weight:700}.scroll{overflow:auto}table{border-collapse:collapse;background:white;font-size:13px}th,td{padding:8px;border:1px solid #d6dfe4;text-align:right}th:first-child,td:first-child{text-align:left}.panels{display:flex;gap:16px;flex-wrap:wrap}figure{margin:0;flex:1;min-width:220px;max-width:320px}svg{max-width:100%;background:white}body>svg{width:720px}figcaption{margin-bottom:6px}li{margin:8px 0}</style>"
    document += "<body>" + "".join(body) + "</body></html>"
    path = Path(path)
    path.write_text(document, encoding="utf-8")
    return path
