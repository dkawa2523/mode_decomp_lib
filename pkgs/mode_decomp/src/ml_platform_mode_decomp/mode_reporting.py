"""Optional views of explicitly selected A coordinates and their field responses."""

from html import escape
from pathlib import Path
from collections.abc import Mapping
import json

import numpy as np
import pandas as pd
from ml_platform_core.io import write_table

from .analysis_reporting import _field_svg
from .evaluation.analysis import mode_reconstruction_curve
from .evaluation.objectives import prepare_objective
from .modes import _mode_responses


def mode_options(value):
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) - {"enabled", "indices", "cumulative_counts"}:
        raise ValueError("analysis.modes accepts enabled, indices and cumulative_counts")
    if not isinstance(value.get("enabled", True), bool):
        raise ValueError("analysis.modes.enabled must be boolean")
    if not value.get("enabled", True):
        return None
    indices = value.get("indices")
    if not isinstance(indices, list) or not indices or any(type(x) is not int or x < 0 for x in indices):
        raise ValueError("analysis.modes.indices must be a nonempty list of nonnegative integers")
    if len(set(indices)) != len(indices):
        raise ValueError("analysis.modes.indices must be distinct")
    counts = value.get("cumulative_counts", list(range(1, len(indices) + 1)))
    if not isinstance(counts, list) or any(type(x) is not int or not 1 <= x <= len(indices) for x in counts):
        raise ValueError("Mode cumulative_counts must lie between 1 and the number of selected indices")
    return {"indices": indices, "counts": sorted(set(counts))}


def write_mode_report(
    representation,
    coefficients,
    reference,
    options,
    settings,
    directory,
    *,
    evaluation_mask=None,
    full_predictions=None,
    prepared=None,
):
    """All reconstruction losses use original reference fields and the main study measure."""
    directory = Path(directory)
    indices = settings["indices"]
    metadata = representation.list_modes(indices, space="a")
    origin = np.zeros(representation.packed_dim)
    directory.mkdir(parents=True, exist_ok=True)
    prepared = prepared or prepare_objective(reference, options, evaluation_mask=evaluation_mask)
    curve = mode_reconstruction_curve(
        representation, coefficients, settings, prepared, full_predictions=full_predictions
    )
    curve["selection_order"] = "explicit_index_order"
    table = pd.DataFrame(
        [
            {key: json.dumps(value) if isinstance(value, (list, tuple, dict)) else value for key, value in item.items()}
            for item in metadata
        ]
    )
    table["selection_order"] = "explicit_index_order"
    mode_path = write_table(table, directory / "modes.csv")
    curve_path = write_table(curve, directory / "reconstruction_curve.csv")
    columns = [
        name
        for name in (
            "index",
            "id",
            "interpretation",
            "channel",
            "component_name",
            "kind",
            "kx",
            "ky",
            "frequency_x",
            "frequency_y",
            "additive",
        )
        if name in table
    ]
    mask = np.ones(representation.field_shape[:-1], dtype=bool)
    if representation.geometry.mask is not None:
        mask &= np.asarray(representation.geometry.mask, dtype=bool).reshape(mask.shape)
    weights = representation.geometry.integration_weights()
    if weights is not None:
        mask &= np.asarray(weights).reshape(mask.shape) > 0
    body = [
        "<h1>Selected coefficient coordinates</h1>",
        "<p>A coordinates are shown before coefficient postprocessing. Selection follows the explicitly supplied "
        "index order; it is not an energy ranking. Responses equal inverse(e_i) − inverse(0). "
        "For nonlinear synthesis they are coordinate perturbations at zero, not additive physical modes. "
        "Coefficient squares are not a common physical energy across methods. Single-channel coordinates display "
        "their channel; joint responses preview up to the first three channels.</p>",
        '<p><a href="modes.csv">Coordinate metadata CSV</a> · '
        '<a href="reconstruction_curve.csv">Reconstruction losses CSV</a></p>',
        table[columns].to_html(index=False, escape=True),
        "<h2>Selected-coordinate reconstruction</h2><p>All reference samples and the main study evaluation support "
        "are scored in the original field. Full A is reconstructed before coefficient postprocessing; "
        "for surrogate studies A comes from predicted targets. A lossy coefficient postprocessor can therefore "
        "make the full-A endpoint differ from the main representation or imputation score.</p>",
        curve.to_html(index=False, escape=True, float_format=lambda value: f"{value:.6g}"),
        "<h2>Unit coordinate responses</h2>",
    ]
    responses = _mode_responses(representation, indices, reference=origin)
    for item, response in zip(metadata, responses, strict=True):
        body.append(f"<h3>{escape(str(item['id']))} · {escape(str(item['interpretation']))}</h3><div class=panels>")
        channels = [item["channel"]] if "channel" in item else range(min(response.shape[-1], 3))
        for channel in channels:
            field = response[..., channel]
            limit = max(float(np.max(np.abs(field[mask]))), np.finfo(float).tiny)
            name = representation.components[channel] if representation.components else f"channel_{channel}"
            body.append(
                f"<figure><figcaption>{escape(name)} · ±{limit:.4g}</figcaption>"
                + _field_svg(field, mask, limit, representation.geometry)
                + "</figure>"
            )
        body.append("</div>")
    document = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
    document += "<title>Selected mode analysis</title><style>body{font:16px system-ui;margin:32px;max-width:1200px;color:#20303a}table{border-collapse:collapse;display:block;overflow:auto}th,td{border:1px solid #ccd7df;padding:8px}.panels{display:flex;flex-wrap:wrap;gap:16px}figure{margin:0;min-width:220px;max-width:320px;flex:1}svg{width:100%}</style><body>"
    report_path = directory / "modes.html"
    report_path.write_text(document + "".join(body) + "</body></html>", encoding="utf-8")
    return {"mode_report": report_path, "mode_metadata": mode_path, "mode_reconstruction": curve_path}
