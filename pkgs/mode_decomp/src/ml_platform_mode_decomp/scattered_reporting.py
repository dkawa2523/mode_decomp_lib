"""Original-point scores shared by direct decomposition and held-out comparisons."""

from .data.input_spec import field_input_spec
import numpy as np

from .evaluation.observations import prepare_scattered_evaluation
from .evaluation.fields import score_weights


def prepare_scattered_report(data, reference):
    _, specification = field_input_spec(data)
    return prepare_scattered_evaluation(data["local_path"], reference, source_manifest=specification)


def add_scattered_report(data, report, reference, options, *, prepared=None):
    evaluator = prepared or prepare_scattered_report(data, reference)
    measure = score_weights(reference, options)
    scores = evaluator.evaluate(report.predictions, sample_weights=measure.sample_weights)
    report.summary["scattered"] = {key: value for key, value in scores.items() if key not in {"samples", "field"}}
    for kind in ("observation", "interpolation"):
        report.metrics[f"{kind}_coverage"] = scores[kind]["coverage"]
        report.per_sample[f"{kind}_coverage"] = [item[kind]["coverage"] for item in scores["samples"]]
        for metric in ("rmse", "mae"):
            values = scores[kind][metric]
            if values is not None:
                for channel, value in enumerate(values):
                    report.metrics[f"{kind}_{metric}_ch{channel}"] = value
                scaled = np.asarray(values) / measure.scales
                report.metrics[f"{kind}_{metric}"] = (
                    float(np.sqrt((scaled**2) @ measure.channel_weights))
                    if metric == "rmse"
                    else float(scaled @ measure.channel_weights)
                )
            for channel in range(reference.fields.shape[-1]):
                report.per_sample[f"{kind}_{metric}_ch{channel}"] = [
                    None if item[kind][metric] is None else item[kind][metric][channel] for item in scores["samples"]
                ]
