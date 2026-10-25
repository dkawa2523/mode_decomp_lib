"""Connect held-out calibration and field coverage to the normal run outputs."""


def add_uncertainty_report(report, model, reference, options):
    calibrator = model.calibrator
    if calibrator is None:
        return
    evaluation = calibrator.evaluate(reference, report.predictions, options)
    report.summary["field_uncertainty"] = {"calibration": calibrator.summary(), "evaluation": evaluation}
    report.per_sample["interval_simultaneous_inside"] = evaluation["sample_simultaneous_inside"]
    report.per_sample["interval_pointwise_coverage"] = evaluation["sample_pointwise_coverage"]
    for channel in range(reference.fields.shape[-1]):
        report.per_sample[f"channel_{channel}_interval_coverage"] = [
            row[channel] for row in evaluation["sample_channel_pointwise_coverage"]
        ]
    for key in (
        "sample_simultaneous_coverage",
        "weighted_sample_simultaneous_coverage",
        "pointwise_coverage",
        "mean_width",
        "mean_scaled_width",
    ):
        if evaluation[key] is not None:
            report.metrics[f"interval_{key}"] = evaluation[key]
