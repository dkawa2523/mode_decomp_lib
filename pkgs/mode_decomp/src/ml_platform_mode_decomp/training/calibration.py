"""Fit held-out field calibration after proper training and selection."""


def validate_calibration_split(dataset, split, settings):
    """Check static band prerequisites before fitting any candidate."""
    if not settings.get("enabled", False):
        return
    from ..uncertainty import _calibration_rank, _require_complete_truth, _support

    support, _ = _support(dataset)
    _calibration_rank(len(split["calibration_indices"]), settings["coverage"])
    _require_complete_truth(dataset, support, indices=split["calibration_indices"])
    _require_complete_truth(dataset, support, indices=split["test_indices"])
    if settings["normalization"] == "channel":
        if not any(dataset._sample_mask(index)[support].any() for index in split["train_indices"]):
            raise ValueError("Channel normalization needs observed proper training values")


def calibrate_from_split(model, dataset, split, settings):
    if settings.get("enabled", False):
        model.calibrate(
            dataset._subset_validated(split["calibration_indices"]),
            training=dataset._subset_validated(split["train_indices"]),
            **{key: settings[key] for key in ("coverage", "scope", "normalization")},
        )
