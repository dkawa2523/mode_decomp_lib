"""Business-owned inputs, grouped by the tasks that actually consume them."""

from ml_platform_core.contracts import PackageManifest, ParameterSpec, StageSpec, TaskSpec

_RUN = (
    ParameterSpec("Run/name", "str", ("run", "name")),
    ParameterSpec("Run/seed", "int", ("run", "seed")),
    ParameterSpec("Runtime/output_dir", "str", ("runtime", "output_dir"), public=False),
)
_INPUT_ROOT = (
    ParameterSpec("Input/local_path", "str", ("data", "local_path"), public=False),
    ParameterSpec("Input/source_manifest", "str", ("data", "source_manifest")),
)
_FIELD_INPUT = _INPUT_ROOT + (
    ParameterSpec("Input/format", "enum", ("data", "format")),
    ParameterSpec("Input/fields", "json", ("data", "fields")),
    ParameterSpec("Input/geometry", "json", ("data", "geometry")),
    ParameterSpec("Input/samples", "json", ("data", "samples")),
    ParameterSpec("Input/conditions", "json", ("data", "conditions")),
    ParameterSpec("Input/masks", "json", ("data", "masks")),
    ParameterSpec("Input/observations", "json", ("data", "observations")),
    ParameterSpec("Input/interpolation", "json", ("data", "interpolation")),
)
_DATASET_INPUT = _FIELD_INPUT + (
    ParameterSpec(
        "Input/clearml_dataset_id",
        "str",
        ("data", "clearml_dataset_id"),
        "Dataset revision label; does not download remote data.",
    ),
)
_SOURCE_INPUT = (ParameterSpec("Input/input", "json", ("data", "input")),)
_METRICS = (
    ParameterSpec("Metrics/reference", "enum", ("metrics", "reference")),
    ParameterSpec("Metrics/names", "json", ("metrics", "names")),
    ParameterSpec("Metrics/channel_scales", "json", ("metrics", "channel_scales")),
    ParameterSpec("Metrics/channel_weights", "json", ("metrics", "channel_weights")),
    ParameterSpec("Metrics/sample_weight_column", "str", ("metrics", "sample_weight_column")),
)
_SPLIT = (
    ParameterSpec("Split/method", "enum", ("split", "method")),
    ParameterSpec("Split/seed", "int", ("split", "seed")),
    ParameterSpec("Split/valid_size", "float", ("split", "valid_size")),
    ParameterSpec("Split/group_column", "str", ("split", "group_column")),
    ParameterSpec("Split/time_column", "str", ("split", "time_column")),
    ParameterSpec("Split/gap_seconds", "float", ("split", "gap_seconds")),
    ParameterSpec("Split/valid_filter_column", "str", ("split", "valid_filter_column")),
    ParameterSpec("Split/valid_filter_value", "str", ("split", "valid_filter_value")),
)
_SELECTION_SPLIT = (
    ParameterSpec("Split/selection_size", "float", ("split", "selection_size")),
    ParameterSpec("Split/selection_method", "enum", ("split", "selection_method")),
)
_CANDIDATES = (ParameterSpec("Model/candidates", "json", ("model", "candidates")),)
_PREDICTIONS = (ParameterSpec("Output/prediction_name", "str", ("output", "prediction_name")),)
_REPORT = (ParameterSpec("Report/config", "json", ("report",)),)
_UNCERTAINTY = (ParameterSpec("Uncertainty/config", "json", ("uncertainty",)),)
_ANALYSIS = (ParameterSpec("Analysis/config", "json", ("analysis",)),)
_TRAINING = (
    _RUN
    + _DATASET_INPUT
    + _SOURCE_INPUT
    + _METRICS
    + _SPLIT
    + _SELECTION_SPLIT
    + _CANDIDATES
    + _PREDICTIONS
    + _UNCERTAINTY
    + (
        ParameterSpec("Model/selection_metric", "enum", ("model", "selection_metric")),
        ParameterSpec("Tuning/config", "json", ("tuning",)),
    )
)
_STAGE = _TRAINING + (
    ParameterSpec("Run/stage", "str", ("run", "stage"), public=False),
    ParameterSpec("StageInputs/canonical_dataset", "str", ("stage_inputs", "canonical_dataset"), public=False),
    ParameterSpec("StageInputs/split", "str", ("stage_inputs", "split"), public=False),
    ParameterSpec("StageInputs/candidate_results", "json", ("stage_inputs", "candidate_results"), public=False),
    ParameterSpec("StageInputs/candidate_id", "str", ("stage_inputs", "candidate_id"), public=False),
    ParameterSpec("Candidate/config", "json", ("candidate",), public=False),
)

MODE_DECOMP_MANIFEST = PackageManifest(
    tasks=(
        TaskSpec("mode_decomp_import", _RUN + _FIELD_INPUT + (ParameterSpec("Import/config", "json", ("import",)),)),
        TaskSpec("mode_decomp_pipeline", _TRAINING),
        TaskSpec("mode_decomp_stage", _STAGE),
        TaskSpec(
            "mode_decomp_infer",
            _RUN
            + _INPUT_ROOT
            + _SOURCE_INPUT
            + _PREDICTIONS
            + _UNCERTAINTY
            + (
                ParameterSpec("Input/dataset_file", "str", ("data", "dataset_file")),
                ParameterSpec("Input/condition_columns", "json", ("data", "condition_columns")),
                ParameterSpec("Input/id_column", "str", ("data", "id_column")),
                ParameterSpec("Model/artifact_path", "str", ("model", "artifact_path"), public=False),
                ParameterSpec("Model/info_path", "str", ("model", "info_path"), public=False),
            ),
        ),
        TaskSpec(
            "mode_decomp_representation",
            _RUN
            + _DATASET_INPUT
            + _METRICS
            + _REPORT
            + _ANALYSIS
            + (ParameterSpec("Representation/config", "json", ("representation",)),),
        ),
        TaskSpec(
            "mode_decomp_analysis",
            _RUN
            + _DATASET_INPUT
            + _SOURCE_INPUT
            + _METRICS
            + _SPLIT
            + _CANDIDATES
            + _REPORT
            + _UNCERTAINTY
            + _ANALYSIS,
        ),
    ),
    stages=(
        StageSpec("prepare_fields", _STAGE, ("canonical_dataset", "split")),
        StageSpec("train_candidate", _STAGE, ("candidate_result", "analysis_summary")),
        StageSpec(
            "finalize_surrogate", _STAGE, ("model", "model_info", "metrics", "result_summary", "analysis_summary")
        ),
    ),
)
