"""Business-owned public inputs and Stage outputs."""

from ml_platform_core.contracts import PackageManifest, ParameterSpec, StageSpec, TaskSpec

_INPUTS = (
    ParameterSpec("Run/name", "str", ("run", "name")),
    ParameterSpec("Run/seed", "int", ("run", "seed")),
    ParameterSpec("Input/local_path", "str", ("data", "local_path"), public=False),
    ParameterSpec("Input/clearml_dataset_id", "str", ("data", "clearml_dataset_id")),
    ParameterSpec("Input/dataset_file", "str", ("data", "dataset_file")),
    ParameterSpec("Input/condition_columns", "json", ("data", "condition_columns")),
    ParameterSpec("Input/source_manifest", "str", ("data", "source_manifest")),
    ParameterSpec("Input/format", "enum", ("data", "format")),
    ParameterSpec("Input/fields", "json", ("data", "fields")),
    ParameterSpec("Input/geometry", "json", ("data", "geometry")),
    ParameterSpec("Input/samples", "json", ("data", "samples")),
    ParameterSpec("Input/conditions", "json", ("data", "conditions")),
    ParameterSpec("Input/masks", "json", ("data", "masks")),
    ParameterSpec("Input/observations", "json", ("data", "observations")),
    ParameterSpec("Input/interpolation", "json", ("data", "interpolation")),
    ParameterSpec("Metrics/names", "json", ("metrics", "names")),
    ParameterSpec("Metrics/channel_scales", "json", ("metrics", "channel_scales")),
    ParameterSpec("Metrics/channel_weights", "json", ("metrics", "channel_weights")),
    ParameterSpec("Metrics/sample_weight_column", "str", ("metrics", "sample_weight_column")),
    ParameterSpec("Output/prediction_name", "str", ("output", "prediction_name")),
    ParameterSpec("Report/config", "json", ("report",)),
    ParameterSpec("Uncertainty/config", "json", ("uncertainty",)),
)
_TRAINING = _INPUTS + (
    ParameterSpec("Run/stage", "str", ("run", "stage"), public=False),
    ParameterSpec("Model/candidates", "json", ("model", "candidates")),
    ParameterSpec("Model/selection_metric", "enum", ("model", "selection_metric")),
    ParameterSpec("Tuning/config", "json", ("tuning",)),
    ParameterSpec("Split/method", "enum", ("split", "method")),
    ParameterSpec("Split/seed", "int", ("split", "seed")),
    ParameterSpec("Split/valid_size", "float", ("split", "valid_size")),
    ParameterSpec("Split/selection_size", "float", ("split", "selection_size")),
    ParameterSpec("Split/selection_method", "enum", ("split", "selection_method")),
    ParameterSpec("Split/group_column", "str", ("split", "group_column")),
    ParameterSpec("Split/time_column", "str", ("split", "time_column")),
    ParameterSpec("Split/gap_seconds", "float", ("split", "gap_seconds")),
    ParameterSpec("Split/valid_filter_column", "str", ("split", "valid_filter_column")),
    ParameterSpec("Split/valid_filter_value", "str", ("split", "valid_filter_value")),
    ParameterSpec("StageInputs/canonical_dataset", "str", ("stage_inputs", "canonical_dataset"), public=False),
    ParameterSpec("StageInputs/split", "str", ("stage_inputs", "split"), public=False),
    ParameterSpec("StageInputs/candidate_results", "json", ("stage_inputs", "candidate_results"), public=False),
    ParameterSpec("StageInputs/candidate_id", "str", ("stage_inputs", "candidate_id"), public=False),
    ParameterSpec("Candidate/config", "json", ("candidate",), public=False),
)
MODE_DECOMP_MANIFEST = PackageManifest(
    tasks=(
        TaskSpec("mode_decomp_import", _INPUTS + (ParameterSpec("Import/config", "json", ("import",)),)),
        TaskSpec("mode_decomp_pipeline", _TRAINING),
        TaskSpec("mode_decomp_stage", _TRAINING),
        TaskSpec(
            "mode_decomp_infer",
            _INPUTS
            + (
                ParameterSpec("Model/artifact_path", "str", ("model", "artifact_path"), public=False),
                ParameterSpec("Model/info_path", "str", ("model", "info_path"), public=False),
                ParameterSpec("Model/source_type", "enum", ("model", "source_type"), public=False),
                ParameterSpec(
                    "Model/source_task_id",
                    "str",
                    ("model", "source_task_id"),
                    "Published training task supplying the field model.",
                ),
                ParameterSpec(
                    "Model/model_selector",
                    "str",
                    ("model", "model_selector"),
                    "Published final model to use for inference.",
                ),
            ),
        ),
        TaskSpec(
            "mode_decomp_representation",
            _INPUTS
            + (
                ParameterSpec("Representation/config", "json", ("representation",)),
                ParameterSpec("Analysis/config", "json", ("analysis",)),
            ),
        ),
        TaskSpec(
            "mode_decomp_analysis",
            _TRAINING + (ParameterSpec("Analysis/config", "json", ("analysis",)),),
        ),
    ),
    stages=(
        StageSpec("prepare_fields", _TRAINING, ("canonical_dataset", "split")),
        StageSpec("train_candidate", _TRAINING, ("candidate_result", "analysis_summary")),
        StageSpec(
            "finalize_surrogate", _TRAINING, ("model", "model_info", "metrics", "result_summary", "analysis_summary")
        ),
    ),
)
