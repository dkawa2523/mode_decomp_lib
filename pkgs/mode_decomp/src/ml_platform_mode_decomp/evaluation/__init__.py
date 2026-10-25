"""Shared field measures and purpose-specific scientific evaluation."""

from .fields import EvaluationReport, ScoreResult, evaluate_fields, mean_field_prediction, score_fields
from .objectives import PreparedObjective, prepare_objective
from .analysis import (
    analyze_imputation,
    analyze_representation,
    analyze_surrogate,
    observation_mask,
    representation_size,
    score_surrogate,
    score_representation,
    load_observation_mask,
)

__all__ = [
    "EvaluationReport",
    "ScoreResult",
    "PreparedObjective",
    "prepare_objective",
    "score_surrogate",
    "score_representation",
    "load_observation_mask",
    "evaluate_fields",
    "score_fields",
    "mean_field_prediction",
    "analyze_imputation",
    "analyze_representation",
    "analyze_surrogate",
    "observation_mask",
    "representation_size",
]
