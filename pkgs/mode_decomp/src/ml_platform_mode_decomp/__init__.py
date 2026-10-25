"""Field representations and surrogates as an ML Platform business package."""

from .runners import run_task
from .config import load_config

__all__ = [
    "run_task",
    "load_config",
    "FieldDataset",
    "FieldRepresentation",
    "FieldSurrogate",
    "FieldCalibrator",
    "HodgeDecomposition",
    "load_model",
    "read_field_output",
]


def __getattr__(name: str):
    if name == "FieldDataset":
        from .data.schema import FieldDataset

        return FieldDataset
    if name == "FieldRepresentation":
        from .representation import FieldRepresentation

        return FieldRepresentation
    if name == "FieldSurrogate":
        from .surrogate import FieldSurrogate

        return FieldSurrogate
    if name == "load_model":
        from .model_artifact import load_model

        return load_model
    if name == "FieldCalibrator":
        from .uncertainty import FieldCalibrator

        return FieldCalibrator
    if name == "HodgeDecomposition":
        from .decomposition.hodge import HodgeDecomposition

        return HodgeDecomposition
    if name == "read_field_output":
        from .field_output import read_field_output

        return read_field_output
    raise AttributeError(name)
