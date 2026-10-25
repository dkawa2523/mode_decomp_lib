"""The field workflow, without SDK placeholders or remote transport."""

from ml_platform_core.contracts import ArtifactRef, DomainStepPlan
from typing import Any

from .manifest import MODE_DECOMP_MANIFEST
from .training.preparation import needs_selection


def build_domain_plan(cfg: dict[str, Any]) -> tuple[DomainStepPlan, ...]:
    stages = MODE_DECOMP_MANIFEST
    prepare_refs = {
        "split": ArtifactRef("prepare_fields", "split"),
        "canonical_dataset": ArtifactRef("prepare_fields", "canonical_dataset"),
    }
    steps = [
        DomainStepPlan(
            "prepare_fields", "prepare_fields", expected_artifacts=stages.stage("prepare_fields").output_artifacts
        )
    ]
    candidate_steps = []
    for candidate in cfg["model"]["candidates"] if needs_selection(cfg) else ():
        name = "train_" + candidate["id"].lower().replace("-", "_")
        candidate_steps.append(name)
        steps.append(
            DomainStepPlan(
                name,
                "train_candidate",
                ("prepare_fields",),
                {"stage_inputs": prepare_refs, "candidate": candidate},
                stages.stage("train_candidate").output_artifacts,
            )
        )
    steps.append(
        DomainStepPlan(
            "finalize_surrogate",
            "finalize_surrogate",
            ("prepare_fields", *candidate_steps),
            {
                "stage_inputs": {
                    **prepare_refs,
                    "candidate_results": [ArtifactRef(name, "candidate_result") for name in candidate_steps],
                }
            },
            stages.stage("finalize_surrogate").output_artifacts,
        )
    )
    return tuple(steps)
