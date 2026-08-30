"""Canonical field names for the data schemas defined in the F2S proposal
(private/proposal.tex, "Standard Data Schemas"). These are intentionally
plain dicts/constants rather than a heavyweight schema library, so every
producer/consumer module can `from f2s.common.schemas import EPISODE_META_FIELDS`
and stay in sync without introducing a new dependency.
"""

EPISODE_META_FIELDS = [
    "episode_id",
    "task",
    "seed",
    "round",
    "success",
    "failure_type",
    "failure_time",
    "failure_stage",
    "episode_length",
]

FAILURE_SEGMENT_FIELDS = [
    "episode_id",
    "failure_time",
    "start_time",
    "end_time",
    "state_window",
    "action_window",
    "feature",
    "failure_type",
    "failure_stage",
]

CANDIDATE_FIELDS = [
    "candidate_id",
    "source_episode_id",
    "failure_mode_id",
    "latent_delta",
    "action_chunk",
    "predicted_success",
    "predicted_risk",
    "actual_success",
    "actual_risk",
]

SKILL_FIELDS = [
    "skill_id",
    "failure_mode_id",
    "latent_delta",
    "precondition",
    "effect",
    "success_rate",
    "recovery_rate",
    "transfer_rate",
    "risk_score",
    "source_candidate_ids",
]

# Failure labels and their priority order (Day 8.1 of the proposal).
FAILURE_LABELS = [
    "success",
    "collision",
    "object_drop",
    "timeout",
    "grasp_failure",
    "placement_failure",
    "pose_error",
    "unknown",
]

FAILURE_STAGES = ["approach", "grasp", "transport", "placement", "unknown"]


def validate_fields(d: dict, required_fields: list, name: str) -> None:
    missing = [f for f in required_fields if f not in d]
    if missing:
        raise ValueError(f"{name} record missing required fields: {missing}")
