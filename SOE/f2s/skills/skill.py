"""Skill data structure (proposal Section 11 / Day 20)."""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class Skill:
    skill_id: str
    failure_mode_id: int
    latent_delta: np.ndarray
    precondition: Dict[str, Any]   # {failure_mode_id, task_stage, object_error_range, goal_error_range}
    effect: Dict[str, Any]         # {final_object_error, final_goal_error, task_progress_change, recovery_success}
    success_rate: float
    recovery_rate: float
    transfer_rate: float
    risk_score: float
    source_candidate_ids: List[str] = field(default_factory=list)
    action_chunk: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        return dict(
            skill_id=self.skill_id,
            failure_mode_id=self.failure_mode_id,
            latent_delta=np.asarray(self.latent_delta).tolist(),
            precondition=self.precondition,
            effect=self.effect,
            success_rate=self.success_rate,
            recovery_rate=self.recovery_rate,
            transfer_rate=self.transfer_rate,
            risk_score=self.risk_score,
            source_candidate_ids=self.source_candidate_ids,
            action_chunk=(np.asarray(self.action_chunk).tolist() if self.action_chunk is not None else None),
        )

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Skill":
        d = dict(d)
        d["latent_delta"] = np.asarray(d["latent_delta"])
        if d.get("action_chunk") is not None:
            d["action_chunk"] = np.asarray(d["action_chunk"])
        return cls(**d)
