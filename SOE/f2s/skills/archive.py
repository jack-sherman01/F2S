"""Skill archive (proposal Section 11 / Day 20): the archive rule, a
duplicate-removal rule, and JSON persistence. Skills that fail the archive
rule are recorded in a separate rejected-skills file with a reason,
instead of being silently dropped."""
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from f2s.skills.skill import Skill

SUCCESS_RATE_THRESHOLD = 0.7
RISK_THRESHOLD = 0.1
DUPLICATE_EPS = 0.5  # L2 distance in latent-delta space (proposal Day 20.2)


class SkillArchive:
    def __init__(self):
        self.skills: List[Skill] = []
        self.rejected: List[Dict[str, Any]] = []

    def add(
        self,
        skill: Skill,
        success_rate_threshold: float = SUCCESS_RATE_THRESHOLD,
        risk_threshold: float = RISK_THRESHOLD,
    ) -> Tuple[bool, Optional[str]]:
        """Archive `skill` if it passes the Day 20.1 rule
        (success_rate > threshold and risk_score < threshold), else record
        it as rejected with a reason. Returns (accepted, reason)."""
        if skill.success_rate <= success_rate_threshold:
            reason = "validation_success_rate_below_threshold"
            self.rejected.append(dict(candidate_id=skill.skill_id, reason=reason))
            return False, reason
        if skill.risk_score >= risk_threshold:
            reason = "validation_risk_above_threshold"
            self.rejected.append(dict(candidate_id=skill.skill_id, reason=reason))
            return False, reason

        self.skills.append(skill)
        self._remove_duplicates()
        return True, None

    def _remove_duplicates(self) -> None:
        """Proposal Day 20.2: two skills are duplicates if
        ||delta_z_i - delta_z_j|| < eps. Keep the one with (1) higher
        success_rate, (2) lower risk_score if tied, (3) higher
        recovery_rate if still tied."""
        kept: List[Skill] = []
        for skill in self.skills:
            duplicate_of = None
            for i, existing in enumerate(kept):
                if existing.failure_mode_id != skill.failure_mode_id:
                    continue
                dist = float(np.linalg.norm(
                    np.asarray(existing.latent_delta).ravel() - np.asarray(skill.latent_delta).ravel()
                ))
                if dist < DUPLICATE_EPS:
                    duplicate_of = i
                    break
            if duplicate_of is None:
                kept.append(skill)
                continue
            existing = kept[duplicate_of]
            better = (
                (skill.success_rate, -skill.risk_score, skill.recovery_rate)
                > (existing.success_rate, -existing.risk_score, existing.recovery_rate)
            )
            if better:
                kept[duplicate_of] = skill
        self.skills = kept

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(dict(
                skills=[s.to_dict() for s in self.skills],
                rejected=self.rejected,
            ), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "SkillArchive":
        archive = cls()
        if not os.path.exists(path):
            return archive
        with open(path, "r") as f:
            data = json.load(f)
        archive.skills = [Skill.from_dict(d) for d in data.get("skills", [])]
        archive.rejected = data.get("rejected", [])
        return archive

    def skills_for_failure_mode(self, failure_mode_id: int) -> List[Skill]:
        return [s for s in self.skills if s.failure_mode_id == failure_mode_id]
