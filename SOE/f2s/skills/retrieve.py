"""Skill retrieval (proposal Section 11.2 / Day 21.1).

    k* = argmax_{k_i in K} [ Match(k_i, f_t) + lambda * PreconditionSimilarity(k_i, s_t) ]

Match(k_i, f_t) is 1.0 if the skill's failure_mode_id equals the current
failure mode, else 0.0 (skills are always generated for a specific
failure mode, so this is an exact-match indicator rather than a learned
similarity). PreconditionSimilarity uses a Gaussian kernel over the
skill's stored object-error precondition midpoint vs. the current
object-error, since that is the scalar precondition feature every skill
records (f2s.skills.archive / Day 20's precondition schema).
"""
import math
from typing import Optional

from f2s.skills.archive import SkillArchive
from f2s.skills.skill import Skill

MIN_MATCH_SCORE = 0.5  # below this, treat as "no matching skill" (Day 21.1)


def precondition_similarity(skill: Skill, current_object_error: float, bandwidth: float = 0.1) -> float:
    lo, hi = skill.precondition.get("object_error_range", (current_object_error, current_object_error))
    mid = (lo + hi) / 2.0
    return float(math.exp(-((current_object_error - mid) ** 2) / (2 * bandwidth ** 2)))


def retrieve(
    archive: SkillArchive,
    failure_mode_id: int,
    current_object_error: float,
    lambd: float = 1.0,
    min_score: float = MIN_MATCH_SCORE,
) -> Optional[Skill]:
    best_skill = None
    best_score = -float("inf")
    for skill in archive.skills:
        match = 1.0 if skill.failure_mode_id == failure_mode_id else 0.0
        sim = precondition_similarity(skill, current_object_error)
        score = match + lambd * sim
        if score > best_score:
            best_score = score
            best_skill = skill
    if best_skill is None or best_score < min_score:
        return None
    return best_skill
