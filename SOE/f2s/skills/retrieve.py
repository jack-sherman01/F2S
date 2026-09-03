"""Skill retrieval (proposal Section 11.2 / Day 21.1; state-aware
Match(S, F_k, s_t) formula per proposal_revised.tex Section 11.2).

    k* = argmax_{k_i in K} [ Match(k_i, f_t) + lambda * PreconditionSimilarity(k_i, s_t) ]

Match(k_i, f_t) is 1.0 if the skill's failure_mode_id equals the current
failure mode, else 0.0 (skills are always generated for a specific
failure mode, so this is an exact-match indicator rather than a learned
similarity). PreconditionSimilarity uses a Gaussian kernel over the
skill's stored object-error precondition midpoint vs. the current
object-error, since that is the scalar precondition feature every skill
records (f2s.skills.archive / Day 20's precondition schema).

**Spatial applicability gate** (added after the diagnostic in
SOE/README_F2S.md's "Controlled diagnostic: is the 0% a gating problem or
a skill-quality problem?" section -- see private/technical_contributions_log.md
section 21 for the full before/after numbers): the two checks above are
*soft* -- with a degenerate `object_error_range=(0.0, 0.0)` precondition
(the offset-sweep discovery scripts' placeholder) and a single failure
mode shared by every skill, `match` alone already clears `MIN_MATCH_SCORE`
regardless of the current state, so retrieval fired on ~every stall
(confirmed: 99/100 episodes in the Day-25 eval, 0% success). This is now
a genuine, previously-missing *hard* gate: a skill is only a retrieval
candidate at all if the current object (x, y) position is within
`position_tolerance` of the skill's own recorded origin state
(`precondition["object_xy"]`) -- the tolerance defaults to exactly the
neighborhood Day-19 validation itself tested
(`f2s.candidates.validator.perturb_object_position_near`'s
`max_offset=0.03`), so the gate is precisely as generous as the actual
validation evidence for the skill, no stricter. Confirmed to take F2S
from a clean 0% (three seeds) to 74.4% +/- 4.2% (matching/slightly
exceeding the 72.2% +/- 1.6% frozen-baseline reference) on the same
in-distribution setup. Skills archived before this change (or via any
path that never recorded `object_xy`) skip the spatial gate entirely
--backward compatible, but see f2s/evolution/loop.py's
`discover_and_archive_skills` for where `object_xy` is now populated for
every newly-archived skill.
"""
import math
from typing import Optional

import numpy as np

from f2s.skills.archive import SkillArchive
from f2s.skills.skill import Skill

MIN_MATCH_SCORE = 0.5  # below this, treat as "no matching skill" (Day 21.1)
DEFAULT_POSITION_TOLERANCE = 0.03  # meters; f2s.candidates.validator.perturb_object_position_near's max_offset


def precondition_similarity(skill: Skill, current_object_error: float, bandwidth: float = 0.1) -> float:
    lo, hi = skill.precondition.get("object_error_range", (current_object_error, current_object_error))
    mid = (lo + hi) / 2.0
    return float(math.exp(-((current_object_error - mid) ** 2) / (2 * bandwidth ** 2)))


def spatial_gate_passed(skill: Skill, current_object_xy) -> bool:
    """Hard applicability gate: True if the skill has no recorded origin
    position (legacy/back-compat -- falls through to the soft checks
    only), or if `current_object_xy` is within the skill's
    `position_tolerance` of its recorded `object_xy`."""
    origin_xy = skill.precondition.get("object_xy")
    if origin_xy is None or current_object_xy is None:
        return True
    tol = skill.precondition.get("position_tolerance", DEFAULT_POSITION_TOLERANCE)
    dist = float(np.linalg.norm(np.asarray(current_object_xy) - np.asarray(origin_xy)))
    return dist < tol


def retrieve(
    archive: SkillArchive,
    failure_mode_id: int,
    current_object_error: float,
    current_object_xy=None,
    lambd: float = 1.0,
    min_score: float = MIN_MATCH_SCORE,
) -> Optional[Skill]:
    best_skill = None
    best_score = -float("inf")
    for skill in archive.skills:
        if not spatial_gate_passed(skill, current_object_xy):
            continue
        match = 1.0 if skill.failure_mode_id == failure_mode_id else 0.0
        sim = precondition_similarity(skill, current_object_error)
        score = match + lambd * sim
        if score > best_score:
            best_score = score
            best_skill = skill
    if best_skill is None or best_score < min_score:
        return None
    return best_skill
