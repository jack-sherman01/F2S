"""Day 20 acceptance test: the archive can add a valid skill, reject an
invalid one, remove duplicates, save to disk, and reload without changing
the surviving records."""
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from f2s.skills.archive import SkillArchive
from f2s.skills.retrieve import retrieve
from f2s.skills.skill import Skill


def make_skill(skill_id, failure_mode_id, success_rate, risk_score, latent_delta, recovery_rate=0.5, object_error_range=(0.1, 0.2)):
    return Skill(
        skill_id=skill_id,
        failure_mode_id=failure_mode_id,
        latent_delta=np.asarray(latent_delta, dtype=np.float32),
        precondition=dict(failure_mode_id=failure_mode_id, task_stage="grasp", object_error_range=object_error_range, goal_error_range=(0.0, 0.3)),
        effect=dict(final_object_error=0.05, final_goal_error=0.05, task_progress_change=0.3, recovery_success=True),
        success_rate=success_rate,
        recovery_rate=recovery_rate,
        transfer_rate=0.0,
        risk_score=risk_score,
        source_candidate_ids=[f"{skill_id}_src"],
    )


def main():
    archive = SkillArchive()

    good = make_skill("skill_A", failure_mode_id=0, success_rate=0.9, risk_score=0.02, latent_delta=[1.0, 0.0, 0.0])
    accepted, reason = archive.add(good)
    assert accepted and reason is None, f"valid skill was rejected: {reason}"
    assert len(archive.skills) == 1

    bad = make_skill("skill_B", failure_mode_id=0, success_rate=0.3, risk_score=0.02, latent_delta=[5.0, 5.0, 5.0])
    accepted, reason = archive.add(bad)
    assert not accepted and reason == "validation_success_rate_below_threshold"
    assert len(archive.skills) == 1
    assert any(r["candidate_id"] == "skill_B" for r in archive.rejected)

    # near-duplicate of `good` (small latent-delta distance, same failure mode):
    # should replace `good` only if it's actually better.
    dup_worse = make_skill("skill_A_dup_worse", failure_mode_id=0, success_rate=0.75, risk_score=0.05, latent_delta=[1.05, 0.0, 0.0])
    archive.add(dup_worse)
    assert len(archive.skills) == 1 and archive.skills[0].skill_id == "skill_A", "worse duplicate should not replace the better skill"

    dup_better = make_skill("skill_A_dup_better", failure_mode_id=0, success_rate=0.95, risk_score=0.01, latent_delta=[1.02, 0.0, 0.0])
    archive.add(dup_better)
    assert len(archive.skills) == 1 and archive.skills[0].skill_id == "skill_A_dup_better", "better duplicate should replace the worse skill"

    # a skill for a *different* failure mode with a similar latent_delta must NOT be deduplicated away
    other_mode = make_skill("skill_C", failure_mode_id=1, success_rate=0.85, risk_score=0.03, latent_delta=[1.02, 0.0, 0.0], object_error_range=(0.3, 0.4))
    archive.add(other_mode)
    assert len(archive.skills) == 2

    tmp_path = "/tmp/f2s_selftest_skill_archive/archive.json"
    if os.path.exists(os.path.dirname(tmp_path)):
        shutil.rmtree(os.path.dirname(tmp_path))
    archive.save(tmp_path)
    reloaded = SkillArchive.load(tmp_path)
    assert len(reloaded.skills) == len(archive.skills)
    assert {s.skill_id for s in reloaded.skills} == {s.skill_id for s in archive.skills}
    assert len(reloaded.rejected) == len(archive.rejected)
    print("Save/reload check passed:", [s.skill_id for s in reloaded.skills])

    # retrieval: object_error close to skill_A's precondition midpoint (0.15) should retrieve it
    retrieved = retrieve(reloaded, failure_mode_id=0, current_object_error=0.15)
    assert retrieved is not None and retrieved.skill_id == "skill_A_dup_better"
    # far outside any precondition range and wrong failure mode -> should retrieve None
    none_retrieved = retrieve(reloaded, failure_mode_id=5, current_object_error=5.0)
    assert none_retrieved is None
    print("Retrieval check passed.")

    shutil.rmtree(os.path.dirname(tmp_path))
    print("\nDay 20 acceptance test PASSED")


if __name__ == "__main__":
    main()
