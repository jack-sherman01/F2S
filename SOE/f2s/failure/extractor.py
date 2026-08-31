"""Failure-time detection and local segment extraction (proposal Day 8),
operating on episodes as saved by f2s.logging.episode_logger.EpisodeLogger.

All thresholds below are heuristics over the *low-dim structured state*
(object height, gripper aperture, end-effector/joint velocity), not a
learned classifier, per the proposal's Day 8 instruction. They are
deliberately conservative and documented so a reviewer can audit exactly
what "collision" / "object_drop" mean in this codebase -- these are proxy
labels for the real robosuite/mujoco contact and joint-limit signals that
robomimic's released low-dim datasets do not log per-timestep.

Failure label priority order (proposal Day 8.1):
    success -> collision -> object_drop -> timeout -> pose_error -> unknown
"""
from typing import Any, Dict, Optional, Tuple

import numpy as np

from f2s.common.schemas import FAILURE_LABELS

# --- heuristic thresholds (documented, not learned) -------------------
TABLE_HEIGHT = 0.8  # robosuite PickPlace bin/table surface default z
LIFT_HEIGHT_MARGIN = 0.05  # object must clear the table by this much to
                            # count as "lifted"
DROP_HEIGHT_MARGIN = 0.02  # after being lifted, object must fall back to
                            # within this margin of the table to count as
                            # "dropped" rather than merely lowered
GRIPPER_CLOSED_THRESHOLD = 0.02
JOINT_VEL_COLLISION_THRESHOLD = 6.0  # rad/s jump between consecutive steps
POSE_ERROR_THRESHOLD = 0.10  # meters, final object-to-goal distance below
                              # which a non-success episode is still
                              # considered "close" (pose_error) rather than
                              # a generic unknown failure


def _joint_vel_jump(joint_vel: np.ndarray) -> float:
    if joint_vel.shape[0] < 2:
        return 0.0
    deltas = np.diff(joint_vel, axis=0)
    return float(np.max(np.linalg.norm(deltas, axis=-1))) if deltas.size else 0.0


def detect_failure_time(meta: Dict[str, Any], arrays: Dict[str, np.ndarray]) -> Tuple[Optional[int], str]:
    """Return (failure_time, failure_type). failure_time is None for a
    successful episode."""
    if meta["success"]:
        return None, "success"

    ep_len = meta["episode_length"]
    if ep_len == 0:
        return None, "unknown"

    joint_vel = arrays.get("obs_robot0_joint_vel")
    if joint_vel is not None and joint_vel.shape[0] >= 2:
        deltas = np.linalg.norm(np.diff(joint_vel, axis=0), axis=-1)
        collision_steps = np.where(deltas > JOINT_VEL_COLLISION_THRESHOLD)[0]
        if len(collision_steps) > 0:
            # +1 because np.diff(joint_vel)[i] is the jump *into* step i+1
            return int(collision_steps[0]) + 1, "collision"

    obj = arrays.get("obs_object")
    gripper = arrays.get("obs_robot0_gripper_qpos")
    if obj is not None and gripper is not None:
        obj_z = obj[:, 2]
        gripper_closed = np.abs(gripper.mean(axis=-1)) < GRIPPER_CLOSED_THRESHOLD
        lifted = obj_z > (TABLE_HEIGHT + LIFT_HEIGHT_MARGIN)
        was_lifted = False
        for t in range(len(obj_z)):
            if lifted[t] and gripper_closed[t]:
                was_lifted = True
                continue
            if was_lifted and gripper_closed[t] and obj_z[t] < (TABLE_HEIGHT + DROP_HEIGHT_MARGIN):
                return t, "object_drop"

    final_delta_p_obj = None
    stall_time = ep_len - 1
    if obj is not None and len(obj) > 0:
        from f2s.failure.features import DEFAULT_GOAL_POS

        delta_p_obj_t = np.linalg.norm(obj[:, 0:3] - DEFAULT_GOAL_POS[None, :], axis=-1)
        final_delta_p_obj = float(delta_p_obj_t[-1])
        stall_time = find_stall_time(delta_p_obj_t)

    if meta.get("failure_type") == "timeout":
        if final_delta_p_obj is not None and final_delta_p_obj < POSE_ERROR_THRESHOLD:
            return stall_time, "pose_error"
        return stall_time, "timeout"

    return stall_time, "unknown"


def find_stall_time(delta_p_obj_t: np.ndarray, min_improvement: float = 1e-4) -> int:
    """The last timestep at which the object got measurably closer to the
    goal than at any earlier point in the episode -- i.e. where real task
    progress last happened. Every step after this is "stalled": the
    episode keeps running (e.g. to a `timeout`) without the object ever
    getting closer to the goal again.

    Using this (rather than the literal final timestep) as the failure
    time is important: a timeout failure's *terminal* state is often a
    dead end with no useful local structure to correct from (e.g. the arm
    idling after having given up), whereas the stall point is exactly
    where a corrective action chunk could plausibly still help."""
    if len(delta_p_obj_t) == 0:
        return 0
    best_so_far = np.minimum.accumulate(delta_p_obj_t)
    last_improvement_t = 0
    for t in range(1, len(best_so_far)):
        if best_so_far[t] < best_so_far[t - 1] - min_improvement:
            last_improvement_t = t
    return last_improvement_t


def assign_failure_stage(meta: Dict[str, Any], arrays: Dict[str, np.ndarray], failure_time: Optional[int]) -> str:
    """Coarse stage-of-task at the moment of failure: approach / grasp /
    transport / placement, based on gripper state and object height, since
    the low-dim dataset carries no explicit task-phase label."""
    if failure_time is None:
        return "none"

    obj = arrays.get("obs_object")
    gripper = arrays.get("obs_robot0_gripper_qpos")
    if obj is None or gripper is None or failure_time >= len(obj):
        return "unknown"

    obj_to_eef = obj[failure_time, 7:10]
    obj_z = obj[failure_time, 2]
    gripper_closed = abs(float(np.mean(gripper[failure_time]))) < GRIPPER_CLOSED_THRESHOLD
    near_object = float(np.linalg.norm(obj_to_eef)) < 0.05
    lifted = obj_z > (TABLE_HEIGHT + LIFT_HEIGHT_MARGIN)

    if not near_object:
        return "approach"
    if near_object and not gripper_closed:
        return "grasp"
    if gripper_closed and lifted:
        return "transport"
    if gripper_closed and not lifted:
        return "placement"
    return "unknown"


def extract_failure_segment(meta: Dict[str, Any], arrays: Dict[str, np.ndarray], failure_time: int, Hf: int = 10):
    """Return (start_time, end_time, state_window, action_window,
    obs_window) for the local failure segment [max(0, t_f - Hf), t_f]."""
    start_time = max(0, failure_time - Hf)
    end_time = failure_time

    state_window = arrays["states"][start_time:end_time + 1]
    action_window = arrays["actions"][start_time:end_time + 1]

    obs_keys = [k[len("obs_"):] for k in arrays.keys() if k.startswith("obs_")]
    obs_window = [
        {k: arrays[f"obs_{k}"][t] for k in obs_keys}
        for t in range(start_time, end_time + 1)
    ]
    return start_time, end_time, state_window, action_window, obs_window


# Default correction intervention offset (steps before the detected
# failure_time to actually start generating candidates from), frozen per
# the Day-22 "freeze the final configuration" rule after the offset sweep
# in SOE/README_F2S.md ("Tested the earlier-intervention hypothesis
# directly" / "Scale-up confirmation"): 0/10/15/20/25/30 were tested on
# 20 then 71 real Can failure states; 15 produced the project's first
# validated, archived skills (2/2 at 100% Day-19 validation, reconfirmed
# at 3.5x scale). Not re-tuned per state -- see that section for the
# measured, still-narrow (~2.8% of 71 states) coverage this single fixed
# value achieves, and why growing coverage (not re-tuning this number)
# is the documented next step, deliberately not pursued further this
# round per the Day-22 "stop tuning, move to the final comparisons" rule.
DEFAULT_INTERVENTION_OFFSET = 15


def process_episode(
    meta: Dict[str, Any],
    arrays: Dict[str, np.ndarray],
    Hf: int = 10,
    intervention_offset: int = DEFAULT_INTERVENTION_OFFSET,
) -> Dict[str, Any]:
    """Full Day-8 pipeline for one episode: detect failure time/type,
    assign stage, extract the local segment. Returns None for successful
    episodes (nothing to extract).

    `failure_time` (returned) is the raw detection point (find_stall_time's
    last-point-of-progress, or the collision/object_drop/pose_error time)
    -- kept for failure-type record-keeping and characterization.
    `intervention_time` is where correction is actually attempted:
    max(0, failure_time - intervention_offset). Stage assignment and the
    local segment/window used for failure-mode clustering are both taken
    at intervention_time, since that's the actual state a generated
    candidate starts from; failure_type/failure_time describe what
    eventually went wrong, not necessarily the state being corrected from.
    """
    failure_time, failure_type = detect_failure_time(meta, arrays)
    if failure_time is None:
        return None
    assert failure_type in FAILURE_LABELS

    intervention_time = max(0, failure_time - intervention_offset)
    failure_stage = assign_failure_stage(meta, arrays, intervention_time)
    start_time, end_time, state_window, action_window, obs_window = extract_failure_segment(
        meta, arrays, intervention_time, Hf=Hf
    )
    return dict(
        episode_id=meta["episode_id"],
        failure_time=failure_time,
        intervention_time=intervention_time,
        intervention_offset=intervention_offset,
        failure_type=failure_type,
        failure_stage=failure_stage,
        start_time=start_time,
        end_time=end_time,
        state_window=state_window,
        action_window=action_window,
        obs_window=obs_window,
    )
