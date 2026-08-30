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
    if obj is not None and len(obj) > 0:
        from f2s.failure.features import DEFAULT_GOAL_POS

        final_delta_p_obj = float(np.linalg.norm(obj[-1, 0:3] - DEFAULT_GOAL_POS))

    if meta.get("failure_type") == "timeout":
        if final_delta_p_obj is not None and final_delta_p_obj < POSE_ERROR_THRESHOLD:
            return ep_len - 1, "pose_error"
        return ep_len - 1, "timeout"

    return ep_len - 1, "unknown"


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


def process_episode(meta: Dict[str, Any], arrays: Dict[str, np.ndarray], Hf: int = 10) -> Dict[str, Any]:
    """Full Day-8 pipeline for one episode: detect failure time/type,
    assign stage, extract the local segment. Returns None for successful
    episodes (nothing to extract)."""
    failure_time, failure_type = detect_failure_time(meta, arrays)
    if failure_time is None:
        return None
    assert failure_type in FAILURE_LABELS

    failure_stage = assign_failure_stage(meta, arrays, failure_time)
    start_time, end_time, state_window, action_window, obs_window = extract_failure_segment(
        meta, arrays, failure_time, Hf=Hf
    )
    return dict(
        episode_id=meta["episode_id"],
        failure_time=failure_time,
        failure_type=failure_type,
        failure_stage=failure_stage,
        start_time=start_time,
        end_time=end_time,
        state_window=state_window,
        action_window=action_window,
        obs_window=obs_window,
    )
