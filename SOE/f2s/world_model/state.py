"""Construction of the low-dimensional world-model state vector

    x_t = [q_t, qdot_t, p_ee_t, p_obj_t, p_goal_t, g_t, r_t]

(proposal Section 9.1) from the raw obs_<key> arrays SOE/robomimic already
expose for RoboMimic Can, reusing the same object-obs layout documented in
f2s/failure/features.py (obj_pos = object[0:3]).
"""
from typing import Dict, List

import numpy as np

from f2s.failure.features import DEFAULT_GOAL_POS, compute_step_features

STATE_COMPONENT_NAMES = [
    "robot0_joint_pos",  # q_t            (7)
    "robot0_joint_vel",  # qdot_t         (7)
    "robot0_eef_pos",    # p_ee_t         (3)
    "object_pos",        # p_obj_t        (3)  (= object[0:3])
    "goal_pos",          # p_goal_t       (3)  (constant, see features.py)
    "gripper_qpos",      # g_t            (2)
    "task_progress",     # r_t            (1)
]
STATE_DIM = 7 + 7 + 3 + 3 + 3 + 2 + 1  # = 26


def build_world_model_state(obs: Dict[str, np.ndarray], goal_pos: np.ndarray = DEFAULT_GOAL_POS) -> np.ndarray:
    q = np.asarray(obs["robot0_joint_pos"], dtype=np.float64)
    qdot = np.asarray(obs["robot0_joint_vel"], dtype=np.float64)
    p_ee = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
    p_obj = np.asarray(obs["object"], dtype=np.float64)[0:3]
    g = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float64)
    r = compute_step_features(obs, goal_pos=goal_pos)["task_progress"]

    x = np.concatenate([q, qdot, p_ee, p_obj, goal_pos, g, [r]])
    assert x.shape[0] == STATE_DIM, f"expected state dim {STATE_DIM}, got {x.shape[0]}"
    return x.astype(np.float32)


def build_world_model_states_for_episode(obs_keys: List[str], arrays: Dict[str, np.ndarray]) -> np.ndarray:
    """Vectorized version of build_world_model_state over a full episode's
    obs_<key> arrays (as loaded from an EpisodeLogger npz)."""
    ep_len = arrays["actions"].shape[0]
    xs = np.zeros((ep_len, STATE_DIM), dtype=np.float32)
    for t in range(ep_len):
        obs_t = {k: arrays[f"obs_{k}"][t] for k in obs_keys}
        xs[t] = build_world_model_state(obs_t)
    return xs
