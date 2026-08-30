"""Failure feature extraction for RoboMimic `Can` (PickPlaceCan, single
active object) episodes logged by f2s.logging.episode_logger.

We deliberately use structured state features read directly out of the
robosuite/robomimic low-dim observation layout, rather than training a
visual/learned failure classifier (per the proposal's Day 9 instruction).

The `object` observation key for this env is exactly 14-dim, built by
robosuite.environments.manipulation.pick_place.PickPlace._setup_observables
as the concatenation of, for the one active object:
    obj_pos            (3)  -- world-frame object position
    obj_quat            (4)  -- world-frame object orientation
    obj_to_eef_pos      (3)  -- object position relative to the gripper
    obj_to_eef_quat     (4)  -- object orientation relative to the gripper
(verified by reading the installed robosuite source, not assumed).

There is no raw contact/collision sensor in the low-dim obs group, so the
"contact" feature `c` is a documented proxy (gripper closed AND end-effector
within a small distance of the object), not a ground-truth contact flag.
The goal position is not an observation key either (RoboMimic's released
low-dim datasets do not log bin2_pos); we use robosuite's PickPlace default
bin2_pos=(0.1, 0.28, 0.8), which is what the "Can" task dataset actually
used (env_kwargs in the dataset's env_args did not override it).
"""
from typing import Any, Dict

import numpy as np

# robosuite.environments.manipulation.pick_place.PickPlace default bin2_pos
# (the target/goal bin for the single-object "Can" task variant).
DEFAULT_GOAL_POS = np.array([0.1, 0.28, 0.8])

# Gripper qpos magnitude below which the Panda gripper is considered closed.
GRIPPER_CLOSED_THRESHOLD = 0.02
# End-effector-to-object distance below which we consider the gripper to be
# "in contact with" the object (proxy for a real contact sensor).
CONTACT_DISTANCE_THRESHOLD = 0.03

FEATURE_NAMES = [
    "delta_p_obj",    # ||object_pos - goal_pos||
    "delta_p_ee",     # ||object_to_eef_pos||
    "delta_theta_ee", # angular distance implied by object_to_eef_quat
    "v_ee",           # ||robot0_eef_vel_lin||
    "gripper_state",  # mean(robot0_gripper_qpos)
    "contact",        # proxy contact indicator in [0, 1]
    "task_progress",  # 1 - normalized delta_p_obj (clipped to [0, 1])
]


def _quat_to_angle(quat_xyzw: np.ndarray) -> float:
    """Angle (radians) of the rotation represented by a wxyz- or xyzw-quat's
    scalar-free part; robust to which convention robosuite used since we
    only need a monotonic "how far from identity rotation" scalar, not the
    exact angle. quat = [x, y, z, w] (robosuite/mujoco convention here uses
    (x,y,z,w) via T.mat2pose -> returns quaternion in (x,y,z,w))."""
    quat = np.asarray(quat_xyzw, dtype=np.float64)
    quat = quat / (np.linalg.norm(quat) + 1e-8)
    w = np.clip(np.abs(quat[-1]), -1.0, 1.0)
    return 2.0 * np.arccos(w)


def compute_step_features(obs: Dict[str, np.ndarray], goal_pos: np.ndarray = DEFAULT_GOAL_POS) -> Dict[str, float]:
    """Compute the raw (un-normalized) feature dict h_i^f for a single
    timestep's observation dict (as stored by EpisodeLogger, i.e. one row
    per obs_<key> array)."""
    object_obs = np.asarray(obs["object"], dtype=np.float64)
    obj_pos = object_obs[0:3]
    obj_to_eef_pos = object_obs[7:10]
    obj_to_eef_quat = object_obs[10:14]

    eef_vel_lin = np.asarray(obs["robot0_eef_vel_lin"], dtype=np.float64)
    gripper_qpos = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float64)

    delta_p_obj = float(np.linalg.norm(obj_pos - goal_pos))
    delta_p_ee = float(np.linalg.norm(obj_to_eef_pos))
    delta_theta_ee = float(_quat_to_angle(obj_to_eef_quat))
    v_ee = float(np.linalg.norm(eef_vel_lin))
    gripper_state = float(np.mean(gripper_qpos))

    gripper_closed = abs(gripper_state) < GRIPPER_CLOSED_THRESHOLD
    contact = float(gripper_closed and (delta_p_ee < CONTACT_DISTANCE_THRESHOLD))

    # normalize by a nominal 1m workspace scale for a bounded progress proxy
    task_progress = float(np.clip(1.0 - delta_p_obj / 1.0, 0.0, 1.0))

    return dict(
        delta_p_obj=delta_p_obj,
        delta_p_ee=delta_p_ee,
        delta_theta_ee=delta_theta_ee,
        v_ee=v_ee,
        gripper_state=gripper_state,
        contact=contact,
        task_progress=task_progress,
    )


def compute_failure_feature_vector(
    state_window_obs: list, goal_pos: np.ndarray = DEFAULT_GOAL_POS
) -> np.ndarray:
    """Proposal Day 9.1: 'Use the final state in the failure window for
    position and pose errors. Use the mean and maximum values over the
    window for velocity and contact features.'

    `state_window_obs` is a list of per-step obs dicts (as returned by
    f2s.failure.extractor.extract_failure_segment). Returns the 7-dim
    feature vector h_i^f in FEATURE_NAMES order, except v_ee/contact use
    (mean, max) so the returned vector is actually 9-dim:
    [delta_p_obj, delta_p_ee, delta_theta_ee,
     v_ee_mean, v_ee_max, gripper_state, contact_mean, contact_max,
     task_progress] -- see FAILURE_FEATURE_NAMES for the exact order.
    """
    per_step = [compute_step_features(obs, goal_pos=goal_pos) for obs in state_window_obs]
    final = per_step[-1]
    v_ee_vals = np.array([s["v_ee"] for s in per_step])
    contact_vals = np.array([s["contact"] for s in per_step])

    return np.array([
        final["delta_p_obj"],
        final["delta_p_ee"],
        final["delta_theta_ee"],
        float(v_ee_vals.mean()),
        float(v_ee_vals.max()),
        final["gripper_state"],
        float(contact_vals.mean()),
        float(contact_vals.max()),
        final["task_progress"],
    ], dtype=np.float64)


FAILURE_FEATURE_NAMES = [
    "delta_p_obj_final",
    "delta_p_ee_final",
    "delta_theta_ee_final",
    "v_ee_mean",
    "v_ee_max",
    "gripper_state_final",
    "contact_mean",
    "contact_max",
    "task_progress_final",
]


def standardize(features: np.ndarray, mu: np.ndarray, sigma: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return (features - mu) / (sigma + eps)
