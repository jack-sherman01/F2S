"""Deterministic safety filtering (proposal Section 10.1 / Day 17) over a
*predicted* world-model rollout, in the same x_t = [q, qdot, p_ee, p_obj,
p_goal, g, r] state layout as f2s.world_model.state (indices below).
"""
from typing import List, Optional, Tuple

import numpy as np

from f2s.failure.extractor import DROP_HEIGHT_MARGIN, LIFT_HEIGHT_MARGIN, TABLE_HEIGHT
from f2s.world_model.state import STATE_DIM

# slices into the 26-dim x_t vector (see f2s/world_model/state.py)
Q_SLICE = slice(0, 7)
QDOT_SLICE = slice(7, 14)
P_EE_SLICE = slice(14, 17)
P_OBJ_SLICE = slice(17, 20)
P_GOAL_SLICE = slice(20, 23)
G_SLICE = slice(23, 25)

# Franka Emika Panda joint limits (radians), from the manufacturer's
# published specification -- used because the low-dim RoboMimic dataset
# does not log simulator joint-limit metadata per timestep.
PANDA_JOINT_LOWER = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
PANDA_JOINT_UPPER = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
# Conservative single-scalar joint-speed limit (Panda's slowest-rated
# joint is ~2.175 rad/s; using that as a uniform bound across all 7).
JOINT_VEL_MAX = 2.175
# A sudden between-step joint-velocity jump above this is treated as a
# collision proxy, matching f2s.failure.extractor's heuristic exactly so
# "collision" means the same thing whether measured on a real or a
# predicted trajectory.
JOINT_VEL_COLLISION_JUMP = 6.0


def collision_detected(predicted_states: np.ndarray) -> bool:
    qdot = predicted_states[:, QDOT_SLICE]
    if qdot.shape[0] < 2:
        return False
    jumps = np.linalg.norm(np.diff(qdot, axis=0), axis=-1)
    return bool(np.any(jumps > JOINT_VEL_COLLISION_JUMP))


def joint_limit_exceeded(predicted_states: np.ndarray) -> bool:
    q = predicted_states[:, Q_SLICE]
    return bool(np.any(q < PANDA_JOINT_LOWER) or np.any(q > PANDA_JOINT_UPPER))


def velocity_limit_exceeded(predicted_states: np.ndarray) -> bool:
    qdot = predicted_states[:, QDOT_SLICE]
    return bool(np.any(np.abs(qdot) > JOINT_VEL_MAX))


def object_drop_predicted(predicted_states: np.ndarray) -> bool:
    obj_z = predicted_states[:, P_OBJ_SLICE][:, 2]
    lifted = obj_z > (TABLE_HEIGHT + LIFT_HEIGHT_MARGIN)
    if not np.any(lifted):
        return False
    first_lift = int(np.argmax(lifted))
    after = obj_z[first_lift:]
    return bool(np.any(after < (TABLE_HEIGHT + DROP_HEIGHT_MARGIN)))


def invalid_action(action_sequence: np.ndarray) -> bool:
    return bool(not np.isfinite(action_sequence).all() or np.any(np.abs(action_sequence) > 1.0 + 1e-3))


def safety_filter(
    predicted_states: np.ndarray,
    action_sequence: np.ndarray,
    max_uncertainty: Optional[float] = None,
    uncertainty: Optional[float] = None,
) -> Tuple[bool, List[str]]:
    """Returns (is_safe, reasons). predicted_states: (H, 26) x_t rollout
    from the world model. action_sequence: (H, action_dim) proposed
    actions. Rejects a candidate if ANY hard constraint is violated
    (proposal Day 17.1)."""
    reasons = []

    if invalid_action(action_sequence):
        reasons.append("invalid_action")
    if collision_detected(predicted_states):
        reasons.append("collision")
    if joint_limit_exceeded(predicted_states):
        reasons.append("joint_limit")
    if velocity_limit_exceeded(predicted_states):
        reasons.append("velocity_limit")
    if object_drop_predicted(predicted_states):
        reasons.append("object_drop")
    if uncertainty is not None and max_uncertainty is not None and uncertainty > max_uncertainty:
        reasons.append("uncertainty")

    return len(reasons) == 0, reasons
