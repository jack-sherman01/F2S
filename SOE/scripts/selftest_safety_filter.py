"""Day 17.3 acceptance test: 5 hand-constructed trajectories (safe,
collision, joint-limit, high-velocity, object-drop); the filter must
accept only the safe one, each unsafe one for its own specific reason.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from f2s.safety.filter import PANDA_JOINT_LOWER, PANDA_JOINT_UPPER, TABLE_HEIGHT, safety_filter
from f2s.world_model.state import STATE_DIM

H = 5
ACTION_DIM = 7
JOINT_MIDPOINT = ((PANDA_JOINT_LOWER + PANDA_JOINT_UPPER) / 2).astype(np.float32)


def base_trajectory():
    x = np.zeros((H, STATE_DIM), dtype=np.float32)
    x[:, 0:7] = JOINT_MIDPOINT           # q: well within limits (midpoint of each joint's range)
    x[:, 7:14] = 0.1                     # qdot: small, safe
    x[:, 17:20] = np.array([0.0, 0.0, TABLE_HEIGHT])  # object stays on the table
    actions = np.random.RandomState(0).uniform(-0.5, 0.5, size=(H, ACTION_DIM)).astype(np.float32)
    return x, actions


def main():
    safe_x, safe_a = base_trajectory()
    ok, reasons = safety_filter(safe_x, safe_a)
    assert ok and reasons == [], f"safe trajectory wrongly rejected: {reasons}"

    collision_x, collision_a = base_trajectory()
    collision_x[2, 7:14] = 20.0  # sudden joint-velocity jump between steps 1->2
    ok, reasons = safety_filter(collision_x, collision_a)
    assert not ok and "collision" in reasons, f"collision trajectory not rejected for the right reason: {reasons}"

    joint_x, joint_a = base_trajectory()
    joint_x[:, 0] = PANDA_JOINT_UPPER[0] + 1.0  # joint 0 out of range on every step
    ok, reasons = safety_filter(joint_x, joint_a)
    assert not ok and "joint_limit" in reasons, f"joint-limit trajectory not rejected for the right reason: {reasons}"

    vel_x, vel_a = base_trajectory()
    vel_x[:, 7:14] = 10.0  # far above JOINT_VEL_MAX
    ok, reasons = safety_filter(vel_x, vel_a)
    assert not ok and "velocity_limit" in reasons, f"velocity trajectory not rejected for the right reason: {reasons}"

    drop_x, drop_a = base_trajectory()
    drop_x[0:2, 17:20] = np.array([0.0, 0.0, TABLE_HEIGHT + 0.15])  # lifted
    drop_x[2:, 17:20] = np.array([0.0, 0.0, TABLE_HEIGHT])          # then dropped back to table height
    ok, reasons = safety_filter(drop_x, drop_a)
    assert not ok and "object_drop" in reasons, f"object-drop trajectory not rejected for the right reason: {reasons}"

    print("Day 17 acceptance test PASSED: all 5 hand-constructed trajectories "
          "were classified correctly (1 safe accepted, 4 unsafe rejected each "
          "for its specific reason).")


if __name__ == "__main__":
    main()
