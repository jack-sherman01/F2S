"""Execute a candidate's action chunk in the real simulator and record the
actual outcome (proposal Day 14.3 / Day 18 / Day 19). Also implements the
Day 19.1 validation-configuration perturbations (object position, goal/
bin position is fixed by the task so we vary object position + friction +
mass instead, since RoboMimic Can has a single fixed goal bin) used to
check whether a successful candidate is a *reusable* skill rather than a
one-off fit to a single exact state.
"""
from typing import Any, Dict, Optional

import numpy as np


def execute_action_chunk(
    env,
    action_chunk: np.ndarray,
    initial_state_dict: Optional[Dict[str, Any]] = None,
    post_reset_hook=None,
) -> Dict[str, Any]:
    """Reset (to `initial_state_dict` if given, else a fresh random reset)
    and open-loop execute `action_chunk`. Returns actual outcome stats.

    `post_reset_hook`, if given, runs right after the internal `env.reset()`
    (which robosuite may hard-reset, i.e. rebuild sim.model from XML) but
    before `reset_to(initial_state_dict)` (which only restores sim *data*,
    not sim.model) -- this is the correct place to apply a physical-
    parameter perturbation (see perturb_friction/perturb_mass below) so it
    survives the subsequent state restore.
    """
    obs = env.reset()
    if post_reset_hook is not None:
        post_reset_hook(env)
    if initial_state_dict is not None:
        obs = env.reset_to(initial_state_dict)
    else:
        initial_state_dict = env.get_state()

    states = [initial_state_dict["states"]]
    success = False
    for t in range(action_chunk.shape[0]):
        obs, reward, done, _ = env.step(action_chunk[t])
        success = bool(env.is_success()["task"])
        states.append(env.get_state()["states"])
        if success or done:
            break

    return dict(
        actual_success=success,
        actual_length=len(states) - 1,
        states=np.stack(states),
        final_obs=obs,
    )


def _raw_env(env):
    """robomimic's EnvRobosuite wrapper does not proxy attribute access to
    the underlying robosuite env (no __getattr__), so `env.sim`,
    `env.obj_body_id`, `env.objects` etc. must go through `env.env`
    explicitly. Centralized here so every perturbation helper below gets
    it right, instead of each one silently AttributeError-ing the first
    time it's actually reached (which is exactly what happened before
    this was fixed -- see SOE/README_F2S.md)."""
    return env.env


def _active_object_and_body_id(raw_env):
    """The attribute names for "the one manipulable object" are not
    consistent across robosuite manipulation envs: PickPlace (used by
    RoboMimic Can) exposes a list `self.objects` + dict `self.obj_body_id`,
    while Lift exposes a single `self.cube` + scalar `self.cube_body_id`
    (confirmed by reading both installed robosuite sources -- see
    SOE/README_F2S.md's Lift diagnostic for why this came up). Try both
    known patterns rather than assuming PickPlace's.

    PickPlace's `self.objects` is always the full 4-object list
    `[Milk, Bread, Cereal, Can]` regardless of task variant --
    `single_object_mode=2` (used by PickPlaceCan) only decides which one
    is actually placed on the table via `self.object_id` (moving the
    other three off-screen), so the active object is
    `self.objects[self.object_id]`, NOT `self.objects[0]` (confirmed by
    inspecting a live PickPlaceCan env: `object_id == 3`, i.e. "Can",
    while `objects[0]` is "Milk", parked at a constant off-screen qpos --
    see SOE/README_F2S.md for the bug this was and its impact)."""
    if hasattr(raw_env, "objects") and hasattr(raw_env, "obj_body_id"):
        object_id = getattr(raw_env, "object_id", 0)
        obj = raw_env.objects[object_id]
        body_id = raw_env.obj_body_id[obj.name]
        return obj, body_id
    if hasattr(raw_env, "cube") and hasattr(raw_env, "cube_body_id"):
        return raw_env.cube, raw_env.cube_body_id
    raise AttributeError(
        f"{type(raw_env).__name__}: don't know how to locate the active "
        "manipulable object (checked objects/obj_body_id and cube/cube_body_id)"
    )


def perturb_object_pose(env) -> Dict[str, Any]:
    """A fresh env.reset() samples a brand new random object placement
    from robosuite's own placement initializer -- the *entire* task
    distribution, not a small perturbation near the original failure
    state. Useful as a hard generalization probe, but NOT what
    Day 19.1's "object-position perturbation" validation configs should
    use (an action chunk tuned to one specific relative gripper-object
    geometry has essentially no chance of succeeding open-loop from an
    unrelated random state -- seek perturb_object_position_near for
    that). Returns the resulting state_dict."""
    env.reset()
    return env.get_state()


def perturb_object_position_near(env, initial_state_dict: Dict[str, Any], max_offset: float = 0.03) -> Dict[str, Any]:
    """Day 19.1's actual intent: a *small* random offset to the object's
    (x, y) position near the original failure state, testing whether a
    candidate's corrective action chunk tolerates minor placement noise
    -- not a resample from the whole task distribution. Restores
    `initial_state_dict` first, then nudges the active object's free-joint
    position in-place via the same sim.data.set_joint_qpos API robosuite's
    own placement initializer uses internally (see
    robosuite.environments.manipulation.pick_place.PickPlace._reset_internal).
    Returns the resulting (perturbed) state_dict."""
    env.reset()
    env.reset_to(initial_state_dict)
    raw = _raw_env(env)
    obj, _ = _active_object_and_body_id(raw)
    joint_name = obj.joints[0]
    qpos = np.array(raw.sim.data.get_joint_qpos(joint_name))
    qpos[0:2] += np.random.uniform(-max_offset, max_offset, size=2)
    raw.sim.data.set_joint_qpos(joint_name, qpos)
    raw.sim.forward()
    return env.get_state()


def sample_object_position_unseen(
    seen_bbox: Dict[str, float], full_bbox: Dict[str, float], margin: float = 0.02, max_tries: int = 200,
) -> np.ndarray:
    """Day 25's "unseen object position": rejection-sample an (x, y) point
    from `full_bbox` (the physically valid placement region) that falls
    OUTSIDE `seen_bbox` expanded by `margin` on every side. `seen_bbox`
    should be the empirical (x_min, x_max, y_min, y_max) footprint of
    object positions actually seen during policy/world-model training
    (see configs/can_unseen_test.yaml, computed from the RoboMimic Can
    demo dataset's t=0 `object` observations) -- not robosuite's abstract
    placement-sampler bounds, which are wider than what was ever actually
    trained on. Falls back to the nearest point on the seen_bbox+margin
    boundary, clipped into full_bbox, if no rejection sample lands outside
    within `max_tries` (keeps the caller from hanging on a degenerate
    bbox)."""
    lo_x, hi_x = full_bbox["x_min"], full_bbox["x_max"]
    lo_y, hi_y = full_bbox["y_min"], full_bbox["y_max"]
    seen_x_min, seen_x_max = seen_bbox["x_min"] - margin, seen_bbox["x_max"] + margin
    seen_y_min, seen_y_max = seen_bbox["y_min"] - margin, seen_bbox["y_max"] + margin

    for _ in range(max_tries):
        x = np.random.uniform(lo_x, hi_x)
        y = np.random.uniform(lo_y, hi_y)
        if not (seen_x_min <= x <= seen_x_max and seen_y_min <= y <= seen_y_max):
            return np.array([x, y])

    # fallback: push the seen-bbox center outward to the nearest full_bbox edge
    cx, cy = (seen_x_min + seen_x_max) / 2, (seen_y_min + seen_y_max) / 2
    x = lo_x if abs(cx - lo_x) > abs(hi_x - cx) else hi_x
    y = lo_y if abs(cy - lo_y) > abs(hi_y - cy) else hi_y
    return np.array([x, y])


def perturb_object_position_unseen(
    env, seen_bbox: Dict[str, float], full_bbox: Dict[str, float], margin: float = 0.02, max_tries: int = 200,
) -> Dict[str, Any]:
    """Force the active object's (x, y) to a position outside the
    training-seen footprint (see sample_object_position_unseen), for use
    right after a fresh `env.reset()` and before any `env.reset_to(...)`
    (same ordering requirement as perturb_friction/perturb_mass -- see
    execute_action_chunk's docstring). Returns the resulting state_dict."""
    raw = _raw_env(env)
    obj, _ = _active_object_and_body_id(raw)
    joint_name = obj.joints[0]
    qpos = np.array(raw.sim.data.get_joint_qpos(joint_name))
    qpos[0:2] = sample_object_position_unseen(seen_bbox, full_bbox, margin=margin, max_tries=max_tries)
    raw.sim.data.set_joint_qpos(joint_name, qpos)
    raw.sim.forward()
    return env.get_state()


def perturb_friction(env, scale: float = 1.5) -> bool:
    """Scale the active object's geom friction coefficients in-place.
    Returns True on success, False if the object/geom introspection
    failed for any reason (in which case the caller should skip this
    validation config rather than crash the pipeline)."""
    try:
        raw = _raw_env(env)
        _, body_id = _active_object_and_body_id(raw)
        geom_ids = [i for i in range(raw.sim.model.ngeom) if raw.sim.model.geom_bodyid[i] == body_id]
        for gid in geom_ids:
            raw.sim.model.geom_friction[gid] *= scale
        raw.sim.forward()
        return True
    except Exception as e:
        print(f"WARNING: perturb_friction failed ({e}); skipping this validation config")
        return False


def perturb_mass(env, scale: float = 1.5) -> bool:
    try:
        raw = _raw_env(env)
        _, body_id = _active_object_and_body_id(raw)
        raw.sim.model.body_mass[body_id] *= scale
        raw.sim.forward()
        return True
    except Exception as e:
        print(f"WARNING: perturb_mass failed ({e}); skipping this validation config")
        return False


def build_validation_configs(n_object_position: int = 5, n_goal_position: int = 3, n_friction: int = 1, n_mass: int = 1):
    """Proposal Day 19.1: 5 object-position + 3 goal-position + 1 friction
    + 1 mass = 10 validation configs. RoboMimic Can has a single fixed
    goal bin (no goal-position randomization available in this task), so
    the 3 "goal-position" slots are realized as additional independent
    small object-position perturbations instead -- documented
    substitution, still 10 total configs probing generalization beyond
    the exact failure state, just via object-placement + physical-
    parameter variation rather than goal variation (which this task does
    not expose)."""
    configs = (
        ["object_position"] * (n_object_position + n_goal_position)
        + ["friction"] * n_friction
        + ["mass"] * n_mass
    )
    return configs
