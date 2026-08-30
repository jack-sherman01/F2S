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
    obj = raw.objects[0]
    joint_name = obj.joints[0]
    qpos = np.array(raw.sim.data.get_joint_qpos(joint_name))
    qpos[0:2] += np.random.uniform(-max_offset, max_offset, size=2)
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
        obj_name = list(raw.obj_body_id.keys())[0]
        body_id = raw.obj_body_id[obj_name]
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
        obj_name = list(raw.obj_body_id.keys())[0]
        body_id = raw.obj_body_id[obj_name]
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
