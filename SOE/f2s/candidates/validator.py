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


def perturb_object_pose(env) -> Dict[str, Any]:
    """A fresh env.reset() already samples a new random object placement
    from robosuite's own placement initializer -- this *is* an object-
    position perturbation, using the environment's own (real,
    unmodified) domain-randomization rather than hand-editing simulator
    state. Returns the resulting state_dict."""
    env.reset()
    return env.get_state()


def perturb_friction(env, scale: float = 1.5) -> bool:
    """Scale the active object's geom friction coefficients in-place.
    Returns True on success, False if the object/geom introspection
    failed for any reason (in which case the caller should skip this
    validation config rather than crash the pipeline)."""
    try:
        obj_name = list(env.obj_body_id.keys())[0]
        body_id = env.obj_body_id[obj_name]
        geom_ids = [i for i in range(env.sim.model.ngeom) if env.sim.model.geom_bodyid[i] == body_id]
        for gid in geom_ids:
            env.sim.model.geom_friction[gid] *= scale
        env.sim.forward()
        return True
    except Exception as e:
        print(f"WARNING: perturb_friction failed ({e}); skipping this validation config")
        return False


def perturb_mass(env, scale: float = 1.5) -> bool:
    try:
        obj_name = list(env.obj_body_id.keys())[0]
        body_id = env.obj_body_id[obj_name]
        env.sim.model.body_mass[body_id] *= scale
        env.sim.forward()
        return True
    except Exception as e:
        print(f"WARNING: perturb_mass failed ({e}); skipping this validation config")
        return False


def build_validation_configs(n_object_position: int = 5, n_goal_position: int = 3, n_friction: int = 1, n_mass: int = 1):
    """Proposal Day 19.1: 5 object-position + 3 goal-position + 1 friction
    + 1 mass = 10 validation configs. RoboMimic Can has a single fixed
    goal bin (no goal-position randomization available in this task), so
    the 3 "goal-position" slots are realized as additional independent
    object-position resamples instead -- documented substitution, still
    10 total configs probing generalization beyond the exact failure
    state, just via object-placement + physical-parameter variation
    rather than goal variation (which this task does not expose)."""
    configs = (
        ["object_position"] * (n_object_position + n_goal_position)
        + ["friction"] * n_friction
        + ["mass"] * n_mass
    )
    return configs
