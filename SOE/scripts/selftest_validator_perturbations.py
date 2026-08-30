"""Targeted regression test for the env.obj_body_id / env.sim
AttributeError bug in f2s.candidates.validator (robomimic's EnvRobosuite
wrapper does not proxy attribute access to the underlying robosuite env)
and for perturb_object_position_near actually producing a *small*, valid
offset rather than a full task resample. Exercises the real simulator.
"""
import json
import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "simulation")))

from easydict import EasyDict

from f2s.candidates.validator import (
    build_validation_configs,
    execute_action_chunk,
    perturb_friction,
    perturb_mass,
    perturb_object_position_near,
)


def main():
    from rollout_utils import dp_load

    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "configs", "soe_can_lowdim_baseline.json"))
    with open(config_path, "r") as f:
        cfg = EasyDict(json.load(f))

    ckpt_path = os.environ.get("F2S_TEST_CKPT")
    assert ckpt_path is not None, "set F2S_TEST_CKPT to any trained SOE checkpoint (only used to instantiate an env)"
    args = SimpleNamespace(
        agent=ckpt_path, critic_agent=None, config=config_path, n_rollouts=1, horizon=None, env=None,
        render=False, render_traj=False, video_dir=None, video_skip=1, camera_names=["agentview"],
        dataset_path=None, dataset_obs=False, seed=0, try_times=1, inference_horizon=None,
        high_noise_eval=False, eta=None, num_inference_steps=None, enable_exploration=False,
        tau1=None, tau2=None, noise_scale=None, enable_exploration_debug=False, disable_styles=False,
        enable_action_noise=False, action_noise_scale=None, enable_cfg=False, cfg_scale=0.5,
        cfg_agent=None, cfg_config=None, abs_action=False, return_intermediate=False,
    )
    # We only need `env` here, not a trained policy -- dp_load builds both,
    # so just use its freshly-initialized (untrained) DP weights.
    policy, env, _, _ = dp_load(args, cfg, enable_exploration_as_args=False)

    env.reset()
    original_state = env.get_state()
    original_obj_pos = np.array(env.env.sim.data.get_joint_qpos(env.env.objects[0].joints[0])[0:2])

    # perturb_friction / perturb_mass must not raise (regression test for
    # the env.obj_body_id -> env.env.obj_body_id fix)
    ok_friction = perturb_friction(env)
    assert ok_friction, "perturb_friction failed -- the env.env fix did not work"
    ok_mass = perturb_mass(env)
    assert ok_mass, "perturb_mass failed -- the env.env fix did not work"
    print("perturb_friction / perturb_mass: no AttributeError, both applied successfully.")

    # perturb_object_position_near must produce a *small* offset near the
    # original state, not a full resample.
    perturbed_state = perturb_object_position_near(env, original_state, max_offset=0.03)
    env.reset_to(perturbed_state)
    perturbed_obj_pos = np.array(env.env.sim.data.get_joint_qpos(env.env.objects[0].joints[0])[0:2])
    offset = np.linalg.norm(perturbed_obj_pos - original_obj_pos)
    assert offset <= 0.03 * np.sqrt(2) + 1e-6, f"offset {offset} exceeds max_offset bound"
    print(f"perturb_object_position_near: offset={offset:.4f}m (bounded by max_offset*sqrt(2)={0.03 * 1.4142:.4f}m) -- PASSED")

    # execute_action_chunk with a friction post_reset_hook must not crash
    # end to end (this is the actual code path run_evolution.py exercises).
    dummy_action_chunk = np.zeros((5, 7), dtype=np.float32)
    result = execute_action_chunk(env, dummy_action_chunk, initial_state_dict=original_state, post_reset_hook=perturb_friction)
    assert "actual_success" in result
    print("execute_action_chunk with post_reset_hook=perturb_friction: ran to completion.")

    configs = build_validation_configs()
    assert len(configs) == 10
    print(f"build_validation_configs(): {configs}")

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
