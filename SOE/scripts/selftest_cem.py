"""Acceptance test for f2s.candidates.cem.cem_search: shape/contract
checks against the real (trained) policy and a real world model, plus a
monotonicity sanity check -- later CEM iterations should not be worse, on
average, than the first (purely random) iteration, since elites only ever
get kept or improved on.
"""
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "simulation")))

from easydict import EasyDict

from f2s.candidates.cem import cem_search
from f2s.common.io import load_all_episode_metadata, load_json
from f2s.failure.extractor import process_episode
from f2s.logging.episode_logger import load_episode
from f2s.world_model.model import WorldModelEnsemble
from f2s.world_model.state import build_world_model_state


def main():
    episode_dir = "results/can/f2s_final/seed_0/round_0/eval/episodes"
    config_path = "configs/soe_can_lowdim_baseline.json"
    ckpt_path = os.environ["F2S_TEST_CKPT"]
    wm_dir = "results/can/f2s_final/seed_0/round_0/world_model"

    from rollout_utils import dp_load

    with open(config_path, "r") as f:
        cfg = EasyDict(json.load(f))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = SimpleNamespace(
        agent=ckpt_path, critic_agent=None, config=config_path, n_rollouts=1, horizon=None, env=None,
        render=False, render_traj=False, video_dir=None, video_skip=1, camera_names=["agentview"],
        dataset_path=None, dataset_obs=False, seed=0, try_times=1, inference_horizon=None,
        high_noise_eval=False, eta=None, num_inference_steps=None, enable_exploration=False,
        tau1=None, tau2=None, noise_scale=None, enable_exploration_debug=False, disable_styles=False,
        enable_action_noise=False, action_noise_scale=None, enable_cfg=False, cfg_scale=0.5,
        cfg_agent=None, cfg_config=None, abs_action=False, return_intermediate=False,
    )
    rollout_policy, env, _, _ = dp_load(args, cfg, enable_exploration_as_args=False)
    dp_module = rollout_policy.policy

    with open(os.path.join(wm_dir, "result.json"), "r") as f:
        wm_result = json.load(f)
    world_model = WorldModelEnsemble(
        state_dim=wm_result["state_dim"], action_dim=wm_result["action_dim"],
        hidden_dim=wm_result["hidden_dim"], ensemble_size=wm_result["ensemble_size"],
    ).to(device)
    world_model.load_state_dict(torch.load(os.path.join(wm_dir, "best_model.pt"), map_location=device))
    world_model.eval()

    metas = load_all_episode_metadata(episode_dir)
    seg, meta = None, None
    for m in metas:
        if m["success"]:
            continue
        _, arrays = load_episode(episode_dir, m["episode_id"])
        s = process_episode(m, arrays, Hf=10)
        if s is not None:
            seg, meta = s, m
            break
    assert seg is not None, "no failure segment found to test against"

    _, arrays = load_episode(episode_dir, seg["episode_id"])
    t_f = seg["failure_time"]
    obs_t = {k: arrays[f"obs_{k}"][t_f] for k in meta["obs_keys"]}
    obs_tensors = {k: torch.from_numpy(np.asarray(v)).float().unsqueeze(0).to(device) for k, v in obs_t.items()}
    x0 = build_world_model_state(obs_t)

    population_size, n_iters = 32, 4
    candidates = cem_search(
        dp_module, world_model, obs_tensors, x0,
        source_episode_id=seg["episode_id"], failure_mode_id=0, device=device,
        population_size=population_size, n_iters=n_iters, seed=0,
    )

    assert len(candidates) == population_size * n_iters, \
        f"expected {population_size * n_iters} candidates, got {len(candidates)}"
    for c in candidates:
        assert c["action_chunk"].shape == (dp_module.action_decoder.horizon, cfg.policy.params.action_dim)
        assert c["latent_delta"].shape[-1] == 59  # obs_feature_dim for this low-dim config
        assert np.isfinite(c["predicted_dist_to_goal"])
        assert c["predicted_risk"] >= 0.0
    print(f"Shape/contract checks passed for {len(candidates)} candidates.")

    # candidates are returned sorted best-first by CEM fitness
    fitnesses = [c["cem_fitness"] for c in candidates]
    assert fitnesses == sorted(fitnesses), "cem_search must return candidates sorted best-first"
    print("Sort-order check passed.")

    # later iterations should, on average, not be worse than the first
    # (purely random) iteration -- that's the entire point of CEM.
    iter0_mean = np.mean([c["predicted_dist_to_goal"] for c in candidates if c["cem_iteration"] == 0])
    iter_last_mean = np.mean([c["predicted_dist_to_goal"] for c in candidates if c["cem_iteration"] == n_iters - 1])
    print(f"mean predicted_dist_to_goal: iter 0 = {iter0_mean:.4f}, iter {n_iters - 1} = {iter_last_mean:.4f}")
    assert iter_last_mean <= iter0_mean, (
        f"CEM should not get worse on average: iter0={iter0_mean:.4f} iter_last={iter_last_mean:.4f}"
    )
    print("Monotonic-improvement check passed.")

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
