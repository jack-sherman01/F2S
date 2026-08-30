"""Does CEM-guided candidate search (f2s.candidates.cem) find working
corrections where pure random perturbation
(scripts/diagnose_candidate_success_rate.py, 0/320 across the same kind
of failure states) did not? Executes the top-K CEM candidates per failure
state in the real simulator and reports actual success.
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
from f2s.candidates.validator import execute_action_chunk
from f2s.common.io import load_all_episode_metadata, load_json
from f2s.failure.extractor import process_episode
from f2s.logging.episode_logger import load_episode
from f2s.world_model.model import WorldModelEnsemble
from f2s.world_model.state import STATE_DIM, build_world_model_state


def main():
    episode_dir = "results/can/f2s_final/seed_0/round_0/eval/episodes"
    config_path = "configs/soe_can_lowdim_baseline.json"
    ckpt_path = os.environ["F2S_TEST_CKPT"]
    world_model_dir = os.environ.get("F2S_WM_DIR", "results/can/f2s_final/seed_0/round_0/world_model")
    n_states = int(os.environ.get("F2S_DIAG_N_STATES", "5"))
    population_size = int(os.environ.get("F2S_CEM_POP", "64"))
    n_iters = int(os.environ.get("F2S_CEM_ITERS", "5"))
    top_k_execute = int(os.environ.get("F2S_CEM_TOPK", "8"))

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

    with open(os.path.join(world_model_dir, "result.json"), "r") as f:
        wm_result = json.load(f)
    world_model = WorldModelEnsemble(
        state_dim=wm_result["state_dim"], action_dim=wm_result["action_dim"],
        hidden_dim=wm_result["hidden_dim"], ensemble_size=wm_result["ensemble_size"],
    ).to(device)
    world_model.load_state_dict(torch.load(os.path.join(world_model_dir, "best_model.pt"), map_location=device))
    world_model.eval()
    print(f"Loaded world model from {world_model_dir} (val MSE {wm_result['best_val_mse']:.6f}, "
          f"beats constant baseline: {wm_result['beats_constant_baseline']})")

    metas = load_all_episode_metadata(episode_dir)
    segments = []
    for meta in metas:
        if meta["success"]:
            continue
        _, arrays = load_episode(episode_dir, meta["episode_id"])
        seg = process_episode(meta, arrays, Hf=10)
        if seg is not None:
            segments.append((seg, meta))

    print(f"{len(segments)} failure segments available; testing {min(n_states, len(segments))} "
          f"with CEM (population={population_size}, iters={n_iters}, executing top {top_k_execute} each).")

    total_executed = 0
    total_successes = 0

    for seg, meta in segments[:n_states]:
        _, arrays = load_episode(episode_dir, seg["episode_id"])
        t_f = seg["failure_time"]
        obs_t = {k: arrays[f"obs_{k}"][t_f] for k in meta["obs_keys"]}
        obs_tensors = {k: torch.from_numpy(np.asarray(v)).float().unsqueeze(0).to(device) for k, v in obs_t.items()}
        x0 = build_world_model_state(obs_t)

        candidates = cem_search(
            dp_module, world_model, obs_tensors, x0,
            source_episode_id=seg["episode_id"], failure_mode_id=0, device=device,
            population_size=population_size, n_iters=n_iters,
        )
        print(f"  episode={seg['episode_id']} t_f={t_f}: best predicted_dist_to_goal="
              f"{candidates[0]['predicted_dist_to_goal']:.4f}, "
              f"worst-of-top-{top_k_execute}={candidates[top_k_execute - 1]['predicted_dist_to_goal']:.4f} "
              f"(iter {candidates[0]['cem_iteration']})")

        initial_state_dict = dict(states=arrays["states"][t_f])
        n_success = 0
        for cand in candidates[:top_k_execute]:
            if not cand["valid"]:
                continue
            result = execute_action_chunk(env, cand["action_chunk"], initial_state_dict=initial_state_dict)
            total_executed += 1
            if result["actual_success"]:
                n_success += 1
                total_successes += 1
        print(f"    -> {n_success}/{top_k_execute} executed CEM candidates succeeded")

    print(f"\nTOTAL: {total_successes}/{total_executed} CEM candidates succeeded "
          f"({100 * total_successes / max(total_executed, 1):.1f}%) across "
          f"{min(n_states, len(segments))} failure states.")


if __name__ == "__main__":
    main()
