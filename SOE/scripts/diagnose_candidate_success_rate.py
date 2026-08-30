"""Ad-hoc diagnostic: of M candidates generated for each failure state, how
many succeed when actually executed in the simulator? Reuses the already-
collected round-0 failure episodes from a completed run_evolution.py run
(no new policy evaluation needed) to see whether the M=16 budget used by
configs/f2s_dev.yaml and f2s_final.yaml is simply too small for the
Gaussian/single-dimension perturbation strategy to ever hit a working
correction, or whether something else is going on.
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

from f2s.candidates.generator import generate_candidates
from f2s.candidates.validator import execute_action_chunk
from f2s.common.io import load_all_episode_metadata, load_json
from f2s.failure.extractor import process_episode
from f2s.logging.episode_logger import load_episode


def main():
    episode_dir = "results/can/f2s_final/seed_0/round_0/eval/episodes"
    config_path = "configs/soe_can_lowdim_baseline.json"
    ckpt_path = os.environ["F2S_TEST_CKPT"]
    M = int(os.environ.get("F2S_DIAG_M", "64"))
    n_states = int(os.environ.get("F2S_DIAG_N_STATES", "5"))

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

    metas = load_all_episode_metadata(episode_dir)
    segments = []
    for meta in metas:
        if meta["success"]:
            continue
        _, arrays = load_episode(episode_dir, meta["episode_id"])
        seg = process_episode(meta, arrays, Hf=10)
        if seg is not None:
            segments.append((seg, meta))

    print(f"{len(segments)} failure segments available; testing {min(n_states, len(segments))} of them with M={M} candidates each.")

    total_candidates = 0
    total_successes = 0
    per_state_success_counts = []

    for seg, meta in segments[:n_states]:
        _, arrays = load_episode(episode_dir, seg["episode_id"])
        t_f = seg["failure_time"]
        obs_t = {k: arrays[f"obs_{k}"][t_f] for k in meta["obs_keys"]}
        obs_tensors = {k: torch.from_numpy(np.asarray(v)).float().unsqueeze(0).to(device) for k, v in obs_t.items()}

        candidates = generate_candidates(
            dp_module, obs_tensors, source_episode_id=seg["episode_id"], failure_mode_id=0,
            M=M, sigma_z=0.5, eta=0.5,
        )
        initial_state_dict = dict(states=arrays["states"][t_f])

        n_success = 0
        for cand in candidates:
            if not cand["valid"]:
                continue
            result = execute_action_chunk(env, cand["action_chunk"], initial_state_dict=initial_state_dict)
            total_candidates += 1
            if result["actual_success"]:
                n_success += 1
                total_successes += 1
        per_state_success_counts.append(n_success)
        print(f"  episode={seg['episode_id']} t_f={t_f}: {n_success}/{M} candidates succeeded")

    print(f"\nTOTAL: {total_successes}/{total_candidates} candidates succeeded "
          f"({100 * total_successes / max(total_candidates, 1):.1f}%) across {len(per_state_success_counts)} failure states.")
    print(f"Per-state success counts: {per_state_success_counts}")


if __name__ == "__main__":
    main()
