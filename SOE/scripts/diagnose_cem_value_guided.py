"""Tests whether *directed* candidate generation (CEM search in latent
space) does better than plain random perturbation once the fitness
signal steering the search is the learned value function V, instead of
raw world-model predicted distance-to-goal.

Context: `f2s.candidates.cem.cem_search` (world-model-distance fitness)
was already tried and made things *worse* than random generation (0/1278
real successes across a 6-offset sweep -- see README_F2S.md's "Re-opened
the search" and "Day 24" sections) -- diagnosed as CEM climbing exactly
the signal shown to have weak within-state correlation with real outcome
(Spearman 0.09-0.30). The closed-loop + value-function experiment (see
README's "Stepping back" section) then showed that swapping *only* the
candidate-scoring signal (world model vs. V) inside an otherwise-random-
candidate pipeline made no difference either -- both single-shot open-
loop numbers landed in the same range (3/71 vs 2/71).

This script isolates the remaining variable: candidate *generation*
itself. Two CEM variants, identical search mechanism, only the fitness
differs:
  (a) f2s.candidates.cem.cem_search           -- world-model distance-to-goal
  (b) f2s.candidates.cem.cem_search_value_guided -- learned V at the
      world-model-predicted final state

Both run single-shot open-loop (execute the single best candidate's full
20-step chunk once, no replanning -- k_exec=20 equivalent) on the same 71
real failure states, same offset=15, so the result is directly comparable
to the already-established baselines:
  - random generation + world-model top-1 score: 3/71 (4.2%)
  - random generation + value-guided top-1 score: 2/71 (2.8%)
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

from f2s.candidates.cem import cem_search, cem_search_value_guided
from f2s.common.io import ensure_fresh_dir, load_all_episode_metadata, save_json
from f2s.failure.extractor import process_episode
from f2s.logging.episode_logger import load_episode
from f2s.value.model import ValueFunction
from f2s.world_model.model import WorldModelEnsemble
from f2s.world_model.state import build_world_model_state

ALL_EPISODE_DIRS = [
    "results/can/f2s_final/seed_0/round_0/eval/episodes",
    "results/can/f2s_final/seed_0/round_1/eval/episodes",
    "results/can/f2s_final/seed_0/round_2/eval/episodes",
    "results/can/f2s_dev/seed_0/round_0/eval/episodes",
    "results/can/f2s_dev/seed_0/round_1/eval/episodes",
    "results/can/f2s_dev/seed_0/round_2/eval/episodes",
    "results/can/f2s_dev_cem/seed_2/round_0/eval/episodes",
    "results/can/f2s_dev_cem/seed_2/round_1/eval/episodes",
    "results/can/f2s_dev_cem/seed_2/round_2/eval/episodes",
    "results/Can/fixed_policy/seed_0/round_0/episodes",
]

OFFSET = 15


def execute_full_chunk(env, action_chunk, initial_state_dict):
    obs = env.reset_to(initial_state_dict)
    success = False
    for t in range(action_chunk.shape[0]):
        obs, reward, done, _ = env.step(action_chunk[t])
        success = bool(env.is_success()["task"])
        if success or done:
            break
    return success


def main():
    config_path = "configs/soe_can_lowdim_baseline.json"
    ckpt_path = os.environ["F2S_CAN_CKPT"]
    wm_dir = "results/can/world_model_h20diag"
    value_dir = "results/can/value_function"
    world_model_horizon = int(os.environ.get("F2S_RANK_HORIZON", "5"))

    output_dir = "results/can/cem_value_guided_diagnostic"
    ensure_fresh_dir(output_dir)

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

    with open(os.path.join(value_dir, "result.json"), "r") as f:
        value_result = json.load(f)
    value_fn = ValueFunction(state_dim=value_result["state_dim"], hidden_dim=value_result["hidden_dim"]).to(device)
    value_fn.load_state_dict(torch.load(os.path.join(value_dir, "best_model.pt"), map_location=device))
    value_fn.eval()
    value_stats = np.load(os.path.join(value_dir, "normalization_stats.npz"))
    value_mu, value_sigma = value_stats["mu"], value_stats["sigma"]

    segments = []
    for ep_dir in ALL_EPISODE_DIRS:
        if not os.path.isdir(ep_dir):
            continue
        metas = load_all_episode_metadata(ep_dir)
        for meta in metas:
            if meta["success"]:
                continue
            _, arrays = load_episode(ep_dir, meta["episode_id"])
            seg = process_episode(meta, arrays, Hf=10)
            if seg is not None:
                segments.append((ep_dir, seg, meta, arrays))
    print(f"Pooled {len(segments)} real failure segments (must be 71, matching prior runs).")

    all_results = {}
    for variant in ["cem_world_model", "cem_value_guided"]:
        print(f"\n===== {variant} (single-shot open-loop, top-1) =====")
        n_success = 0
        per_state = []
        for state_idx, (ep_dir, seg, meta, arrays) in enumerate(segments):
            t_f = max(0, seg["failure_time"] - OFFSET)
            obs_t = {k: arrays[f"obs_{k}"][t_f] for k in meta["obs_keys"]}
            initial_state_dict = dict(states=arrays["states"][t_f])
            obs_tensors = {k: torch.from_numpy(np.asarray(obs_t[k])).float().unsqueeze(0).to(device)
                           for k in meta["obs_keys"]}
            x0 = build_world_model_state(obs_t)

            if variant == "cem_world_model":
                candidates = cem_search(
                    dp_module, world_model, obs_tensors, x0, source_episode_id=seg["episode_id"],
                    failure_mode_id=0, device=device, horizon_wm=world_model_horizon, seed=state_idx,
                )
            else:
                candidates = cem_search_value_guided(
                    dp_module, world_model, value_fn, value_mu, value_sigma, obs_tensors, x0,
                    source_episode_id=seg["episode_id"], failure_mode_id=0, device=device,
                    horizon_wm=world_model_horizon, seed=state_idx,
                )
            valid_candidates = [c for c in candidates if c["valid"]]
            best = valid_candidates[0] if valid_candidates else None

            success = False
            if best is not None:
                success = execute_full_chunk(env, best["action_chunk"], initial_state_dict)
            n_success += int(success)
            per_state.append(dict(
                state_idx=state_idx, episode_id=seg["episode_id"], success=success,
                best_cem_fitness=(float(best["cem_fitness"]) if best is not None else None),
            ))
            if (state_idx + 1) % 20 == 0 or state_idx == len(segments) - 1:
                print(f"  [{state_idx + 1}/{len(segments)}] processed, {n_success} real successes so far")
        summary = dict(variant=variant, n_states=len(segments), n_success=n_success,
                        success_rate=n_success / len(segments))
        print(f"{variant}: {n_success}/{len(segments)} states recovered ({100 * n_success / len(segments):.1f}%)")
        all_results[variant] = dict(summary=summary, per_state=per_state)
        save_json(os.path.join(output_dir, f"{variant}.json"), all_results[variant])

    save_json(os.path.join(output_dir, "summary.json"), {k: v["summary"] for k, v in all_results.items()})
    print(f"\n{'='*70}")
    for variant in all_results:
        s = all_results[variant]["summary"]
        print(f"{variant}: {s['n_success']}/{s['n_states']} states recovered ({100*s['success_rate']:.1f}%)")
    print("(reference, already established) random generation + world-model top-1: 3/71 (4.2%)")
    print("(reference, already established) random generation + value-guided top-1: 2/71 (2.8%)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
