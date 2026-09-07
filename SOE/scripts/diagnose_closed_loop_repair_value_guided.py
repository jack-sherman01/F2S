"""Same closed-loop (receding-horizon) repair mechanism as
scripts/diagnose_closed_loop_repair.py, but candidate scoring at each
re-plan step now uses the learned value function
(f2s.value.model.ValueFunction, trained by scripts/train_value_function.py
on real outcomes across the whole project -- see
f2s/value/dataset.py) instead of the short-horizon world-model score.

Concretely: roll the world model forward `world_model_horizon` steps
under each candidate (same as before -- the world model's own short-
horizon dynamics prediction is not in question, only using it *alone* as
the ranking signal was), then evaluate V at the *predicted final state*
and use that success-probability estimate as the candidate's score. This
is model-based value expansion: short-horizon dynamics (locally
trustworthy, per sections 3-4's own accuracy numbers) composed with a
long-horizon value estimate trained directly on whether real episodes
eventually succeeded -- exactly the signal a 5-step world-model rollout
structurally cannot see on its own, and the diagnosed root cause of the
within-state ranking paradox (README_F2S.md's "Why guided search
backfires" section).

Same 71-state pool, same offset=15, same k_exec sweep as the world-model-
scored version, for a direct, apples-to-apples comparison: does replacing
the scoring signal (holding the closed-loop mechanism itself fixed) change
the outcome versus scripts/diagnose_closed_loop_repair.py's results?
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
from f2s.candidates.scorer import rank_candidate
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
MAX_TOTAL_STEPS = 20
M = 16
K_EXEC_VALUES = [20, 10, 5]


def execute_action_prefix(env, action_chunk, k_exec, initial_state_dict):
    import robomimic.utils.obs_utils as ObsUtils

    obs = env.reset_to(initial_state_dict)
    success = False
    steps_taken = 0
    for t in range(min(k_exec, action_chunk.shape[0])):
        obs, reward, done, _ = env.step(action_chunk[t])
        steps_taken += 1
        success = bool(env.is_success()["task"])
        if success or done:
            break
    return dict(
        state_dict=env.get_state(),
        obs=ObsUtils.unprocess_obs_dict(obs),
        actual_success=success,
        steps_taken=steps_taken,
    )


def closed_loop_repair_value_guided(env, dp_module, world_model, value_fn, value_mu, value_sigma, device,
                                     initial_state_dict, initial_obs_t, obs_keys,
                                     k_exec, seed, M=M, max_total_steps=MAX_TOTAL_STEPS, world_model_horizon=5):
    state_dict = initial_state_dict
    obs_t = initial_obs_t
    total_steps = 0
    success = False
    iterations = 0
    trace = []
    while total_steps < max_total_steps and not success:
        obs_tensors = {k: torch.from_numpy(np.asarray(obs_t[k])).float().unsqueeze(0).to(device) for k in obs_keys}
        x0 = build_world_model_state(obs_t)
        candidates = generate_candidates(
            dp_module, obs_tensors, source_episode_id="closed_loop_value", failure_mode_id=0,
            M=M, sigma_z=0.5, eta=0.5, seed=seed * 1000 + iterations,
        )
        best, best_score = None, -float("inf")
        for cand in candidates:
            if not cand["valid"]:
                continue
            rank_result = rank_candidate(world_model, x0, cand["action_chunk"], world_model_horizon, device)
            final_state = rank_result["predicted_states"][-1]
            final_state_norm = (final_state - value_mu) / value_sigma
            with torch.no_grad():
                v_score = value_fn.predict_proba(
                    torch.from_numpy(final_state_norm).float().unsqueeze(0).to(device)
                ).item()
            if v_score > best_score:
                best_score = v_score
                best = cand
        if best is None:
            break
        remaining = max_total_steps - total_steps
        result = execute_action_prefix(env, best["action_chunk"], min(k_exec, remaining), state_dict)
        total_steps += result["steps_taken"]
        success = result["actual_success"]
        state_dict = result["state_dict"]
        obs_t = result["obs"]
        trace.append(dict(iteration=iterations, best_value_score=float(best_score), steps_taken=result["steps_taken"]))
        iterations += 1
        if result["steps_taken"] < k_exec:
            break
    return dict(success=success, total_steps=total_steps, iterations=iterations, trace=trace)


def main():
    config_path = "configs/soe_can_lowdim_baseline.json"
    ckpt_path = os.environ["F2S_CAN_CKPT"]
    wm_dir = "results/can/world_model_h20diag"
    value_dir = "results/can/value_function"
    world_model_horizon = int(os.environ.get("F2S_RANK_HORIZON", "5"))

    output_dir = "results/can/closed_loop_repair_value_guided_diagnostic"
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
    for k_exec in K_EXEC_VALUES:
        print(f"\n===== k_exec = {k_exec} (value-guided) =====")
        n_success = 0
        per_state = []
        for state_idx, (ep_dir, seg, meta, arrays) in enumerate(segments):
            t_f = max(0, seg["failure_time"] - OFFSET)
            obs_t = {k: arrays[f"obs_{k}"][t_f] for k in meta["obs_keys"]}
            initial_state_dict = dict(states=arrays["states"][t_f])
            result = closed_loop_repair_value_guided(
                env, dp_module, world_model, value_fn, value_mu, value_sigma, device,
                initial_state_dict, obs_t, meta["obs_keys"], k_exec=k_exec, seed=state_idx,
            )
            n_success += int(result["success"])
            per_state.append(dict(
                state_idx=state_idx, episode_id=seg["episode_id"], success=result["success"],
                total_steps=result["total_steps"], iterations=result["iterations"],
            ))
            if (state_idx + 1) % 20 == 0 or state_idx == len(segments) - 1:
                print(f"  [{state_idx + 1}/{len(segments)}] processed, {n_success} real successes so far")
        summary = dict(k_exec=k_exec, n_states=len(segments), n_success=n_success,
                        success_rate=n_success / len(segments))
        print(f"k_exec={k_exec}: {n_success}/{len(segments)} states recovered "
              f"({100 * n_success / len(segments):.1f}%)")
        all_results[k_exec] = dict(summary=summary, per_state=per_state)
        save_json(os.path.join(output_dir, f"k_exec_{k_exec}.json"), all_results[k_exec])

    save_json(os.path.join(output_dir, "summary.json"), {str(k): v["summary"] for k, v in all_results.items()})
    print(f"\n{'='*70}")
    for k_exec in K_EXEC_VALUES:
        s = all_results[k_exec]["summary"]
        print(f"k_exec={k_exec:>2}: {s['n_success']}/{s['n_states']} states recovered ({100*s['success_rate']:.1f}%)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
