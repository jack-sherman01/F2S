"""Tests the core hypothesis behind pivoting F2S's intervention mechanism
from open-loop chunk replacement to closed-loop (MPC-style) correction.

Motivation: every real-success rate measured so far in this project used
a single decision at the intervention point -- perturb the latent once,
decode a full 20-step action chunk, execute it entirely open-loop, check
success/failure only at the end (f2s.candidates.validator.execute_action_chunk).
Note this isn't even "more open-loop than normal" -- the base diffusion
policy itself only replans every `inference_horizon` steps, which every
config in this project leaves at its default of 20 (see
simulation/rollout_utils.py's RolloutDP.__call__: `if self.ac is None or
self.t >= self.inference_horizon`). So the standing hypothesis is that
committing to ONE 20-step decision, at the SAME granularity the base
policy itself uses, is why coverage is so narrow (~2-7% of failure states
correctable at all) and why candidates that succeed once rarely survive
even a small state perturbation (Day-19 validation).

This script tests a genuinely finer-grained alternative: re-plan (generate
M fresh candidates, world-model-rank, execute only the best one) every
`k_exec` real steps, for `k_exec` values below the base policy's own
20-step cadence, versus a k_exec=20 control (which reduces to exactly the
original open-loop mechanism, on the same code path, as an internal
sanity check that closed-loop framing alone doesn't already change
anything when there's no room to re-plan). Same total step budget
(max_total_steps=20) and same states/offset across conditions, so the
only thing that varies is replanning granularity.

Candidate selection at each re-plan step is by world-model score alone
(no real-simulator search over candidates) -- executing only the
single best-ranked candidate per iteration, not all M, both because (a)
that's what a real deployed closed-loop controller would actually do
(no oracle access to try everything in the real world) and (b) it keeps
this initial test cheap. If this shows a real effect, scaling up
(execute top-k per iteration, sweep offsets, more states) is the natural
next step.
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
from f2s.candidates.validator import execute_action_chunk
from f2s.common.io import ensure_fresh_dir, load_all_episode_metadata, save_json
from f2s.failure.extractor import process_episode
from f2s.logging.episode_logger import load_episode
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

OFFSET = 15  # the project's originally frozen default -- also the offset with the weakest
             # (though nonzero) open-loop coverage (3/1136 candidates, see section 12.1),
             # so it's the sharpest test of whether closed-loop replanning helps.
MAX_TOTAL_STEPS = 20  # same total step budget as one open-loop chunk, for a fair comparison
M = 16
K_EXEC_VALUES = [20, 10, 5]  # 20 = control (reduces to the original open-loop mechanism)


def execute_action_prefix(env, action_chunk, k_exec, initial_state_dict):
    """Continue a rollout from `initial_state_dict` (the real current
    state, not necessarily the original intervention state), executing
    only the first `k_exec` steps of `action_chunk` (fewer if the episode
    succeeds/terminates first). Unlike execute_action_chunk, this is
    meant to be called repeatedly in sequence, each time continuing from
    where the previous call left off."""
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


def closed_loop_repair(env, dp_module, world_model, device, initial_state_dict, initial_obs_t, obs_keys,
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
            dp_module, obs_tensors, source_episode_id="closed_loop", failure_mode_id=0,
            M=M, sigma_z=0.5, eta=0.5, seed=seed * 1000 + iterations,
        )
        best, best_score = None, -float("inf")
        for cand in candidates:
            if not cand["valid"]:
                continue
            rank_result = rank_candidate(world_model, x0, cand["action_chunk"], world_model_horizon, device)
            if rank_result["score"] > best_score:
                best_score = rank_result["score"]
                best = cand
        if best is None:
            break
        remaining = max_total_steps - total_steps
        result = execute_action_prefix(env, best["action_chunk"], min(k_exec, remaining), state_dict)
        total_steps += result["steps_taken"]
        success = result["actual_success"]
        state_dict = result["state_dict"]
        obs_t = result["obs"]
        trace.append(dict(iteration=iterations, best_score=float(best_score), steps_taken=result["steps_taken"]))
        iterations += 1
        if result["steps_taken"] < k_exec:
            break  # env terminated (timeout/done) before finishing this segment
    return dict(success=success, total_steps=total_steps, iterations=iterations, trace=trace)


def main():
    config_path = "configs/soe_can_lowdim_baseline.json"
    ckpt_path = os.environ["F2S_CAN_CKPT"]
    wm_dir = "results/can/world_model_h20diag"
    world_model_horizon = int(os.environ.get("F2S_RANK_HORIZON", "5"))

    output_dir = "results/can/closed_loop_repair_diagnostic"
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
        print(f"\n===== k_exec = {k_exec} ({'control, = open-loop' if k_exec == MAX_TOTAL_STEPS else 'closed-loop'}) =====")
        n_success = 0
        per_state = []
        for state_idx, (ep_dir, seg, meta, arrays) in enumerate(segments):
            t_f = max(0, seg["failure_time"] - OFFSET)
            obs_t = {k: arrays[f"obs_{k}"][t_f] for k in meta["obs_keys"]}
            initial_state_dict = dict(states=arrays["states"][t_f])
            result = closed_loop_repair(
                env, dp_module, world_model, device, initial_state_dict, obs_t, meta["obs_keys"],
                k_exec=k_exec, seed=state_idx,
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

    save_json(os.path.join(output_dir, "summary.json"), {
        str(k): v["summary"] for k, v in all_results.items()
    })
    print(f"\n{'='*70}")
    for k_exec in K_EXEC_VALUES:
        s = all_results[k_exec]["summary"]
        print(f"k_exec={k_exec:>2}: {s['n_success']}/{s['n_states']} states recovered ({100*s['success_rate']:.1f}%)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
