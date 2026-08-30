"""Follow-up to scripts/evaluate_candidate_ranking.py's finding: 17/20 Can
failure states showed ~zero real-outcome variance across 16 genuinely
diverse candidates, suggesting `find_stall_time` (the last point of
measurable progress) may already be too late to correct from.

Tests this directly: same 20 failure states, same M=16 candidates, same
world model, same everything -- except the intervention point is moved
earlier by a fixed offset before the stall time, at two offsets (10 and
20 steps). If within-state outcome variance and ranking quality improve
as the intervention point moves earlier, that confirms the hypothesis
and points at a concrete fix (change where correction is attempted). If
they don't improve, that rules out timing as the explanation and points
elsewhere (e.g. genuine task/reachability limits, independent of when
correction starts).

Controlled comparison: reuses the exact same 20 chosen failure
states/order as the original test (same seed for np.random.RandomState(0)
episode selection) so results are directly comparable state-by-state
against results/can/candidate_ranking_eval/records.json.
"""
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "simulation")))

from easydict import EasyDict

from f2s.candidates.generator import generate_candidates
from f2s.candidates.scorer import rank_candidate
from f2s.candidates.validator import execute_action_chunk
from f2s.common.io import ensure_fresh_dir, load_all_episode_metadata, load_json, save_json
from f2s.failure.extractor import process_episode
from f2s.failure.features import DEFAULT_GOAL_POS
from f2s.logging.episode_logger import load_episode
from f2s.world_model.model import WorldModelEnsemble
from f2s.world_model.state import build_world_model_state


def main():
    config_path = "configs/soe_can_lowdim_baseline.json"
    ckpt_path = os.environ["F2S_CAN_CKPT"]
    wm_dir = "results/can/world_model_h20diag"
    episode_dirs = [
        "results/can/f2s_final/seed_0/round_0/eval/episodes",
        "results/can/f2s_final/seed_0/round_1/eval/episodes",
        "results/can/f2s_final/seed_0/round_2/eval/episodes",
    ]
    n_states = int(os.environ.get("F2S_RANK_N_STATES", "20"))
    M = int(os.environ.get("F2S_RANK_M", "16"))
    world_model_horizon = int(os.environ.get("F2S_RANK_HORIZON", "5"))
    offsets = [int(x) for x in os.environ.get("F2S_EARLY_OFFSETS", "10,20").split(",")]

    output_dir = os.environ.get("F2S_EARLY_OUTPUT_DIR", "results/can/candidate_ranking_eval_early")
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

    # --- identical failure-state selection as evaluate_candidate_ranking.py ---
    segments = []
    for ep_dir in episode_dirs:
        metas = load_all_episode_metadata(ep_dir)
        for meta in metas:
            if meta["success"]:
                continue
            _, arrays = load_episode(ep_dir, meta["episode_id"])
            seg = process_episode(meta, arrays, Hf=10)
            if seg is not None:
                segments.append((ep_dir, seg, meta))
    rng = np.random.RandomState(0)
    chosen = [segments[i] for i in rng.choice(len(segments), size=n_states, replace=False)]

    all_offset_results = {}

    for offset in offsets:
        print(f"\n===== offset = {offset} steps before stall time =====")
        all_records = []
        near_zero_variance_count = 0

        for state_idx, (ep_dir, seg, meta) in enumerate(chosen):
            _, arrays = load_episode(ep_dir, seg["episode_id"])
            t_f = max(0, seg["failure_time"] - offset)
            obs_t = {k: arrays[f"obs_{k}"][t_f] for k in meta["obs_keys"]}
            obs_tensors = {k: torch.from_numpy(np.asarray(v)).float().unsqueeze(0).to(device) for k, v in obs_t.items()}
            x0 = build_world_model_state(obs_t)
            initial_state_dict = dict(states=arrays["states"][t_f])

            candidates = generate_candidates(
                dp_module, obs_tensors, source_episode_id=seg["episode_id"], failure_mode_id=0,
                M=M, sigma_z=0.5, eta=0.5, seed=state_idx,
            )

            state_records = []
            for cand in candidates:
                if not cand["valid"]:
                    continue
                rank_result = rank_candidate(world_model, x0, cand["action_chunk"], world_model_horizon, device)
                predicted_final_error = float(np.linalg.norm(
                    rank_result["predicted_states"][-1, 17:20] - DEFAULT_GOAL_POS
                ))
                exec_result = execute_action_chunk(env, cand["action_chunk"], initial_state_dict=initial_state_dict)
                actual_final_obj = np.asarray(exec_result["final_obs"]["object"])[0:3]
                actual_final_error = float(np.linalg.norm(actual_final_obj - DEFAULT_GOAL_POS))
                record = dict(
                    state_idx=state_idx, episode_id=seg["episode_id"], original_t_f=int(seg["failure_time"]),
                    used_t_f=int(t_f), predicted_final_error=predicted_final_error,
                    actual_success=bool(exec_result["actual_success"]), actual_final_error=actual_final_error,
                )
                state_records.append(record)
                all_records.append(record)

            actual_errs = [r["actual_final_error"] for r in state_records]
            std = float(np.std(actual_errs)) if actual_errs else 0.0
            if std < 0.01:
                near_zero_variance_count += 1
            print(f"[state {state_idx + 1}/{n_states}] {seg['episode_id']} "
                  f"orig_t_f={seg['failure_time']} used_t_f={t_f}: std={std:.4f}")

        save_json(os.path.join(output_dir, f"records_offset{offset}.json"), all_records)

        pred = np.array([r["predicted_final_error"] for r in all_records])
        actual = np.array([r["actual_final_error"] for r in all_records])
        rho, pval = spearmanr(pred, actual)

        within_state_rhos = []
        stds = []
        for state_idx in range(n_states):
            state_records = [r for r in all_records if r["state_idx"] == state_idx]
            p = [r["predicted_final_error"] for r in state_records]
            a = [r["actual_final_error"] for r in state_records]
            stds.append(float(np.std(a)))
            if len(set(p)) < 2 or len(set(a)) < 2:
                continue
            r2, _ = spearmanr(p, a)
            within_state_rhos.append(r2)

        offset_summary = dict(
            offset=offset, n_candidates_total=len(all_records),
            pooled_rho=float(rho), pooled_pval=float(pval),
            within_state_mean_rho=float(np.mean(within_state_rhos)) if within_state_rhos else None,
            within_state_median_rho=float(np.median(within_state_rhos)) if within_state_rhos else None,
            n_states_near_zero_variance=near_zero_variance_count,
            mean_within_state_actual_error_std=float(np.mean(stds)),
            median_within_state_actual_error_std=float(np.median(stds)),
        )
        all_offset_results[offset] = offset_summary
        print(f"\noffset={offset}: pooled_rho={rho:.4f}, within_state_mean_rho={offset_summary['within_state_mean_rho']}, "
              f"near_zero_variance_states={near_zero_variance_count}/{n_states}, "
              f"mean_std={offset_summary['mean_within_state_actual_error_std']:.4f}")

    # --- baseline (offset=0) for direct comparison, loaded from the original committed run ---
    with open("results/can/candidate_ranking_eval/summary.json", "r") as f:
        baseline_summary = json.load(f)
    baseline_row = dict(
        offset=0,
        within_state_mean_rho=baseline_summary["decomposition"]["within_state_mean_rho"],
        n_states_near_zero_variance=baseline_summary["decomposition"]["n_states_near_zero_outcome_variance"],
        mean_within_state_actual_error_std=baseline_summary["decomposition"]["mean_within_state_actual_error_std"],
    )

    print("\n===== SUMMARY: does intervening earlier help? =====")
    print(f"{'offset':>8} {'within_state_rho':>18} {'near_zero_states':>18} {'mean_std':>10}")
    print(f"{0:>8} {baseline_row['within_state_mean_rho']:>18.4f} "
          f"{baseline_row['n_states_near_zero_variance']:>15}/{n_states} {baseline_row['mean_within_state_actual_error_std']:>10.4f}")
    for offset in offsets:
        r = all_offset_results[offset]
        print(f"{offset:>8} {r['within_state_mean_rho']:>18.4f} "
              f"{r['n_states_near_zero_variance']:>15}/{n_states} {r['mean_within_state_actual_error_std']:>10.4f}")

    save_json(os.path.join(output_dir, "summary.json"), dict(
        baseline_offset0=baseline_row, offsets=all_offset_results, n_states=n_states,
    ))


if __name__ == "__main__":
    main()
