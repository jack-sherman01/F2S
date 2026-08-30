"""Day 14 acceptance test (Final Acceptance Criteria item 6): is
world-model candidate ranking better than random ranking?

Distinct from every "does the world model beat the constant-state
baseline" check already done (Day 12) -- this one asks whether the
world model's *relative ordering* of candidates for a given failure
state agrees with which candidates actually turn out better when
executed for real. Uses continuous final object-to-goal error (not
binary success, which has near-zero variance given the negative results
already found on Can -- Day 14.2 asks for both, and error is the
informative one here) as the outcome measure, on Can, the proposal's
primary task.

Procedure (matches Day 14.1-14.3 exactly):
  1. Select >=20 real Can failure states.
  2. For each, generate M=16 latent-space candidates (plain Gaussian/
     single-dimension perturbation, not CEM -- Day 14.1 specifies 16
     perturbations of the base kind, and this test is about the world
     model's ranking quality in isolation, independent of search
     strategy).
  3. Predict each candidate's outcome with the world model.
  4. Execute *all* 16 in the real simulator, recording actual outcomes.
  5. Compute Spearman rho between predicted and actual final error.
  6. Compare world-model-guided top-B selection against random top-B
     selection, averaged over many random draws, on actual outcome --
     the decision-relevant form of "beats random ranking."
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
    n_random_draws = 200
    top_b = 2  # matches the pipeline's default num_executed_candidates_per_failure_mode

    output_dir = "results/can/candidate_ranking_eval"
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
    print(f"World model: val_mse={wm_result['best_val_mse']:.6f} (properly episode-split, see prior commits)")

    # --- Day 14.1: select >= 20 real failure states, pooled across rounds ---
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
    print(f"{len(segments)} failure segments available across {len(episode_dirs)} rounds; using {n_states}.")
    assert len(segments) >= n_states, f"need >= {n_states} failure states, found {len(segments)}"

    rng = np.random.RandomState(0)
    chosen = [segments[i] for i in rng.choice(len(segments), size=n_states, replace=False)]

    all_records = []
    per_state_summaries = []

    for state_idx, (ep_dir, seg, meta) in enumerate(chosen):
        _, arrays = load_episode(ep_dir, seg["episode_id"])
        t_f = seg["failure_time"]
        obs_t = {k: arrays[f"obs_{k}"][t_f] for k in meta["obs_keys"]}
        obs_tensors = {k: torch.from_numpy(np.asarray(v)).float().unsqueeze(0).to(device) for k, v in obs_t.items()}
        x0 = build_world_model_state(obs_t)
        initial_state_dict = dict(states=arrays["states"][t_f])

        # Day 14.1: 16 candidates (plain perturbation, not CEM)
        candidates = generate_candidates(
            dp_module, obs_tensors, source_episode_id=seg["episode_id"], failure_mode_id=0,
            M=M, sigma_z=0.5, eta=0.5, seed=state_idx,
        )

        state_records = []
        for cand in candidates:
            if not cand["valid"]:
                continue
            # Day 14.2: predict
            rank_result = rank_candidate(world_model, x0, cand["action_chunk"], world_model_horizon, device)
            predicted_final_error = float(np.linalg.norm(
                rank_result["predicted_states"][-1, 17:20] - DEFAULT_GOAL_POS
            ))

            # Day 14.3: execute for real
            exec_result = execute_action_chunk(env, cand["action_chunk"], initial_state_dict=initial_state_dict)
            actual_final_obj = np.asarray(exec_result["final_obs"]["object"])[0:3]
            actual_final_error = float(np.linalg.norm(actual_final_obj - DEFAULT_GOAL_POS))

            record = dict(
                state_idx=state_idx, episode_id=seg["episode_id"], candidate_id=cand["candidate_id"],
                predicted_success=rank_result["predicted_success"], predicted_risk=rank_result["predicted_risk"],
                predicted_score=rank_result["score"], predicted_final_error=predicted_final_error,
                actual_success=bool(exec_result["actual_success"]), actual_final_error=actual_final_error,
            )
            state_records.append(record)
            all_records.append(record)

        n_valid = len(state_records)
        n_actual_success = sum(1 for r in state_records if r["actual_success"])
        print(f"[state {state_idx + 1}/{n_states}] {seg['episode_id']} t_f={t_f}: "
              f"{n_valid} valid candidates, {n_actual_success} actually succeeded, "
              f"mean actual_final_error={np.mean([r['actual_final_error'] for r in state_records]):.4f}")
        per_state_summaries.append(dict(
            state_idx=state_idx, episode_id=seg["episode_id"], n_candidates=n_valid,
            n_actual_success=n_actual_success,
        ))

    save_json(os.path.join(output_dir, "records.json"), all_records)

    # --- Spearman correlation: predicted vs. actual final error (pooled) ---
    predicted_errors = np.array([r["predicted_final_error"] for r in all_records])
    actual_errors = np.array([r["actual_final_error"] for r in all_records])
    rho, pval = spearmanr(predicted_errors, actual_errors)
    print(f"\nSpearman(predicted_final_error, actual_final_error) over {len(all_records)} candidates: "
          f"rho={rho:.4f}, p={pval:.4g}")

    # also correlate predicted_score (higher = better) against -actual_final_error (higher = better)
    predicted_scores = np.array([r["predicted_score"] for r in all_records])
    rho_score, pval_score = spearmanr(predicted_scores, -actual_errors)
    print(f"Spearman(predicted_score, -actual_final_error): rho={rho_score:.4f}, p={pval_score:.4g}")

    # --- decision-relevant comparison: world-model top-B selection vs. random top-B ---
    wm_top_b_errors = []
    random_top_b_errors = []
    for state_idx in range(n_states):
        state_records = [r for r in all_records if r["state_idx"] == state_idx]
        if len(state_records) < top_b:
            continue
        sorted_by_wm = sorted(state_records, key=lambda r: r["predicted_score"], reverse=True)
        wm_top_b_errors.append(np.mean([r["actual_final_error"] for r in sorted_by_wm[:top_b]]))

        draws = []
        rng2 = np.random.RandomState(state_idx)
        for _ in range(n_random_draws):
            idx = rng2.choice(len(state_records), size=top_b, replace=False)
            draws.append(np.mean([state_records[i]["actual_final_error"] for i in idx]))
        random_top_b_errors.append(float(np.mean(draws)))

    wm_mean = float(np.mean(wm_top_b_errors))
    random_mean = float(np.mean(random_top_b_errors))
    print(f"\nMean actual final error of world-model top-{top_b} selection: {wm_mean:.4f}")
    print(f"Mean actual final error of random top-{top_b} selection (averaged over {n_random_draws} draws/state): {random_mean:.4f}")
    print(f"World-model selection {'BEATS' if wm_mean < random_mean else 'DOES NOT beat'} random selection "
          f"({'lower error is better' if wm_mean < random_mean else 'higher error than random'}).")

    save_json(os.path.join(output_dir, "summary.json"), dict(
        n_states=n_states, n_candidates_total=len(all_records),
        spearman_predicted_vs_actual_error=dict(rho=float(rho), pval=float(pval)),
        spearman_predicted_score_vs_neg_actual_error=dict(rho=float(rho_score), pval=float(pval_score)),
        world_model_top_b_mean_actual_error=wm_mean,
        random_top_b_mean_actual_error=random_mean,
        world_model_beats_random=bool(wm_mean < random_mean),
        top_b=top_b, per_state_summaries=per_state_summaries,
    ))
    print(f"\nSaved: {output_dir}/summary.json, {output_dir}/records.json")


if __name__ == "__main__":
    main()
