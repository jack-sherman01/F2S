"""Scale-up confirmation of the earlier-intervention finding
(scripts/evaluate_candidate_ranking_early_intervention.py, tested on 20
states). Pools *every* real Can failure segment available across all
episode directories collected so far (no new policy evaluation needed --
pure reuse of already-collected real rollouts) and re-runs the same
offset=0 vs. offset=15 comparison at ~3.5x the sample size, to check
whether the earlier finding (offset=15 restores real outcome diversity,
improves ranking, and produced the project's first two archived skills)
holds up or was luck on 2/20 states.

Any candidate that succeeds is immediately run through the real Day-19
validation protocol and the real SkillArchive, exactly as
scripts/validate_can_successful_candidates.py did for the original 3.
"""
import glob
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
from f2s.candidates.validator import build_validation_configs, execute_action_chunk, perturb_friction, perturb_mass, perturb_object_position_near
from f2s.common.io import ensure_fresh_dir, load_all_episode_metadata, load_json, save_json
from f2s.failure.extractor import process_episode
from f2s.failure.features import DEFAULT_GOAL_POS
from f2s.logging.episode_logger import load_episode
from f2s.skills.archive import SkillArchive
from f2s.skills.skill import Skill
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


def run_condition(offset, segments, dp_module, world_model, env, device, world_model_horizon, M, output_dir):
    print(f"\n===== offset = {offset} steps before stall time, {len(segments)} states =====")
    all_records = []
    near_zero_variance_count = 0
    successes = []

    for state_idx, (ep_dir, seg, meta) in enumerate(segments):
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
            predicted_final_error = float(np.linalg.norm(rank_result["predicted_states"][-1, 17:20] - DEFAULT_GOAL_POS))
            exec_result = execute_action_chunk(env, cand["action_chunk"], initial_state_dict=initial_state_dict)
            actual_final_obj = np.asarray(exec_result["final_obs"]["object"])[0:3]
            actual_final_error = float(np.linalg.norm(actual_final_obj - DEFAULT_GOAL_POS))
            record = dict(
                state_idx=state_idx, episode_id=seg["episode_id"], ep_dir=ep_dir, offset=offset,
                used_t_f=int(t_f), original_t_f=int(seg["failure_time"]),
                predicted_final_error=predicted_final_error,
                actual_success=bool(exec_result["actual_success"]), actual_final_error=actual_final_error,
            )
            state_records.append(record)
            all_records.append(record)
            if exec_result["actual_success"]:
                successes.append(dict(record=record, action_chunk=cand["action_chunk"], candidate_id=cand["candidate_id"],
                                       latent_delta=cand["latent_delta"], initial_state_dict=initial_state_dict))

        actual_errs = [r["actual_final_error"] for r in state_records]
        std = float(np.std(actual_errs)) if actual_errs else 0.0
        if std < 0.01:
            near_zero_variance_count += 1
        if (state_idx + 1) % 10 == 0 or state_idx == len(segments) - 1:
            print(f"  [{state_idx + 1}/{len(segments)}] processed, {len(successes)} successes so far")

    save_json(os.path.join(output_dir, f"records_offset{offset}.json"),
              [{k: v for k, v in r.items()} for r in all_records])

    pred = np.array([r["predicted_final_error"] for r in all_records])
    actual = np.array([r["actual_final_error"] for r in all_records])
    rho, pval = spearmanr(pred, actual)

    within_state_rhos, stds = [], []
    for state_idx in range(len(segments)):
        state_records = [r for r in all_records if r["state_idx"] == state_idx]
        p = [r["predicted_final_error"] for r in state_records]
        a = [r["actual_final_error"] for r in state_records]
        stds.append(float(np.std(a)))
        if len(set(p)) < 2 or len(set(a)) < 2:
            continue
        r2, _ = spearmanr(p, a)
        within_state_rhos.append(r2)

    n_total_success = sum(1 for r in all_records if r["actual_success"])
    summary = dict(
        offset=offset, n_states=len(segments), n_candidates_total=len(all_records),
        pooled_rho=float(rho), pooled_pval=float(pval),
        within_state_mean_rho=float(np.mean(within_state_rhos)) if within_state_rhos else None,
        within_state_median_rho=float(np.median(within_state_rhos)) if within_state_rhos else None,
        n_states_near_zero_variance=near_zero_variance_count,
        mean_within_state_actual_error_std=float(np.mean(stds)),
        n_total_success=n_total_success,
        success_rate=n_total_success / len(all_records),
    )
    print(f"offset={offset}: n_states={len(segments)}, near_zero_variance={near_zero_variance_count}/{len(segments)}, "
          f"within_state_median_rho={summary['within_state_median_rho']}, "
          f"total_successes={n_total_success}/{len(all_records)}")
    return summary, successes


def main():
    config_path = "configs/soe_can_lowdim_baseline.json"
    ckpt_path = os.environ["F2S_CAN_CKPT"]
    wm_dir = "results/can/world_model_h20diag"
    M = int(os.environ.get("F2S_RANK_M", "16"))
    world_model_horizon = int(os.environ.get("F2S_RANK_HORIZON", "5"))
    offsets = [int(x) for x in os.environ.get("F2S_FULLSCALE_OFFSETS", "0,15").split(",")]

    output_dir = "results/can/candidate_ranking_full_scale"
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

    # pool every real failure segment available, in a fixed, reproducible order
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
                segments.append((ep_dir, seg, meta))
    print(f"Pooled {len(segments)} real failure segments across {len(ALL_EPISODE_DIRS)} episode directories.")

    all_summaries = {}
    all_successes = {}
    for offset in offsets:
        summary, successes = run_condition(offset, segments, dp_module, world_model, env, device,
                                            world_model_horizon, M, output_dir)
        all_summaries[offset] = summary
        all_successes[offset] = successes

    print(f"\n{'='*70}\nSUMMARY: offset=0 (baseline) vs. offset=15 at full scale (n={len(segments)} states)\n{'='*70}")
    for offset in offsets:
        s = all_summaries[offset]
        print(f"offset={offset:>3}: near_zero_variance={s['n_states_near_zero_variance']:>3}/{s['n_states']}  "
              f"within_state_median_rho={s['within_state_median_rho']:.4f}  "
              f"successes={s['n_total_success']}/{s['n_candidates_total']} ({100*s['success_rate']:.2f}%)")

    # --- run every success found through the real Day-19 validation protocol ---
    print(f"\n{'='*70}\nValidating every real success found against Day-19 protocol\n{'='*70}")
    archive = SkillArchive()
    validation_results = []
    for offset in offsets:
        for succ in all_successes[offset]:
            r = succ["record"]
            action_chunk = succ["action_chunk"]
            initial_state_dict = succ["initial_state_dict"]
            print(f"\nValidating: {r['episode_id']} offset={offset} used_t_f={r['used_t_f']}")
            configs = build_validation_configs()
            n_valid = 0
            for cfg_kind in configs:
                if cfg_kind == "object_position":
                    perturbed_state = perturb_object_position_near(env, initial_state_dict)
                    ok = execute_action_chunk(env, action_chunk, initial_state_dict=perturbed_state)["actual_success"]
                elif cfg_kind == "friction":
                    ok = execute_action_chunk(env, action_chunk, initial_state_dict=initial_state_dict,
                                               post_reset_hook=perturb_friction)["actual_success"]
                else:
                    ok = execute_action_chunk(env, action_chunk, initial_state_dict=initial_state_dict,
                                               post_reset_hook=perturb_mass)["actual_success"]
                n_valid += int(ok)
            skill_success_rate = n_valid / len(configs)
            print(f"  Day-19 validation: {n_valid}/{len(configs)} = {skill_success_rate:.1%}")

            skill = Skill(
                skill_id=f"{r['episode_id']}_offset{offset}_fullscale_skill", failure_mode_id=0,
                latent_delta=succ["latent_delta"],
                precondition=dict(failure_mode_id=0, task_stage="unknown", object_error_range=(0.0, 0.0), goal_error_range=(0.0, 0.0)),
                effect=dict(final_object_error=None, final_goal_error=None, task_progress_change=None, recovery_success=True),
                success_rate=skill_success_rate, recovery_rate=skill_success_rate, transfer_rate=skill_success_rate,
                risk_score=0.0, source_candidate_ids=[succ["candidate_id"]], action_chunk=action_chunk,
            )
            accepted, reason = archive.add(skill)
            print(f"  Archive decision: {'ACCEPTED' if accepted else f'REJECTED ({reason})'}")
            validation_results.append(dict(
                episode_id=r["episode_id"], offset=offset, used_t_f=r["used_t_f"],
                validation_success_rate=skill_success_rate, archived=accepted, rejection_reason=reason,
            ))

    archive.save(os.path.join(output_dir, "skill_archive.json"))
    save_json(os.path.join(output_dir, "validation_results.json"), validation_results)
    save_json(os.path.join(output_dir, "summary.json"), dict(
        n_states_pooled=len(segments), offsets=all_summaries,
        n_validated_and_archived=len(archive.skills),
    ))
    print(f"\n{'='*70}\nFINAL: {len(archive.skills)} skills archived out of {len(validation_results)} candidates tested\n{'='*70}")


if __name__ == "__main__":
    main()
