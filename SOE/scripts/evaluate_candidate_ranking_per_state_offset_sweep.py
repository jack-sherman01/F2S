"""Per-state offset sweep (the one lever from the section-11/16 diagnosis
chain not yet tried): instead of one fixed intervention offset for every
state, try several offsets *per state* and see whether any of them
produces a real success that clears the corrected Day-19 validation
threshold. CEM-guided search was already shown (offset=15, 0/213, see
README's "Re-opened the search after the fix" section) to optimize a
metric that doesn't transfer real success; pure random perturbation
(f2s.candidates.generator.generate_candidates, same M=16 as every earlier
offset test) is what actually found the only 3 real successes to date, so
that's what this sweep uses too.

offset in {0, 15} was already fully executed across this exact 71-state
pool by scripts/evaluate_candidate_ranking_full_scale.py
(results/can/candidate_ranking_full_scale/) -- real execution outcomes
are unaffected by the Day-19 validator bug (only the downstream
skill-validation step was wrong), so those two offsets are NOT re-run
here; this script covers the remaining offsets from section 11's original
grid: {10, 20, 25, 30}. Every real success found (at any offset, for any
state) is immediately run through the real, now-fixed Day-19 validation
protocol and the real SkillArchive.
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
from f2s.candidates.validator import (
    build_validation_configs,
    execute_action_chunk,
    perturb_friction,
    perturb_mass,
    perturb_object_position_near,
)
from f2s.common.io import ensure_fresh_dir, load_all_episode_metadata, load_json, save_json
from f2s.failure.extractor import process_episode
from f2s.failure.features import DEFAULT_GOAL_POS
from f2s.logging.episode_logger import load_episode
from f2s.skills.archive import SkillArchive
from f2s.skills.skill import Skill
from f2s.world_model.model import WorldModelEnsemble
from f2s.world_model.state import build_world_model_state

# Must match scripts/evaluate_candidate_ranking_full_scale.py exactly so
# state_idx lines up with the already-executed offset=0/15 results.
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

NEW_OFFSETS = [10, 20, 25, 30]  # 0 and 15 already fully executed elsewhere


def validate_and_maybe_archive(env, cand, initial_state_dict, archive, episode_id, offset, output_dir, validation_results):
    configs = build_validation_configs()
    n_valid = 0
    for cfg_kind in configs:
        if cfg_kind == "object_position":
            perturbed_state = perturb_object_position_near(env, initial_state_dict)
            ok = execute_action_chunk(env, cand["action_chunk"], initial_state_dict=perturbed_state)["actual_success"]
        elif cfg_kind == "friction":
            ok = execute_action_chunk(env, cand["action_chunk"], initial_state_dict=initial_state_dict,
                                       post_reset_hook=perturb_friction)["actual_success"]
        else:
            ok = execute_action_chunk(env, cand["action_chunk"], initial_state_dict=initial_state_dict,
                                       post_reset_hook=perturb_mass)["actual_success"]
        n_valid += int(ok)
    skill_success_rate = n_valid / len(configs)
    print(f"  Day-19 validation: {n_valid}/{len(configs)} = {skill_success_rate:.1%}")

    skill = Skill(
        skill_id=f"{episode_id}_offset{offset}_perstate_skill", failure_mode_id=0,
        latent_delta=cand["latent_delta"],
        precondition=dict(failure_mode_id=0, task_stage="unknown", object_error_range=(0.0, 0.0), goal_error_range=(0.0, 0.0)),
        effect=dict(final_object_error=None, final_goal_error=None, task_progress_change=None, recovery_success=True),
        success_rate=skill_success_rate, recovery_rate=skill_success_rate, transfer_rate=skill_success_rate,
        risk_score=0.0, source_candidate_ids=[cand["candidate_id"]], action_chunk=cand["action_chunk"],
    )
    accepted, reason = archive.add(skill)
    print(f"  Archive decision: {'ACCEPTED' if accepted else f'REJECTED ({reason})'}")
    validation_results.append(dict(
        episode_id=episode_id, offset=offset, validation_success_rate=skill_success_rate,
        archived=accepted, rejection_reason=reason,
    ))
    save_json(os.path.join(output_dir, "validation_results.json"), validation_results)
    archive.save(os.path.join(output_dir, "skill_archive.json"))
    return accepted


def main():
    config_path = "configs/soe_can_lowdim_baseline.json"
    ckpt_path = os.environ["F2S_CAN_CKPT"]
    wm_dir = "results/can/world_model_h20diag"
    M = int(os.environ.get("F2S_RANK_M", "16"))
    world_model_horizon = int(os.environ.get("F2S_RANK_HORIZON", "5"))

    output_dir = "results/can/candidate_ranking_per_state_offset_sweep"
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
                segments.append((ep_dir, seg, meta))
    print(f"Pooled {len(segments)} real failure segments (must be 71, matching candidate_ranking_full_scale).")
    print(f"Sweeping offsets {NEW_OFFSETS} per state ({len(segments) * len(NEW_OFFSETS) * M} candidate executions budgeted).")

    archive = SkillArchive()
    validation_results = []
    all_records = []
    n_real_success = 0

    for state_idx, (ep_dir, seg, meta) in enumerate(segments):
        _, arrays = load_episode(ep_dir, seg["episode_id"])
        for offset in NEW_OFFSETS:
            t_f = max(0, seg["failure_time"] - offset)
            obs_t = {k: arrays[f"obs_{k}"][t_f] for k in meta["obs_keys"]}
            obs_tensors = {k: torch.from_numpy(np.asarray(v)).float().unsqueeze(0).to(device) for k, v in obs_t.items()}
            x0 = build_world_model_state(obs_t)
            initial_state_dict = dict(states=arrays["states"][t_f])

            candidates = generate_candidates(
                dp_module, obs_tensors, source_episode_id=seg["episode_id"], failure_mode_id=0,
                M=M, sigma_z=0.5, eta=0.5, seed=state_idx * 1000 + offset,
            )
            for cand in candidates:
                if not cand["valid"]:
                    continue
                exec_result = execute_action_chunk(env, cand["action_chunk"], initial_state_dict=initial_state_dict)
                record = dict(
                    state_idx=state_idx, episode_id=seg["episode_id"], offset=offset, used_t_f=int(t_f),
                    actual_success=bool(exec_result["actual_success"]),
                )
                all_records.append(record)
                if not exec_result["actual_success"]:
                    continue
                n_real_success += 1
                print(f"\n[state {state_idx + 1}/{len(segments)}, offset={offset}] real success: "
                      f"{seg['episode_id']}; running Day-19 validation...")
                validate_and_maybe_archive(env, cand, initial_state_dict, archive, seg["episode_id"], offset,
                                            output_dir, validation_results)

        if (state_idx + 1) % 10 == 0 or state_idx == len(segments) - 1:
            print(f"  [{state_idx + 1}/{len(segments)}] processed, {n_real_success} real successes so far, "
                  f"{len(archive.skills)} archived")
            save_json(os.path.join(output_dir, "records.json"), all_records)

    save_json(os.path.join(output_dir, "records.json"), all_records)
    save_json(os.path.join(output_dir, "summary.json"), dict(
        n_states=len(segments), offsets_swept=NEW_OFFSETS, M=M,
        n_candidates_executed=len(all_records), n_real_success=n_real_success,
        n_validated_and_archived=len(archive.skills),
    ))
    print(f"\n{'='*70}\nFINAL: {n_real_success} real successes out of {len(all_records)} candidates "
          f"executed across offsets {NEW_OFFSETS}; {len(archive.skills)} skills archived (>70% Day-19 threshold)\n{'='*70}")


if __name__ == "__main__":
    main()
