"""Re-opens the candidate search after the Day-19 validator bug fix (see
SOE/README_F2S.md, "Critical bug found while building Day 25"): the three
candidates that previously looked validated were found with pure random
perturbation (f2s.candidates.generator.generate_candidates) at
offset=15. That combination is now confirmed to score 50-70% under the
*correct* validator -- below the 70% archive threshold. This script tries
the one lever not yet combined with offset=15: CEM-guided search
(f2s.candidates.cem.cem_search), which f2s/evolution/loop.py's production
path already defaults to (use_cem=True) but which was only ever tried,
pre-bugfix, at offset=0 (see README's CEM section: 0/40 real transfer
there, before the intervention-timing finding existed).

Same 71-state pool as scripts/evaluate_candidate_ranking_full_scale.py,
same frozen offset=15, CEM instead of random search, top-3 CEM candidates
per state executed for real (instead of top-1), every real success
immediately run through the real (now-fixed) Day-19 validation protocol
and the real SkillArchive.
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
from f2s.candidates.validator import (
    build_validation_configs,
    execute_action_chunk,
    perturb_friction,
    perturb_mass,
    perturb_object_position_near,
)
from f2s.common.io import ensure_fresh_dir, load_all_episode_metadata, load_json, save_json
from f2s.failure.extractor import process_episode
from f2s.logging.episode_logger import load_episode
from f2s.safety.filter import safety_filter
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

OFFSET = 15
TOP_K_EXECUTED = 3


def main():
    config_path = "configs/soe_can_lowdim_baseline.json"
    ckpt_path = os.environ["F2S_CAN_CKPT"]
    wm_dir = "results/can/world_model_h20diag"
    world_model_horizon = int(os.environ.get("F2S_RANK_HORIZON", "5"))

    output_dir = "results/can/candidate_ranking_cem_offset15"
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
    print(f"Pooled {len(segments)} real failure segments across {len(ALL_EPISODE_DIRS)} episode directories.")
    print(f"CEM search at offset={OFFSET}, top-{TOP_K_EXECUTED} candidates executed per state "
          f"({len(segments) * TOP_K_EXECUTED} real executions budgeted).")

    archive = SkillArchive()
    all_records = []
    validation_results = []
    n_real_success = 0

    for state_idx, (ep_dir, seg, meta) in enumerate(segments):
        _, arrays = load_episode(ep_dir, seg["episode_id"])
        t_f = max(0, seg["failure_time"] - OFFSET)
        obs_t = {k: arrays[f"obs_{k}"][t_f] for k in meta["obs_keys"]}
        obs_tensors = {k: torch.from_numpy(np.asarray(v)).float().unsqueeze(0).to(device) for k, v in obs_t.items()}
        x0 = build_world_model_state(obs_t)
        initial_state_dict = dict(states=arrays["states"][t_f])

        cem_candidates = cem_search(
            dp_module, world_model, obs_tensors, x0,
            source_episode_id=seg["episode_id"], failure_mode_id=0, device=device,
            population_size=64, n_iters=5, horizon_wm=world_model_horizon, seed=state_idx,
        )

        ranked = []
        for cand in cem_candidates:
            if not cand["valid"]:
                continue
            is_safe, reasons = safety_filter(cand["predicted_states"], cand["action_chunk"][:world_model_horizon])
            if not is_safe:
                continue
            ranked.append(cand)
        ranked.sort(key=lambda c: c["predicted_dist_to_goal"])

        for cand in ranked[:TOP_K_EXECUTED]:
            exec_result = execute_action_chunk(env, cand["action_chunk"], initial_state_dict=initial_state_dict)
            record = dict(
                state_idx=state_idx, episode_id=seg["episode_id"], ep_dir=ep_dir, offset=OFFSET,
                used_t_f=int(t_f), predicted_dist_to_goal=cand["predicted_dist_to_goal"],
                actual_success=bool(exec_result["actual_success"]),
            )
            all_records.append(record)
            if not exec_result["actual_success"]:
                continue
            n_real_success += 1

            print(f"\n[{state_idx + 1}/{len(segments)}] real success: {seg['episode_id']} "
                  f"(predicted_dist_to_goal={cand['predicted_dist_to_goal']:.4f}); running Day-19 validation...")
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
                skill_id=f"{seg['episode_id']}_cem_offset{OFFSET}_skill", failure_mode_id=0,
                latent_delta=cand["latent_delta"],
                precondition=dict(failure_mode_id=0, task_stage="unknown", object_error_range=(0.0, 0.0), goal_error_range=(0.0, 0.0)),
                effect=dict(final_object_error=None, final_goal_error=None, task_progress_change=None, recovery_success=True),
                success_rate=skill_success_rate, recovery_rate=skill_success_rate, transfer_rate=skill_success_rate,
                risk_score=0.0, source_candidate_ids=[cand["candidate_id"]], action_chunk=cand["action_chunk"],
            )
            accepted, reason = archive.add(skill)
            print(f"  Archive decision: {'ACCEPTED' if accepted else f'REJECTED ({reason})'}")
            validation_results.append(dict(
                episode_id=seg["episode_id"], offset=OFFSET, used_t_f=int(t_f),
                validation_success_rate=skill_success_rate, archived=accepted, rejection_reason=reason,
            ))

        if (state_idx + 1) % 10 == 0 or state_idx == len(segments) - 1:
            print(f"  [{state_idx + 1}/{len(segments)}] processed, {n_real_success} real successes so far, "
                  f"{len(archive.skills)} archived")

    save_json(os.path.join(output_dir, "records.json"), all_records)
    archive.save(os.path.join(output_dir, "skill_archive.json"))
    save_json(os.path.join(output_dir, "validation_results.json"), validation_results)
    save_json(os.path.join(output_dir, "summary.json"), dict(
        n_states=len(segments), offset=OFFSET, top_k_executed=TOP_K_EXECUTED,
        n_candidates_executed=len(all_records), n_real_success=n_real_success,
        n_validated_and_archived=len(archive.skills),
    ))
    print(f"\n{'='*70}\nFINAL: {n_real_success} real successes out of {len(all_records)} CEM candidates "
          f"executed; {len(archive.skills)} skills archived (>{0.7:.0%} Day-19 threshold)\n{'='*70}")


if __name__ == "__main__":
    main()
