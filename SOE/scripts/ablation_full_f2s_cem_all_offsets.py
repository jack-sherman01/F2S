"""Day 24 required ablation comparison: SOE vs Failure Replay vs
F2S w/o World Model vs Full F2S (proposal_revised.tex Section 13 /
"Day 24: Run Ablations").

Realization that simplifies this: the two *minimum-required* ablations
("F2S without failure clustering" and "F2S without world-model ranking")
both describe exactly the method already used throughout this project
for the real archive (scripts/evaluate_candidate_ranking_per_state_offset_sweep.py:
flat failure_mode_id=0 for every state, pure random generate_candidates,
every valid candidate executed for real, no ranking/selection step) --
i.e. the existing 3-seed "F2S" result (74.4% +/- 4.2%, section 20) IS
the "F2S w/o World Model" data point. So the only genuinely new run
needed to complete the required comparison is "Full F2S": failure-mode
clustering + world-model-guided (CEM) candidate ranking, matching
f2s.evolution.loop.discover_and_archive_skills's actual defaults
(use_cem=True).

One more realization, confirmed by reading f2s/candidates/cem.py: CEM's
signature has no `skill_deltas` parameter, so clustering's only real
functional lever in this codebase (f2s.candidates.generator.generate_candidates's
"historical skill perturbation reuse" from existing_skills of the same
cluster) has NO effect on the CEM-based discovery path -- clustering
would only change which `failure_mode_id` label gets attached to a
result, never which candidates get generated/executed. So this script
does compute real failure-mode clusters (for an honest, literal "with
clustering" condition and to report cluster assignments), but the
discovery method itself is exactly section 16's CEM approach
(scripts/evaluate_candidate_ranking_cem_offset15.py), now extended from
a single frozen offset=15 to the full 6-offset sweep {0,10,15,20,25,30}
used everywhere else in this project, on the same 71-state pool, for a
fair, complete comparison -- not just re-testing the same single point.
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
from f2s.common.io import ensure_fresh_dir, load_all_episode_metadata, save_json
from f2s.failure.clustering import choose_k
from f2s.failure.extractor import process_episode
from f2s.failure.features import compute_failure_feature_vector, standardize
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

OFFSETS = [0, 10, 15, 20, 25, 30]
TOP_K_EXECUTED = 3


def main():
    config_path = "configs/soe_can_lowdim_baseline.json"
    ckpt_path = os.environ["F2S_CAN_CKPT"]
    wm_dir = "results/can/world_model_h20diag"
    world_model_horizon = int(os.environ.get("F2S_RANK_HORIZON", "5"))

    output_dir = "results/can/ablation_full_f2s_cem_all_offsets"
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

    # pool the same 71 real failure segments used throughout this project
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

    # real failure-mode clustering (K chosen same way as the production
    # pipeline: f2s_final.yaml's num_failure_clusters=4)
    features = np.stack([compute_failure_feature_vector(seg["obs_window"]) for _, seg, _, _ in segments])
    failure_types = [seg["failure_type"] for _, seg, _, _ in segments]
    failure_stages = [seg["failure_stage"] for _, seg, _, _ in segments]
    object_errors = features[:, 0]
    mu, sigma = features.mean(axis=0), features.std(axis=0)
    features_std = standardize(features, mu, sigma)
    chosen_k, all_results = choose_k(
        features_std, failure_types, failure_stages, object_errors,
        k_candidates=(2, 4, 6), preferred_k=4, min_cluster_size=2,
    )
    cluster_model = all_results[chosen_k]["model"]
    cluster_labels = cluster_model.labels_
    print(f"Clustering: chosen_k={chosen_k}, cluster sizes={np.bincount(cluster_labels)}")
    save_json(os.path.join(output_dir, "clusters.json"), dict(
        chosen_k=int(chosen_k), cluster_sizes=np.bincount(cluster_labels).tolist(),
        clusters=all_results[chosen_k]["summary"],
    ))

    print(f"Sweeping offsets {OFFSETS} per state via CEM, top-{TOP_K_EXECUTED} candidates executed per "
          f"(state, offset) ({len(segments) * len(OFFSETS) * TOP_K_EXECUTED} real executions budgeted).")

    archive = SkillArchive()
    all_records = []
    validation_results = []
    n_real_success = 0

    for state_idx, (ep_dir, seg, meta, arrays) in enumerate(segments):
        cluster_id = int(cluster_labels[state_idx])
        for offset in OFFSETS:
            t_f = max(0, seg["failure_time"] - offset)
            obs_t = {k: arrays[f"obs_{k}"][t_f] for k in meta["obs_keys"]}
            obs_tensors = {k: torch.from_numpy(np.asarray(v)).float().unsqueeze(0).to(device) for k, v in obs_t.items()}
            x0 = build_world_model_state(obs_t)
            initial_state_dict = dict(states=arrays["states"][t_f])
            object_xy = obs_t["object"][0:2]

            cem_candidates = cem_search(
                dp_module, world_model, obs_tensors, x0,
                source_episode_id=seg["episode_id"], failure_mode_id=cluster_id, device=device,
                population_size=64, n_iters=5, horizon_wm=world_model_horizon, seed=state_idx * 1000 + offset,
            )
            ranked = []
            for cand in cem_candidates:
                if not cand["valid"]:
                    continue
                is_safe, _ = safety_filter(cand["predicted_states"], cand["action_chunk"][:world_model_horizon])
                if not is_safe:
                    continue
                ranked.append(cand)
            ranked.sort(key=lambda c: c["predicted_dist_to_goal"])

            for cand in ranked[:TOP_K_EXECUTED]:
                exec_result = execute_action_chunk(env, cand["action_chunk"], initial_state_dict=initial_state_dict)
                record = dict(
                    state_idx=state_idx, episode_id=seg["episode_id"], offset=offset, cluster_id=cluster_id,
                    predicted_dist_to_goal=cand["predicted_dist_to_goal"],
                    actual_success=bool(exec_result["actual_success"]),
                )
                all_records.append(record)
                if not exec_result["actual_success"]:
                    continue
                n_real_success += 1
                print(f"\n[state {state_idx + 1}/{len(segments)}, offset={offset}, cluster={cluster_id}] "
                      f"real success: {seg['episode_id']}; running Day-19 validation...")

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
                    skill_id=f"{seg['episode_id']}_offset{offset}_fullf2s_skill", failure_mode_id=cluster_id,
                    latent_delta=cand["latent_delta"],
                    precondition=dict(
                        failure_mode_id=cluster_id, task_stage=seg["failure_stage"],
                        object_error_range=(0.0, 0.0), goal_error_range=(0.0, 0.0),
                        object_xy=[float(object_xy[0]), float(object_xy[1])], position_tolerance=0.03,
                    ),
                    effect=dict(final_object_error=None, final_goal_error=None, task_progress_change=None, recovery_success=True),
                    success_rate=skill_success_rate, recovery_rate=skill_success_rate, transfer_rate=skill_success_rate,
                    risk_score=0.0, source_candidate_ids=[cand["candidate_id"]], action_chunk=cand["action_chunk"],
                )
                accepted, reason = archive.add(skill)
                print(f"  Archive decision: {'ACCEPTED' if accepted else f'REJECTED ({reason})'}")
                validation_results.append(dict(
                    episode_id=seg["episode_id"], offset=offset, cluster_id=cluster_id,
                    validation_success_rate=skill_success_rate, archived=accepted, rejection_reason=reason,
                ))
                save_json(os.path.join(output_dir, "validation_results.json"), validation_results)
                archive.save(os.path.join(output_dir, "skill_archive.json"))

        if (state_idx + 1) % 10 == 0 or state_idx == len(segments) - 1:
            print(f"  [{state_idx + 1}/{len(segments)}] processed, {n_real_success} real successes so far, "
                  f"{len(archive.skills)} archived")
            save_json(os.path.join(output_dir, "records.json"), all_records)

    save_json(os.path.join(output_dir, "records.json"), all_records)
    save_json(os.path.join(output_dir, "summary.json"), dict(
        n_states=len(segments), offsets_swept=OFFSETS, top_k_executed=TOP_K_EXECUTED, chosen_k=int(chosen_k),
        n_candidates_executed=len(all_records), n_real_success=n_real_success,
        n_validated_and_archived=len(archive.skills),
    ))
    print(f"\n{'='*70}\nFINAL (Full F2S: clustering + CEM ranking): {n_real_success} real successes "
          f"out of {len(all_records)} candidates executed; {len(archive.skills)} skills archived\n{'='*70}")


if __name__ == "__main__":
    main()
