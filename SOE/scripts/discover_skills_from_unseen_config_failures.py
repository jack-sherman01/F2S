"""Validates "Option A" for closing H3's upside gap (see
README_F2S.md's "Fresh three-seed + Day-25 numbers with the gating fix
in production" section): the current 2 archived skills' validated 3cm
neighborhoods don't overlap the Day-25 unseen-configuration test region,
so the gate (correctly) never fires there and H3 (does the archive help,
especially under unseen configs) remains untested.

Rather than discovering skills from in-distribution failures (as
scripts/evaluate_candidate_ranking_per_state_offset_sweep.py did) and
hoping their neighborhood happens to overlap the unseen-config space,
this sources failure states directly from the Day-25 unseen-config eval's
own failures (results/Can/fixed_policy/seed_0/unseen/episodes -- the
"position" and "combined" categories failed 25/25 and 25/25 there,
exactly the unseen-position-forcing categories). 50 of that eval's 55
failures have an intervention-time object position genuinely outside
`can_unseen_test.yaml`'s seen_object_bbox -- i.e. these intervention
states are *already* inside the unseen-config region. Running the exact
same offset-sweep + real-execution + Day-19-style small-neighborhood
validation as section 17 on these states means any resulting skill's
validated 3cm neighborhood is centered *inside* the unseen-config space
by construction -- so the existing spatial gate (f2s/skills/retrieve.py)
would have a real chance to fire during a Day-25 unseen-config rollout
without any change to the gate itself.

Same method as scripts/evaluate_candidate_ranking_per_state_offset_sweep.py:
pure random generate_candidates (M=16), offsets swept per state, every
real success immediately Day-19-validated (perturb_object_position_near,
the *ordinary* small-neighborhood test -- not perturb_object_position_unseen;
the "unseen-ness" comes entirely from where these states already are, not
from changing the validation protocol) and archived if it clears the
0.7 threshold, with the real object_xy recorded so f2s/skills/retrieve.py's
spatial gate applies correctly.
"""
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
import yaml

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
from f2s.common.io import ensure_fresh_dir, load_all_episode_metadata, save_json
from f2s.failure.extractor import process_episode
from f2s.logging.episode_logger import load_episode
from f2s.skills.archive import SkillArchive
from f2s.skills.skill import Skill

SOURCE_EPISODE_DIR = "results/Can/fixed_policy/seed_0/unseen/episodes"
OFFSETS = [0, 10, 15, 20, 25, 30]
M = 16


def main():
    config_path = "configs/soe_can_lowdim_baseline.json"
    ckpt_path = os.environ["F2S_CAN_CKPT"]

    with open("configs/can_unseen_test.yaml", "r") as f:
        ucfg = yaml.safe_load(f)
    seen_bbox = ucfg["seen_object_bbox"]
    margin = ucfg["position_margin"]

    output_dir = "results/can/skills_from_unseen_config_failures"
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

    # Pool failure states from the unseen-config eval whose intervention-time
    # object position is genuinely outside the training-seen footprint.
    segments = []
    metas = load_all_episode_metadata(SOURCE_EPISODE_DIR)
    for meta in metas:
        if meta["success"]:
            continue
        _, arrays = load_episode(SOURCE_EPISODE_DIR, meta["episode_id"])
        seg = process_episode(meta, arrays, Hf=10)
        if seg is None:
            continue
        t_f = seg["intervention_time"]
        xy = arrays["obs_object"][t_f][0:2]
        inside = (seen_bbox["x_min"] - margin <= xy[0] <= seen_bbox["x_max"] + margin) and \
                 (seen_bbox["y_min"] - margin <= xy[1] <= seen_bbox["y_max"] + margin)
        if inside:
            continue  # this failure's intervention state isn't actually "unseen" -- skip
        segments.append((seg, meta, arrays))
    print(f"Pooled {len(segments)} unseen-config failure states (of {sum(1 for m in metas if not m['success'])} "
          f"total failures) with intervention-time position outside the seen footprint.")
    print(f"Sweeping offsets {OFFSETS} per state ({len(segments) * len(OFFSETS) * M} candidate executions budgeted).")

    archive = SkillArchive()
    all_records = []
    validation_results = []
    n_real_success = 0

    for state_idx, (seg, meta, arrays) in enumerate(segments):
        for offset in OFFSETS:
            t_f = max(0, seg["failure_time"] - offset)
            obs_t = {k: arrays[f"obs_{k}"][t_f] for k in meta["obs_keys"]}
            obs_tensors = {k: torch.from_numpy(np.asarray(v)).float().unsqueeze(0).to(device) for k, v in obs_t.items()}
            initial_state_dict = dict(states=arrays["states"][t_f])
            object_xy = obs_t["object"][0:2]

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
                    object_xy=[float(object_xy[0]), float(object_xy[1])],
                    actual_success=bool(exec_result["actual_success"]),
                )
                all_records.append(record)
                if not exec_result["actual_success"]:
                    continue
                n_real_success += 1
                print(f"\n[state {state_idx + 1}/{len(segments)}, offset={offset}] real success: "
                      f"{seg['episode_id']} at unseen position {object_xy}; running Day-19 validation...")

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
                    skill_id=f"{seg['episode_id']}_offset{offset}_unseenregion_skill", failure_mode_id=0,
                    latent_delta=cand["latent_delta"],
                    precondition=dict(
                        failure_mode_id=0, task_stage="unknown",
                        object_error_range=(0.0, 0.0), goal_error_range=(0.0, 0.0),
                        object_xy=[float(object_xy[0]), float(object_xy[1])],
                        position_tolerance=0.03,
                    ),
                    effect=dict(final_object_error=None, final_goal_error=None, task_progress_change=None, recovery_success=True),
                    success_rate=skill_success_rate, recovery_rate=skill_success_rate, transfer_rate=skill_success_rate,
                    risk_score=0.0, source_candidate_ids=[cand["candidate_id"]], action_chunk=cand["action_chunk"],
                )
                accepted, reason = archive.add(skill)
                print(f"  Archive decision: {'ACCEPTED' if accepted else f'REJECTED ({reason})'}")
                validation_results.append(dict(
                    episode_id=seg["episode_id"], offset=offset, object_xy=[float(object_xy[0]), float(object_xy[1])],
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
        n_states=len(segments), offsets_swept=OFFSETS, M=M,
        n_candidates_executed=len(all_records), n_real_success=n_real_success,
        n_validated_and_archived=len(archive.skills),
    ))
    print(f"\n{'='*70}\nFINAL: {n_real_success} real successes out of {len(all_records)} candidates "
          f"executed across offsets {OFFSETS}; {len(archive.skills)} skills archived, "
          f"validated regions inside the unseen-config space\n{'='*70}")


if __name__ == "__main__":
    main()
