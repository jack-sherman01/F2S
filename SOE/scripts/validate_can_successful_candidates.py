"""Takes the successful candidates found across the offset sweep
(scripts/evaluate_candidate_ranking_early_intervention.py) and runs each
through the actual Day-19 cross-configuration validation protocol and the
real, unmodified SkillArchive/Day-20 archive rule -- the direct
continuation of "does the earlier-intervention fix actually let us
archive a validated skill on Can."

Regenerates the exact same 16 candidates (same seed=state_idx, same
generate_candidates call) for each (episode, offset) pair with a known
success, then matches the successful one by its predicted_final_error
value (recorded in the sweep's records.json) since the sweep didn't
separately persist action_chunks (they're large; only scalars were kept).
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
from f2s.candidates.validator import build_validation_configs, execute_action_chunk, perturb_friction, perturb_mass, perturb_object_position_near
from f2s.common.io import ensure_fresh_dir, load_all_episode_metadata, load_json, save_json
from f2s.failure.extractor import process_episode
from f2s.failure.features import DEFAULT_GOAL_POS
from f2s.logging.episode_logger import load_episode
from f2s.skills.archive import SkillArchive
from f2s.skills.skill import Skill
from f2s.world_model.model import WorldModelEnsemble
from f2s.world_model.state import build_world_model_state


TARGETS = [
    dict(episode_id="episode_000040", offset=15, expected_predicted_error=0.2322, seed=0),
    dict(episode_id="episode_000048", offset=15, expected_predicted_error=0.2304, seed=7),
    dict(episode_id="episode_000040", offset=20, expected_predicted_error=0.2624, seed=0),
]

EPISODE_DIRS = [
    "results/can/f2s_final/seed_0/round_0/eval/episodes",
    "results/can/f2s_final/seed_0/round_1/eval/episodes",
    "results/can/f2s_final/seed_0/round_2/eval/episodes",
]


def main():
    config_path = "configs/soe_can_lowdim_baseline.json"
    ckpt_path = os.environ["F2S_CAN_CKPT"]
    wm_dir = "results/can/world_model_h20diag"
    output_dir = "results/can/skill_validation_early_intervention"
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

    # find each target episode across the pooled round dirs
    ep_lookup = {}
    for ep_dir in EPISODE_DIRS:
        for meta in load_all_episode_metadata(ep_dir):
            if not meta["success"]:
                ep_lookup.setdefault(meta["episode_id"], ep_dir)

    archive = SkillArchive()
    results = []

    for target in TARGETS:
        ep_dir = ep_lookup[target["episode_id"]]
        meta = load_json(os.path.join(ep_dir, f"{target['episode_id']}.json"))
        _, arrays = load_episode(ep_dir, target["episode_id"])
        seg = process_episode(meta, arrays, Hf=10)
        t_f = max(0, seg["failure_time"] - target["offset"])
        obs_t = {k: arrays[f"obs_{k}"][t_f] for k in meta["obs_keys"]}
        obs_tensors = {k: torch.from_numpy(np.asarray(v)).float().unsqueeze(0).to(device) for k, v in obs_t.items()}
        x0 = build_world_model_state(obs_t)
        initial_state_dict = dict(states=arrays["states"][t_f])

        candidates = generate_candidates(
            dp_module, obs_tensors, source_episode_id=target["episode_id"], failure_mode_id=0,
            M=16, sigma_z=0.5, eta=0.5, seed=target["seed"],
        )

        matched = None
        for cand in candidates:
            if not cand["valid"]:
                continue
            rank_result = rank_candidate(world_model, x0, cand["action_chunk"], 5, device)
            predicted_final_error = float(np.linalg.norm(rank_result["predicted_states"][-1, 17:20] - DEFAULT_GOAL_POS))
            if abs(predicted_final_error - target["expected_predicted_error"]) < 1e-3:
                matched = cand
                break
        assert matched is not None, f"could not re-identify the successful candidate for {target}"

        # confirm it still succeeds from the exact state (determinism check)
        confirm = execute_action_chunk(env, matched["action_chunk"], initial_state_dict=initial_state_dict)
        print(f"\n{target['episode_id']} offset={target['offset']}: re-identified candidate, "
              f"direct re-execution success={confirm['actual_success']}")

        if not confirm["actual_success"]:
            results.append(dict(**target, matched=True, reconfirmed_success=False))
            continue

        print("Running Day-19 validation (10 configs)...")
        configs = build_validation_configs()
        n_valid_success = 0
        per_config = []
        for cfg_kind in configs:
            if cfg_kind == "object_position":
                perturbed_state = perturb_object_position_near(env, initial_state_dict)
                ok = execute_action_chunk(env, matched["action_chunk"], initial_state_dict=perturbed_state)["actual_success"]
            elif cfg_kind == "friction":
                ok = execute_action_chunk(env, matched["action_chunk"], initial_state_dict=initial_state_dict,
                                           post_reset_hook=perturb_friction)["actual_success"]
            else:
                ok = execute_action_chunk(env, matched["action_chunk"], initial_state_dict=initial_state_dict,
                                           post_reset_hook=perturb_mass)["actual_success"]
            per_config.append(dict(config=cfg_kind, success=ok))
            n_valid_success += int(ok)
            print(f"  {cfg_kind}: {'SUCCESS' if ok else 'fail'}")

        skill_success_rate = n_valid_success / len(configs)
        print(f"Validation: {n_valid_success}/{len(configs)} = {skill_success_rate:.1%}")

        skill = Skill(
            skill_id=f"{target['episode_id']}_offset{target['offset']}_skill", failure_mode_id=0,
            latent_delta=matched["latent_delta"],
            precondition=dict(failure_mode_id=0, task_stage="unknown",
                               object_error_range=(0.0, 0.0), goal_error_range=(0.0, 0.0)),
            effect=dict(final_object_error=None, final_goal_error=None, task_progress_change=None, recovery_success=True),
            success_rate=skill_success_rate, recovery_rate=skill_success_rate, transfer_rate=skill_success_rate,
            risk_score=0.0, source_candidate_ids=[matched["candidate_id"]], action_chunk=matched["action_chunk"],
        )
        accepted, reason = archive.add(skill)
        print(f"Archive decision: {'ACCEPTED' if accepted else f'REJECTED ({reason})'}")

        results.append(dict(
            **target, matched=True, reconfirmed_success=True, per_config_results=per_config,
            validation_success_rate=skill_success_rate, archived=accepted, rejection_reason=reason,
        ))

    archive.save(os.path.join(output_dir, "skill_archive.json"))
    save_json(os.path.join(output_dir, "validation_results.json"), results)
    print(f"\n{'='*60}\nFinal archive size: {len(archive.skills)} / {len(TARGETS)} tested\n{'='*60}")


if __name__ == "__main__":
    main()
