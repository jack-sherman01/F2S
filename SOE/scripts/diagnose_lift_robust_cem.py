"""Follow-up to scripts/diagnose_lift_skill_archiving.py's finding: a
successful candidate existed but was brittle (tolerance collapsed between
0.5cm and 1cm of object-position offset -- see SOE/README_F2S.md).

Tests one of the two concrete next steps from that writeup: score
candidates by predicted success *averaged over a small neighborhood of
nearby states*, not a single point estimate, so CEM is directly selecting
for the property we actually want (generalizes past the exact failure
state) instead of hoping it falls out incidentally.

Deliberately isolated from diagnose_lift_full_pipeline.py (imports its
few generic helpers -- ZERO_GOAL, find_lift_stall_time -- but does not
modify it) so the already-validated single-point CEM path is untouched;
this is new, separately-tested functionality, not a rewrite of working
code. Same target failure state and same evaluation protocol (the
0.5/1/2/3cm sweep) as the brittleness diagnostic, for a controlled,
apples-to-apples before/after comparison.
"""
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "simulation")))
sys.path.insert(0, os.path.dirname(__file__))

from easydict import EasyDict

from diagnose_lift_full_pipeline import ZERO_GOAL, find_lift_stall_time
from f2s.candidates.generator import get_latent
from f2s.candidates.validator import build_validation_configs, execute_action_chunk, perturb_friction, perturb_mass, perturb_object_position_near
from f2s.common.io import load_json, save_json
from f2s.logging.episode_logger import load_episode
from f2s.safety.filter import P_OBJ_SLICE, collision_detected, joint_limit_exceeded, velocity_limit_exceeded
from f2s.skills.archive import SkillArchive
from f2s.skills.skill import Skill
from f2s.world_model.model import WorldModelEnsemble, rollout_world_model
from f2s.world_model.state import build_world_model_state


def lift_cem_search_robust(
    dp_module, world_model, obs_dict, x0, device,
    population_size=64, n_iters=5, sigma_init=0.5, sigma_min=0.05,
    horizon_wm=5, risk_weight=0.2, seed=0,
    neighborhood_jitter_xy=0.01, n_neighbors=4,
):
    """Same CEM structure as diagnose_lift_full_pipeline.lift_cem_search,
    but fitness is the MEAN predicted height across `n_neighbors` random
    xy offsets (uniform in [-neighborhood_jitter_xy, +neighborhood_jitter_xy])
    of the failure state, plus the exact state itself, rather than a
    single point estimate. The neighborhood is fixed once at the start
    (not resampled every iteration) so every candidate and every
    iteration is compared on the same set of nearby states."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    z_f = get_latent(dp_module, obs_dict)
    D = z_f.shape[-1]
    device_t = torch.device(device)
    mean = torch.zeros(D, device=device_t)
    std = torch.full((D,), sigma_init, device=device_t)
    n_elite = max(1, int(round(population_size * 0.25)))

    neighbor_offsets = [np.zeros(2, dtype=np.float32)]
    for _ in range(n_neighbors):
        neighbor_offsets.append(
            np.random.uniform(-neighborhood_jitter_xy, neighborhood_jitter_xy, size=2).astype(np.float32)
        )
    x0_neighbors = []
    for off in neighbor_offsets:
        x = x0.copy()
        x[P_OBJ_SLICE.start:P_OBJ_SLICE.start + 2] += off
        x0_neighbors.append(torch.from_numpy(x).float().to(device_t))

    all_candidates = []
    for it in range(n_iters):
        deltas = mean.unsqueeze(0) + std.unsqueeze(0) * torch.randn(population_size, D, device=device_t)
        z_batch = z_f.repeat(population_size, 1) + deltas
        with torch.no_grad():
            action_chunks = dp_module.action_decoder.predict_action(z_batch)
        h = min(horizon_wm, action_chunks.shape[1])
        a_batch = action_chunks[:, :h, :].permute(1, 0, 2)
        action_chunks_np = action_chunks.detach().cpu().numpy()

        heights_per_nb = []
        pred_states_exact_np = None
        risk_np = None
        for nb_i, x0_nb in enumerate(x0_neighbors):
            x0_batch = x0_nb.unsqueeze(0).repeat(population_size, 1)
            with torch.no_grad():
                pred_states = rollout_world_model(world_model, x0_batch, a_batch, horizon=h)
            heights_per_nb.append(pred_states[-1, :, P_OBJ_SLICE][:, 2])  # (K,)
            if nb_i == 0:
                pred_states_exact_np = pred_states.permute(1, 0, 2).cpu().numpy()
                risk_np = np.zeros(population_size, dtype=np.float32)
                for k in range(population_size):
                    r = 0.0
                    r += float(collision_detected(pred_states_exact_np[k]))
                    r += float(velocity_limit_exceeded(pred_states_exact_np[k]))
                    r += float(joint_limit_exceeded(pred_states_exact_np[k]))
                    risk_np[k] = r

        mean_height = torch.stack(heights_per_nb, dim=0).mean(dim=0)  # (K,)
        min_height = torch.stack(heights_per_nb, dim=0).min(dim=0).values

        fitness = -(mean_height.cpu().numpy()) + risk_weight * risk_np  # lower is better
        elite_idx = np.argsort(fitness)[:n_elite]
        elite_deltas = deltas[torch.as_tensor(elite_idx, device=device_t)]
        mean = elite_deltas.mean(dim=0)
        std = elite_deltas.std(dim=0).clamp(min=sigma_min)

        for k in range(population_size):
            valid = bool(np.isfinite(action_chunks_np[k]).all())
            all_candidates.append(dict(
                action_chunk=action_chunks_np[k],
                predicted_height_exact=float(heights_per_nb[0][k].item()),
                predicted_mean_neighborhood_height=float(mean_height[k].item()),
                predicted_min_neighborhood_height=float(min_height[k].item()),
                predicted_risk=float(risk_np[k]),
                cem_fitness=float(fitness[k]), valid=valid, cem_iteration=it,
            ))

    all_candidates.sort(key=lambda c: c["cem_fitness"])
    return all_candidates


def sweep_tolerance(env, action_chunk, initial_state_dict, offsets_cm, trials=10, seed=0):
    results = {}
    for offset_cm in offsets_cm:
        np.random.seed(seed)
        n_ok = 0
        for _ in range(trials):
            perturbed = perturb_object_position_near(env, initial_state_dict, max_offset=offset_cm / 100.0)
            ok = execute_action_chunk(env, action_chunk, initial_state_dict=perturbed)["actual_success"]
            n_ok += int(ok)
        results[offset_cm] = (n_ok, trials)
        print(f"  offset<= {offset_cm}cm: {n_ok}/{trials} succeeded")
    return results


def main():
    config_path = "configs/soe_lift_lowdim_baseline.json"
    ckpt_path = os.environ["F2S_LIFT_CKPT"]
    episode_dir = "results/lift/soe/seed_0/round_0/eval/episodes"
    wm_dir = "results/lift/world_model_diag"
    target_episode_id = os.environ.get("F2S_LIFT_TARGET_EPISODE", "episode_000005")

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

    meta = load_json(os.path.join(episode_dir, f"{target_episode_id}.json"))
    _, arrays = load_episode(episode_dir, target_episode_id)
    cube_z = arrays["obs_object"][:, 2]
    t_f = find_lift_stall_time(cube_z)
    print(f"Target failure state: {target_episode_id} t_f={t_f} cube_z={cube_z[t_f]:.4f}")

    obs_t = {k: arrays[f"obs_{k}"][t_f] for k in meta["obs_keys"]}
    obs_tensors = {k: torch.from_numpy(np.asarray(v)).float().unsqueeze(0).to(device) for k, v in obs_t.items()}
    x0 = build_world_model_state(obs_t, goal_pos=ZERO_GOAL)
    initial_state_dict = dict(states=arrays["states"][t_f])

    print("\nRunning neighborhood-robust CEM (jitter=1cm, 4 neighbors + exact state)...")
    candidates = lift_cem_search_robust(
        dp_module, world_model, obs_tensors, x0, device=device,
        population_size=64, n_iters=5, horizon_wm=5, neighborhood_jitter_xy=0.01, n_neighbors=4, seed=0,
    )

    print("\nExecuting top 8 robust-CEM candidates directly (from the exact failure state)...")
    successful = None
    n_direct_success = 0
    for i, cand in enumerate(candidates[:8]):
        if not cand["valid"]:
            continue
        result = execute_action_chunk(env, cand["action_chunk"], initial_state_dict=initial_state_dict)
        print(f"  candidate {i}: mean_nb_height={cand['predicted_mean_neighborhood_height']:.4f} "
              f"min_nb_height={cand['predicted_min_neighborhood_height']:.4f} -> actual_success={result['actual_success']}")
        if result["actual_success"]:
            n_direct_success += 1
            if successful is None:
                successful = cand
    print(f"Direct (exact-state) success: {n_direct_success}/8")

    if successful is None:
        print("\nNo successful candidate found -- cannot test tolerance. Stopping.")
        save_json("results/lift/world_model_diag/robust_cem_summary.json", dict(
            target_episode_id=target_episode_id, direct_success_count=0, found_candidate=False,
        ))
        return

    print("\nTolerance sweep for the robust-CEM-selected candidate (same protocol as the single-point diagnostic):")
    tolerance = sweep_tolerance(env, successful["action_chunk"], initial_state_dict, [0.5, 1.0, 2.0, 3.0], trials=10, seed=0)

    print("\nFull Day-19 validation (10 configs) for comparison with the single-point result (2/10)...")
    configs = build_validation_configs()
    n_valid = 0
    for cfg_kind in configs:
        if cfg_kind == "object_position":
            perturbed_state = perturb_object_position_near(env, initial_state_dict)
            ok = execute_action_chunk(env, successful["action_chunk"], initial_state_dict=perturbed_state)["actual_success"]
        elif cfg_kind == "friction":
            ok = execute_action_chunk(env, successful["action_chunk"], initial_state_dict=initial_state_dict,
                                       post_reset_hook=perturb_friction)["actual_success"]
        else:
            ok = execute_action_chunk(env, successful["action_chunk"], initial_state_dict=initial_state_dict,
                                       post_reset_hook=perturb_mass)["actual_success"]
        n_valid += int(ok)
    skill_success_rate = n_valid / len(configs)
    print(f"Day-19 validation: {n_valid}/{len(configs)} = {skill_success_rate:.1%}")

    archive = SkillArchive()
    skill = Skill(
        skill_id=f"{target_episode_id}_lift_robust_skill", failure_mode_id=0,
        latent_delta=np.zeros(1),
        precondition=dict(failure_mode_id=0, task_stage="grasp", object_error_range=(0.0, 0.02), goal_error_range=(0.0, 0.0)),
        effect=dict(final_object_error=None, final_goal_error=None,
                    task_progress_change=float(successful["predicted_height_exact"] - cube_z[t_f]),
                    recovery_success=True),
        success_rate=skill_success_rate, recovery_rate=skill_success_rate, transfer_rate=skill_success_rate,
        risk_score=0.0, source_candidate_ids=[f"{target_episode_id}_cand"], action_chunk=successful["action_chunk"],
    )
    accepted, reason = archive.add(skill)
    print(f"Archive decision: {'ACCEPTED' if accepted else f'REJECTED ({reason})'}")

    save_json("results/lift/world_model_diag/robust_cem_summary.json", dict(
        target_episode_id=target_episode_id, direct_success_count=n_direct_success,
        found_candidate=True,
        tolerance_sweep={str(k): dict(succeeded=v[0], trials=v[1]) for k, v in tolerance.items()},
        day19_validation_success_rate=skill_success_rate, archived=accepted, rejection_reason=reason,
    ))


if __name__ == "__main__":
    main()
