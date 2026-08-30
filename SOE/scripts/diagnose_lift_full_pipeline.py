"""Cheap, decisive check (per the plan in SOE/README_F2S.md): does the F2S
correction mechanism (generate a latent-space candidate from a real
failure state, predict its outcome with a world model, execute it) find
even ONE working correction on an easier primary task, before concluding
anything more general about the approach itself?

RoboMimic Lift is a good choice: the proposal explicitly allows Lift,
Push, or Transport as the primary task instead of Can, and Lift's success
condition is materially simpler -- cube_height > table_height + 0.04
(robosuite Lift._check_success), no target position, no placement phase,
no orientation requirement. If corrections still can't be found here, the
finding is about the correction *mechanism*, not about Can's specific
difficulty.

This is deliberately a self-contained script rather than changes to the
shared f2s modules: f2s.candidates.cem and f2s.failure.extractor hardcode
Can's goal-relative fitness (distance to a fixed target position) and
14-dim `object` obs layout, neither of which apply to Lift (10-dim
`object`, height-based success). Reuses f2s.world_model.state.
build_world_model_state (which only touches obs["object"][0:3], valid for
any single-object robosuite task) and every other already-validated
piece; only the fitness/success functions below are Lift-specific.
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

from f2s.candidates.generator import get_latent
from f2s.candidates.validator import execute_action_chunk
from f2s.common.io import ensure_fresh_dir, load_all_episode_metadata, load_json, save_json
from f2s.logging.episode_logger import EpisodeLogger, load_episode
from f2s.safety.filter import P_OBJ_SLICE, collision_detected, joint_limit_exceeded, safety_filter, velocity_limit_exceeded
from f2s.world_model.dataset import build_transitions, compute_normalization_stats
from f2s.world_model.model import rollout_world_model
from f2s.world_model.state import build_world_model_state, build_world_model_states_for_episode
from f2s.world_model.train import train_world_model

TABLE_HEIGHT = 0.8  # robosuite default, confirmed via Lift._check_success's table_offset
LIFT_SUCCESS_MARGIN = 0.04  # robosuite Lift._check_success: cube_height > table_height + 0.04
ZERO_GOAL = np.zeros(3, dtype=np.float32)  # Lift has no target *position*; build_world_model_state's
                                            # p_goal slot is an unused constant feature for this task


def find_lift_stall_time(cube_z_t: np.ndarray) -> int:
    """Mirrors f2s.failure.extractor.find_stall_time but for a
    *maximization* objective (cube height) instead of minimizing
    distance-to-goal -- the last timestep the cube reached a new maximum
    height; everything after is a stall with no further lift progress."""
    if len(cube_z_t) == 0:
        return 0
    best_so_far = np.maximum.accumulate(cube_z_t)
    last_improvement_t = 0
    for t in range(1, len(best_so_far)):
        if best_so_far[t] > best_so_far[t - 1] + 1e-4:
            last_improvement_t = t
    return last_improvement_t


def lift_cem_search(dp_module, world_model, obs_dict, x0, device, population_size=64, n_iters=5,
                     sigma_init=0.5, sigma_min=0.05, horizon_wm=5, risk_weight=0.2, seed=0):
    """Same CEM structure as f2s.candidates.cem.cem_search, but fitness
    maximizes predicted cube height instead of minimizing distance to a
    fixed goal position."""
    torch.manual_seed(seed)
    z_f = get_latent(dp_module, obs_dict)
    D = z_f.shape[-1]
    device_t = torch.device(device)
    mean = torch.zeros(D, device=device_t)
    std = torch.full((D,), sigma_init, device=device_t)
    n_elite = max(1, int(round(population_size * 0.25)))
    x0_t = torch.from_numpy(x0).float().to(device_t)

    all_candidates = []
    for it in range(n_iters):
        deltas = mean.unsqueeze(0) + std.unsqueeze(0) * torch.randn(population_size, D, device=device_t)
        z_batch = z_f.repeat(population_size, 1) + deltas
        with torch.no_grad():
            action_chunks = dp_module.action_decoder.predict_action(z_batch)
        h = min(horizon_wm, action_chunks.shape[1])
        x0_batch = x0_t.unsqueeze(0).repeat(population_size, 1)
        a_batch = action_chunks[:, :h, :].permute(1, 0, 2)
        with torch.no_grad():
            pred_states = rollout_world_model(world_model, x0_batch, a_batch, horizon=h)

        pred_height = pred_states[-1, :, P_OBJ_SLICE][:, 2]  # (K,)
        pred_states_np = pred_states.permute(1, 0, 2).cpu().numpy()
        action_chunks_np = action_chunks.detach().cpu().numpy()

        risk_np = np.zeros(population_size, dtype=np.float32)
        for k in range(population_size):
            r = 0.0
            r += float(collision_detected(pred_states_np[k]))
            r += float(velocity_limit_exceeded(pred_states_np[k]))
            r += float(joint_limit_exceeded(pred_states_np[k]))
            risk_np[k] = r

        fitness = -(pred_height.cpu().numpy()) + risk_weight * risk_np  # lower is better (maximize height)
        elite_idx = np.argsort(fitness)[:n_elite]
        elite_deltas = deltas[torch.as_tensor(elite_idx, device=device_t)]
        mean = elite_deltas.mean(dim=0)
        std = elite_deltas.std(dim=0).clamp(min=sigma_min)

        for k in range(population_size):
            valid = bool(np.isfinite(action_chunks_np[k]).all())
            all_candidates.append(dict(
                action_chunk=action_chunks_np[k], predicted_states=pred_states_np[k],
                predicted_height=float(pred_height[k].item()), predicted_risk=float(risk_np[k]),
                cem_fitness=float(fitness[k]), valid=valid, cem_iteration=it,
            ))

    all_candidates.sort(key=lambda c: c["cem_fitness"])
    return all_candidates


def main():
    config_path = "configs/soe_lift_lowdim_baseline.json"
    ckpt_path = os.environ["F2S_LIFT_CKPT"]
    num_eval_episodes = int(os.environ.get("F2S_LIFT_EVAL_EPISODES", "30"))
    n_states = int(os.environ.get("F2S_LIFT_N_STATES", "5"))
    top_k_execute = int(os.environ.get("F2S_LIFT_TOPK", "8"))

    from rollout_utils import dp_load

    with open(config_path, "r") as f:
        cfg = EasyDict(json.load(f))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = SimpleNamespace(
        agent=ckpt_path, critic_agent=None, config=config_path, n_rollouts=num_eval_episodes, horizon=None,
        env=None, render=False, render_traj=False, video_dir=None, video_skip=1, camera_names=["agentview"],
        dataset_path=None, dataset_obs=False, seed=0, try_times=1, inference_horizon=None,
        high_noise_eval=False, eta=None, num_inference_steps=None, enable_exploration=False,
        tau1=None, tau2=None, noise_scale=None, enable_exploration_debug=False, disable_styles=False,
        enable_action_noise=False, action_noise_scale=None, enable_cfg=False, cfg_scale=0.5,
        cfg_agent=None, cfg_config=None, abs_action=False, return_intermediate=False,
    )
    rollout_policy, env, _, rollout_horizon = dp_load(args, cfg, enable_exploration_as_args=False)
    dp_module = rollout_policy.policy

    # --- Step 1: evaluate the baseline policy, log episodes ---
    from rollout_utils import rollout as soe_rollout

    eval_dir = "results/lift/soe/seed_0/round_0/eval"
    ensure_fresh_dir(eval_dir)
    logger = EpisodeLogger(output_dir=eval_dir, task="Lift", seed=0, round_id=0)
    n_success = 0
    for i in range(num_eval_episodes):
        stats, traj = soe_rollout(rollout_policy, env, horizon=rollout_horizon, return_obs=True)
        success = bool(stats["Success_Rate"])
        n_success += int(success)
        ep_len = int(stats["Horizon"])
        logger.start_episode()
        for t in range(ep_len):
            obs_t = {k: traj["obs"][k][t] for k in traj["obs"]}
            logger.add_step(obs_t, traj["states"][t], traj["actions"][t], float(traj["rewards"][t]), bool(traj["dones"][t]))
        if success:
            logger.finish_episode(True, "success", None, "none")
        else:
            logger.finish_episode(False, "timeout", ep_len - 1, "unknown")
        print(f"[eval {i + 1}/{num_eval_episodes}] success={success} len={ep_len}")
    print(f"\nBaseline Lift policy: {n_success}/{num_eval_episodes} = {100 * n_success / num_eval_episodes:.1f}% success")

    # --- Step 2: build + train a world model on these episodes ---
    episode_dir = os.path.join(eval_dir, "episodes")
    ids = [os.path.splitext(os.path.basename(p))[0]
           for p in __import__("glob").glob(os.path.join(episode_dir, "episode_*.json"))]
    rng = np.random.RandomState(0)
    perm = rng.permutation(len(ids))
    n_train = int(round(0.8 * len(ids)))
    train_ids = [ids[i] for i in perm[:n_train]]
    val_ids = [ids[i] for i in perm[n_train:]]

    train_states, train_actions, train_next_states = build_transitions(episode_dir, train_ids)
    val_states, val_actions, val_next_states = build_transitions(episode_dir, val_ids)
    wm_dir = "results/lift/world_model_diag"
    ensure_fresh_dir(wm_dir)
    world_model, wm_result = train_world_model(
        train_states, train_actions, train_next_states, val_states, val_actions, val_next_states,
        output_dir=wm_dir, hidden_dim=256, epochs=50, seed=0,
    )
    save_json(os.path.join(wm_dir, "result.json"), wm_result)
    print(f"\nWorld model: val_mse={wm_result['best_val_mse']:.6f}, "
          f"constant_baseline={wm_result['constant_state_val_mse']:.6f}, "
          f"beats_constant={wm_result['beats_constant_baseline']}")

    # --- Step 3: find real failure states, run CEM, execute in sim ---
    metas = load_all_episode_metadata(episode_dir)
    candidates_states = []
    for meta in metas:
        if meta["success"]:
            continue
        _, arrays = load_episode(episode_dir, meta["episode_id"])
        cube_z = arrays["obs_object"][:, 2]
        t_f = find_lift_stall_time(cube_z)
        candidates_states.append((meta, arrays, t_f))

    print(f"\n{len(candidates_states)} failure episodes available; testing {min(n_states, len(candidates_states))} "
          f"with CEM (population=64, iters=5, executing top {top_k_execute} each).")

    total_exec, total_succ = 0, 0
    for meta, arrays, t_f in candidates_states[:n_states]:
        obs_t = {k: arrays[f"obs_{k}"][t_f] for k in meta["obs_keys"]}
        obs_tensors = {k: torch.from_numpy(np.asarray(v)).float().unsqueeze(0).to(device) for k, v in obs_t.items()}
        x0 = build_world_model_state(obs_t, goal_pos=ZERO_GOAL)

        candidates = lift_cem_search(dp_module, world_model, obs_tensors, x0, device=device,
                                      population_size=64, n_iters=5, horizon_wm=5)
        print(f"  episode={meta['episode_id']} t_f={t_f} cube_z_at_tf={arrays['obs_object'][t_f, 2]:.4f} "
              f"(success needs > {TABLE_HEIGHT + LIFT_SUCCESS_MARGIN:.4f}): "
              f"best predicted_height={candidates[0]['predicted_height']:.4f}")

        initial_state_dict = dict(states=arrays["states"][t_f])
        n_success_here = 0
        for cand in candidates[:top_k_execute]:
            if not cand["valid"]:
                continue
            result = execute_action_chunk(env, cand["action_chunk"], initial_state_dict=initial_state_dict)
            total_exec += 1
            if result["actual_success"]:
                n_success_here += 1
                total_succ += 1
        print(f"    -> {n_success_here}/{top_k_execute} executed CEM candidates succeeded")

    print(f"\nTOTAL: {total_succ}/{total_exec} CEM candidates succeeded "
          f"({100 * total_succ / max(total_exec, 1):.1f}%) across "
          f"{min(n_states, len(candidates_states))} Lift failure states.")

    save_json("results/lift/world_model_diag/diagnostic_summary.json", dict(
        baseline_success_rate=n_success / num_eval_episodes,
        world_model_result=wm_result,
        n_failure_states_tested=min(n_states, len(candidates_states)),
        total_candidates_executed=total_exec,
        total_candidates_succeeded=total_succ,
    ))


if __name__ == "__main__":
    main()
