"""The F2S self-evolution loop (proposal Section 12 / Days 15-21).

Each round:
  1. evaluate the current policy (skill-augmented: a stall detector
     monitors task progress online; on a stall it retrieves a matching
     skill from the archive and plays back its action chunk instead of
     querying the diffusion policy for those steps -- this is how the
     skill archive is actually used at evaluation time, rather than
     folded into policy weights via fine-tuning);
  2. extract + cluster failures from that round's episodes;
  3. for each failure mode's representative failure states: retrieve an
     existing skill, or generate candidates, rank with the world model,
     safety-filter, execute the top candidates in the real simulator,
     validate successful ones across held-out configurations, and
     archive the ones that pass;
  4. retrain the world model on all transitions collected so far.

Policy weight fine-tuning (the "Policy Update" box in the proposal's
Figure 1) is intentionally *not* implemented in this loop -- see
SOE/README_F2S.md for why, and scope this as the natural next extension.
"""
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from f2s.candidates.generator import generate_candidates
from f2s.candidates.scorer import rank_candidate
from f2s.candidates.validator import (
    build_validation_configs,
    execute_action_chunk,
    perturb_friction,
    perturb_mass,
)
from f2s.common.io import ensure_fresh_dir, load_all_episode_metadata, load_json, save_json
from f2s.failure.clustering import choose_k
from f2s.failure.extractor import process_episode
from f2s.failure.features import DEFAULT_GOAL_POS, FAILURE_FEATURE_NAMES, compute_failure_feature_vector, standardize
from f2s.logging.episode_logger import EpisodeLogger, load_episode
from f2s.logging.metrics import compute_round_metrics
from f2s.safety.filter import safety_filter
from f2s.skills.archive import SkillArchive
from f2s.skills.retrieve import retrieve
from f2s.skills.skill import Skill
from f2s.world_model.dataset import build_transitions, compute_normalization_stats, split_episodes_by_id
from f2s.world_model.state import build_world_model_state, build_world_model_states_for_episode
from f2s.world_model.train import train_world_model

STALL_WINDOW = 20  # steps with no task-progress improvement -> "stalled"
STALL_PROGRESS_EPS = 0.01


def rollout_with_skills(
    policy,          # RolloutDP wrapper (simulation/rollout_utils.RolloutDP)
    env,
    horizon: int,
    archive: SkillArchive,
    failure_cluster_model=None,   # sklearn KMeans fit on standardized failure features (or None -> no retrieval)
    failure_cluster_norm=None,    # (mu, sigma) for standardizing features before predicting cluster id
):
    """Same contract as simulation.rollout_utils.rollout(return_obs=True),
    plus online skill retrieval on a stall. Every simulator step and every
    normal-mode policy action goes through the real, unmodified SOE
    env/policy code; only the skill-override branch substitutes an
    archived action chunk for the diffusion policy's own action."""
    from robomimic.utils import tensor_utils as TensorUtils
    import robomimic.utils.obs_utils as ObsUtils

    policy.start_episode()
    obs = env.reset()
    state_dict = env.get_state()
    obs = env.reset_to(state_dict)

    traj = dict(actions=[], rewards=[], dones=[], states=[], obs=[], next_obs=[])
    progress_history = []
    skill_playback_remaining = 0
    skill_playback_chunk = None
    skill_playback_t = 0
    skills_used = []

    total_reward = 0.0
    success = False
    step_i = 0
    for step_i in range(horizon):
        progress = compute_failure_feature_vector([ObsUtils.unprocess_obs_dict(obs)])[-1]  # task_progress_final
        progress_history.append(progress)
        stalled = (
            len(progress_history) > STALL_WINDOW
            and (progress_history[-1] - progress_history[-STALL_WINDOW]) < STALL_PROGRESS_EPS
        )

        if skill_playback_remaining == 0 and stalled and failure_cluster_model is not None:
            feat = compute_failure_feature_vector([ObsUtils.unprocess_obs_dict(obs)])
            mu, sigma = failure_cluster_norm
            feat_std = standardize(feat, mu, sigma)
            cluster_id = int(failure_cluster_model.predict(feat_std.reshape(1, -1))[0])
            object_error = float(feat[0])
            skill = retrieve(archive, failure_mode_id=cluster_id, current_object_error=object_error)
            if skill is not None and skill.action_chunk is not None:
                skill_playback_chunk = np.asarray(skill.action_chunk)
                skill_playback_remaining = skill_playback_chunk.shape[0]
                skill_playback_t = 0
                skills_used.append(skill.skill_id)

        if skill_playback_remaining > 0:
            act = skill_playback_chunk[skill_playback_t]
            skill_playback_t += 1
            skill_playback_remaining -= 1
            if skill_playback_remaining == 0:
                policy.start_episode()  # resync the diffusion policy's internal chunk cache
        else:
            act = policy(ob=obs)

        next_obs, r, done, _ = env.step(act)
        total_reward += r
        success = bool(env.is_success()["task"])
        done = done or success

        traj["actions"].append(act)
        traj["rewards"].append(r)
        traj["dones"].append(done)
        traj["states"].append(state_dict["states"])
        traj["obs"].append(ObsUtils.unprocess_obs_dict(obs))
        traj["next_obs"].append(ObsUtils.unprocess_obs_dict(next_obs))

        if done or success:
            break
        obs = next_obs
        state_dict = env.get_state()

    stats = dict(Return=total_reward, Horizon=step_i + 1, Success_Rate=float(success), Skills_Used=skills_used)
    traj["obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["obs"])
    traj["next_obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["next_obs"])
    for k in ["actions", "rewards", "dones", "states"]:
        traj[k] = np.array(traj[k])
    for k in ["obs", "next_obs"]:
        for kp in traj[k]:
            traj[k][kp] = np.array(traj[k][kp])
    return stats, traj


def evaluate_with_skills(
    policy, env, num_episodes: int, output_dir: str, task: str, seed: int, round_id: int,
    archive: SkillArchive, failure_cluster_model=None, failure_cluster_norm=None, horizon: int = 400,
    method_name: str = "F2S",
):
    os.makedirs(output_dir, exist_ok=True)
    logger = EpisodeLogger(output_dir=output_dir, task=task, seed=seed, round_id=round_id)
    for i in range(num_episodes):
        stats, traj = rollout_with_skills(
            policy, env, horizon, archive, failure_cluster_model, failure_cluster_norm
        )
        success = bool(stats["Success_Rate"])
        ep_len = int(stats["Horizon"])
        eid = logger.start_episode()
        for t in range(ep_len):
            obs_t = {k: traj["obs"][k][t] for k in traj["obs"]}
            logger.add_step(obs_t, traj["states"][t], traj["actions"][t], float(traj["rewards"][t]), bool(traj["dones"][t]))
        if success:
            logger.finish_episode(True, "success", None, "none")
        else:
            logger.finish_episode(False, "timeout", ep_len - 1, "unknown")
        print(f"[{method_name} eval {i + 1}/{num_episodes}] success={success} len={ep_len} skills_used={stats['Skills_Used']}")

    metas = load_all_episode_metadata(os.path.join(output_dir, "episodes"))
    metrics = compute_round_metrics(metas, task=task, method=method_name, seed=seed, round_id=round_id)
    save_json(os.path.join(output_dir, "metrics.json"), metrics)
    return metrics


def discover_and_archive_skills(
    episode_dir: str,
    failure_dir: str,
    failure_mode_dir: str,
    dp_module: torch.nn.Module,
    world_model: torch.nn.Module,
    env,
    archive: SkillArchive,
    device: str,
    failure_window: int = 10,
    num_clusters: int = 4,
    M: int = 16,
    sigma_z: float = 0.5,
    eta: float = 0.5,
    world_model_horizon: int = 5,
    candidates_executed_per_mode: int = 2,
    skill_validation_episodes: int = 10,
    max_states_per_mode: int = 5,
) -> Dict[str, Any]:
    """Days 8-10 + 15-20 for one round's episodes: extract failures,
    cluster into modes, generate/rank/filter/validate/archive candidates."""
    ensure_fresh_dir(failure_dir)
    ensure_fresh_dir(failure_mode_dir)

    metas = load_all_episode_metadata(episode_dir)
    segments = []
    for meta in metas:
        if meta["success"]:
            continue
        _, arrays = load_episode(episode_dir, meta["episode_id"])
        seg = process_episode(meta, arrays, Hf=failure_window)
        if seg is not None:
            segments.append(seg)

    save_json(os.path.join(failure_dir, "summary.json"), dict(
        total_episodes=len(metas),
        failed_episodes=sum(1 for m in metas if not m["success"]),
        episodes_with_failure_segments=len(segments),
    ))

    if len(segments) < 2:
        return dict(status="too_few_failures", n_segments=len(segments), n_skills_added=0,
                    cluster_model=None, cluster_norm=None)

    features = np.stack([compute_failure_feature_vector(seg["obs_window"]) for seg in segments])
    failure_types = [seg["failure_type"] for seg in segments]
    failure_stages = [seg["failure_stage"] for seg in segments]
    object_errors = features[:, 0]

    mu, sigma = features.mean(axis=0), features.std(axis=0)
    features_std = standardize(features, mu, sigma)
    np.savez(os.path.join(failure_mode_dir, "normalization_stats.npz"), mu=mu, sigma=sigma)

    chosen_k, all_results = choose_k(
        features_std, failure_types, failure_stages, object_errors,
        k_candidates=(2, 4, 6), preferred_k=num_clusters, min_cluster_size=2,
    )
    model = all_results[chosen_k]["model"]
    save_json(os.path.join(failure_mode_dir, "clusters.json"), dict(k=chosen_k, clusters=all_results[chosen_k]["summary"]))

    n_skills_added = 0
    n_candidates_generated = 0
    n_safety_rejected = 0
    n_executed = 0

    for cluster_id in range(chosen_k):
        idx = np.where(model.labels_ == cluster_id)[0]
        if len(idx) == 0:
            continue
        chosen_idx = idx[:max_states_per_mode]

        for i in chosen_idx:
            seg = segments[i]
            meta = next(m for m in metas if m["episode_id"] == seg["episode_id"])
            _, arrays = load_episode(episode_dir, seg["episode_id"])
            t_f = seg["failure_time"]
            obs_t = {k: arrays[f"obs_{k}"][t_f] for k in meta["obs_keys"]}
            obs_tensors = {k: torch.from_numpy(np.asarray(v)).float().unsqueeze(0).to(device) for k, v in obs_t.items()}

            existing_skills = archive.skills_for_failure_mode(cluster_id)
            skill_deltas = [s.latent_delta for s in existing_skills]

            candidates = generate_candidates(
                dp_module, obs_tensors, source_episode_id=seg["episode_id"], failure_mode_id=cluster_id,
                M=M, sigma_z=sigma_z, eta=eta, skill_deltas=skill_deltas,
            )
            n_candidates_generated += len(candidates)

            x0 = build_world_model_state(obs_t)
            ranked = []
            for cand in candidates:
                if not cand["valid"]:
                    continue
                rank_result = rank_candidate(world_model, x0, cand["action_chunk"], world_model_horizon, device)
                is_safe, reasons = safety_filter(rank_result["predicted_states"], cand["action_chunk"][:world_model_horizon])
                if not is_safe:
                    n_safety_rejected += 1
                    continue
                ranked.append(dict(**cand, **rank_result, safety_reasons=reasons))
            ranked.sort(key=lambda c: c["score"], reverse=True)

            for cand in ranked[:candidates_executed_per_mode]:
                initial_state_dict = dict(states=arrays["states"][t_f])
                exec_result = execute_action_chunk(env, cand["action_chunk"], initial_state_dict=initial_state_dict)
                n_executed += 1
                cand["actual_success"] = exec_result["actual_success"]
                if not exec_result["actual_success"]:
                    continue

                # validate across held-out configurations (Day 19)
                configs = build_validation_configs()
                n_valid_success = 0
                n_valid_unsafe = 0
                for cfg_kind in configs:
                    if cfg_kind == "object_position":
                        # a fresh reset() already resamples object placement; run
                        # the candidate open-loop from that new random state.
                        ok = execute_action_chunk(env, cand["action_chunk"], initial_state_dict=None)["actual_success"]
                    elif cfg_kind == "friction":
                        result = execute_action_chunk(
                            env, cand["action_chunk"], initial_state_dict=initial_state_dict,
                            post_reset_hook=perturb_friction,
                        )
                        ok = result["actual_success"]
                    else:  # mass
                        result = execute_action_chunk(
                            env, cand["action_chunk"], initial_state_dict=initial_state_dict,
                            post_reset_hook=perturb_mass,
                        )
                        ok = result["actual_success"]
                    n_valid_success += int(ok)

                skill_success_rate = n_valid_success / len(configs)
                # NOTE: n_valid_unsafe is always 0 here -- execute_action_chunk
                # (Day 19 validation) only returns final success/length, not a
                # full per-step trajectory, so the safety-filter predicates
                # (which need a state *sequence*) are not re-applied to real
                # execution. Risk is therefore only checked at candidate-
                # selection time (the world-model-predicted rollout, before
                # execution) and not re-verified against the real simulator
                # during skill validation. Documented simplification, not a
                # silent gap: skill_risk will read 0.0 for every archived
                # skill until this is closed.
                skill_risk = n_valid_unsafe / len(configs)

                skill = Skill(
                    skill_id=f"{cand['candidate_id']}_skill",
                    failure_mode_id=cluster_id,
                    latent_delta=cand["latent_delta"],
                    precondition=dict(
                        failure_mode_id=cluster_id, task_stage=seg["failure_stage"],
                        object_error_range=(float(object_errors[i]) * 0.5, float(object_errors[i]) * 1.5),
                        goal_error_range=(0.0, float(object_errors.max())),
                    ),
                    effect=dict(
                        final_object_error=float(cand["predicted_states"][-1, 17] if cand["predicted_states"].shape[-1] > 17 else -1),
                        final_goal_error=None, task_progress_change=None, recovery_success=True,
                    ),
                    success_rate=skill_success_rate,
                    recovery_rate=skill_success_rate,
                    transfer_rate=skill_success_rate,
                    risk_score=skill_risk,
                    source_candidate_ids=[cand["candidate_id"]],
                    action_chunk=cand["action_chunk"],
                )
                accepted, reason = archive.add(skill)
                if accepted:
                    n_skills_added += 1

    return dict(
        status="ok", n_segments=len(segments), chosen_k=chosen_k,
        n_candidates_generated=n_candidates_generated, n_safety_rejected=n_safety_rejected,
        n_executed=n_executed, n_skills_added=n_skills_added,
        cluster_model=model, cluster_norm=(mu, sigma),
    )


def retrain_world_model_from_episodes(episode_dirs: List[str], output_dir: str, seed: int = 0, epochs: int = 50, hidden_dim: int = 256):
    """Aggregate transitions from every episode directory listed in
    `episode_dirs` (so the world model keeps improving across rounds) and
    retrain from scratch (Section 9.2: "updated once per evolution
    round")."""
    ensure_fresh_dir(output_dir)
    all_states, all_actions, all_next_states = [], [], []
    for ep_dir in episode_dirs:
        ids = [
            os.path.splitext(os.path.basename(p))[0]
            for p in __import__("glob").glob(os.path.join(ep_dir, "episode_*.json"))
        ]
        s, a, ns = build_transitions(ep_dir, ids)
        if s.shape[0] > 0:
            all_states.append(s)
            all_actions.append(a)
            all_next_states.append(ns)
    states = np.concatenate(all_states)
    actions = np.concatenate(all_actions)
    next_states = np.concatenate(all_next_states)

    n = states.shape[0]
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    n_train = int(round(0.8 * n))
    train_idx, val_idx = perm[:n_train], perm[n_train:]

    model, result = train_world_model(
        states[train_idx], actions[train_idx], next_states[train_idx],
        states[val_idx], actions[val_idx], next_states[val_idx],
        output_dir=output_dir, hidden_dim=hidden_dim, epochs=epochs, seed=seed,
    )
    save_json(os.path.join(output_dir, "result.json"), result)
    return model, result
