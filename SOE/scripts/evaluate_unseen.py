"""Day 25: evaluate a method on configurations not used for policy,
world-model, or skill training/validation -- unseen object positions,
unseen friction, unseen mass, and unseen combinations of the above (see
configs/can_unseen_test.yaml for exactly what "unseen" means here and
why). No retraining happens anywhere in this script.

Usage:
    python scripts/evaluate_unseen.py \
        --method fixed_policy \
        --checkpoint <ckpt> \
        --config configs/soe_can_lowdim_baseline.json \
        --unseen_config configs/can_unseen_test.yaml \
        --num_episodes 100 --seed 0 \
        --output_dir results/Can/fixed_policy/seed_0/unseen

--method f2s additionally takes --archive_path/--cluster_model_path (same
contract as scripts/run_method.py); with no archive or an empty one, F2S
degrades to the plain closed-loop policy (no skill ever retrieved), and
skill_transfer_rate is reported as null rather than fabricated.

--method unguided_latent_repair additionally takes --world_model_dir
(same contract as scripts/run_method.py): on each stall it generates a
fresh corrective candidate on the spot (same candidate-generation +
world-model-ranking machinery as f2s) instead of retrieving from an
archive -- see f2s.evolution.loop.rollout_with_unguided_repair.
"""
import argparse
import json
import os
import pickle
import sys
import time
from types import SimpleNamespace

import numpy as np
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
SOE_SIMULATION_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "simulation"))
sys.path.insert(0, SOE_SIMULATION_ROOT)
sys.path.insert(0, os.path.dirname(__file__))

from f2s.candidates.validator import perturb_friction, perturb_mass, perturb_object_position_unseen
from f2s.common.io import ensure_fresh_dir, git_commit_hash, load_all_episode_metadata, save_json
from f2s.common.seeds import set_seed
from f2s.failure.extractor import detect_failure_time
from f2s.failure.features import standardize
from f2s.logging.episode_logger import EpisodeLogger, load_episode
from f2s.logging.metrics import compute_round_metrics
from f2s.safety.filter import collision_detected, joint_limit_exceeded, velocity_limit_exceeded
from f2s.skills.archive import SkillArchive
from f2s.skills.retrieve import retrieve
from f2s.world_model.state import build_world_model_states_for_episode

STALL_WINDOW = 20
STALL_PROGRESS_EPS = 0.01


def apply_unseen_condition(env, category: str, unseen_cfg: dict):
    """Must run right after a fresh env.reset() and before any
    env.reset_to(...) -- same ordering constraint as every other
    physical-parameter perturbation in f2s.candidates.validator."""
    if category in ("position", "combined"):
        perturb_object_position_unseen(
            env, unseen_cfg["seen_object_bbox"], unseen_cfg["full_object_bbox"],
            margin=unseen_cfg["position_margin"],
        )
    if category in ("friction", "combined"):
        perturb_friction(env, scale=unseen_cfg["friction_scale_unseen"])
    if category in ("mass", "combined"):
        perturb_mass(env, scale=unseen_cfg["mass_scale_unseen"])


def rollout_unseen_episode(
    policy, env, horizon: int, category: str, unseen_cfg: dict,
    archive=None, failure_cluster_model=None, failure_cluster_norm=None,
    world_model=None, device=None, world_model_horizon: int = 5, use_cem: bool = False,
):
    """Closed-loop policy rollout starting from an unseen initial
    configuration. Mirrors simulation/rollout_utils.rollout() /
    f2s.evolution.loop.rollout_with_skills / rollout_with_unguided_repair,
    except the initial env.reset() is followed immediately by
    apply_unseen_condition() (which needs to run before any env.reset_to,
    per the physical-parameter-perturbation ordering constraint -- see
    execute_action_chunk's docstring), instead of those functions' own
    unconditional in-distribution reset.

    Exactly one of (archive, world_model) should be given: `archive` runs
    the f2s skill-retrieval branch, `world_model` runs the
    unguided_latent_repair on-the-spot-generation branch; neither gives
    plain closed-loop rollout (fixed_policy/soe/failure_replay)."""
    import robomimic.utils.obs_utils as ObsUtils
    from robomimic.utils import tensor_utils as TensorUtils

    from f2s.failure.features import compute_failure_feature_vector

    policy.start_episode()
    env.reset()
    apply_unseen_condition(env, category, unseen_cfg)
    state_dict = env.get_state()
    obs = env.reset_to(state_dict)

    traj = dict(actions=[], rewards=[], dones=[], states=[], obs=[], next_obs=[])
    progress_history = []
    skill_playback_remaining = 0
    skill_playback_chunk = None
    skill_playback_t = 0
    skills_used = []
    n_repairs_attempted = 0

    total_reward = 0.0
    success = False
    step_i = 0
    for step_i in range(horizon):
        if archive is not None and failure_cluster_model is not None:
            progress = compute_failure_feature_vector([ObsUtils.unprocess_obs_dict(obs)])[-1]
            progress_history.append(progress)
            stalled = (
                len(progress_history) > STALL_WINDOW
                and (progress_history[-1] - progress_history[-STALL_WINDOW]) < STALL_PROGRESS_EPS
            )
            if skill_playback_remaining == 0 and stalled:
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
        elif world_model is not None:
            import torch

            from f2s.candidates.cem import cem_search
            from f2s.candidates.generator import generate_candidates
            from f2s.candidates.scorer import rank_candidate
            from f2s.safety.filter import safety_filter
            from f2s.world_model.state import build_world_model_state

            progress = compute_failure_feature_vector([ObsUtils.unprocess_obs_dict(obs)])[-1]
            progress_history.append(progress)
            stalled = (
                len(progress_history) > STALL_WINDOW
                and (progress_history[-1] - progress_history[-STALL_WINDOW]) < STALL_PROGRESS_EPS
            )
            if skill_playback_remaining == 0 and stalled:
                obs_np = ObsUtils.unprocess_obs_dict(obs)
                obs_tensors = {k: torch.from_numpy(np.asarray(v)).float().unsqueeze(0).to(device)
                               for k, v in obs_np.items()}
                x0 = build_world_model_state(obs_np)
                dp_module = policy.policy
                n_repairs_attempted += 1

                if use_cem:
                    candidates = cem_search(
                        dp_module, world_model, obs_tensors, x0, source_episode_id="unguided", failure_mode_id=0,
                        device=device, population_size=16, n_iters=5, horizon_wm=world_model_horizon,
                        seed=n_repairs_attempted,
                    )
                    ranked = [c for c in candidates if c["valid"]]
                    ranked.sort(key=lambda c: c["predicted_dist_to_goal"])
                else:
                    candidates = generate_candidates(
                        dp_module, obs_tensors, source_episode_id="unguided", failure_mode_id=0,
                        M=16, sigma_z=0.5, eta=0.5, seed=n_repairs_attempted,
                    )
                    ranked = []
                    for cand in candidates:
                        if not cand["valid"]:
                            continue
                        rank_result = rank_candidate(world_model, x0, cand["action_chunk"], world_model_horizon, device)
                        ranked.append(dict(**cand, **rank_result))
                    ranked.sort(key=lambda c: c["score"], reverse=True)

                best = None
                for cand in ranked:
                    is_safe, _ = safety_filter(cand["predicted_states"], cand["action_chunk"][:world_model_horizon])
                    if is_safe:
                        best = cand
                        break
                if best is not None:
                    skill_playback_chunk = np.asarray(best["action_chunk"])
                    skill_playback_remaining = skill_playback_chunk.shape[0]
                    skill_playback_t = 0
                    skills_used.append(f"unguided_repair_{n_repairs_attempted}")

        if skill_playback_remaining > 0:
            act = skill_playback_chunk[skill_playback_t]
            skill_playback_t += 1
            skill_playback_remaining -= 1
            if skill_playback_remaining == 0:
                policy.start_episode()
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


def compute_failure_mode_coverage(episodes_dir: str, reference_episodes_dir: str) -> "float | None":
    """Fraction of the distinct failure_type values seen in the reference
    (in-distribution) eval for this method that also appear among this
    unseen eval's failures. 1.0 = every in-distribution failure mode still
    shows up here (no qualitatively new or missing failure surface); < 1.0
    = the unseen configs collapsed failure diversity (e.g. everything
    becomes a timeout); the metric is undefined (None) if the reference
    run had zero failures to compare against."""
    def failure_types(ep_dir):
        types = set()
        for meta in load_all_episode_metadata(ep_dir):
            if meta["success"]:
                continue
            _, arrays = load_episode(ep_dir, meta["episode_id"])
            _, refined_type = detect_failure_time(meta, arrays)
            types.add(refined_type)
        return types

    ref_types = failure_types(reference_episodes_dir) if os.path.isdir(reference_episodes_dir) else set()
    if len(ref_types) == 0:
        return None
    unseen_types = failure_types(episodes_dir)
    return len(unseen_types & ref_types) / len(ref_types)


def compute_safety_violation_rate(episodes_dir: str) -> "float | None":
    metas = load_all_episode_metadata(episodes_dir)
    if len(metas) == 0:
        return None
    n_violation = 0
    for meta in metas:
        _, arrays = load_episode(episodes_dir, meta["episode_id"])
        if arrays["actions"].shape[0] < 2:
            continue
        states = build_world_model_states_for_episode(meta["obs_keys"], arrays)
        if collision_detected(states) or joint_limit_exceeded(states) or velocity_limit_exceeded(states):
            n_violation += 1
    return n_violation / len(metas)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True,
                         choices=["fixed_policy", "soe", "failure_replay", "f2s", "unguided_latent_repair"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True, help="SOE DP policy config json")
    parser.add_argument("--unseen_config", default="configs/can_unseen_test.yaml")
    parser.add_argument("--task", default="Can")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num_episodes", type=int, default=100)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--reference_episodes_dir", default=None,
                         help="in-distribution eval episodes dir for this method, for failure_mode_coverage; "
                              "defaults to results/<task>/<method>/seed_<seed>/round_0/episodes")
    parser.add_argument("--archive_path", default=None, help="for --method f2s")
    parser.add_argument("--cluster_model_path", default=None, help="for --method f2s")
    parser.add_argument("--world_model_dir", default=None, help="for --method unguided_latent_repair")
    parser.add_argument("--world_model_horizon", type=int, default=5, help="for --method unguided_latent_repair")
    parser.add_argument("--use_cem", action="store_true", help="for --method unguided_latent_repair")
    args = parser.parse_args()

    with open(args.unseen_config, "r") as f:
        unseen_cfg = yaml.safe_load(f)
    assert unseen_cfg.get("do_not_retrain", False), "can_unseen_test.yaml must set do_not_retrain: true"

    ensure_fresh_dir(args.output_dir)
    set_seed(args.seed)

    for src, dst in [(args.config, "config.yaml"), (args.unseen_config, "unseen_config.yaml")]:
        with open(src, "r") as f_in, open(os.path.join(args.output_dir, dst), "w") as f_out:
            f_out.write(f_in.read())
    with open(os.path.join(args.output_dir, "git_commit.txt"), "w") as f:
        f.write(git_commit_hash(os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))) + "\n")

    from easydict import EasyDict

    from rollout_utils import dp_load

    with open(args.config, "r") as f:
        cfg = EasyDict(json.load(f))
    run_args = SimpleNamespace(
        agent=args.checkpoint, critic_agent=None, config=args.config, n_rollouts=args.num_episodes,
        horizon=None, env=None, render=False, render_traj=False, video_dir=None, video_skip=1,
        camera_names=["agentview"], dataset_path=None, dataset_obs=False, seed=args.seed, try_times=1,
        inference_horizon=None, high_noise_eval=False, eta=None, num_inference_steps=None,
        enable_exploration=(args.method == "soe"), tau1=None, tau2=None,
        noise_scale=(2.0 if args.method == "soe" else None), enable_exploration_debug=False,
        disable_styles=False, enable_action_noise=False, action_noise_scale=None, enable_cfg=False,
        cfg_scale=0.5, cfg_agent=None, cfg_config=None, abs_action=False, return_intermediate=False,
    )
    policy, env, _, rollout_horizon = dp_load(run_args, cfg, enable_exploration_as_args=False)
    if args.method == "soe":
        policy.enable_exploration_as_args(run_args, cfg)

    archive, cluster_model, cluster_norm = None, None, None
    if args.method == "f2s" and args.archive_path is not None and os.path.exists(args.archive_path):
        archive = SkillArchive.load(args.archive_path)
        if args.cluster_model_path is not None and os.path.exists(args.cluster_model_path):
            with open(args.cluster_model_path, "rb") as f:
                cluster_model, cluster_norm = pickle.load(f)

    world_model, device = None, None
    if args.method == "unguided_latent_repair":
        assert args.world_model_dir is not None, "--method unguided_latent_repair requires --world_model_dir"
        import torch

        from f2s.world_model.model import WorldModelEnsemble

        device = "cuda" if torch.cuda.is_available() else "cpu"
        with open(os.path.join(args.world_model_dir, "result.json"), "r") as f:
            wm_result = json.load(f)
        world_model = WorldModelEnsemble(
            state_dim=wm_result["state_dim"], action_dim=wm_result["action_dim"],
            hidden_dim=wm_result["hidden_dim"], ensemble_size=wm_result["ensemble_size"],
        ).to(device)
        world_model.load_state_dict(torch.load(os.path.join(args.world_model_dir, "best_model.pt"), map_location=device))
        world_model.eval()

    schedule = unseen_cfg["category_schedule"]
    logger = EpisodeLogger(output_dir=args.output_dir, task=args.task, seed=args.seed, round_id=0)
    categories_used = []
    skills_used_by_episode = {}

    t_start = time.time()
    for i in range(args.num_episodes):
        category = schedule[i % len(schedule)]
        categories_used.append(category)
        stats, traj = rollout_unseen_episode(
            policy, env, rollout_horizon, category, unseen_cfg,
            archive=archive, failure_cluster_model=cluster_model, failure_cluster_norm=cluster_norm,
            world_model=world_model, device=device, world_model_horizon=args.world_model_horizon,
            use_cem=args.use_cem,
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
        skills_used_by_episode[eid] = stats["Skills_Used"]
        print(f"[{args.method} unseen {i + 1}/{args.num_episodes}] category={category} success={success} "
              f"len={ep_len} skills_used={stats['Skills_Used']}")

    save_json(os.path.join(args.output_dir, "categories_used.json"),
              dict(zip([f"episode_{idx:06d}" for idx in range(args.num_episodes)], categories_used)))
    save_json(os.path.join(args.output_dir, "skills_used_by_episode.json"), skills_used_by_episode)

    episodes_dir = os.path.join(args.output_dir, "episodes")
    metas = load_all_episode_metadata(episodes_dir)
    metrics = compute_round_metrics(metas, task=args.task, method=args.method, seed=args.seed, round_id=0)
    metrics["eval_wall_time_seconds"] = time.time() - t_start
    metrics["checkpoint"] = args.checkpoint
    metrics["config"] = args.config
    metrics["unseen_config"] = args.unseen_config
    metrics["category_counts"] = {c: categories_used.count(c) for c in set(categories_used)}

    reference_dir = args.reference_episodes_dir or f"results/{args.task}/{args.method}/seed_{args.seed}/round_0/episodes"
    metrics["failure_mode_coverage"] = compute_failure_mode_coverage(episodes_dir, reference_dir)
    metrics["safety_violation_rate"] = compute_safety_violation_rate(episodes_dir)

    if args.method in ("f2s", "unguided_latent_repair"):
        # For f2s this is skill-transfer rate (retrieved skills that led to
        # success); for unguided_latent_repair it's the analogous on-the-
        # spot-repair success rate (freshly generated candidates that led
        # to success) -- same computation, different meaning per method,
        # reported under the same field for direct comparability.
        n_skill_episodes = sum(1 for v in skills_used_by_episode.values() if len(v) > 0)
        if n_skill_episodes > 0:
            success_by_eid = {m["episode_id"]: m["success"] for m in metas}
            n_skill_success = sum(
                1 for eid, used in skills_used_by_episode.items() if len(used) > 0 and success_by_eid.get(eid, False)
            )
            metrics["skill_transfer_rate"] = n_skill_success / n_skill_episodes
            metrics["skill_episodes"] = n_skill_episodes
        else:
            metrics["skill_transfer_rate"] = None
            metrics["skill_episodes"] = 0
    else:
        metrics["skill_transfer_rate"] = None

    save_json(os.path.join(args.output_dir, "metrics.json"), metrics)
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    main()
