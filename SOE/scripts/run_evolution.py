"""Day 21.2 / Section 12 CLI: run N F2S evolution rounds on top of a
trained SOE baseline checkpoint.

    python scripts/run_evolution.py \
        --config configs/f2s_dev.yaml \
        --task Can --seed 0 --rounds 3 \
        --output_dir results/can/f2s_dev/seed_0
"""
import argparse
import json
import os
import sys
import time

import torch
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
SOE_SIMULATION_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "simulation"))
sys.path.insert(0, SOE_SIMULATION_ROOT)

from f2s.common.io import ensure_fresh_dir, git_commit_hash, save_json
from f2s.common.seeds import set_seed
from f2s.evolution.loop import discover_and_archive_skills, evaluate_with_skills, retrain_world_model_from_episodes
from f2s.skills.archive import SkillArchive
from f2s.world_model.model import WorldModelEnsemble
from f2s.world_model.state import STATE_DIM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="f2s_dev.yaml / f2s_final.yaml")
    parser.add_argument("--task", default="Can")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=None, help="override config's evolution_rounds")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--policy_config", required=True, help="SOE DP policy config json (e.g. configs/soe_can_lowdim_baseline.json)")
    parser.add_argument("--policy_ckpt", required=True, help="trained SOE baseline checkpoint to start from")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    num_rounds = args.rounds if args.rounds is not None else cfg["evolution_rounds"]

    ensure_fresh_dir(args.output_dir)
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from types import SimpleNamespace

    from rollout_utils import dp_load

    with open(args.policy_config, "r") as f:
        policy_cfg_raw = json.load(f)
    from easydict import EasyDict

    policy_cfg = EasyDict(policy_cfg_raw)

    run_args = SimpleNamespace(
        agent=args.policy_ckpt, critic_agent=None, config=args.policy_config, n_rollouts=cfg["episodes_per_round"],
        horizon=None, env=None, render=False, render_traj=False, video_dir=None, video_skip=1,
        camera_names=["agentview"], dataset_path=None, dataset_obs=False, seed=args.seed, try_times=1,
        inference_horizon=None, high_noise_eval=False, eta=None, num_inference_steps=None,
        enable_exploration=False, tau1=None, tau2=None, noise_scale=None, enable_exploration_debug=False,
        disable_styles=False, enable_action_noise=False, action_noise_scale=None, enable_cfg=False,
        cfg_scale=0.5, cfg_agent=None, cfg_config=None, abs_action=False, return_intermediate=False,
    )
    rollout_policy, env, _, rollout_horizon = dp_load(run_args, policy_cfg, enable_exploration_as_args=False)
    dp_module = rollout_policy.policy  # the raw DP nn.Module, for candidate generation

    world_model = WorldModelEnsemble(
        state_dim=STATE_DIM, action_dim=policy_cfg.policy.params.action_dim,
        hidden_dim=cfg["world_model_hidden_dim"], ensemble_size=cfg["world_model_ensemble_size"],
    ).to(device)

    archive = SkillArchive()
    all_episode_dirs = []
    round_summaries = []
    t_start = time.time()
    # for round 0, the world model is untrained (random init) and no
    # cluster model exists yet, so evaluate_with_skills naturally degrades
    # to plain SOE rollout (no skill retrieval matches are attempted).
    cluster_model = None
    cluster_norm = None

    for round_id in range(num_rounds):
        round_dir = os.path.join(args.output_dir, f"round_{round_id}")
        os.makedirs(round_dir, exist_ok=True)
        print(f"\n===== F2S evolution round {round_id} =====")

        eval_dir = os.path.join(round_dir, "eval")
        metrics = evaluate_with_skills(
            rollout_policy, env, num_episodes=cfg["episodes_per_round"], output_dir=eval_dir,
            task=args.task, seed=args.seed, round_id=round_id, archive=archive,
            failure_cluster_model=cluster_model, failure_cluster_norm=cluster_norm,
            horizon=rollout_horizon, method_name="F2S",
        )
        all_episode_dirs.append(os.path.join(eval_dir, "episodes"))

        failure_dir = os.path.join(round_dir, "failures")
        failure_mode_dir = os.path.join(round_dir, "failure_modes")
        discovery_result = discover_and_archive_skills(
            episode_dir=os.path.join(eval_dir, "episodes"),
            failure_dir=failure_dir, failure_mode_dir=failure_mode_dir,
            dp_module=dp_module, world_model=world_model, env=env, archive=archive, device=device,
            failure_window=cfg["failure_window"], num_clusters=cfg["num_failure_clusters"],
            M=cfg["num_candidates_per_failure_mode"], world_model_horizon=cfg["world_model_horizon"],
            candidates_executed_per_mode=cfg["num_executed_candidates_per_failure_mode"],
            skill_validation_episodes=cfg["skill_validation_episodes"],
        )
        print("skill discovery:", discovery_result)

        if discovery_result["status"] == "ok":
            cluster_model = discovery_result["cluster_model"]
            cluster_norm = discovery_result["cluster_norm"]

        wm_dir = os.path.join(round_dir, "world_model")
        wm_model, wm_result = retrain_world_model_from_episodes(
            episode_dirs=all_episode_dirs, output_dir=wm_dir, seed=args.seed,
            epochs=50, hidden_dim=cfg["world_model_hidden_dim"],
        )
        world_model = wm_model
        print("world model:", wm_result)

        archive.save(os.path.join(round_dir, "skill_archive.json"))

        discovery_summary = {k: v for k, v in discovery_result.items() if k not in ("cluster_model", "cluster_norm")}
        round_summary = dict(
            round=round_id, git_commit=git_commit_hash(os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))),
            eval_metrics=metrics, discovery=discovery_summary, world_model_result=wm_result,
            archive_size=len(archive.skills),
        )
        save_json(os.path.join(round_dir, "round_summary.json"), round_summary)
        round_summaries.append(round_summary)

    save_json(os.path.join(args.output_dir, "evolution_summary.json"), dict(
        rounds=round_summaries, total_wall_time_seconds=time.time() - t_start,
        final_archive_size=len(archive.skills),
    ))
    print(f"\nF2S evolution complete: {num_rounds} rounds, final archive size {len(archive.skills)}.")


if __name__ == "__main__":
    main()
