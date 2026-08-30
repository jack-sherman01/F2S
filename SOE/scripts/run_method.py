"""Day 23 CLI: evaluate one method on the trained SOE baseline checkpoint,
under identical rollout budgets/seeds, and save results/<task>/<method>/seed_<seed>/round_0/metrics.json.

Supported --method values, and exactly what each one actually does (no
method here is a stub -- every one runs real, unmodified SOE/robosuite
simulation code):

  fixed_policy   Evaluate the trained DP checkpoint with no exploration
                 and no skill archive. The "do nothing extra" baseline.

  soe            Evaluate the trained DP checkpoint with SOE's own
                 exploration enabled (--enable_exploration; see
                 SOE/src/policy/exploration.py -- CADS-style noise
                 injected into the diffusion conditioning during
                 sampling). This is "SOE-only" from the proposal's
                 baseline list.

  f2s            Evaluate with the skill archive from a completed
                 run_evolution.py run (--archive_path), using the online
                 stall-detector + skill-retrieval rollout
                 (f2s.evolution.loop.rollout_with_skills).

Two methods named in the proposal's baseline list -- Success-only and
Failure Replay -- require a full data-aggregation + policy-retraining
loop that is not implemented in this codebase (f2s.evolution.loop
deliberately does not retrain policy weights; see SOE/README_F2S.md for
why). Rather than fabricate numbers for an unimplemented method, --method
success_only/failure_replay are refused with an explicit NotImplementedError.
"""
import argparse
import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
SOE_SIMULATION_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "simulation"))
sys.path.insert(0, SOE_SIMULATION_ROOT)
sys.path.insert(0, os.path.dirname(__file__))  # so "from evaluate_policy import ..." resolves

from f2s.common.io import ensure_fresh_dir, git_commit_hash
from f2s.common.seeds import set_seed
from f2s.skills.archive import SkillArchive

UNIMPLEMENTED_METHODS = {
    "success_only": "requires the policy-retraining loop (collect successful trajectories, "
                     "retrain DP weights), not implemented in this codebase.",
    "failure_replay": "requires the policy-retraining loop (replay failure trajectories into "
                       "training data, retrain DP weights), not implemented in this codebase.",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=["fixed_policy", "soe", "f2s", "success_only", "failure_replay"])
    parser.add_argument("--config", required=True, help="SOE DP policy config json")
    parser.add_argument("--task", default="Can")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num_episodes", type=int, default=30)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--noise_scale", type=float, default=2.0, help="for --method soe")
    parser.add_argument("--archive_path", default=None, help="for --method f2s: skill_archive.json from run_evolution.py")
    parser.add_argument("--cluster_model_path", default=None, help="for --method f2s: pickled sklearn cluster model + norm (see run_evolution.py)")
    args = parser.parse_args()

    if args.method in UNIMPLEMENTED_METHODS:
        raise NotImplementedError(
            f"--method {args.method}: {UNIMPLEMENTED_METHODS[args.method]} "
            "Refusing to report a fabricated number for an unimplemented method."
        )

    ensure_fresh_dir(args.output_dir)
    set_seed(args.seed)

    config_copy_path = os.path.join(args.output_dir, "config.yaml")
    with open(args.config, "r") as f_in, open(config_copy_path, "w") as f_out:
        f_out.write(f_in.read())
    with open(os.path.join(args.output_dir, "git_commit.txt"), "w") as f:
        f.write(git_commit_hash(os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))) + "\n")

    if args.method in ("fixed_policy", "soe"):
        from evaluate_policy import evaluate_policy  # noqa: E402  (sys.path set above)
        metrics = evaluate_policy(
            agent_ckpt=args.checkpoint,
            config_path=args.config,
            task=args.task,
            num_episodes=args.num_episodes,
            seed=args.seed,
            output_dir=args.output_dir,
            enable_exploration=(args.method == "soe"),
            noise_scale=(args.noise_scale if args.method == "soe" else None),
            method_name=args.method,
        )
    elif args.method == "f2s":
        assert args.archive_path is not None, "--method f2s requires --archive_path"
        import numpy as np
        import torch
        from easydict import EasyDict

        from f2s.evolution.loop import evaluate_with_skills
        from rollout_utils import dp_load

        with open(args.config, "r") as f:
            cfg = EasyDict(json.load(f))
        run_args = SimpleNamespace(
            agent=args.checkpoint, critic_agent=None, config=args.config, n_rollouts=args.num_episodes,
            horizon=None, env=None, render=False, render_traj=False, video_dir=None, video_skip=1,
            camera_names=["agentview"], dataset_path=None, dataset_obs=False, seed=args.seed, try_times=1,
            inference_horizon=None, high_noise_eval=False, eta=None, num_inference_steps=None,
            enable_exploration=False, tau1=None, tau2=None, noise_scale=None, enable_exploration_debug=False,
            disable_styles=False, enable_action_noise=False, action_noise_scale=None, enable_cfg=False,
            cfg_scale=0.5, cfg_agent=None, cfg_config=None, abs_action=False, return_intermediate=False,
        )
        policy, env, _, rollout_horizon = dp_load(run_args, cfg, enable_exploration_as_args=False)
        archive = SkillArchive.load(args.archive_path)

        cluster_model, cluster_norm = None, None
        if args.cluster_model_path is not None and os.path.exists(args.cluster_model_path):
            import pickle
            with open(args.cluster_model_path, "rb") as f:
                cluster_model, cluster_norm = pickle.load(f)

        metrics = evaluate_with_skills(
            policy, env, num_episodes=args.num_episodes, output_dir=args.output_dir,
            task=args.task, seed=args.seed, round_id=0, archive=archive,
            failure_cluster_model=cluster_model, failure_cluster_norm=cluster_norm,
            horizon=rollout_horizon, method_name="f2s",
        )
    else:
        raise AssertionError("unreachable")

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
