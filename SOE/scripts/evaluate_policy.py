"""Method-independent evaluation harness (proposal Day 7):

    evaluate_policy(policy_ckpt, config_path, task, num_episodes, seed, output_dir)

Runs the *actual* SOE simulator + policy code (robomimic/robosuite env,
RolloutDP wrapper, rollout() in SOE/simulation/rollout_utils.py) -- nothing
about the physics or the policy forward pass is reimplemented here -- and
logs every episode through f2s.logging.episode_logger.EpisodeLogger so the
exact same downstream code (failure extraction, metrics, world-model
dataset construction) works regardless of which method produced the
rollouts.

Usage:
    python scripts/evaluate_policy.py \
        --agent <ckpt.ckpt> --config <config.json> \
        --task Can --num_episodes 50 --seed 0 \
        --output_dir results/can/soe/seed_0/round_0/eval_after \
        [--enable_exploration --noise_scale 2.0]
"""
import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
SOE_SIMULATION_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "simulation"))
sys.path.insert(0, SOE_SIMULATION_ROOT)

from f2s.common.io import load_all_episode_metadata, save_json
from f2s.common.seeds import set_seed
from f2s.logging.episode_logger import EpisodeLogger
from f2s.logging.metrics import compute_round_metrics


def _default_run_args(**overrides):
    """Mirror simulation/run.py's argparse defaults exactly, as a
    SimpleNamespace, so we can call dp_load()/rollout() in-process without
    going through a subprocess + hdf5 round trip."""
    defaults = dict(
        agent=None,
        critic_agent=None,
        config=None,
        n_rollouts=10,
        horizon=None,
        env=None,
        render=False,
        render_traj=False,
        video_dir=None,
        video_skip=1,
        camera_names=["agentview"],
        dataset_path=None,
        dataset_obs=False,
        seed=None,
        try_times=1,
        inference_horizon=None,
        high_noise_eval=False,
        eta=None,
        num_inference_steps=None,
        enable_exploration=False,
        tau1=None,
        tau2=None,
        noise_scale=None,
        enable_exploration_debug=False,
        disable_styles=False,
        enable_action_noise=False,
        action_noise_scale=None,
        enable_cfg=False,
        cfg_scale=0.5,
        cfg_agent=None,
        cfg_config=None,
        abs_action=False,
        return_intermediate=False,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def evaluate_policy(
    agent_ckpt: str,
    config_path: str,
    task: str,
    num_episodes: int,
    seed: int,
    output_dir: str,
    round_id: int = 0,
    horizon=None,
    enable_exploration: bool = False,
    noise_scale=None,
    method_name: str = "SOE",
):
    """Reset env, run policy, record each step, detect success/failure,
    save the episode, compute metrics, save a summary JSON file."""
    import torch

    from rollout_utils import dp_load, rollout

    set_seed(seed)

    with open(config_path, "r") as f:
        cfg_raw = json.load(f)
    from easydict import EasyDict

    cfg = EasyDict(cfg_raw)

    args = _default_run_args(
        agent=agent_ckpt,
        config=config_path,
        n_rollouts=num_episodes,
        horizon=horizon,
        seed=seed,
        enable_exploration=enable_exploration,
        noise_scale=noise_scale,
    )

    policy, env, rollout_num_episodes, rollout_horizon = dp_load(
        args, cfg, enable_exploration_as_args=False
    )
    if enable_exploration:
        policy.enable_exploration_as_args(args, cfg)

    os.makedirs(output_dir, exist_ok=True)
    logger = EpisodeLogger(output_dir=output_dir, task=task, seed=seed, round_id=round_id)

    t_start = time.time()
    for i in range(num_episodes):
        stats, traj = rollout(
            policy=policy,
            env=env,
            horizon=rollout_horizon,
            render=False,
            video_dir=None,
            return_obs=True,
            camera_names=args.camera_names,
            initial_state_dict=None,
            traj_renderer=None,
            abs_action=False,
            rotation_transformer=None,
        )
        success = bool(stats["Success_Rate"])
        ep_len = int(stats["Horizon"])

        eid = logger.start_episode()
        for t in range(ep_len):
            obs_t = {k: traj["obs"][k][t] for k in traj["obs"]}
            logger.add_step(
                observation=obs_t,
                state=traj["states"][t],
                action=traj["actions"][t],
                reward=float(traj["rewards"][t]),
                done=bool(traj["dones"][t]),
            )
        if success:
            logger.finish_episode(success=True, failure_type="success", failure_time=None, failure_stage="none")
        else:
            # SOE's env terminates only on success or on hitting `horizon`
            # (see rollout_utils.rollout: `done = done or success`), so a
            # failed episode that used the full horizon is a timeout by
            # construction; failure-mode refinement (collision/object_drop/
            # etc.) happens later in f2s.failure.extractor using the
            # logged state trajectory.
            logger.finish_episode(
                success=False, failure_type="timeout", failure_time=ep_len - 1, failure_stage="unknown"
            )
        print(f"[{i + 1}/{num_episodes}] episode={eid} success={success} len={ep_len}")

    training_time = None  # evaluation, not training
    wall_time = time.time() - t_start

    episodes_dir = os.path.join(output_dir, "episodes")
    metas = load_all_episode_metadata(episodes_dir)
    metrics = compute_round_metrics(
        metas, task=task, method=method_name, seed=seed, round_id=round_id,
        training_time_seconds=training_time,
    )
    metrics["eval_wall_time_seconds"] = wall_time
    metrics["checkpoint"] = agent_ckpt
    metrics["config"] = config_path
    metrics["enable_exploration"] = enable_exploration
    save_json(os.path.join(output_dir, "metrics.json"), metrics)
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--num_episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--round_id", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--enable_exploration", action="store_true")
    parser.add_argument("--noise_scale", type=float, default=None)
    parser.add_argument("--method_name", type=str, default="SOE")
    args = parser.parse_args()

    evaluate_policy(
        agent_ckpt=args.agent,
        config_path=args.config,
        task=args.task,
        num_episodes=args.num_episodes,
        seed=args.seed,
        output_dir=args.output_dir,
        round_id=args.round_id,
        horizon=args.horizon,
        enable_exploration=args.enable_exploration,
        noise_scale=args.noise_scale,
        method_name=args.method_name,
    )
