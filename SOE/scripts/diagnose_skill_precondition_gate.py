"""Controlled diagnostic: is F2S's 0% (see README_F2S.md's three-seed
results / Day-25 sections) caused by the skill being retrieved in states
it was never meant for (a gating problem), or would it still fail even
when genuinely applicable (a skill-quality problem)? The two are
currently confounded -- retrieval fires on ~every stall regardless of
match quality, so "the skill never once succeeded when used" doesn't by
itself tell us which explanation is true.

This adds a REAL precondition gate -- not the k=1 always-match placeholder
used elsewhere -- and only allows retrieval when the current object (x, y)
position is within the exact tolerance Day-19 validation itself used
(f2s.candidates.validator.perturb_object_position_near's max_offset=0.03)
of the skill's own origin state (the real state it was discovered and
validated from). This is the "should apply" test as strict as the
evidence we actually have for the skill (Day-19 only validated a 3cm
neighborhood, so this gate is exactly as generous as the evidence
supports, not stricter).

Same in-distribution setup as the three-seed final results (same
checkpoint, same 30 episodes x seeds {0,1,2}), so success rates are
directly comparable to results/Can/f2s/seed_*/round_0/metrics.json.
"""
import json
import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "simulation")))

from easydict import EasyDict

from f2s.common.io import ensure_fresh_dir, git_commit_hash, load_all_episode_metadata, save_json
from f2s.common.seeds import set_seed
from f2s.logging.episode_logger import EpisodeLogger
from f2s.logging.metrics import compute_round_metrics
from f2s.skills.archive import SkillArchive

# (x, y) object position at the exact intervention state each skill was
# discovered and Day-19-validated from (see private/technical_contributions_log.md
# section 20 for how these were recovered: state_idx 25 -> episode_000040
# in results/can/f2s_final/seed_0/round_1/eval/episodes at t_f=89 (offset=10);
# state_idx 55 -> episode_000008 in results/can/f2s_dev_cem/seed_2/round_0/eval/episodes
# at t_f=73 (offset=25)).
SKILL_ORIGIN_XY = {
    "episode_000040_offset10_perstate_skill": np.array([0.21621729, 0.29207194]),
    "episode_000008_offset25_perstate_skill": np.array([0.17561927, 0.2146325]),
}
GATE_TOLERANCE = 0.03  # meters; exactly f2s.candidates.validator.perturb_object_position_near's max_offset

STALL_WINDOW = 20
STALL_PROGRESS_EPS = 0.01


def rollout_gated(policy, env, horizon: int, archive):
    """Same stall-detection + retrieval loop as
    f2s.evolution.loop.rollout_with_skills, except retrieval additionally
    requires the current object (x, y) to be within GATE_TOLERANCE of one
    of the archive's skills' real origin state -- i.e. the skill is only
    used where the actual evidence (Day-19 validation) says it should
    apply."""
    import robomimic.utils.obs_utils as ObsUtils
    from robomimic.utils import tensor_utils as TensorUtils

    from f2s.failure.features import compute_failure_feature_vector

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
    n_stalls = 0
    n_gate_blocked = 0

    total_reward = 0.0
    success = False
    step_i = 0
    for step_i in range(horizon):
        progress = compute_failure_feature_vector([ObsUtils.unprocess_obs_dict(obs)])[-1]
        progress_history.append(progress)
        stalled = (
            len(progress_history) > STALL_WINDOW
            and (progress_history[-1] - progress_history[-STALL_WINDOW]) < STALL_PROGRESS_EPS
        )

        if skill_playback_remaining == 0 and stalled:
            n_stalls += 1
            obs_np = ObsUtils.unprocess_obs_dict(obs)
            current_xy = np.asarray(obs_np["object"])[0:2]

            best_skill, best_dist = None, float("inf")
            for skill in archive.skills:
                origin_xy = SKILL_ORIGIN_XY.get(skill.skill_id)
                if origin_xy is None:
                    continue
                dist = float(np.linalg.norm(current_xy - origin_xy))
                if dist < GATE_TOLERANCE and dist < best_dist:
                    best_skill, best_dist = skill, dist

            if best_skill is not None:
                skill_playback_chunk = np.asarray(best_skill.action_chunk)
                skill_playback_remaining = skill_playback_chunk.shape[0]
                skill_playback_t = 0
                skills_used.append(dict(skill_id=best_skill.skill_id, dist=best_dist, step=step_i))
            else:
                n_gate_blocked += 1

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

    stats = dict(
        Return=total_reward, Horizon=step_i + 1, Success_Rate=float(success),
        Skills_Used=skills_used, N_Stalls=n_stalls, N_Gate_Blocked=n_gate_blocked,
    )
    traj["obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["obs"])
    traj["next_obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["next_obs"])
    for k in ["actions", "rewards", "dones", "states"]:
        traj[k] = np.array(traj[k])
    for k in ["obs", "next_obs"]:
        for kp in traj[k]:
            traj[k][kp] = np.array(traj[k][kp])
    return stats, traj


def main():
    config_path = "configs/soe_can_lowdim_baseline.json"
    ckpt_path = os.environ["F2S_CAN_CKPT"]
    archive_path = "results/can/candidate_ranking_per_state_offset_sweep/skill_archive.json"
    num_episodes = int(os.environ.get("F2S_GATE_NUM_EPISODES", "30"))
    seeds = [int(x) for x in os.environ.get("F2S_GATE_SEEDS", "0,1,2").split(",")]

    from rollout_utils import dp_load

    with open(config_path, "r") as f:
        cfg = EasyDict(json.load(f))
    archive = SkillArchive.load(archive_path)

    all_seed_summaries = {}
    for seed in seeds:
        output_dir = f"results/can/skill_precondition_gate_diagnostic/seed_{seed}"
        ensure_fresh_dir(output_dir)
        set_seed(seed)

        run_args = SimpleNamespace(
            agent=ckpt_path, critic_agent=None, config=config_path, n_rollouts=num_episodes,
            horizon=None, env=None, render=False, render_traj=False, video_dir=None, video_skip=1,
            camera_names=["agentview"], dataset_path=None, dataset_obs=False, seed=seed, try_times=1,
            inference_horizon=None, high_noise_eval=False, eta=None, num_inference_steps=None,
            enable_exploration=False, tau1=None, tau2=None, noise_scale=None, enable_exploration_debug=False,
            disable_styles=False, enable_action_noise=False, action_noise_scale=None, enable_cfg=False,
            cfg_scale=0.5, cfg_agent=None, cfg_config=None, abs_action=False, return_intermediate=False,
        )
        policy, env, _, rollout_horizon = dp_load(run_args, cfg, enable_exploration_as_args=False)

        logger = EpisodeLogger(output_dir=output_dir, task="Can", seed=seed, round_id=0)
        total_stalls, total_gate_blocked, total_gate_passed = 0, 0, 0
        gate_used_by_episode = {}
        for i in range(num_episodes):
            stats, traj = rollout_gated(policy, env, rollout_horizon, archive)
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
            total_stalls += stats["N_Stalls"]
            total_gate_blocked += stats["N_Gate_Blocked"]
            total_gate_passed += len(stats["Skills_Used"])
            gate_used_by_episode[eid] = dict(success=success, skills_used=stats["Skills_Used"])
            print(f"[gated seed={seed} {i + 1}/{num_episodes}] success={success} len={ep_len} "
                  f"stalls={stats['N_Stalls']} gate_passed={len(stats['Skills_Used'])} "
                  f"skills_used={stats['Skills_Used']}")

        save_json(os.path.join(output_dir, "gate_used_by_episode.json"), gate_used_by_episode)
        n_episodes_gate_passed = sum(1 for v in gate_used_by_episode.values() if len(v["skills_used"]) > 0)
        n_episodes_gate_passed_success = sum(
            1 for v in gate_used_by_episode.values() if len(v["skills_used"]) > 0 and v["success"]
        )

        metas = load_all_episode_metadata(os.path.join(output_dir, "episodes"))
        metrics = compute_round_metrics(metas, task="Can", method="f2s_gated", seed=seed, round_id=0)
        metrics["total_stalls"] = total_stalls
        metrics["total_gate_blocked"] = total_gate_blocked
        metrics["total_gate_passed_invocations"] = total_gate_passed
        metrics["gate_tolerance_m"] = GATE_TOLERANCE
        metrics["n_episodes_with_gate_pass"] = n_episodes_gate_passed
        metrics["n_episodes_with_gate_pass_and_success"] = n_episodes_gate_passed_success
        save_json(os.path.join(output_dir, "metrics.json"), metrics)
        all_seed_summaries[seed] = metrics
        print(json.dumps(metrics, indent=2))

    save_json("results/can/skill_precondition_gate_diagnostic/summary.json", all_seed_summaries)
    with open("results/can/skill_precondition_gate_diagnostic/git_commit.txt", "w") as f:
        f.write(git_commit_hash(os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))) + "\n")


if __name__ == "__main__":
    main()
