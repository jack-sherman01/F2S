"""Day 4-5 acceptance test: exercise EpisodeLogger + metrics.py end to end
on a small synthetic dataset with hand-computed expected values, so the
core logging/metrics layer is verified before any real method is built on
top of it.

Usage: python scripts/selftest_logging.py
"""
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from f2s.common.io import load_all_episode_metadata
from f2s.logging.episode_logger import EpisodeLogger, load_episode
from f2s.logging.metrics import (
    compute_collision_rate,
    compute_episode_length,
    compute_failure_rate,
    compute_object_drop_rate,
    compute_recovery_rate,
    compute_rollout_count,
    compute_round_metrics,
    compute_rollouts_to_threshold,
    compute_success_rate,
)


def make_synthetic_episode(logger, length, success, failure_type, failure_time, failure_stage):
    logger.start_episode()
    for t in range(length):
        obs = {
            "object": np.full((14,), t, dtype=np.float32),
            "robot0_eef_pos": np.full((3,), t, dtype=np.float32),
        }
        state = np.full((10,), t, dtype=np.float32)
        action = np.full((7,), 0.1, dtype=np.float32)
        logger.add_step(obs, state, action, reward=float(t == length - 1 and success), done=(t == length - 1))
    return logger.finish_episode(success, failure_type, failure_time, failure_stage)


def main():
    tmp_dir = "/tmp/f2s_selftest_logging"
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir)

    logger = EpisodeLogger(output_dir=tmp_dir, task="Can", seed=0, round_id=0)

    # 5 episodes: 2 success (no failure event), 1 success-after-failure
    # (recovered), 2 failures (collision, object_drop).
    make_synthetic_episode(logger, length=50, success=True, failure_type="success", failure_time=None, failure_stage="none")
    make_synthetic_episode(logger, length=60, success=True, failure_type="success", failure_time=None, failure_stage="none")
    make_synthetic_episode(logger, length=80, success=True, failure_type="grasp_failure", failure_time=20, failure_stage="grasp")
    make_synthetic_episode(logger, length=40, success=False, failure_type="collision", failure_time=15, failure_stage="approach")
    make_synthetic_episode(logger, length=100, success=False, failure_type="object_drop", failure_time=55, failure_stage="transport")

    # --- replay check: load episode back and verify array/metadata agreement ---
    episodes_dir = os.path.join(tmp_dir, "episodes")
    metas = load_all_episode_metadata(episodes_dir)
    assert len(metas) == 5, f"expected 5 episode metadata files, got {len(metas)}"
    for meta in metas:
        _, arrays = load_episode(episodes_dir, meta["episode_id"])
        assert arrays["actions"].shape[0] == meta["episode_length"], "actions length mismatch"
        assert arrays["states"].shape[0] == meta["episode_length"], "states length mismatch"
        assert arrays["rewards"].shape[0] == meta["episode_length"], "rewards length mismatch"
        assert arrays["dones"].shape[0] == meta["episode_length"], "dones length mismatch"
        assert arrays["obs_object"].shape[0] == meta["episode_length"], "obs length mismatch"
    print("Replay check passed: all 5 episodes reloaded with consistent shapes.")

    # --- metrics check against hand-computed expected values ---
    success_rate = compute_success_rate(metas)
    failure_rate = compute_failure_rate(metas)
    recovery_rate = compute_recovery_rate(metas)
    mean_len = compute_episode_length(metas)
    rollout_count = compute_rollout_count(metas)
    collision_rate = compute_collision_rate(metas)
    drop_rate = compute_object_drop_rate(metas)

    assert success_rate == 3 / 5, f"success_rate expected 0.6, got {success_rate}"
    assert failure_rate == 2 / 5, f"failure_rate expected 0.4, got {failure_rate}"
    # failure episodes (failure_time is not None): episodes 3,4,5 -> 3 total;
    # of those, only episode 3 (grasp_failure, recovered) is success == True.
    assert recovery_rate == 1 / 3, f"recovery_rate expected 1/3, got {recovery_rate}"
    assert mean_len == (50 + 60 + 80 + 40 + 100) / 5, f"mean_len wrong: {mean_len}"
    assert rollout_count == 5
    assert collision_rate == 1 / 5, f"collision_rate expected 0.2, got {collision_rate}"
    assert drop_rate == 1 / 5, f"object_drop_rate expected 0.2, got {drop_rate}"
    print("Metrics check passed: success_rate=%.3f failure_rate=%.3f recovery_rate=%.3f "
          "mean_len=%.1f collision_rate=%.3f object_drop_rate=%.3f" % (
              success_rate, failure_rate, recovery_rate, mean_len, collision_rate, drop_rate
          ))

    round_metrics = compute_round_metrics(metas, task="Can", method="selftest", seed=0, round_id=0)
    assert round_metrics["success_rate"] == success_rate
    print("compute_round_metrics output:", round_metrics)

    # --- budget-based metrics: never invent a value if target unreached ---
    running_rates = [0.0, 0.5, 0.5, 2 / 3, 0.75]
    budget = compute_rollouts_to_threshold(running_rates, thresholds=(0.5, 0.7, 0.9))
    assert budget["rollouts_to_50_percent"] == 1
    assert budget["rollouts_to_70_percent"] == 4
    assert budget["rollouts_to_90_percent"] is None, "90% target never reached, must stay None"
    print("Budget-based metrics check passed:", budget)

    # --- "never overwrite an experiment directory" check ---
    from f2s.common.io import ensure_fresh_dir

    fresh_dir = os.path.join(tmp_dir, "round_0")
    ensure_fresh_dir(fresh_dir)
    try:
        ensure_fresh_dir(episodes_dir)  # non-empty -> must raise
        raise AssertionError("ensure_fresh_dir should have refused a non-empty directory")
    except FileExistsError:
        print("ensure_fresh_dir correctly refused to reuse a non-empty directory.")

    shutil.rmtree(tmp_dir)
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
