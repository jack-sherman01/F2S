"""Method-independent metric computation over episode metadata dicts
(as produced by f2s.logging.episode_logger.EpisodeLogger / loaded via
f2s.common.io.load_all_episode_metadata).

Every function takes a list[dict] of episode metadata (or, where noted,
the corresponding per-episode arrays) and returns a plain float, so the
same code path evaluates SOE, Success-only, Failure Replay, and F2S.
"""

from typing import Any, Dict, List, Optional

import numpy as np


def _rate(numerator: int, denominator: int) -> Optional[float]:
    if denominator == 0:
        return None
    return numerator / denominator


def compute_success_rate(episodes: List[Dict[str, Any]]) -> Optional[float]:
    if len(episodes) == 0:
        return None
    return _rate(sum(1 for e in episodes if e["success"]), len(episodes))


def compute_failure_rate(episodes: List[Dict[str, Any]]) -> Optional[float]:
    sr = compute_success_rate(episodes)
    return None if sr is None else 1.0 - sr


def compute_recovery_rate(episodes: List[Dict[str, Any]]) -> Optional[float]:
    """Recovery rate is defined only over episodes that contain a failure
    signal partway through and then still recover (success == True despite
    a non-null failure_time). Per the proposal: denominator is the number of
    "failure episodes" (episodes with a detected failure event), numerator
    is how many of those still end successfully.
    """
    failure_episodes = [e for e in episodes if e.get("failure_time") is not None]
    if len(failure_episodes) == 0:
        return None
    recovered = sum(1 for e in failure_episodes if e["success"])
    return _rate(recovered, len(failure_episodes))


def compute_episode_length(episodes: List[Dict[str, Any]]) -> Optional[float]:
    if len(episodes) == 0:
        return None
    return float(np.mean([e["episode_length"] for e in episodes]))


def compute_rollout_count(episodes: List[Dict[str, Any]]) -> int:
    return len(episodes)


def compute_collision_rate(episodes: List[Dict[str, Any]]) -> Optional[float]:
    if len(episodes) == 0:
        return None
    n_collision = sum(1 for e in episodes if e.get("failure_type") == "collision")
    return _rate(n_collision, len(episodes))


def compute_object_drop_rate(episodes: List[Dict[str, Any]]) -> Optional[float]:
    if len(episodes) == 0:
        return None
    n_drop = sum(1 for e in episodes if e.get("failure_type") == "object_drop")
    return _rate(n_drop, len(episodes))


def compute_rollouts_to_threshold(
    running_success_rates: List[float], thresholds=(0.5, 0.7, 0.9)
) -> Dict[str, Optional[int]]:
    """running_success_rates[n] = success rate using the first n rollouts.
    Returns, for each threshold tau, the smallest n with rate >= tau, or
    None if the target is never reached (never invented -- proposal Day 5.3)."""
    out = {}
    for tau in thresholds:
        n_tau = None
        for n, rate in enumerate(running_success_rates):
            if rate >= tau:
                n_tau = n
                break
        out[f"rollouts_to_{int(tau * 100)}_percent"] = n_tau
    return out


def compute_round_metrics(
    episodes: List[Dict[str, Any]],
    task: str,
    method: str,
    seed: int,
    round_id: int,
    training_time_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    return dict(
        task=task,
        method=method,
        seed=seed,
        round=round_id,
        success_rate=compute_success_rate(episodes),
        failure_rate=compute_failure_rate(episodes),
        recovery_rate=compute_recovery_rate(episodes),
        collision_rate=compute_collision_rate(episodes),
        object_drop_rate=compute_object_drop_rate(episodes),
        mean_episode_length=compute_episode_length(episodes),
        rollout_count=compute_rollout_count(episodes),
        training_time_seconds=training_time_seconds,
    )
