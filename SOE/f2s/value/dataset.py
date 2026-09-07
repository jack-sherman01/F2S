"""Value-function training data: (state, outcome) pairs pooled across
every real episode this project has logged, labeled by whether *that
episode* ultimately succeeded -- a Monte-Carlo/outcome-supervision
target (every timestep in a successful episode is labeled 1.0, every
timestep in a failed one is labeled 0.0), not a discounted TD target.
This is the simplest defensible target that directly answers the
question the closed-loop controller needs at each step ("does this
state look like ones that led to eventual success"), and it requires no
new rollouts -- only real, already-collected trajectories from methods
across the whole project (fixed_policy, SOE, Failure Replay, F2S both
pre- and post-gating-fix, Unguided Latent Repair, the evolution loop,
and the gating diagnostic), each of which already carries a real
success/failure label per episode.

Known, documented limitation: some source episodes ran under altered
friction/mass (the "unseen configuration" evaluations) or forced
off-distribution object positions. The 26-dim world-model state
(f2s.world_model.state) does not encode friction/mass directly, so two
identical-looking states from different physical conditions can carry
different true outcomes -- an unavoidable aliasing given the current
state representation, not a bug in this dataset construction. Noted
here rather than silently ignored.

Train/val split is done *per source directory, at the episode level*
(never by timestep) -- the same discipline as
f2s.world_model.dataset.split_episodes_by_id, extended to pool many
directories: leaking a transition from one timestep of an episode into
val while other timesteps of the *same* episode are in train would let
the value function memorize episode identity rather than learn from
state features.
"""
import glob
import os
from typing import List, Tuple

import numpy as np

from f2s.common.io import load_json
from f2s.logging.episode_logger import load_episode
from f2s.world_model.state import STATE_DIM, build_world_model_states_for_episode

# Every Can-task episode directory logged across this project as of the
# value-function work (2026-09-07) -- deliberately excludes Lift (a
# different task, different goal/state semantics not handled by this
# single-task value function) and nothing else. See
# private/technical_contributions_log.md for how this list was compiled
# (a full `find results -type d -name episodes` scan, task-filtered).
ALL_CAN_EPISODE_DIRS = [
    "results/can/f2s_dev_cem/seed_2/round_0/eval/episodes",
    "results/can/f2s_dev_cem/seed_2/round_1/eval/episodes",
    "results/can/f2s_dev_cem/seed_2/round_2/eval/episodes",
    "results/can/f2s_dev/seed_0/round_0/eval/episodes",
    "results/can/f2s_dev/seed_0/round_1/eval/episodes",
    "results/can/f2s_dev/seed_0/round_2/eval/episodes",
    "results/can/f2s_evolution_postfix/seed_0/round_0/eval/episodes",
    "results/can/f2s_evolution_postfix/seed_0/round_1/eval/episodes",
    "results/can/f2s_evolution_postfix/seed_0/round_2/eval/episodes",
    "results/can/f2s_final/seed_0/round_0/eval/episodes",
    "results/can/f2s_final/seed_0/round_1/eval/episodes",
    "results/can/f2s_final/seed_0/round_2/eval/episodes",
    "results/Can/f2s_PRE_GATING_FIX/seed_0/round_0/episodes",
    "results/Can/f2s_PRE_GATING_FIX/seed_0/unseen/episodes",
    "results/Can/f2s_PRE_GATING_FIX/seed_1/round_0/episodes",
    "results/Can/f2s_PRE_GATING_FIX/seed_2/round_0/episodes",
    "results/Can/f2s/seed_0/round_0/episodes",
    "results/Can/f2s/seed_0/unseen/episodes",
    "results/Can/f2s/seed_1/round_0/episodes",
    "results/Can/f2s/seed_2/round_0/episodes",
    "results/Can/failure_replay/seed_0/round_0/episodes",
    "results/Can/failure_replay/seed_0/unseen/episodes",
    "results/Can/failure_replay/seed_1/round_0/episodes",
    "results/Can/failure_replay/seed_2/round_0/episodes",
    "results/Can/fixed_policy/seed_0/round_0/episodes",
    "results/Can/fixed_policy/seed_0/unseen/episodes",
    "results/Can/fixed_policy/seed_1/round_0/episodes",
    "results/Can/fixed_policy/seed_2/round_0/episodes",
    "results/can/skill_precondition_gate_diagnostic/seed_0/episodes",
    "results/can/skill_precondition_gate_diagnostic/seed_1/episodes",
    "results/can/skill_precondition_gate_diagnostic/seed_2/episodes",
    "results/Can/soe/seed_0/round_0/episodes",
    "results/Can/soe/seed_0/unseen/episodes",
    "results/Can/soe/seed_1/round_0/episodes",
    "results/Can/soe/seed_2/round_0/episodes",
    "results/Can/unguided_latent_repair/seed_0/round_0/episodes",
    "results/Can/unguided_latent_repair/seed_0/unseen/episodes",
    "results/Can/unguided_latent_repair/seed_1/round_0/episodes",
    "results/Can/unguided_latent_repair/seed_2/round_0/episodes",
]


def split_episodes_by_id(episodes_dir: str, train_ratio: float = 0.8, seed: int = 0):
    episode_ids = sorted(
        os.path.splitext(os.path.basename(p))[0]
        for p in glob.glob(os.path.join(episodes_dir, "episode_*.json"))
    )
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(episode_ids))
    n_train = int(round(train_ratio * len(episode_ids)))
    train_ids = [episode_ids[i] for i in perm[:n_train]]
    val_ids = [episode_ids[i] for i in perm[n_train:]]
    assert set(train_ids).isdisjoint(val_ids), "train/val episode split must be disjoint"
    return train_ids, val_ids


def build_value_examples(episodes_dir: str, episode_ids: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    states, labels = [], []
    for eid in episode_ids:
        meta_path = os.path.join(episodes_dir, f"{eid}.json")
        if not os.path.exists(meta_path):
            continue
        meta = load_json(meta_path)
        if meta["episode_length"] < 1:
            continue
        _, arrays = load_episode(episodes_dir, eid)
        x = build_world_model_states_for_episode(meta["obs_keys"], arrays)  # (T, state_dim)
        label = 1.0 if meta["success"] else 0.0
        states.append(x)
        labels.append(np.full(x.shape[0], label, dtype=np.float32))
    if len(states) == 0:
        return np.zeros((0, STATE_DIM), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    return np.concatenate(states), np.concatenate(labels)


def build_pooled_dataset(episode_dirs: List[str], train_ratio: float = 0.8, seed: int = 0):
    """Splits each directory's episodes into train/val independently
    (so every directory contributes to both splits, rather than e.g. all
    of one method landing in val by chance), then pools. Returns
    (train_states, train_labels, val_states, val_labels, stats) where
    stats records exactly how many episodes/states came from where, for
    a fully traceable dataset provenance record."""
    train_states_list, train_labels_list = [], []
    val_states_list, val_labels_list = [], []
    per_dir_stats = []

    for d in episode_dirs:
        if not os.path.isdir(d):
            per_dir_stats.append(dict(dir=d, found=False))
            continue
        train_ids, val_ids = split_episodes_by_id(d, train_ratio=train_ratio, seed=seed)
        tr_x, tr_y = build_value_examples(d, train_ids)
        va_x, va_y = build_value_examples(d, val_ids)
        train_states_list.append(tr_x)
        train_labels_list.append(tr_y)
        val_states_list.append(va_x)
        val_labels_list.append(va_y)
        per_dir_stats.append(dict(
            dir=d, found=True, n_episodes=len(train_ids) + len(val_ids),
            n_train_episodes=len(train_ids), n_val_episodes=len(val_ids),
            n_train_states=int(tr_x.shape[0]), n_val_states=int(va_x.shape[0]),
        ))

    train_states = np.concatenate(train_states_list) if train_states_list else np.zeros((0, STATE_DIM), dtype=np.float32)
    train_labels = np.concatenate(train_labels_list) if train_labels_list else np.zeros((0,), dtype=np.float32)
    val_states = np.concatenate(val_states_list) if val_states_list else np.zeros((0, STATE_DIM), dtype=np.float32)
    val_labels = np.concatenate(val_labels_list) if val_labels_list else np.zeros((0,), dtype=np.float32)

    return train_states, train_labels, val_states, val_labels, per_dir_stats


def compute_normalization_stats(x: np.ndarray, eps: float = 1e-6):
    mu = x.mean(axis=0)
    sigma = x.std(axis=0) + eps
    return mu, sigma
