"""World-model transition dataset construction (proposal Day 11): builds
(x_t, a_t, x_{t+1}) triples from logged episodes, split by *episode* (not
by timestep) into train/val so no episode leaks across the split.
"""
import glob
import os
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from f2s.common.io import load_json
from f2s.logging.episode_logger import load_episode
from f2s.world_model.state import STATE_DIM, build_world_model_states_for_episode


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


def build_transitions(episodes_dir: str, episode_ids) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    states, actions, next_states = [], [], []
    for eid in episode_ids:
        meta = load_json(os.path.join(episodes_dir, f"{eid}.json"))
        if meta["episode_length"] < 2:
            continue
        _, arrays = load_episode(episodes_dir, eid)
        obs_keys = meta["obs_keys"]
        x = build_world_model_states_for_episode(obs_keys, arrays)  # (T, state_dim)
        a = arrays["actions"]  # (T, action_dim)
        states.append(x[:-1])
        actions.append(a[:-1])
        next_states.append(x[1:])

    if len(states) == 0:
        return (
            np.zeros((0, STATE_DIM), dtype=np.float32),
            np.zeros((0, 0), dtype=np.float32),
            np.zeros((0, STATE_DIM), dtype=np.float32),
        )
    return np.concatenate(states), np.concatenate(actions), np.concatenate(next_states)


def compute_normalization_stats(x: np.ndarray, eps: float = 1e-6):
    mu = x.mean(axis=0)
    sigma = x.std(axis=0) + eps
    return mu, sigma


class TransitionDataset(Dataset):
    def __init__(self, states: np.ndarray, actions: np.ndarray, next_states: np.ndarray):
        assert states.shape[0] == actions.shape[0] == next_states.shape[0]
        self.states = torch.from_numpy(states).float()
        self.actions = torch.from_numpy(actions).float()
        self.next_states = torch.from_numpy(next_states).float()

    def __len__(self):
        return self.states.shape[0]

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx], self.next_states[idx]
