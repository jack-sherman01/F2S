"""Multi-step world-model evaluation (proposal Day 13): autoregressive
rollout error as a function of horizon h in {1,3,5}, checked against the
same val transitions used during training."""
from typing import Dict, List

import numpy as np
import torch

from f2s.world_model.model import rollout_world_model


def build_multistep_windows(states: np.ndarray, actions: np.ndarray, episode_boundaries: List[int], horizon: int):
    """From per-episode-concatenated (states, actions) arrays plus the
    index (in the concatenated array) where each episode starts, build
    windows of length `horizon` that do not cross an episode boundary:
    (x_t, a_t:t+h, x_{t+1:t+h+1})."""
    boundaries = sorted(episode_boundaries) + [states.shape[0]]
    x0_list, a_list, xtruth_list = [], [], []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        for t in range(start, end - horizon):
            x0_list.append(states[t])
            a_list.append(actions[t:t + horizon])
            xtruth_list.append(states[t + 1:t + horizon + 1])
    if len(x0_list) == 0:
        return None, None, None
    return np.stack(x0_list), np.stack(a_list), np.stack(xtruth_list)


@torch.no_grad()
def evaluate_multistep(model, x0: np.ndarray, actions: np.ndarray, x_truth: np.ndarray, device: str, horizons=(1, 3, 5)):
    """Returns {h: mse} for each h in horizons (h <= actions.shape[1])."""
    model.eval()
    x0_t = torch.from_numpy(x0).float().to(device)
    a_t = torch.from_numpy(actions).float().to(device).permute(1, 0, 2)  # (H, N, action_dim)
    x_truth_t = torch.from_numpy(x_truth).float().to(device).permute(1, 0, 2)  # (H, N, state_dim)

    max_h = actions.shape[1]
    pred = rollout_world_model(model, x0_t, a_t, horizon=max_h)  # (H, N, state_dim)

    results: Dict[int, float] = {}
    for h in horizons:
        if h > max_h:
            continue
        mse = torch.mean(torch.sum((pred[h - 1] - x_truth_t[h - 1]) ** 2, dim=-1)).item()
        results[h] = float(mse)
    return results
