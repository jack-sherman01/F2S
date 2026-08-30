"""Lightweight action-conditioned world model (proposal Section 9.2):

    x_hat_{t+1} = x_t + f_phi(x_t, a_t)

3 hidden layers, hidden dim 256, ReLU -- matching the proposal's default
config exactly. An optional ensemble (E>1, independently initialized and
independently shuffled per training call) supports the uncertainty-based
safety filtering described in Section 10.1; the main experiments use E=1.
"""
from typing import List, Optional

import torch
import torch.nn as nn


class ResidualDynamicsMLP(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        delta = self.net(torch.cat([state, action], dim=-1))
        return state + delta


class WorldModelEnsemble(nn.Module):
    """E independently-initialized ResidualDynamicsMLPs. With E=1 this is
    exactly the deterministic model; with E>1, forward() returns the mean
    prediction and per-member predictions are available via
    forward_members() for the uncertainty term u(k) in Section 10.1."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, ensemble_size: int = 1):
        super().__init__()
        self.ensemble_size = ensemble_size
        self.members = nn.ModuleList([
            ResidualDynamicsMLP(state_dim, action_dim, hidden_dim) for _ in range(ensemble_size)
        ])

    def forward_members(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # (ensemble_size, batch, state_dim)
        return torch.stack([m(state, action) for m in self.members], dim=0)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        preds = self.forward_members(state, action)
        return preds.mean(dim=0)


@torch.no_grad()
def rollout_world_model(
    model: nn.Module,
    initial_state: torch.Tensor,
    action_sequence: torch.Tensor,
    horizon: int,
) -> torch.Tensor:
    """Autoregressive multi-step prediction (proposal Day 13.1).

    initial_state: (batch, state_dim)
    action_sequence: (horizon, batch, action_dim)
    returns: (horizon, batch, state_dim)
    """
    predicted_states = []
    current_state = initial_state
    for t in range(horizon):
        current_state = model(current_state, action_sequence[t])
        predicted_states.append(current_state)
    return torch.stack(predicted_states)


@torch.no_grad()
def rollout_world_model_with_uncertainty(
    model: WorldModelEnsemble,
    initial_state: torch.Tensor,
    action_sequence: torch.Tensor,
    horizon: int,
):
    """Same as rollout_world_model, but also returns per-step ensemble
    variance (proposal Section 10.1: u(k) = mean_t Var_e[x_hat_t^(e)])."""
    predicted_states = []
    predicted_vars = []
    current_states = initial_state.unsqueeze(0).repeat(model.ensemble_size, 1, 1)  # (E, batch, state_dim)
    for t in range(horizon):
        action_t = action_sequence[t].unsqueeze(0).repeat(model.ensemble_size, 1, 1)
        next_states = torch.stack([
            model.members[e](current_states[e], action_t[e]) for e in range(model.ensemble_size)
        ], dim=0)
        predicted_states.append(next_states.mean(dim=0))
        predicted_vars.append(next_states.var(dim=0).mean(dim=-1))  # scalar variance per batch element
        current_states = next_states
    return torch.stack(predicted_states), torch.stack(predicted_vars)
