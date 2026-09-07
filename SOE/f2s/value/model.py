"""Learned state-value function V(x_t) = P(this episode eventually
succeeds | x_t), trained on real outcomes (proposal-adjacent addition,
not in the original F2S design -- added specifically to fix the
diagnosed root cause of the world-model-ranking paradox: the world
model's own short-horizon (H=5) rollout has no way to see whether a
candidate correction eventually leads to success, only whether it looks
locally plausible 5 steps out. V is trained directly on whether the
*episode* succeeded, so it can supply exactly the long-horizon signal
short-horizon dynamics prediction structurally cannot.

Same 3-hidden-layer, ReLU MLP shape as f2s.world_model.model's
ResidualDynamicsMLP, for consistency -- the only architectural
difference is the output head (scalar logit instead of a state
residual)."""
import torch
import torch.nn as nn


class ValueFunction(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Returns raw logits, shape (batch,) -- use predict_proba for
        the actual [0, 1] success-probability estimate."""
        return self.net(state).squeeze(-1)

    def predict_proba(self, state: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(state))
