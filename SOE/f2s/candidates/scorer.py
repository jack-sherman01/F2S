"""Candidate scoring via world-model prediction (proposal Section 9.3 /
Day 16). Uses the initial score J(k) = S_hat(k) - R(k) (Day 16.2 -- novelty
and transfer terms are deliberately not added until this base scorer is
verified, per the proposal's own instruction)."""
from typing import Dict, Tuple

import numpy as np
import torch

from f2s.failure.features import DEFAULT_GOAL_POS
from f2s.safety.filter import (
    P_OBJ_SLICE,
    collision_detected,
    joint_limit_exceeded,
    object_drop_predicted,
    velocity_limit_exceeded,
)
from f2s.world_model.model import rollout_world_model

EPS_P = 0.05  # meters; SuccessCondition position threshold (Section 9.3)


def compute_predicted_success(predicted_state_sequence: np.ndarray, goal_pos: np.ndarray = DEFAULT_GOAL_POS) -> float:
    """S_hat(k) = I[||p_obj_hat(T) - p_goal|| < eps_p] (position-only
    SuccessCondition, Section 9.3 -- Can has no orientation requirement)."""
    final_obj = predicted_state_sequence[-1, P_OBJ_SLICE]
    return float(np.linalg.norm(final_obj - goal_pos) < EPS_P)


def compute_predicted_risk(predicted_state_sequence: np.ndarray, action_sequence: np.ndarray) -> float:
    """R(k) = sum of the same hard-constraint indicators used by the
    safety filter (Section 10.1), unweighted (lambda_* = 1) for this base
    scorer."""
    r = 0.0
    r += float(collision_detected(predicted_state_sequence))
    r += float(velocity_limit_exceeded(predicted_state_sequence))
    r += float(joint_limit_exceeded(predicted_state_sequence))
    r += float(object_drop_predicted(predicted_state_sequence))
    return r


def score_candidate(predicted_state_sequence: np.ndarray, action_sequence: np.ndarray) -> Tuple[float, float, float]:
    predicted_success = compute_predicted_success(predicted_state_sequence)
    predicted_risk = compute_predicted_risk(predicted_state_sequence, action_sequence)
    score = predicted_success - predicted_risk
    return score, predicted_success, predicted_risk


@torch.no_grad()
def rank_candidate(world_model: torch.nn.Module, x0: np.ndarray, action_chunk: np.ndarray, horizon_wm: int, device: str) -> Dict:
    """Roll the world model forward `horizon_wm` steps from x0 under the
    first `horizon_wm` actions of `action_chunk`, then score the result."""
    h = min(horizon_wm, action_chunk.shape[0])
    x0_t = torch.from_numpy(x0).float().unsqueeze(0).to(device)  # (1, state_dim)
    a_t = torch.from_numpy(action_chunk[:h]).float().unsqueeze(1).to(device)  # (h, 1, action_dim)
    pred = rollout_world_model(world_model, x0_t, a_t, horizon=h)  # (h, 1, state_dim)
    pred_np = pred.squeeze(1).cpu().numpy()

    score, predicted_success, predicted_risk = score_candidate(pred_np, action_chunk[:h])
    return dict(
        predicted_states=pred_np,
        score=score,
        predicted_success=predicted_success,
        predicted_risk=predicted_risk,
    )
