"""Candidate scoring via world-model prediction (proposal Section 9.3 /
Day 16). Uses the initial score J(k) = S_hat(k) - R(k) (Day 16.2 -- novelty
and transfer terms are deliberately not added until this base scorer is
verified, per the proposal's own instruction, unchanged in
proposal_revised.tex's Day 16.2).

Also implements the recovery-value term V_repair(k) added in
proposal_revised.tex Section 9.4 ("Predicting Counterfactual Recovery
Value"): a dense, continuous signal (goal-error improvement + task-
progress improvement - risk) rather than the binary S_hat(k), motivated
by exactly the failure mode this project already diagnosed the hard way
(SOE/README_F2S.md's scoring-formula section: S_hat is 0 for nearly every
candidate on hard states, giving CEM/ranking nothing to climb; the
project's own CEM fitness and several offset-sweep scripts already used
"predicted distance to goal" ad hoc as a workaround -- V_repair formalizes
that into the real scorer, as an additional field alongside the existing,
already-verified base score rather than replacing it)."""
from typing import Dict, Tuple

import numpy as np
import torch

from f2s.failure.features import DEFAULT_GOAL_POS
from f2s.safety.filter import (
    P_GOAL_SLICE,
    P_OBJ_SLICE,
    collision_detected,
    joint_limit_exceeded,
    object_drop_predicted,
    velocity_limit_exceeded,
)
from f2s.world_model.model import rollout_world_model

EPS_P = 0.05  # meters; SuccessCondition position threshold (Section 9.3)

# Recovery-value weights (proposal_revised.tex Section 9.4). Not tuned --
# beta_r intentionally kept small relative to beta_e/beta_p since risk is
# already a hard reject via the safety filter (Section 10.1); this term
# only needs to break ties among already-safety-passing candidates.
BETA_E = 1.0
BETA_P = 1.0
BETA_R = 0.2


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


def compute_goal_error(x: np.ndarray) -> float:
    """e_goal(x) = ||p_obj(x) - p_goal(x)||, reading both from the state
    vector itself (not the DEFAULT_GOAL_POS constant) so this stays
    correct for any task's goal position, not just Can's fixed one."""
    return float(np.linalg.norm(x[P_OBJ_SLICE] - x[P_GOAL_SLICE]))


def compute_task_progress(x: np.ndarray) -> float:
    """r(x): the world-model state's own task_progress component (last
    element of the 26-dim x_t layout, see f2s/world_model/state.py)."""
    return float(x[-1])


def compute_predicted_recovery_value(
    x0: np.ndarray,
    predicted_final_state: np.ndarray,
    predicted_risk: float,
    beta_e: float = BETA_E,
    beta_p: float = BETA_P,
    beta_r: float = BETA_R,
) -> float:
    """V_repair(k) = beta_e*(e_goal(x_t) - e_goal(x_hat_{t+H})) +
    beta_p*(r(x_hat_{t+H}) - r(x_t)) - beta_r*R(k) (proposal_revised.tex
    Section 9.4). Positive = predicted to move closer to the goal and/or
    make more task progress than the failure state it was generated from,
    net of predicted risk."""
    delta_goal_error = compute_goal_error(x0) - compute_goal_error(predicted_final_state)
    delta_progress = compute_task_progress(predicted_final_state) - compute_task_progress(x0)
    return float(beta_e * delta_goal_error + beta_p * delta_progress - beta_r * predicted_risk)


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
    predicted_recovery_value = compute_predicted_recovery_value(x0, pred_np[-1], predicted_risk)
    return dict(
        predicted_states=pred_np,
        score=score,
        predicted_success=predicted_success,
        predicted_risk=predicted_risk,
        predicted_recovery_value=predicted_recovery_value,
    )
