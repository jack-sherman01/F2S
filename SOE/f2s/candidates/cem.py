"""Guided candidate search via the Cross-Entropy Method (CEM), replacing
pure random (Gaussian / single-dimension) perturbation of the latent
readout vector with an iterative search guided by the world model.

Motivation (see SOE/README_F2S.md, "a key finding"): pure random
perturbation at M up to 64 found 0/320 successful candidates across real
failure states. Two problems with that approach:

1. It never adapts -- every candidate is drawn from the same fixed-width
   Gaussian / axis-aligned single-dimension distribution regardless of
   which region of latent space the world model thinks is promising.
2. `compute_predicted_success` (f2s.candidates.scorer) is a *binary*
   position-threshold indicator. If nothing in the initial population
   is predicted to succeed -- the common case here -- every candidate
   scores identically and there is no gradient to climb.

This module fixes both: CEM iteratively refines a Gaussian search
distribution over Delta z using the world model's *continuous* predicted
final object-to-goal distance as the fitness signal (informative even
when literally 0 candidates are predicted to succeed), while still
reporting the existing binary predicted_success/predicted_risk (Section
9.3 / 10.1) for every candidate so the rest of the pipeline (safety
filtering, archiving, logging) is unchanged.
"""
from typing import Any, Dict, List

import numpy as np
import torch

from f2s.candidates.generator import get_latent
from f2s.candidates.scorer import compute_predicted_risk, compute_predicted_success
from f2s.failure.features import DEFAULT_GOAL_POS
from f2s.safety.filter import P_OBJ_SLICE
from f2s.world_model.model import rollout_world_model


def cem_search(
    dp_module: torch.nn.Module,
    world_model: torch.nn.Module,
    obs_dict: Dict[str, torch.Tensor],
    x0: np.ndarray,
    source_episode_id: str,
    failure_mode_id: int,
    device: str,
    population_size: int = 64,
    n_iters: int = 5,
    elite_frac: float = 0.25,
    sigma_init: float = 0.5,
    sigma_min: float = 0.05,
    horizon_wm: int = 5,
    risk_weight: float = 0.2,
    seed: int = 0,
) -> List[Dict[str, Any]]:
    """Search Delta z around the current failure state's readout vector to
    minimize the world model's predicted final object-to-goal distance.
    Returns every candidate evaluated across all iterations, sorted best
    (lowest CEM fitness, i.e. most promising) first -- the caller (e.g.
    f2s.evolution.loop.discover_and_archive_skills) is responsible for
    safety-filtering and deciding how many of the top candidates to
    actually execute in the real simulator, exactly as it already does
    for f2s.candidates.generator.generate_candidates().
    """
    torch.manual_seed(seed)
    z_f = get_latent(dp_module, obs_dict)  # (1, D)
    D = z_f.shape[-1]
    device_t = torch.device(device)

    mean = torch.zeros(D, device=device_t)
    std = torch.full((D,), sigma_init, device=device_t)
    n_elite = max(1, int(round(population_size * elite_frac)))

    x0_t = torch.from_numpy(x0).float().to(device_t)  # (state_dim,)
    goal_pos_t = torch.from_numpy(DEFAULT_GOAL_POS.astype(np.float32)).to(device_t)
    obj_slice = P_OBJ_SLICE

    all_candidates: List[Dict[str, Any]] = []

    for it in range(n_iters):
        deltas = mean.unsqueeze(0) + std.unsqueeze(0) * torch.randn(population_size, D, device=device_t)
        z_batch = z_f.repeat(population_size, 1) + deltas  # (K, D)

        with torch.no_grad():
            action_chunks = dp_module.action_decoder.predict_action(z_batch)  # (K, T, action_dim)

        h = min(horizon_wm, action_chunks.shape[1])
        x0_batch = x0_t.unsqueeze(0).repeat(population_size, 1)
        a_batch = action_chunks[:, :h, :].permute(1, 0, 2)  # (h, K, action_dim)
        with torch.no_grad():
            pred_states = rollout_world_model(world_model, x0_batch, a_batch, horizon=h)  # (h, K, state_dim)

        # continuous CEM fitness: predicted final object-to-goal distance
        # (lower is better), plus a soft risk penalty from the same
        # hard-constraint indicators the safety filter uses.
        final_obj_pred = pred_states[-1, :, obj_slice]  # (K, 3)
        dist_to_goal = torch.linalg.norm(final_obj_pred - goal_pos_t.unsqueeze(0), dim=-1)  # (K,)

        pred_states_np = pred_states.permute(1, 0, 2).cpu().numpy()  # (K, h, state_dim)
        action_chunks_np = action_chunks.detach().cpu().numpy()
        dist_np = dist_to_goal.cpu().numpy()

        risk_np = np.zeros(population_size, dtype=np.float32)
        success_np = np.zeros(population_size, dtype=np.float32)
        for k in range(population_size):
            risk_np[k] = compute_predicted_risk(pred_states_np[k], action_chunks_np[k, :h])
            success_np[k] = compute_predicted_success(pred_states_np[k])

        fitness = dist_np + risk_weight * risk_np  # lower is better

        elite_idx = np.argsort(fitness)[:n_elite]
        elite_deltas = deltas[torch.as_tensor(elite_idx, device=device_t)]
        mean = elite_deltas.mean(dim=0)
        std = elite_deltas.std(dim=0).clamp(min=sigma_min)

        for k in range(population_size):
            valid = bool(np.isfinite(action_chunks_np[k]).all())
            all_candidates.append(dict(
                candidate_id=f"{source_episode_id}_cem{it}_{k:03d}",
                source_episode_id=source_episode_id,
                failure_mode_id=failure_mode_id,
                kind="cem",
                cem_iteration=it,
                latent_delta=deltas[k].cpu().numpy(),
                action_chunk=action_chunks_np[k],
                valid=valid,
                predicted_states=pred_states_np[k],
                predicted_dist_to_goal=float(dist_np[k]),
                predicted_risk=float(risk_np[k]),
                predicted_success=float(success_np[k]),
                score=float(success_np[k] - risk_np[k]),  # same score convention as scorer.score_candidate
                cem_fitness=float(fitness[k]),
            ))

    all_candidates.sort(key=lambda c: c["cem_fitness"])
    return all_candidates
