"""Counterfactual candidate-skill generation in SOE's latent space
(proposal Section 8 / Day 15).

SOE's actual DP policy (src/policy/dp.py) does not implement a stochastic
(mu, sigma) VAE-style encoder -- for the low-dim observation setting used
here, `img_encoder(obs_dict)` (a.k.a. `DP.get_latent_action`) is a
deterministic concatenation of the raw low-dim features (q_phi in the
proposal's notation collapses to the identity, since our config sets
`readout_dim=None` so DP's optional bottleneck MLP is unused). We treat
this deterministic readout vector as the "latent" z_f the proposal
perturbs, and decode candidates with SOE's own, unmodified diffusion
action decoder (`DiffusionUNetPolicy.predict_action`) -- no part of the
diffusion sampling process is reimplemented here.
"""
from typing import Any, Dict, List, Optional

import numpy as np
import torch


def get_latent(dp_module: torch.nn.Module, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    """z_f = E_psi(o_t) (proposal Section 6.2), realized as SOE's own
    img_encoder readout for observation dict `obs_dict` (each tensor
    shaped (1, D_key), i.e. batch size 1)."""
    with torch.no_grad():
        return dp_module.get_latent_action(obs_dict)


def _gaussian_deltas(z: torch.Tensor, n: int, sigma_z: float) -> List[torch.Tensor]:
    return [sigma_z * torch.randn_like(z) for _ in range(n)]


def _single_dim_deltas(z: torch.Tensor, n: int, eta: float) -> List[torch.Tensor]:
    dim = z.shape[-1]
    deltas = []
    d, sign = 0, 1.0
    for _ in range(n):
        delta = torch.zeros_like(z)
        delta[..., d % dim] = sign * eta
        deltas.append(delta)
        if sign < 0:
            d += 1
        sign = -sign
    return deltas


def _historical_deltas(z: torch.Tensor, skill_deltas: List[np.ndarray], n: int) -> List[torch.Tensor]:
    out = []
    for arr in skill_deltas[:n]:
        t = torch.as_tensor(np.asarray(arr), dtype=z.dtype, device=z.device).reshape(z.shape)
        out.append(t)
    return out


def generate_candidates(
    dp_module: torch.nn.Module,
    obs_dict: Dict[str, torch.Tensor],
    source_episode_id: str,
    failure_mode_id: int,
    M: int = 16,
    sigma_z: float = 0.5,
    eta: float = 0.5,
    skill_deltas: Optional[List[np.ndarray]] = None,
    seed: int = 0,
) -> List[Dict[str, Any]]:
    """Generate M candidate skills for one failure state. Proposal Day
    15.2 default (no matching skills in the archive): 8 Gaussian + 8
    single-dimension perturbations. When `skill_deltas` from matching
    archived skills are available, up to M//4 of the M candidates are
    historical-perturbation reuses (Section 8, "Historical Skill
    Perturbation"), with the Gaussian budget reduced accordingly so the
    total stays exactly M."""
    torch.manual_seed(seed)
    z_f = get_latent(dp_module, obs_dict)

    skill_deltas = skill_deltas or []
    n_historical = min(len(skill_deltas), M // 4)
    n_single = M // 2
    n_gaussian = M - n_single - n_historical

    deltas: List[torch.Tensor] = []
    kinds: List[str] = []
    for d in _historical_deltas(z_f, skill_deltas, n_historical):
        deltas.append(d)
        kinds.append("historical")
    for d in _gaussian_deltas(z_f, n_gaussian, sigma_z):
        deltas.append(d)
        kinds.append("gaussian")
    for d in _single_dim_deltas(z_f, n_single, eta):
        deltas.append(d)
        kinds.append("single_dim")
    assert len(deltas) == M

    candidates = []
    for j, (kind, delta) in enumerate(zip(kinds, deltas)):
        z_j = z_f + delta
        valid = bool(torch.isfinite(z_j).all().item())
        action_chunk = None
        if valid:
            with torch.no_grad():
                action_chunk = dp_module.action_decoder.predict_action(z_j)
            valid = bool(torch.isfinite(action_chunk).all().item())
        candidates.append(dict(
            candidate_id=f"{source_episode_id}_cand_{j:03d}",
            source_episode_id=source_episode_id,
            failure_mode_id=failure_mode_id,
            kind=kind,
            latent_delta=delta.squeeze(0).cpu().numpy(),
            action_chunk=(action_chunk.squeeze(0).cpu().numpy() if action_chunk is not None else None),
            valid=valid,
        ))
    return candidates
