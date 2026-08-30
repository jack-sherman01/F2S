"""Cheap check before committing to closed-loop replanning (see
SOE/README_F2S.md's robust-CEM finding): does an ensemble world model's
*disagreement* between members spike near the 0.5-1cm position offset
where the real simulator's success rate is already known to collapse
(scripts/diagnose_lift_robust_cem.py: 2/10 at 0.5cm, 0/10 at 1cm+)?

If ensemble disagreement tracks that boundary, it's a usable signal for
rejecting brittle candidates (Section 10.1's u(k), already implemented in
f2s.world_model.model.rollout_world_model_with_uncertainty but not
previously exercised) without the larger architectural change of
closed-loop replanning. If it doesn't track the boundary, that's useful
too -- it would mean the single mean-predicting MLP's smoothness bias
isn't something ensembling three of the same architecture fixes, and
replanning is the more promising remaining lever.

Deliberately read-only with respect to every previously-committed Lift
artifact: trains a *new* ensemble model into its own output directory,
touches no shared f2s module.
"""
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "simulation")))
sys.path.insert(0, os.path.dirname(__file__))

from easydict import EasyDict

from diagnose_lift_full_pipeline import ZERO_GOAL, find_lift_stall_time
from f2s.common.io import ensure_fresh_dir, load_json, save_json
from f2s.logging.episode_logger import load_episode
from f2s.world_model.dataset import build_transitions
from f2s.world_model.model import WorldModelEnsemble, rollout_world_model_with_uncertainty
from f2s.world_model.state import build_world_model_state
from f2s.world_model.train import train_world_model
from f2s.safety.filter import P_OBJ_SLICE


def main():
    config_path = "configs/soe_lift_lowdim_baseline.json"
    episode_dir = "results/lift/soe/seed_0/round_0/eval/episodes"
    target_episode_id = os.environ.get("F2S_LIFT_TARGET_EPISODE", "episode_000005")
    horizon_wm = int(os.environ.get("F2S_LIFT_HORIZON", "5"))

    with open(config_path, "r") as f:
        cfg = EasyDict(json.load(f))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- train a fresh E=3 ensemble on the same episode-level split used
    # for the single-model diagnostics (same seed, same data) ---
    ids = [os.path.splitext(os.path.basename(p))[0]
           for p in __import__("glob").glob(os.path.join(episode_dir, "episode_*.json"))]
    rng = np.random.RandomState(0)
    perm = rng.permutation(len(ids))
    n_train = int(round(0.8 * len(ids)))
    train_ids = [ids[i] for i in perm[:n_train]]
    val_ids = [ids[i] for i in perm[n_train:]]

    train_states, train_actions, train_next_states = build_transitions(episode_dir, train_ids)
    val_states, val_actions, val_next_states = build_transitions(episode_dir, val_ids)

    wm_dir = "results/lift/world_model_ensemble_diag"
    ensure_fresh_dir(wm_dir)
    ensemble_model, wm_result = train_world_model(
        train_states, train_actions, train_next_states, val_states, val_actions, val_next_states,
        output_dir=wm_dir, hidden_dim=256, ensemble_size=3, epochs=50, seed=0,
    )
    save_json(os.path.join(wm_dir, "result.json"), wm_result)
    print(f"Ensemble (E=3) world model: val_mse={wm_result['best_val_mse']:.6f}, "
          f"beats_constant={wm_result['beats_constant_baseline']}")

    # --- load the target failure state ---
    meta = load_json(os.path.join(episode_dir, f"{target_episode_id}.json"))
    _, arrays = load_episode(episode_dir, target_episode_id)
    cube_z = arrays["obs_object"][:, 2]
    t_f = find_lift_stall_time(cube_z)
    obs_t = {k: arrays[f"obs_{k}"][t_f] for k in meta["obs_keys"]}
    x0 = build_world_model_state(obs_t, goal_pos=ZERO_GOAL)

    # A fixed, representative action chunk to probe with: use the actual
    # action chunk the policy took from this state (a real, plausible
    # trajectory), rather than a synthetic one, so the probe reflects
    # actions the pipeline would actually consider.
    action_chunk = arrays["actions"][t_f:t_f + horizon_wm]
    if action_chunk.shape[0] < horizon_wm:
        pad = np.repeat(action_chunk[-1:], horizon_wm - action_chunk.shape[0], axis=0)
        action_chunk = np.concatenate([action_chunk, pad], axis=0)

    action_t = torch.from_numpy(action_chunk).float().unsqueeze(1).to(device)  # (H, 1, action_dim)

    print(f"\nTarget: {target_episode_id} t_f={t_f}, probing with the policy's own "
          f"{horizon_wm}-step action chunk from that state.")
    print(f"{'offset(cm)':>10} {'mean_final_height':>18} {'ensemble_var(x)':>16} {'known_real_success_rate':>24}")

    # known real success rates from the earlier tolerance sweep, for
    # direct visual comparison in this table (not recomputed here --
    # that used a *different*, CEM-found successful action chunk, not
    # this diagnostic action; included only as a reference point for
    # "does disagreement grow in the same region where reality is fragile").
    known_real_success = {0.0: None, 0.5: "2/10", 1.0: "0/10", 2.0: "0/10", 3.0: "0/10"}

    results = {}
    n_probes_per_offset = 8
    for offset_cm in [0.0, 0.5, 1.0, 2.0, 3.0]:
        offset_m = offset_cm / 100.0
        final_heights = []
        ensemble_vars = []
        rng2 = np.random.RandomState(hash(offset_cm) % (2**31))
        n_probes = 1 if offset_cm == 0.0 else n_probes_per_offset
        for _ in range(n_probes):
            x_probe = x0.copy()
            if offset_cm > 0.0:
                jitter = rng2.uniform(-offset_m, offset_m, size=2).astype(np.float32)
                x_probe[P_OBJ_SLICE.start:P_OBJ_SLICE.start + 2] += jitter
            x0_t = torch.from_numpy(x_probe).float().unsqueeze(0).to(device)
            with torch.no_grad():
                pred_states, pred_vars = rollout_world_model_with_uncertainty(ensemble_model, x0_t, action_t, horizon=horizon_wm)
            final_heights.append(float(pred_states[-1, 0, P_OBJ_SLICE][2].item()))
            ensemble_vars.append(float(pred_vars[-1, 0].item()))

        results[offset_cm] = dict(
            mean_final_height=float(np.mean(final_heights)),
            std_final_height_across_probes=float(np.std(final_heights)),
            mean_ensemble_variance=float(np.mean(ensemble_vars)),
        )
        print(f"{offset_cm:>10.1f} {results[offset_cm]['mean_final_height']:>18.4f} "
              f"{results[offset_cm]['mean_ensemble_variance']:>16.8f} {str(known_real_success[offset_cm]):>24}")

    var_at_0 = results[0.0]["mean_ensemble_variance"]
    var_at_boundary = results[1.0]["mean_ensemble_variance"]
    ratio = var_at_boundary / max(var_at_0, 1e-12)
    print(f"\nEnsemble variance ratio (1cm vs exact state): {ratio:.2f}x")
    if ratio > 3.0:
        print("Ensemble disagreement DOES grow substantially near the boundary where real success "
              "collapses -- worth pursuing as a candidate-rejection signal.")
    else:
        print("Ensemble disagreement does NOT grow substantially near the boundary -- three "
              "instances of the same small-MLP architecture share the same smoothness bias, "
              "so this signal would not have caught the brittleness found earlier.")

    save_json("results/lift/world_model_ensemble_diag/uncertainty_vs_offset.json", dict(
        target_episode_id=target_episode_id, t_f=int(t_f), horizon_wm=horizon_wm,
        results={str(k): v for k, v in results.items()}, variance_ratio_1cm_vs_exact=ratio,
    ))


if __name__ == "__main__":
    main()
