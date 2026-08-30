"""Day 13.2: 'error grouped by task stage.' Stratifies the world model's
multi-step object-position prediction error (see
results/can/world_model_h20diag/multistep_eval/object_position_rmse_by_horizon.json
for the un-stratified version) by the task stage of the *window's
starting state* -- approach / grasp / transport / placement, using the
same thresholds as f2s.failure.extractor.assign_failure_stage, applied to
the world-model state vector x_t directly (which already carries p_ee,
p_obj, and gripper qpos -- see f2s/world_model/state.py) rather than the
raw obs dict, since these windows are built directly from the world
model's own state representation.

Answers the question raised in SOE/README_F2S.md: are contact-adjacent
transitions (grasp/transport/placement) predicted systematically worse
than free-space ones (approach), which would support the "world model
hasn't learned contact dynamics well" hypothesis over "it's just
generic horizon noise"?
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from f2s.common.io import load_json, save_json
from f2s.failure.extractor import GRIPPER_CLOSED_THRESHOLD, LIFT_HEIGHT_MARGIN, TABLE_HEIGHT
from f2s.logging.episode_logger import load_episode
from f2s.safety.filter import G_SLICE, P_EE_SLICE, P_OBJ_SLICE
from f2s.world_model.evaluate import build_multistep_windows
from f2s.world_model.model import WorldModelEnsemble, rollout_world_model
from f2s.world_model.state import build_world_model_states_for_episode

NEAR_OBJECT_THRESHOLD = 0.05


def classify_stage(x: np.ndarray) -> str:
    """x: (26,) world-model state vector. Mirrors
    f2s.failure.extractor.assign_failure_stage's categories and
    thresholds exactly, but reads them off x_t instead of raw obs."""
    p_ee = x[P_EE_SLICE]
    p_obj = x[P_OBJ_SLICE]
    gripper = x[G_SLICE]
    near_object = float(np.linalg.norm(p_ee - p_obj)) < NEAR_OBJECT_THRESHOLD
    gripper_closed = abs(float(np.mean(gripper))) < GRIPPER_CLOSED_THRESHOLD
    lifted = p_obj[2] > (TABLE_HEIGHT + LIFT_HEIGHT_MARGIN)

    if not near_object:
        return "approach"
    if near_object and not gripper_closed:
        return "grasp"
    if gripper_closed and lifted:
        return "transport"
    if gripper_closed and not lifted:
        return "placement"
    return "unknown"


def main():
    wm_dir = "results/can/world_model_h20diag"
    episode_dir = "results/can/f2s_final/seed_0/round_0/eval/episodes"
    split = load_json("results/can/world_model_dataset_h20diag/episode_split.json")
    val_ids = split["val_episode_ids"]

    with open(os.path.join(wm_dir, "result.json"), "r") as f:
        wm_result = json.load(f)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = WorldModelEnsemble(
        state_dim=wm_result["state_dim"], action_dim=wm_result["action_dim"],
        hidden_dim=wm_result["hidden_dim"], ensemble_size=wm_result["ensemble_size"],
    ).to(device)
    model.load_state_dict(torch.load(os.path.join(wm_dir, "best_model.pt"), map_location=device))
    model.eval()

    max_h = 20
    all_states, all_actions, boundaries = [], [], [0]
    for eid in val_ids:
        meta = load_json(os.path.join(episode_dir, f"{eid}.json"))
        if meta["episode_length"] < max_h + 1:
            continue
        _, arrays = load_episode(episode_dir, eid)
        x = build_world_model_states_for_episode(meta["obs_keys"], arrays)
        all_states.append(x)
        all_actions.append(arrays["actions"])
        boundaries.append(boundaries[-1] + x.shape[0])
    states = np.concatenate(all_states)
    actions = np.concatenate(all_actions)

    x0, a_windows, x_truth = build_multistep_windows(states, actions, boundaries[:-1], horizon=max_h)
    stages = np.array([classify_stage(x0[i]) for i in range(x0.shape[0])])
    stage_counts = {s: int((stages == s).sum()) for s in np.unique(stages)}
    print(f"{x0.shape[0]} windows total. Stage counts at window start: {stage_counts}")

    x0_t = torch.from_numpy(x0).float().to(device)
    a_t = torch.from_numpy(a_windows).float().to(device).permute(1, 0, 2)
    x_truth_t = torch.from_numpy(x_truth).float().to(device).permute(1, 0, 2)
    with torch.no_grad():
        pred = rollout_world_model(model, x0_t, a_t, horizon=max_h)

    horizons = [1, 3, 5, 8, 12, 16, 20]
    out = dict(num_windows=int(x0.shape[0]), stage_counts=stage_counts, by_horizon={})

    for h in horizons:
        obj_err = (pred[h - 1][:, P_OBJ_SLICE] - x_truth_t[h - 1][:, P_OBJ_SLICE])
        per_window_sq_err = (obj_err ** 2).sum(dim=-1).cpu().numpy()  # (N,)
        per_window_rmse = np.sqrt(per_window_sq_err)

        overall_rmse = float(per_window_rmse.mean())
        by_stage = {}
        for stage in sorted(set(stages)):
            mask = stages == stage
            if mask.sum() == 0:
                continue
            by_stage[stage] = dict(rmse_m=float(per_window_rmse[mask].mean()), n=int(mask.sum()))
        out["by_horizon"][str(h)] = dict(overall_rmse_m=overall_rmse, by_stage=by_stage)

        stage_str = "  ".join(f"{s}={v['rmse_m']*1000:.1f}mm(n={v['n']})" for s, v in by_stage.items())
        print(f"h={h:2d}  overall={overall_rmse*1000:6.1f}mm   {stage_str}")

    save_json("results/can/world_model_h20diag/multistep_eval/error_by_stage.json", out)

    # summary comparison: contact-adjacent (grasp/transport/placement) vs
    # free-space (approach), at the largest horizon
    h_key = str(max(horizons))
    contact_stages = {"grasp", "transport", "placement"}
    contact_rmses = [v["rmse_m"] for s, v in out["by_horizon"][h_key]["by_stage"].items() if s in contact_stages]
    freespace_rmse = out["by_horizon"][h_key]["by_stage"].get("approach", {}).get("rmse_m")
    if contact_rmses and freespace_rmse is not None:
        contact_mean = float(np.mean(contact_rmses))
        print(f"\nAt h={h_key}: contact-adjacent stages mean RMSE = {contact_mean*1000:.1f}mm, "
              f"approach (free-space) RMSE = {freespace_rmse*1000:.1f}mm "
              f"({'CONFIRMS' if contact_mean > freespace_rmse else 'DOES NOT CONFIRM'} "
              f"the contact-dynamics hypothesis)")


if __name__ == "__main__":
    main()
