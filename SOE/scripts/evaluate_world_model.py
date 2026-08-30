"""Day 13 CLI: multi-step world-model prediction error vs horizon.

    python scripts/evaluate_world_model.py \
        --model_dir results/can/world_model/round_0 \
        --episode_dir results/can/soe/seed_0/round_0/episodes \
        --episode_split results/can/world_model_dataset/episode_split.json \
        --output_dir results/can/world_model/round_0/multistep_eval
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from f2s.common.io import ensure_fresh_dir, load_json
from f2s.logging.episode_logger import load_episode
from f2s.world_model.evaluate import build_multistep_windows, evaluate_multistep
from f2s.world_model.model import WorldModelEnsemble
from f2s.world_model.state import STATE_DIM, build_world_model_states_for_episode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, help="dir with best_model.pt + config.yaml from train_world_model.py")
    parser.add_argument("--episode_dir", required=True)
    parser.add_argument("--episode_split", required=True, help="episode_split.json from build_world_model_dataset.py")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 3, 5])
    args = parser.parse_args()

    ensure_fresh_dir(args.output_dir)

    with open(os.path.join(args.model_dir, "config.yaml"), "r") as f:
        train_config = json.load(f)
    with open(os.path.join(args.model_dir, "result.json"), "r") as f:
        train_result = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = WorldModelEnsemble(
        state_dim=train_result["state_dim"],
        action_dim=train_result["action_dim"],
        hidden_dim=train_result["hidden_dim"],
        ensemble_size=train_result["ensemble_size"],
    ).to(device)
    model.load_state_dict(torch.load(os.path.join(args.model_dir, "best_model.pt"), map_location=device))
    model.eval()

    split = load_json(args.episode_split)
    val_ids = split["val_episode_ids"]
    max_horizon = max(args.horizons)

    all_states, all_actions = [], []
    boundaries = [0]
    for eid in val_ids:
        meta = load_json(os.path.join(args.episode_dir, f"{eid}.json"))
        if meta["episode_length"] < max_horizon + 1:
            continue
        _, arrays = load_episode(args.episode_dir, eid)
        x = build_world_model_states_for_episode(meta["obs_keys"], arrays)
        all_states.append(x)
        all_actions.append(arrays["actions"])
        boundaries.append(boundaries[-1] + x.shape[0])

    if len(all_states) == 0:
        raise RuntimeError("no validation episodes long enough for the requested max horizon")

    states = np.concatenate(all_states)
    actions = np.concatenate(all_actions)

    x0, a_windows, x_truth = build_multistep_windows(states, actions, boundaries[:-1], horizon=max_horizon)
    results = evaluate_multistep(model, x0, a_windows, x_truth, device=device, horizons=tuple(args.horizons))

    out = dict(
        horizons_mse=results,
        num_windows=int(x0.shape[0]),
        max_horizon_evaluated=max_horizon,
        single_step_val_mse_from_training=train_result["best_val_mse"],
    )
    if 5 in results and results[5] > 3 * results.get(1, results[5]):
        out["note"] = ("5-step MSE diverges relative to 1-step (>3x); per proposal Day 13.2, "
                        "consider setting the main experimental horizon H_WM=3 instead of 5.")

    with open(os.path.join(args.output_dir, "multistep_mse.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
