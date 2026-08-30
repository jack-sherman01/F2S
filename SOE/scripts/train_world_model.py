"""Day 12 CLI.

    python scripts/train_world_model.py \
        --train_states results/can/world_model_dataset/train_states.npy \
        --train_actions results/can/world_model_dataset/train_actions.npy \
        --train_next_states results/can/world_model_dataset/train_next_states.npy \
        --val_states results/can/world_model_dataset/val_states.npy \
        --val_actions results/can/world_model_dataset/val_actions.npy \
        --val_next_states results/can/world_model_dataset/val_next_states.npy \
        --hidden_dim 256 --horizon 1 --epochs 50 --batch_size 256 --lr 3e-4 \
        --output_dir results/can/world_model/round_0
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from f2s.common.io import ensure_fresh_dir, git_commit_hash
from f2s.common.seeds import set_seed
from f2s.world_model.train import train_world_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_states", required=True)
    parser.add_argument("--train_actions", required=True)
    parser.add_argument("--train_next_states", required=True)
    parser.add_argument("--val_states", required=True)
    parser.add_argument("--val_actions", required=True)
    parser.add_argument("--val_next_states", required=True)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=1, help="kept for CLI parity with the proposal; "
                         "single-step loss is always used here, multi-step evaluation is Day 13")
    parser.add_argument("--ensemble_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--gradient_clip_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    ensure_fresh_dir(args.output_dir)
    set_seed(args.seed)

    train_states = np.load(args.train_states)
    train_actions = np.load(args.train_actions)
    train_next_states = np.load(args.train_next_states)
    val_states = np.load(args.val_states)
    val_actions = np.load(args.val_actions)
    val_next_states = np.load(args.val_next_states)

    normalization_src = os.path.join(os.path.dirname(args.train_states), "world_model_normalization.npz")
    if os.path.exists(normalization_src):
        shutil.copy(normalization_src, os.path.join(args.output_dir, "normalization.npz"))

    model, result = train_world_model(
        train_states, train_actions, train_next_states,
        val_states, val_actions, val_next_states,
        output_dir=args.output_dir,
        hidden_dim=args.hidden_dim,
        ensemble_size=args.ensemble_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        gradient_clip_norm=args.gradient_clip_norm,
        seed=args.seed,
    )

    config = vars(args)
    config["git_commit"] = git_commit_hash(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    with open(os.path.join(args.output_dir, "config.yaml"), "w") as f:
        json.dump(config, f, indent=2)  # yaml-compatible superset (json is valid yaml)

    with open(os.path.join(args.output_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
    if result["beats_constant_baseline"] is False:
        print("WARNING: Day 12 acceptance test FAILED -- learned model did not "
              "beat the constant-state baseline on validation data.")
    elif result["beats_constant_baseline"] is True:
        print("Day 12 acceptance test PASSED: learned model beats the constant-state baseline.")


if __name__ == "__main__":
    main()
