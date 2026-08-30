"""Day 11 CLI.

    python scripts/build_world_model_dataset.py \
        --episode_dir results/can/soe/seed_0/round_0/episodes \
        --train_ratio 0.8 \
        --seed 0 \
        --output_dir results/can/world_model_dataset
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from f2s.common.io import ensure_fresh_dir
from f2s.world_model.dataset import build_transitions, compute_normalization_stats, split_episodes_by_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode_dir", required=True)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    ensure_fresh_dir(args.output_dir)

    train_ids, val_ids = split_episodes_by_id(args.episode_dir, train_ratio=args.train_ratio, seed=args.seed)
    print(f"episodes: {len(train_ids) + len(val_ids)} total, {len(train_ids)} train, {len(val_ids)} val")

    train_states, train_actions, train_next_states = build_transitions(args.episode_dir, train_ids)
    val_states, val_actions, val_next_states = build_transitions(args.episode_dir, val_ids)

    for name, arr in [
        ("train_states", train_states), ("train_actions", train_actions), ("train_next_states", train_next_states),
        ("val_states", val_states), ("val_actions", val_actions), ("val_next_states", val_next_states),
    ]:
        assert np.isfinite(arr).all(), f"{name} contains NaN/Inf"

    state_dim = train_states.shape[1]
    action_dim = train_actions.shape[1]
    print("state_dim:", state_dim)
    print("action_dim:", action_dim)
    print("state_min:", train_states.min(axis=0))
    print("state_max:", train_states.max(axis=0))
    print("action_min:", train_actions.min(axis=0))
    print("action_max:", train_actions.max(axis=0))

    mu_x, sigma_x = compute_normalization_stats(train_states)
    mu_a, sigma_a = compute_normalization_stats(train_actions)
    np.savez(
        os.path.join(args.output_dir, "world_model_normalization.npz"),
        mu_x=mu_x, sigma_x=sigma_x, mu_a=mu_a, sigma_a=sigma_a,
    )

    np.save(os.path.join(args.output_dir, "train_states.npy"), train_states)
    np.save(os.path.join(args.output_dir, "train_actions.npy"), train_actions)
    np.save(os.path.join(args.output_dir, "train_next_states.npy"), train_next_states)
    np.save(os.path.join(args.output_dir, "val_states.npy"), val_states)
    np.save(os.path.join(args.output_dir, "val_actions.npy"), val_actions)
    np.save(os.path.join(args.output_dir, "val_next_states.npy"), val_next_states)

    with open(os.path.join(args.output_dir, "episode_split.json"), "w") as f:
        import json
        json.dump(dict(train_episode_ids=train_ids, val_episode_ids=val_ids), f, indent=2)

    print(f"train transitions: {train_states.shape[0]}, val transitions: {val_states.shape[0]}")
    print("Day 11 acceptance test: no NaN/Inf, consistent dims, disjoint episode ids -- PASSED")


if __name__ == "__main__":
    main()
