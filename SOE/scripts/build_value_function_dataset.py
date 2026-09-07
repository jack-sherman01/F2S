"""Builds and saves the pooled value-function training dataset (state,
eventual-success-label pairs) from every real Can-task episode logged
across this project. See f2s/value/dataset.py for exactly what counts as
a source, the per-episode train/val split discipline, and the documented
friction/mass aliasing limitation.

    python scripts/build_value_function_dataset.py \
        --output_dir results/can/value_function_dataset
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from f2s.common.io import ensure_fresh_dir, save_json
from f2s.value.dataset import ALL_CAN_EPISODE_DIRS, build_pooled_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="results/can/value_function_dataset")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    ensure_fresh_dir(args.output_dir)

    train_states, train_labels, val_states, val_labels, per_dir_stats = build_pooled_dataset(
        ALL_CAN_EPISODE_DIRS, train_ratio=args.train_ratio, seed=args.seed,
    )

    np.savez(
        os.path.join(args.output_dir, "dataset.npz"),
        train_states=train_states, train_labels=train_labels,
        val_states=val_states, val_labels=val_labels,
    )

    n_dirs_found = sum(1 for d in per_dir_stats if d.get("found"))
    n_dirs_missing = sum(1 for d in per_dir_stats if not d.get("found"))
    summary = dict(
        n_dirs_listed=len(ALL_CAN_EPISODE_DIRS), n_dirs_found=n_dirs_found, n_dirs_missing=n_dirs_missing,
        n_train_states=int(train_states.shape[0]), n_val_states=int(val_states.shape[0]),
        train_positive_rate=float(train_labels.mean()) if len(train_labels) > 0 else None,
        val_positive_rate=float(val_labels.mean()) if len(val_labels) > 0 else None,
        per_dir=per_dir_stats,
    )
    save_json(os.path.join(args.output_dir, "dataset_summary.json"), summary)
    print(f"Pooled from {n_dirs_found}/{len(ALL_CAN_EPISODE_DIRS)} directories "
          f"({n_dirs_missing} not found, skipped).")
    print(f"train: {train_states.shape[0]} states, positive rate {summary['train_positive_rate']:.3f}")
    print(f"val:   {val_states.shape[0]} states, positive rate {summary['val_positive_rate']:.3f}")


if __name__ == "__main__":
    main()
