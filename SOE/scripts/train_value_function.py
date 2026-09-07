"""Trains the value function on the pooled dataset from
scripts/build_value_function_dataset.py.

    python scripts/train_value_function.py \
        --dataset_dir results/can/value_function_dataset \
        --output_dir results/can/value_function
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from f2s.common.io import ensure_fresh_dir
from f2s.value.train import train_value_function


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default="results/can/value_function_dataset")
    parser.add_argument("--output_dir", default="results/can/value_function")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    ensure_fresh_dir(args.output_dir)

    data = np.load(os.path.join(args.dataset_dir, "dataset.npz"))
    _, result = train_value_function(
        train_states=data["train_states"], train_labels=data["train_labels"],
        val_states=data["val_states"], val_labels=data["val_labels"],
        output_dir=args.output_dir, hidden_dim=args.hidden_dim, epochs=args.epochs, seed=args.seed,
    )
    with open(os.path.join(args.output_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
