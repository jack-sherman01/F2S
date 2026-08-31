"""Builds the training dataset for the "Failure Replay" baseline (proposal
Final Acceptance Criteria item 10: "F2S is compared with SOE and Failure
Replay"; also listed as a required baseline in Days 23-24 and Figure 3).

The proposal never spells out Failure Replay's exact mechanism (only
lists it by name alongside Success-only and Full F2S as contrastive
baselines). The most standard, defensible reading, consistent with how
it's used contrastively throughout the proposal (Success-only: retrain
on new *successful* rollouts only; Failure Replay: also incorporate
*failed* rollouts, but naively; Full F2S: process failures through the
whole structured pipeline -- clustering, candidate generation,
world-model ranking, safety filtering, validation): Failure Replay
fine-tunes the baseline policy on the original demos *plus the raw
failed episode trajectories, replayed as-is* -- no failure-mode
clustering, no candidate generation, no world-model, no safety filter,
no skill validation. It isolates "does merely exposing the policy to
more of its own failure states help" from "does F2S's structured
processing of those failures help," which is exactly the ablation-style
comparison the proposal's Figure 3 (SOE vs. Failure Replay vs. F2S w/o
clustering vs. Full F2S) is built around.

This script builds a combined RoboMimic-schema low-dim hdf5 (the
original demos, filtered by --original_mask_key, plus one new "demo"
group per failed episode) so SOE's own, completely unmodified
train_single_gpu.py / RoboMimicDataset can fine-tune on it exactly like
any other robomimic dataset -- no new training code, only new *data*.
"""
import argparse
import glob
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from f2s.common.io import load_all_episode_metadata, load_json
from f2s.logging.episode_logger import load_episode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_hdf5", required=True, help="original RoboMimic low_dim dataset")
    parser.add_argument("--original_mask_key", default="20_percent_train",
                         help="mask selecting which original demos to include (same one the baseline used)")
    parser.add_argument("--failure_episode_dirs", nargs="+", required=True,
                         help="one or more EpisodeLogger episodes/ directories to pull failed episodes from")
    parser.add_argument("--output_hdf5", required=True)
    parser.add_argument("--new_mask_key", default="train_with_failure_replay")
    parser.add_argument("--max_failure_episodes", type=int, default=None,
                         help="cap on how many failed episodes to append (None = all found)")
    args = parser.parse_args()

    if os.path.exists(args.output_hdf5):
        raise FileExistsError(f"refusing to overwrite existing dataset: {args.output_hdf5}")
    os.makedirs(os.path.dirname(args.output_hdf5), exist_ok=True)

    src = h5py.File(args.original_hdf5, "r")
    dst = h5py.File(args.output_hdf5, "w")
    data_grp = dst.create_group("data")

    original_demo_names = [d.decode("utf-8") for d in src[f"mask/{args.original_mask_key}"]]
    print(f"Copying {len(original_demo_names)} original demos (mask={args.original_mask_key})...")
    for name in original_demo_names:
        src.copy(f"data/{name}", data_grp, name=name)

    # --- append failed episodes as new demo groups ---
    # "failrep_N" rather than "demo_N": the original dataset's demo names
    # are not contiguous within a mask subset (e.g. 20_percent_train picks
    # 36 *scattered* names out of demo_0..demo_199), so a sequential
    # demo_0/1/2... counter can collide with an existing original demo
    # name -- a distinct prefix sidesteps that entirely rather than having
    # to compute the true max index across a sparse, non-contiguous set.
    failed_names = []
    n_appended = 0
    for ep_dir in args.failure_episode_dirs:
        if not os.path.isdir(ep_dir):
            print(f"WARNING: {ep_dir} does not exist, skipping")
            continue
        metas = load_all_episode_metadata(ep_dir)
        for meta in metas:
            if meta["success"]:
                continue
            if args.max_failure_episodes is not None and n_appended >= args.max_failure_episodes:
                break
            _, arrays = load_episode(ep_dir, meta["episode_id"])
            obs_keys = meta["obs_keys"]
            demo_name = f"failrep_{n_appended}"
            grp = data_grp.create_group(demo_name)
            grp.create_dataset("actions", data=arrays["actions"])
            obs_grp = grp.create_group("obs")
            for k in obs_keys:
                obs_grp.create_dataset(k, data=arrays[f"obs_{k}"])
            grp.attrs["num_samples"] = arrays["actions"].shape[0]
            failed_names.append(demo_name)
            n_appended += 1
        if args.max_failure_episodes is not None and n_appended >= args.max_failure_episodes:
            break

    print(f"Appended {n_appended} failed episodes (from {', '.join(args.failure_episode_dirs)}).")

    mask_grp = dst.create_group("mask")
    combined = original_demo_names + failed_names
    mask_grp.create_dataset(args.new_mask_key, data=np.array(combined, dtype="S"))
    mask_grp.create_dataset(f"{args.new_mask_key}_original_only", data=np.array(original_demo_names, dtype="S"))
    mask_grp.create_dataset(f"{args.new_mask_key}_failures_only", data=np.array(failed_names, dtype="S"))

    data_grp.attrs["total"] = sum(data_grp[n].attrs["num_samples"] for n in combined)
    if "env_args" in src["data"].attrs:
        data_grp.attrs["env_args"] = src["data"].attrs["env_args"]

    src.close()
    dst.close()
    print(f"Wrote {args.output_hdf5}: {len(original_demo_names)} original demos + "
          f"{n_appended} failure-replay demos = {len(combined)} total, under mask '{args.new_mask_key}'.")


if __name__ == "__main__":
    main()
