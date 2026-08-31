"""Day 8 CLI: extract failure segments from a directory of logged episodes.

    python scripts/extract_failures.py \
        --episode_dir results/can/soe/seed_0/round_0/eval_after/episodes \
        --output_dir results/can/soe/seed_0/round_0/failures \
        --failure_window 10
"""
import argparse
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from f2s.common.io import ensure_fresh_dir, load_all_episode_metadata, save_json
from f2s.failure.extractor import process_episode
from f2s.logging.episode_logger import load_episode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode_dir", required=True, help="directory containing episode_*.npz/.json")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--failure_window", type=int, default=10)
    args = parser.parse_args()

    ensure_fresh_dir(args.output_dir)
    segments_dir = os.path.join(args.output_dir, "segments")
    os.makedirs(segments_dir, exist_ok=True)

    metas = load_all_episode_metadata(args.episode_dir)
    total_episodes = len(metas)
    successful_episodes = sum(1 for m in metas if m["success"])
    failed_episodes = total_episodes - successful_episodes

    n_segments = 0
    failure_type_hist = Counter()
    failure_stage_hist = Counter()

    for meta in metas:
        if meta["success"]:
            continue
        _, arrays = load_episode(args.episode_dir, meta["episode_id"])
        segment = process_episode(meta, arrays, Hf=args.failure_window)
        if segment is None:
            continue

        n_segments += 1
        failure_type_hist[segment["failure_type"]] += 1
        failure_stage_hist[segment["failure_stage"]] += 1

        seg_id = f"{segment['episode_id']}_seg"
        np.savez_compressed(
            os.path.join(segments_dir, f"{seg_id}.npz"),
            state_window=segment["state_window"],
            action_window=segment["action_window"],
            **{f"obs_{k}": np.stack([o[k] for o in segment["obs_window"]]) for k in segment["obs_window"][0]},
        )
        save_json(os.path.join(segments_dir, f"{seg_id}.json"), dict(
            episode_id=segment["episode_id"],
            failure_time=segment["failure_time"],
            intervention_time=segment["intervention_time"],
            intervention_offset=segment["intervention_offset"],
            failure_type=segment["failure_type"],
            failure_stage=segment["failure_stage"],
            start_time=segment["start_time"],
            end_time=segment["end_time"],
        ))

    summary = dict(
        total_episodes=total_episodes,
        successful_episodes=successful_episodes,
        failed_episodes=failed_episodes,
        episodes_with_failure_segments=n_segments,
        failure_type_histogram=dict(failure_type_hist),
        failure_stage_histogram=dict(failure_stage_hist),
    )
    save_json(os.path.join(args.output_dir, "summary.json"), summary)
    print(summary)

    if failed_episodes > 0:
        coverage = n_segments / failed_episodes
        print(f"failure-segment coverage: {coverage:.1%}")
        if coverage < 0.8:
            print("WARNING: Day 8 acceptance test requires >= 80% of failed "
                  "episodes to produce a non-empty failure segment.")


if __name__ == "__main__":
    main()
