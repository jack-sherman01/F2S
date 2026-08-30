"""Day 9-10 CLI: compute failure features for every extracted segment,
standardize them (fit on this data), and cluster into failure modes.

    python scripts/cluster_failures.py \
        --failure_dir results/can/soe/seed_0/round_0/failures \
        --num_clusters 4 \
        --seed 0 \
        --output_dir results/can/soe/seed_0/round_0/failure_modes
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from f2s.common.io import ensure_fresh_dir, load_json, save_json
from f2s.failure.clustering import choose_k, fit_failure_clusters, summarize_clusters
from f2s.failure.features import FAILURE_FEATURE_NAMES, compute_failure_feature_vector, standardize


def _load_segment(segments_dir: str, seg_json_path: str):
    meta = load_json(seg_json_path)
    seg_id = os.path.splitext(os.path.basename(seg_json_path))[0]
    npz = np.load(os.path.join(segments_dir, f"{seg_id}.npz"))
    obs_keys = [k[len("obs_"):] for k in npz.files if k.startswith("obs_")]
    window_len = npz["state_window"].shape[0]
    obs_window = [{k: npz[f"obs_{k}"][t] for k in obs_keys} for t in range(window_len)]
    return meta, obs_window


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--failure_dir", required=True, help="output_dir from extract_failures.py")
    parser.add_argument("--num_clusters", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--try_multiple_k", action="store_true",
                         help="also fit K in {2,4,6} and report which was chosen (Day 10.1-10.2)")
    args = parser.parse_args()

    ensure_fresh_dir(args.output_dir)

    segments_dir = os.path.join(args.failure_dir, "segments")
    seg_json_paths = sorted(glob.glob(os.path.join(segments_dir, "*.json")))
    if len(seg_json_paths) == 0:
        raise RuntimeError(f"no failure segments found under {segments_dir}; run extract_failures.py first")

    features = []
    failure_types = []
    failure_stages = []
    episode_ids = []
    object_errors = []
    for p in seg_json_paths:
        meta, obs_window = _load_segment(segments_dir, p)
        feat = compute_failure_feature_vector(obs_window)
        features.append(feat)
        failure_types.append(meta["failure_type"])
        failure_stages.append(meta["failure_stage"])
        episode_ids.append(meta["episode_id"])
        object_errors.append(feat[0])  # delta_p_obj_final is feature index 0

    features = np.stack(features)
    object_errors = np.array(object_errors)
    assert features.ndim == 2
    assert np.isfinite(features).all(), "failure features contain NaN/Inf"
    assert features.shape[0] == len(seg_json_paths)

    mu = features.mean(axis=0)
    sigma = features.std(axis=0)
    features_std = standardize(features, mu, sigma)
    np.savez(os.path.join(args.output_dir, "normalization_stats.npz"), mu=mu, sigma=sigma)

    if args.try_multiple_k:
        chosen_k, all_results = choose_k(
            features_std, failure_types, failure_stages, object_errors,
            k_candidates=(2, 4, 6), seed=args.seed, preferred_k=args.num_clusters,
        )
        save_json(os.path.join(args.output_dir, "k_selection.json"), dict(
            chosen_k=chosen_k,
            candidates={str(k): [c["cluster_size"] for c in r["summary"]] for k, r in all_results.items()},
        ))
        model = all_results[chosen_k]["model"]
        clusters = all_results[chosen_k]["summary"]
        k_used = chosen_k
    else:
        model = fit_failure_clusters(features_std, args.num_clusters, seed=args.seed)
        clusters = summarize_clusters(model, features_std, failure_types, failure_stages, object_errors)
        k_used = args.num_clusters

    assignments = model.labels_.tolist()
    save_json(os.path.join(args.output_dir, "assignments.json"), dict(
        k=k_used,
        feature_names=FAILURE_FEATURE_NAMES,
        episode_ids=episode_ids,
        cluster_id=assignments,
    ))
    save_json(os.path.join(args.output_dir, "clusters.json"), dict(k=k_used, clusters=clusters))

    print(f"Fit K={k_used} failure-mode clusters over {features.shape[0]} segments.")
    for c in clusters:
        print(f"  cluster {c['cluster_id']}: size={c['cluster_size']} "
              f"type_hist={c['failure_type_histogram']} stage_hist={c['failure_stage_histogram']} "
              f"mean_object_error={c['mean_object_error']}")

    n_ge_10 = sum(1 for c in clusters if c["cluster_size"] >= 10)
    if n_ge_10 < 2:
        print(f"WARNING: Days 8-10 exit test requires >= 2 clusters with >= 10 samples; got {n_ge_10}.")


if __name__ == "__main__":
    main()
