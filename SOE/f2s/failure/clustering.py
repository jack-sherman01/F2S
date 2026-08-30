"""Failure-mode clustering (proposal Day 10): group standardized failure
feature vectors (f2s.failure.features.compute_failure_feature_vector) into
K clusters via K-means, and attach interpretable per-cluster statistics.
"""
from collections import Counter
from typing import Any, Dict, List

import numpy as np
from sklearn.cluster import KMeans


def fit_failure_clusters(features: np.ndarray, k: int, seed: int = 0) -> KMeans:
    assert features.ndim == 2, f"expected 2D feature matrix, got shape {features.shape}"
    assert np.isfinite(features).all(), "failure features contain NaN/Inf"
    model = KMeans(n_clusters=k, random_state=seed, n_init=10)
    model.fit(features)
    return model


def summarize_clusters(
    model: KMeans,
    features: np.ndarray,
    failure_types: List[str],
    failure_stages: List[str],
    raw_object_errors: np.ndarray,
) -> List[Dict[str, Any]]:
    labels = model.labels_
    clusters = []
    for c in range(model.n_clusters):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            clusters.append(dict(
                cluster_id=c, cluster_size=0, cluster_center=model.cluster_centers_[c].tolist(),
                failure_type_histogram={}, failure_stage_histogram={},
                mean_object_error=None, mean_feature=None, std_feature=None,
            ))
            continue
        type_hist = dict(Counter(failure_types[i] for i in idx))
        stage_hist = dict(Counter(failure_stages[i] for i in idx))
        clusters.append(dict(
            cluster_id=c,
            cluster_size=int(len(idx)),
            cluster_center=model.cluster_centers_[c].tolist(),
            failure_type_histogram=type_hist,
            failure_stage_histogram=stage_hist,
            mean_object_error=float(np.mean(raw_object_errors[idx])),
            mean_feature=features[idx].mean(axis=0).tolist(),
            std_feature=features[idx].std(axis=0).tolist(),
        ))
    return clusters


def choose_k(
    features: np.ndarray,
    failure_types: List[str],
    failure_stages: List[str],
    raw_object_errors: np.ndarray,
    k_candidates=(2, 4, 6),
    seed: int = 0,
    preferred_k: int = 4,
    min_cluster_size: int = 2,
):
    """Fit K in k_candidates, prefer preferred_k unless it produces a
    degenerate clustering (some cluster smaller than min_cluster_size while
    a smaller K would not), per proposal Day 10.1-10.2."""
    results = {}
    for k in k_candidates:
        if k > len(features):
            continue
        model = fit_failure_clusters(features, k, seed=seed)
        summary = summarize_clusters(model, features, failure_types, failure_stages, raw_object_errors)
        results[k] = dict(model=model, summary=summary)

    if preferred_k in results:
        sizes = [c["cluster_size"] for c in results[preferred_k]["summary"]]
        if min(sizes) >= min_cluster_size:
            return preferred_k, results

    # fall back to the largest candidate K whose smallest cluster is not
    # degenerate; if none qualify, keep preferred_k anyway (documented,
    # not silently substituted).
    for k in sorted(results.keys(), reverse=True):
        sizes = [c["cluster_size"] for c in results[k]["summary"]]
        if min(sizes) >= min_cluster_size:
            return k, results

    return preferred_k, results
