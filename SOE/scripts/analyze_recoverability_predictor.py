"""Capstone diagnostic: is recovery success predicted by which corrective
action is chosen, or almost entirely by how far along the failure state
already was before any correction is attempted?

Motivation: sections on candidate scoring (world model, value function),
closed-loop correction, and directed search (CEM, both fitness signals)
all independently failed to beat plain random generation. Before
concluding the correction *mechanism* has a hard ceiling, check the more
basic possibility: maybe correction success is gated almost entirely by
the STARTING STATE's own proximity to the goal, not by anything the
candidate-selection machinery does -- in which case no amount of
cleverness in scoring/search could have helped, because the real lever
(which failure states get attempted) was never the free variable.

Uses `results/can/candidate_ranking_per_state_offset_sweep/records.json`
-- the full per-state-per-offset offset sweep where all M=16 candidates
were executed for real at every (state, offset) combination (284 combos,
4544 real executions, 15 real successes) -- so both the outcome and the
starting-state features are already grounded in real simulator rollouts,
no new execution needed.

For each (state, offset) combo, computes the failure state's own
goal_error (object-to-goal distance) and task_progress (world-model
state's own progress feature) BEFORE any candidate is generated, and
checks how well those two features alone predict whether at least one of
the 16 real candidates from that state succeeded.
"""
import json
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score
from scipy import stats

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from f2s.candidates.scorer import compute_goal_error, compute_task_progress
from f2s.common.io import load_all_episode_metadata, save_json
from f2s.failure.extractor import process_episode
from f2s.logging.episode_logger import load_episode
from f2s.world_model.state import build_world_model_state

ALL_EPISODE_DIRS = [
    "results/can/f2s_final/seed_0/round_0/eval/episodes",
    "results/can/f2s_final/seed_0/round_1/eval/episodes",
    "results/can/f2s_final/seed_0/round_2/eval/episodes",
    "results/can/f2s_dev/seed_0/round_0/eval/episodes",
    "results/can/f2s_dev/seed_0/round_1/eval/episodes",
    "results/can/f2s_dev/seed_0/round_2/eval/episodes",
    "results/can/f2s_dev_cem/seed_2/round_0/eval/episodes",
    "results/can/f2s_dev_cem/seed_2/round_1/eval/episodes",
    "results/can/f2s_dev_cem/seed_2/round_2/eval/episodes",
    "results/Can/fixed_policy/seed_0/round_0/episodes",
]

RECORDS_PATH = "results/can/candidate_ranking_per_state_offset_sweep/records.json"
OUTPUT_DIR = "results/can/recoverability_predictor_analysis"


def main():
    segments = []
    for ep_dir in ALL_EPISODE_DIRS:
        if not os.path.isdir(ep_dir):
            continue
        metas = load_all_episode_metadata(ep_dir)
        for meta in metas:
            if meta["success"]:
                continue
            _, arrays = load_episode(ep_dir, meta["episode_id"])
            seg = process_episode(meta, arrays, Hf=10)
            if seg is not None:
                segments.append((ep_dir, seg, meta, arrays))
    assert len(segments) == 71, f"expected 71 pooled states, got {len(segments)}"

    records = json.load(open(RECORDS_PATH))
    from collections import defaultdict
    succ_count = defaultdict(int)
    attempt_count = defaultdict(int)
    for r in records:
        key = (r["state_idx"], r["offset"])
        attempt_count[key] += 1
        if r.get("actual_success"):
            succ_count[key] += 1

    rows = []
    for (state_idx, offset), n_att in attempt_count.items():
        ep_dir, seg, meta, arrays = segments[state_idx]
        t_f = max(0, seg["failure_time"] - offset)
        obs_t = {k: arrays[f"obs_{k}"][t_f] for k in meta["obs_keys"]}
        x = build_world_model_state(obs_t)
        rows.append(dict(
            state_idx=state_idx, episode_id=seg["episode_id"], offset=offset,
            n_attempts=n_att, n_success=succ_count[(state_idx, offset)],
            success_rate=succ_count[(state_idx, offset)] / n_att,
            goal_error=compute_goal_error(x), task_progress=compute_task_progress(x),
        ))

    goal_errors = np.array([r["goal_error"] for r in rows])
    task_progress = np.array([r["task_progress"] for r in rows])
    success_rate = np.array([r["success_rate"] for r in rows])
    any_success = (success_rate > 0).astype(float)

    spearman_ge = stats.spearmanr(goal_errors, success_rate)
    spearman_tp = stats.spearmanr(task_progress, success_rate)
    pb_ge = stats.pointbiserialr(any_success, goal_errors)
    pb_tp = stats.pointbiserialr(any_success, task_progress)
    auc_ge = roc_auc_score(any_success, -goal_errors)
    auc_tp = roc_auc_score(any_success, task_progress)

    result = dict(
        n_combos=len(rows),
        n_combos_with_any_success=int(any_success.sum()),
        n_combos_zero_success=int((any_success == 0).sum()),
        spearman_goal_error_vs_success_rate=dict(statistic=spearman_ge.statistic, pvalue=spearman_ge.pvalue),
        spearman_task_progress_vs_success_rate=dict(statistic=spearman_tp.statistic, pvalue=spearman_tp.pvalue),
        point_biserial_goal_error_vs_any_success=dict(statistic=pb_ge.statistic, pvalue=pb_ge.pvalue),
        point_biserial_task_progress_vs_any_success=dict(statistic=pb_tp.statistic, pvalue=pb_tp.pvalue),
        auc_neg_goal_error_predicts_any_success=float(auc_ge),
        auc_task_progress_predicts_any_success=float(auc_tp),
        goal_error_mean_success_combos=float(goal_errors[any_success == 1].mean()),
        goal_error_mean_zero_success_combos=float(goal_errors[any_success == 0].mean()),
        task_progress_mean_success_combos=float(task_progress[any_success == 1].mean()),
        task_progress_mean_zero_success_combos=float(task_progress[any_success == 0].mean()),
        rows=rows,
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_json(os.path.join(OUTPUT_DIR, "result.json"), result)

    print(f"n (state, offset) combos: {len(rows)} "
          f"({result['n_combos_with_any_success']} with >=1 real success, "
          f"{result['n_combos_zero_success']} with zero)")
    print(f"Spearman(goal_error, success_rate)     = {spearman_ge.statistic:.4f} (p={spearman_ge.pvalue:.2e})")
    print(f"Spearman(task_progress, success_rate)  = {spearman_tp.statistic:.4f} (p={spearman_tp.pvalue:.2e})")
    print(f"AUC(-goal_error -> any_success)         = {auc_ge:.4f}")
    print(f"AUC(task_progress -> any_success)       = {auc_tp:.4f}")
    print(f"goal_error   mean | success combos = {result['goal_error_mean_success_combos']:.4f} | "
          f"zero-success combos = {result['goal_error_mean_zero_success_combos']:.4f}")
    print(f"task_progress mean | success combos = {result['task_progress_mean_success_combos']:.4f} | "
          f"zero-success combos = {result['task_progress_mean_zero_success_combos']:.4f}")


if __name__ == "__main__":
    main()
