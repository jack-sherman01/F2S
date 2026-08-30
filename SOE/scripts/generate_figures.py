"""Day 26 CLI: turn the raw results/ directory into figures + tables.
Every plotted number is read from a committed metrics.json/evolution_summary.json
-- nothing here is synthesized. Figures/tables that need data we don't
have yet (a given method/seed hasn't been run) are simply skipped, with a
printed note, rather than filled with placeholder numbers.

    python scripts/generate_figures.py --results_root results --output_dir results/figures
"""
import argparse
import csv
import glob
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from f2s.common.io import load_json


def maybe_load(path):
    return load_json(path) if os.path.exists(path) else None


def fig_success_vs_round(evolution_summary_paths, output_dir):
    """Figure 1: success rate vs evolution round, one line per (method,seed) run found."""
    fig, ax = plt.subplots(figsize=(6, 4))
    plotted = False
    for label, path in evolution_summary_paths:
        data = maybe_load(path)
        if data is None:
            print(f"[fig1] skip {label}: {path} not found")
            continue
        rounds = [r["round"] for r in data["rounds"]]
        rates = [r["eval_metrics"]["success_rate"] for r in data["rounds"]]
        ax.plot(rounds, rates, marker="o", label=label)
        plotted = True
    if not plotted:
        print("[fig1] nothing to plot, skipping")
        plt.close(fig)
        return
    ax.set_xlabel("Evolution round")
    ax.set_ylabel("Success rate")
    ax.set_ylim(0, 1)
    ax.set_title("Success rate vs. evolution round")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig1_success_vs_round.png"), dpi=150)
    plt.close(fig)
    print("[fig1] saved")


def fig_world_model_accuracy(evolution_summary_paths, output_dir):
    """Figure derived from world-model result.json per round: best_val_mse
    vs. constant_state_val_mse, i.e. Day 12's acceptance test visualized."""
    fig, ax = plt.subplots(figsize=(6, 4))
    plotted = False
    for label, path in evolution_summary_paths:
        data = maybe_load(path)
        if data is None:
            continue
        rounds = [r["round"] for r in data["rounds"]]
        wm_mse = [r["world_model_result"]["best_val_mse"] for r in data["rounds"]]
        const_mse = [r["world_model_result"]["constant_state_val_mse"] for r in data["rounds"]]
        ax.plot(rounds, wm_mse, marker="o", label=f"{label}: learned model")
        ax.plot(rounds, const_mse, marker="x", linestyle="--", label=f"{label}: constant-state baseline")
        plotted = True
    if not plotted:
        print("[fig_wm] nothing to plot, skipping")
        plt.close(fig)
        return
    ax.set_xlabel("Evolution round")
    ax.set_ylabel("Validation MSE (log scale)")
    ax.set_yscale("log")
    ax.set_title("World model vs. constant-state baseline (Day 12 acceptance test)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig5_world_model_accuracy.png"), dpi=150)
    plt.close(fig)
    print("[fig_wm] saved")


def fig_skill_archive_growth(evolution_summary_paths, output_dir):
    fig, ax = plt.subplots(figsize=(6, 4))
    plotted = False
    for label, path in evolution_summary_paths:
        data = maybe_load(path)
        if data is None:
            continue
        rounds = [r["round"] for r in data["rounds"]]
        sizes = [r["archive_size"] for r in data["rounds"]]
        ax.plot(rounds, sizes, marker="o", label=label)
        plotted = True
    if not plotted:
        print("[fig6] nothing to plot, skipping")
        plt.close(fig)
        return
    ax.set_xlabel("Evolution round")
    ax.set_ylabel("Skill archive size |K_r|")
    ax.set_title("Skill archive growth")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig6_skill_archive_growth.png"), dpi=150)
    plt.close(fig)
    print("[fig6] saved")


def fig_safety(evolution_summary_paths, output_dir):
    fig, ax = plt.subplots(figsize=(6, 4))
    plotted = False
    for label, path in evolution_summary_paths:
        data = maybe_load(path)
        if data is None:
            continue
        rounds, rejection_rate = [], []
        for r in data["rounds"]:
            disc = r["discovery"]
            n_gen = disc.get("n_candidates_generated", 0)
            if n_gen == 0:
                continue
            rounds.append(r["round"])
            rejection_rate.append(disc["n_safety_rejected"] / n_gen)
        if rounds:
            ax.plot(rounds, rejection_rate, marker="o", label=label)
            plotted = True
    if not plotted:
        print("[fig7] nothing to plot, skipping")
        plt.close(fig)
        return
    ax.set_xlabel("Evolution round")
    ax.set_ylabel("Fraction of generated candidates safety-rejected")
    ax.set_ylim(0, 1)
    ax.set_title("Safety-filter rejection rate")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig7_safety_rejection_rate.png"), dpi=150)
    plt.close(fig)
    print("[fig7] saved")


def table_main_comparison(method_dirs, output_dir):
    """Day 26 'main performance table': one row per (method, seed) with
    mean +/- std where multiple seeds exist for the same method."""
    rows = []
    for method, pattern in method_dirs:
        for metrics_path in sorted(glob.glob(pattern)):
            m = maybe_load(metrics_path)
            if m is None:
                continue
            rows.append(dict(
                method=method, seed=m.get("seed"), success_rate=m.get("success_rate"),
                recovery_rate=m.get("recovery_rate"), rollout_count=m.get("rollout_count"),
                source=metrics_path,
            ))
    if not rows:
        print("[table_main] nothing found, skipping")
        return
    csv_path = os.path.join(output_dir, "table_main_comparison.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "seed", "success_rate", "recovery_rate", "rollout_count", "source"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[table_main] saved {csv_path} ({len(rows)} rows)")
    for row in rows:
        print(f"  {row['method']:16s} seed={row['seed']} success_rate={row['success_rate']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", default="results")
    parser.add_argument("--output_dir", default="results/figures")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    evolution_summaries = [
        ("F2S (dev, 10 ep/round)", os.path.join(args.results_root, "can/f2s_dev/seed_0/evolution_summary.json")),
        ("F2S (final, 50 ep/round)", os.path.join(args.results_root, "can/f2s_final/seed_0/evolution_summary.json")),
    ]
    fig_success_vs_round(evolution_summaries, args.output_dir)
    fig_world_model_accuracy(evolution_summaries, args.output_dir)
    fig_skill_archive_growth(evolution_summaries, args.output_dir)
    fig_safety(evolution_summaries, args.output_dir)

    method_dirs = [
        ("fixed_policy", os.path.join(args.results_root, "Can/fixed_policy/seed_*/round_0/metrics.json")),
        ("soe", os.path.join(args.results_root, "Can/soe/seed_*/round_0/metrics.json")),
    ]
    table_main_comparison(method_dirs, args.output_dir)


if __name__ == "__main__":
    main()
