#!/bin/bash
# F2S on Alex: regenerate the three evolution episode directories the
# per-state offset sweep's failure pool is built from, each on the candidate
# generation path its original run actually used:
#   f2s_final  seed 0, --no_use_cem  (original: 96/160/128 candidates per round)
#   f2s_dev    seed 0, --no_use_cem  (pre-CEM commit f71e5fd)
#   f2s_dev_cem seed 2, CEM default  (commit dad9507)
# The already-completed CEM run of f2s_final is kept separately as
# results/can/f2s_final_alex/ (the Day-24 "Full F2S" configuration).
#   sbatch /anvme/workspace/b306dd11-f2s/F2S/SOE/slurm/repro_evolution_dirs.sbatch
#SBATCH --job-name=f2s_repro_evo
#SBATCH --partition=a40,a100
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --export=NONE
#SBATCH --output=/anvme/workspace/b306dd11-f2s/F2S/SOE/install_logs/alex_repro_evolution_dirs_%j.log
unset SLURM_EXPORT_ENV
set -eo pipefail
source /anvme/workspace/b306dd11-f2s/F2S/SOE/env.sh
cd "$SOE_ROOT"
echo "=== $(date) | node $(hostname) | job $SLURM_JOB_ID | $(nvidia-smi --query-gpu=name --format=csv,noheader) ==="

CKPT="$SOE_ROOT/results/can/soe/seed_0/round_0/logs/soe_can_lowdim_baseline/2026-09-04-22-44-00/ckpt/policy_last.ckpt"
test -f "$CKPT"
PC=configs/soe_can_lowdim_baseline.json

echo; echo "=== [1/3] f2s_final seed 0, random path (--no_use_cem) ==="
python -u scripts/run_evolution.py --config configs/f2s_final.yaml --task Can --seed 0 \
    --output_dir results/can/f2s_final_alex_random/seed_0 \
    --policy_config $PC --policy_ckpt "$CKPT" --no_use_cem

echo; echo "=== [2/3] f2s_dev seed 0, random path (--no_use_cem) ==="
python -u scripts/run_evolution.py --config configs/f2s_dev.yaml --task Can --seed 0 \
    --output_dir results/can/f2s_dev_alex/seed_0 \
    --policy_config $PC --policy_ckpt "$CKPT" --no_use_cem

echo; echo "=== [3/3] f2s_dev_cem seed 2, CEM path (default) ==="
python -u scripts/run_evolution.py --config configs/f2s_dev.yaml --task Can --seed 2 \
    --output_dir results/can/f2s_dev_cem_alex/seed_2 \
    --policy_config $PC --policy_ckpt "$CKPT"

echo; echo "=== candidate counts vs original machine ==="
python - <<'PY'
import json, os
pairs = [("f2s_final","f2s_final_alex_random",0), ("f2s_dev","f2s_dev_alex",0), ("f2s_dev_cem","f2s_dev_cem_alex",2)]
for orig, new, seed in pairs:
    print(f"--- {orig}")
    for r in range(3):
        po = f"results/can/{orig}/seed_{seed}/round_{r}/round_summary.json"
        pn = f"results/can/{new}/seed_{seed}/round_{r}/round_summary.json"
        o = json.load(open(po))["discovery"] if os.path.exists(po) else {}
        n = json.load(open(pn))["discovery"] if os.path.exists(pn) else {}
        print(f"  round {r}: n_candidates 原={o.get('n_candidates_generated')} Alex={n.get('n_candidates_generated')} | "
              f"skills 原={o.get('n_skills_added')} Alex={n.get('n_skills_added')}")
PY
echo "=== DONE $(date) ==="
