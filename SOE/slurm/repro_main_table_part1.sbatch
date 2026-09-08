#!/bin/bash
# F2S on Alex: reproduce the three-seed main-table rows that do NOT depend on
# the offset-sweep skill archive -- fixed_policy, soe, unguided_latent_repair.
# Same parameters as install_logs/run_three_seed_final.sh (30 episodes, seeds
# 0/1/2), with the Alex-retrained baseline checkpoint and the Alex-reproduced
# world model. Existing scripts are called unmodified.
# Writes to results/Can/<method>_alex/ -- the original machine's numbers in
# results/Can/<method>/ are left untouched for comparison.
#   sbatch /anvme/workspace/b306dd11-f2s/F2S/SOE/slurm/repro_main_table_part1.sbatch
#SBATCH --job-name=f2s_repro_main1
#SBATCH --partition=a40,a100
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --export=NONE
#SBATCH --output=/anvme/workspace/b306dd11-f2s/F2S/SOE/install_logs/alex_repro_main_table_part1_%j.log
unset SLURM_EXPORT_ENV
set -eo pipefail
source /anvme/workspace/b306dd11-f2s/F2S/SOE/env.sh
cd "$SOE_ROOT"
echo "=== $(date) | node $(hostname) | job $SLURM_JOB_ID | $(nvidia-smi --query-gpu=name --format=csv,noheader) ==="

CKPT_BASE="$SOE_ROOT/results/can/soe/seed_0/round_0/logs/soe_can_lowdim_baseline/2026-09-04-22-44-00/ckpt/policy_last.ckpt"
WM_DIR="$SOE_ROOT/results/can/world_model_h20diag_alex"
CONFIG=configs/soe_can_lowdim_baseline.json
NUM_EP=30
test -f "$CKPT_BASE"; test -f "$WM_DIR/best_model.pt"

for seed in 0 1 2; do
  for spec in "fixed_policy|" "soe|" "unguided_latent_repair|--world_model_dir $WM_DIR"; do
    method="${spec%%|*}"; extra="${spec#*|}"
    out="results/Can/${method}_alex/seed_${seed}/round_0"
    echo; echo "=== ${method} seed=${seed} ($(date +%H:%M:%S)) ==="
    python -u scripts/run_method.py --method "$method" --checkpoint "$CKPT_BASE" \
        --config "$CONFIG" --seed "$seed" --num_episodes $NUM_EP \
        --output_dir "$out" $extra
    cat "$out/metrics.json"
  done
done

echo; echo "=== SUMMARY (Alex) vs original machine ==="
python - <<'PY'
import json, os
ref = {"fixed_policy": [0.7333,0.7333,0.70], "soe": [0.0,0.0,0.0],
       "unguided_latent_repair": [0.0,0.0,0.0]}
for m, r in ref.items():
    got = []
    for s in (0,1,2):
        p = f"results/Can/{m}_alex/seed_{s}/round_0/metrics.json"
        got.append(json.load(open(p))["success_rate"] if os.path.exists(p) else None)
    print(f"{m:26s} Alex={got}  original={r}")
PY
echo "=== DONE $(date) ==="
