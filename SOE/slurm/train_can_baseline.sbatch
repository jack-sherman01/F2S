#!/bin/bash
# F2S on NHR@FAU Alex: retrain the Day-3 SOE baseline policy on Can
# (500 epochs, 36 demos = 20_percent_train, configs/soe_can_lowdim_baseline.json,
# unmodified SOE train_single_gpu.py), then evaluate it with the F2S harness
# (30 in-distribution episodes, seed 0) for comparison with the recorded
# 73.3% Fixed Policy result from the original machine. The original
# checkpoint did not survive the machine move (*.ckpt is gitignored), and
# every F2S experiment needs this checkpoint as its starting point.
# Writes a NEW timestamped dir under results/can/soe/seed_0/round_0/logs/
# soe_can_lowdim_baseline/ and a new results/Can/fixed_policy_alex_retrain/
# dir; nothing existing is overwritten.
#   sbatch /anvme/workspace/b306dd11-f2s/F2S/SOE/slurm/train_can_baseline.sbatch
#SBATCH --job-name=f2s_can_baseline
#SBATCH --partition=a40,a100
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --export=NONE
#SBATCH --output=/anvme/workspace/b306dd11-f2s/F2S/SOE/install_logs/alex_train_can_baseline_%j.log
unset SLURM_EXPORT_ENV
set -eo pipefail
source /anvme/workspace/b306dd11-f2s/F2S/SOE/env.sh
cd "$SOE_ROOT"
echo "=== $(date) | node $(hostname) | job $SLURM_JOB_ID | $(nvidia-smi --query-gpu=name --format=csv,noheader) ==="
T0=$(date +%s)
(cd src && python train_single_gpu.py --config ../configs/soe_can_lowdim_baseline.json)
LOGDIR=$(ls -td results/can/soe/seed_0/round_0/logs/soe_can_lowdim_baseline/*/ | head -1)
CKPT=${LOGDIR}ckpt/policy_last.ckpt
test -f "$CKPT"
echo "trained checkpoint: $CKPT | training wall time: $(( $(date +%s) - T0 )) s"
tail -3 "${LOGDIR}log.txt"

echo "=== Fixed Policy eval, 30 episodes, seed 0 (recorded original-machine result: 73.3%) ==="
python scripts/run_method.py --method fixed_policy --checkpoint "$CKPT" \
    --config configs/soe_can_lowdim_baseline.json --seed 0 --num_episodes 30 \
    --output_dir results/Can/fixed_policy_alex_retrain/seed_0/round_0
cat results/Can/fixed_policy_alex_retrain/seed_0/round_0/metrics.json
echo "=== DONE $(date) ==="
