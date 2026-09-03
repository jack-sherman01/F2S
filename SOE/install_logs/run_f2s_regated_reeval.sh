#!/bin/bash
set -e
source /data/heng/F2S/SOE/env.sh
export CUDA_VISIBLE_DEVICES=1
cd /data/heng/F2S/SOE

CKPT_BASE=/data/heng/F2S/SOE/results/can/soe/seed_0/round_0/logs/soe_can_lowdim_baseline/2026-08-29-22-22-54/ckpt/policy_epoch_500_seed_0.ckpt
ARCHIVE=/data/heng/F2S/SOE/results/can/candidate_ranking_per_state_offset_sweep/skill_archive.json
CLUSTER=/data/heng/F2S/SOE/results/can/candidate_ranking_per_state_offset_sweep/single_mode_cluster_model.pkl
CONFIG=configs/soe_can_lowdim_baseline.json

echo "=== f2s three-seed in-distribution (gated) ==="
for seed in 0 1 2; do
  echo "--- seed=$seed ---"
  python -u scripts/run_method.py --method f2s --checkpoint "$CKPT_BASE" \
    --config "$CONFIG" --seed "$seed" --num_episodes 30 \
    --archive_path "$ARCHIVE" --cluster_model_path "$CLUSTER" \
    --output_dir results/Can/f2s/seed_${seed}/round_0
done

echo "=== f2s Day-25 unseen-config (gated) ==="
python -u scripts/evaluate_unseen.py --method f2s --checkpoint "$CKPT_BASE" \
  --config "$CONFIG" --seed 0 --num_episodes 100 \
  --archive_path "$ARCHIVE" --cluster_model_path "$CLUSTER" \
  --output_dir results/Can/f2s/seed_0/unseen

echo "=== ALL DONE ==="
