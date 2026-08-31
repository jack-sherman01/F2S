#!/bin/bash
set -e
source /data/heng/F2S/SOE/env.sh
export CUDA_VISIBLE_DEVICES=1
cd /data/heng/F2S/SOE

CKPT_BASE=/data/heng/F2S/SOE/results/can/soe/seed_0/round_0/logs/soe_can_lowdim_baseline/2026-08-29-22-22-54/ckpt/policy_epoch_500_seed_0.ckpt
CKPT_REPLAY=/data/heng/F2S/SOE/results/can/failure_replay/seed_0/round_0/logs/soe_can_lowdim_failure_replay/2026-08-30-18-32-47/ckpt/policy_last.ckpt
ARCHIVE=/data/heng/F2S/SOE/results/can/candidate_ranking_per_state_offset_sweep/skill_archive.json
CLUSTER=/data/heng/F2S/SOE/results/can/candidate_ranking_per_state_offset_sweep/single_mode_cluster_model.pkl

echo "=== fixed_policy ==="
python -u scripts/evaluate_unseen.py --method fixed_policy --checkpoint "$CKPT_BASE" \
  --config configs/soe_can_lowdim_baseline.json --seed 0 --num_episodes 100 \
  --output_dir results/Can/fixed_policy/seed_0/unseen

echo "=== soe ==="
python -u scripts/evaluate_unseen.py --method soe --checkpoint "$CKPT_BASE" \
  --config configs/soe_can_lowdim_baseline.json --seed 0 --num_episodes 100 \
  --output_dir results/Can/soe/seed_0/unseen

echo "=== failure_replay ==="
python -u scripts/evaluate_unseen.py --method failure_replay --checkpoint "$CKPT_REPLAY" \
  --config configs/soe_can_lowdim_baseline.json --seed 0 --num_episodes 100 \
  --output_dir results/Can/failure_replay/seed_0/unseen

echo "=== f2s ==="
python -u scripts/evaluate_unseen.py --method f2s --checkpoint "$CKPT_BASE" \
  --config configs/soe_can_lowdim_baseline.json --seed 0 --num_episodes 100 \
  --archive_path "$ARCHIVE" --cluster_model_path "$CLUSTER" \
  --output_dir results/Can/f2s/seed_0/unseen

echo "=== ALL DONE ==="
