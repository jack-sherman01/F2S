#!/bin/bash
set -e
source /data/heng/F2S/SOE/env.sh
export CUDA_VISIBLE_DEVICES=1
cd /data/heng/F2S/SOE

CKPT_BASE=/data/heng/F2S/SOE/results/can/soe/seed_0/round_0/logs/soe_can_lowdim_baseline/2026-08-29-22-22-54/ckpt/policy_epoch_500_seed_0.ckpt
CKPT_REPLAY=/data/heng/F2S/SOE/results/can/failure_replay/seed_0/round_0/logs/soe_can_lowdim_failure_replay/2026-08-30-18-32-47/ckpt/policy_last.ckpt
ARCHIVE=/data/heng/F2S/SOE/results/can/candidate_ranking_per_state_offset_sweep/skill_archive.json
CLUSTER=/data/heng/F2S/SOE/results/can/candidate_ranking_per_state_offset_sweep/single_mode_cluster_model.pkl
WM_DIR=/data/heng/F2S/SOE/results/can/world_model_h20diag
CONFIG=configs/soe_can_lowdim_baseline.json
NUM_EP=30

run() {
  local method=$1 ckpt=$2 seed=$3 extra=$4
  local out=results/Can/${method}/seed_${seed}/round_0
  if [ -f "${out}/metrics.json" ]; then
    echo "=== SKIP ${method} seed=${seed} (already has metrics.json) ==="
    return
  fi
  echo "=== ${method} seed=${seed} ==="
  python -u scripts/run_method.py --method "$method" --checkpoint "$ckpt" \
    --config "$CONFIG" --seed "$seed" --num_episodes $NUM_EP \
    --output_dir "$out" $extra
}

for seed in 0 1 2; do
  run fixed_policy "$CKPT_BASE" $seed ""
  run soe "$CKPT_BASE" $seed ""
  run failure_replay "$CKPT_REPLAY" $seed ""
  run f2s "$CKPT_BASE" $seed "--archive_path $ARCHIVE --cluster_model_path $CLUSTER"
  run unguided_latent_repair "$CKPT_BASE" $seed "--world_model_dir $WM_DIR"
done

echo "=== ALL DONE ==="
