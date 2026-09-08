#!/bin/bash
# F2S on Alex: reproduce the Failure Replay baseline (main-table row 3).
# Procedure per SOE/README_F2S.md "Filling the remaining Final Acceptance
# Criteria gaps": build a combined RoboMimic-schema hdf5 from the original 36
# training demos plus the raw failed episodes of f2s_final round 0, then
# fine-tune the baseline checkpoint on it for 150 epochs with SOE's own
# unmodified train_single_gpu.py, then evaluate.
# Original machine: 13 failed episodes -> 63.3% / 66.7% / 66.7% in-distribution
# (vs. the 73.3% frozen baseline -- naive replay HURT by 10 points) and 36.0%
# on the Day-25 unseen configurations.
# configs/soe_can_lowdim_failure_replay_alex.json is a copy of the committed
# config differing only in three absolute paths (resume_ckpt now points at the
# Alex-retrained checkpoint; the original still points at the old machine's
# 2026-08-29 timestamped directory, which does not exist here). The committed
# config is left untouched.
#   sbatch /anvme/workspace/b306dd11-f2s/F2S/SOE/slurm/repro_failure_replay.sbatch
#SBATCH --job-name=f2s_repro_replay
#SBATCH --partition=a40,a100
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --export=NONE
#SBATCH --output=/anvme/workspace/b306dd11-f2s/F2S/SOE/install_logs/alex_repro_failure_replay_%j.log
unset SLURM_EXPORT_ENV
set -eo pipefail
source /anvme/workspace/b306dd11-f2s/F2S/SOE/env.sh
cd "$SOE_ROOT"
echo "=== $(date) | node $(hostname) | job $SLURM_JOB_ID | $(nvidia-smi --query-gpu=name --format=csv,noheader) ==="

FAILDIR=results/can/f2s_final_alex_random/seed_0/round_0/eval/episodes
echo "failed episodes available in $FAILDIR: $(python -c "
import glob,json
print(sum(1 for f in glob.glob('$FAILDIR/episode_*.json') if not json.load(open(f))['success']))")"

echo; echo "=== [1/3] build combined failure-replay dataset ==="
python -u scripts/build_failure_replay_dataset.py \
    --original_hdf5 "$SOE_ROOT/simulation/datasets/can/ph/low_dim_v141.hdf5" \
    --original_mask_key 20_percent_train \
    --failure_episode_dirs "$FAILDIR" \
    --output_hdf5 "$SOE_ROOT/results/can/failure_replay_dataset_alex.hdf5" \
    --new_mask_key train_with_failure_replay

echo; echo "=== [2/3] fine-tune baseline for 150 epochs (unmodified train_single_gpu.py) ==="
T0=$(date +%s)
(cd src && python -u train_single_gpu.py --config ../configs/soe_can_lowdim_failure_replay_alex.json)
echo "fine-tune wall time: $(( $(date +%s) - T0 )) s"
LOGDIR=$(ls -td results/can/failure_replay_alex/seed_0/round_0/logs/soe_can_lowdim_failure_replay/*/ | head -1)
CKPT="${LOGDIR}ckpt/policy_last.ckpt"
test -f "$CKPT"; echo "fine-tuned checkpoint: $CKPT"

echo; echo "=== [3/3] evaluate (original: 63.3 / 66.7 / 66.7 in-dist, 36.0 unseen) ==="
for seed in 0 1 2; do
  out="results/Can/failure_replay_alex/seed_${seed}/round_0"
  echo; echo "--- failure_replay seed=$seed ($(date +%H:%M:%S)) ---"
  python -u scripts/run_method.py --method failure_replay --checkpoint "$CKPT" \
      --config configs/soe_can_lowdim_baseline.json --seed "$seed" --num_episodes 30 \
      --output_dir "$out"
  cat "$out/metrics.json"
done
echo; echo "--- failure_replay unseen, 100 episodes ---"
python -u scripts/evaluate_unseen.py --method failure_replay --checkpoint "$CKPT" \
    --config configs/soe_can_lowdim_baseline.json --seed 0 --num_episodes 100 \
    --output_dir results/Can/failure_replay_alex/seed_0/unseen
cat results/Can/failure_replay_alex/seed_0/unseen/metrics.json
echo; echo "=== DONE $(date) ==="
