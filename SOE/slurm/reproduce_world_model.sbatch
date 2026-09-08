#!/bin/bash
# F2S on NHR@FAU Alex: reproduce the world-model chain from the Alex-retrained
# Day-3 baseline checkpoint. Runs EXISTING scripts with their originally
# recorded parameters -- no project code is modified by this job.
#
# Chain (matches SOE/README_F2S.md "Follow-up: quantified the world model's
# accuracy vs. horizon out to h=20"):
#   1. scripts/run_evolution.py, configs/f2s_final.yaml, 3 rounds, seed 0
#      -> round_0/eval/episodes = the 50 real episodes the world model was
#         trained on originally (original machine: 848 s total, round_0
#         success_rate 0.74, final archive size 0)
#   2. scripts/build_world_model_dataset.py  (train_ratio 0.8, seed 0)
#   3. scripts/train_world_model.py          (hyperparameters copied verbatim
#      from results/can/world_model_h20diag/config.yaml)
#   4. scripts/evaluate_world_model.py       (horizons 1 2 3 5 8 12 16 20)
#
# Writes to NEW *_alex output dirs; nothing recorded from the original
# machine is overwritten, so the numbers stay directly comparable.
#   sbatch /anvme/workspace/b306dd11-f2s/F2S/SOE/slurm/reproduce_world_model.sbatch
#SBATCH --job-name=f2s_repro_wm
#SBATCH --partition=a40,a100
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --export=NONE
#SBATCH --output=/anvme/workspace/b306dd11-f2s/F2S/SOE/install_logs/alex_reproduce_world_model_%j.log
unset SLURM_EXPORT_ENV
set -eo pipefail
source /anvme/workspace/b306dd11-f2s/F2S/SOE/env.sh
cd "$SOE_ROOT"
echo "=== $(date) | node $(hostname) | job $SLURM_JOB_ID | $(nvidia-smi --query-gpu=name --format=csv,noheader) ==="

CKPT="$SOE_ROOT/results/can/soe/seed_0/round_0/logs/soe_can_lowdim_baseline/2026-09-04-22-44-00/ckpt/policy_last.ckpt"
test -f "$CKPT" || { echo "missing Alex baseline checkpoint: $CKPT"; exit 1; }
echo "baseline checkpoint: $CKPT"

EVO_DIR=results/can/f2s_final_alex/seed_0
DS_DIR=results/can/world_model_dataset_h20diag_alex
WM_DIR=results/can/world_model_h20diag_alex

echo; echo "=== [1/4] 3-round evolution loop (original machine: 848 s, archive 0) ==="
T0=$(date +%s)
python scripts/run_evolution.py \
    --config configs/f2s_final.yaml \
    --task Can --seed 0 \
    --output_dir "$EVO_DIR" \
    --policy_config configs/soe_can_lowdim_baseline.json \
    --policy_ckpt "$CKPT"
echo "evolution wall time: $(( $(date +%s) - T0 )) s"
cat "$EVO_DIR/round_0/eval/metrics.json"

EP_DIR="$EVO_DIR/round_0/eval/episodes"
echo "episodes produced: $(ls -1d "$EP_DIR"/episode_* 2>/dev/null | wc -l)"

echo; echo "=== [2/4] build world-model dataset (80/20 episode-level split, seed 0) ==="
python scripts/build_world_model_dataset.py \
    --episode_dir "$EP_DIR" --train_ratio 0.8 --seed 0 --output_dir "$DS_DIR"

echo; echo "=== [3/4] train world model (hyperparams from world_model_h20diag/config.yaml) ==="
python scripts/train_world_model.py \
    --train_states "$DS_DIR/train_states.npy" \
    --train_actions "$DS_DIR/train_actions.npy" \
    --train_next_states "$DS_DIR/train_next_states.npy" \
    --val_states "$DS_DIR/val_states.npy" \
    --val_actions "$DS_DIR/val_actions.npy" \
    --val_next_states "$DS_DIR/val_next_states.npy" \
    --hidden_dim 256 --horizon 1 --ensemble_size 1 --epochs 50 \
    --batch_size 256 --lr 0.0003 --weight_decay 0.0001 \
    --gradient_clip_norm 1.0 --seed 0 --output_dir "$WM_DIR"
cat "$WM_DIR/result.json"

echo; echo "=== [4/4] multi-step evaluation out to h=20 ==="
python scripts/evaluate_world_model.py \
    --model_dir "$WM_DIR" --episode_dir "$EP_DIR" \
    --episode_split "$DS_DIR/episode_split.json" \
    --output_dir "$WM_DIR/multistep_eval" \
    --horizons 1 2 3 5 8 12 16 20
cat "$WM_DIR/multistep_eval/object_position_rmse_by_horizon.json"

echo; echo "=== reference (original machine) ==="
echo "  h=1 0.0039 | h=5 0.0183 | h=8 0.0279 | h=12 0.0394 | h=16 0.0501 | h=20 0.0604 (m)"
echo "=== DONE $(date) ==="
