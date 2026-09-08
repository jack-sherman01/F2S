#!/bin/bash
# F2S on Alex: the main-table f2s row and the Day-25 unseen-configuration
# evaluation, both unblocked by the reproduced offset-sweep skill archive
# (results/can/candidate_ranking_per_state_offset_sweep_alex/skill_archive.json,
# 2 Day-19-validated skills, same count as the original machine).
# Parameters follow install_logs/run_three_seed_final.sh and
# run_day25_unseen_all_methods.sh (30 in-distribution episodes per seed,
# 100 unseen episodes, seeds 0/1/2).
# --cluster_model_path reuses the original single_mode_cluster_model.pkl: it is
# a degenerate KMeans(n_clusters=1) with identity normalization, so predict()
# returns cluster 0 for any input, which is exactly what the Alex archive's
# failure_mode_id=0 skills need; no script in the repo generates this artifact.
#   sbatch /anvme/workspace/b306dd11-f2s/F2S/SOE/slurm/repro_f2s_and_unseen.sbatch
#SBATCH --job-name=f2s_repro_f2s
#SBATCH --partition=a40,a100
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --export=NONE
#SBATCH --output=/anvme/workspace/b306dd11-f2s/F2S/SOE/install_logs/alex_repro_f2s_and_unseen_%j.log
unset SLURM_EXPORT_ENV
set -eo pipefail
source /anvme/workspace/b306dd11-f2s/F2S/SOE/env.sh
cd "$SOE_ROOT"
echo "=== $(date) | node $(hostname) | job $SLURM_JOB_ID | $(nvidia-smi --query-gpu=name --format=csv,noheader) ==="

CKPT="$SOE_ROOT/results/can/soe/seed_0/round_0/logs/soe_can_lowdim_baseline/2026-09-04-22-44-00/ckpt/policy_last.ckpt"
ARCHIVE="$SOE_ROOT/results/can/candidate_ranking_per_state_offset_sweep_alex/skill_archive.json"
CLUSTER="$SOE_ROOT/results/can/candidate_ranking_per_state_offset_sweep/single_mode_cluster_model.pkl"
CONFIG=configs/soe_can_lowdim_baseline.json
test -f "$CKPT"; test -f "$ARCHIVE"; test -f "$CLUSTER"
echo "archive skills: $(python -c "import json;a=json.load(open('$ARCHIVE'));print(len(a.get('skills',a)))")"

echo; echo "########## f2s in-distribution, 30 episodes, seeds 0/1/2 (original: 80.0 / 73.3 / 70.0) ##########"
for seed in 0 1 2; do
  out="results/Can/f2s_alex/seed_${seed}/round_0"
  echo; echo "=== f2s seed=$seed ($(date +%H:%M:%S)) ==="
  python -u scripts/run_method.py --method f2s --checkpoint "$CKPT" \
      --config "$CONFIG" --seed "$seed" --num_episodes 30 \
      --archive_path "$ARCHIVE" --cluster_model_path "$CLUSTER" \
      --output_dir "$out"
  cat "$out/metrics.json"
done

echo; echo "########## Day-25 unseen configuration, 100 episodes, seed 0 ##########"
for spec in "fixed_policy|" "soe|" "f2s|--archive_path $ARCHIVE --cluster_model_path $CLUSTER"; do
  method="${spec%%|*}"; extra="${spec#*|}"
  out="results/Can/${method}_alex/seed_0/unseen"
  echo; echo "=== unseen ${method} ($(date +%H:%M:%S)) ==="
  python -u scripts/evaluate_unseen.py --method "$method" --checkpoint "$CKPT" \
      --config "$CONFIG" --seed 0 --num_episodes 100 \
      --output_dir "$out" $extra
  cat "$out/metrics.json"
done

echo; echo "=== SUMMARY: Alex vs original machine ==="
python - <<'PY'
import json, os
print("in-distribution (30 episodes/seed)")
for m, ref in (("fixed_policy",[0.7333,0.7333,0.70]), ("soe",[0.0]*3),
               ("failure_replay",[0.6333,0.6667,0.6667]), ("f2s",[0.80,0.7333,0.70]),
               ("unguided_latent_repair",[0.0]*3)):
    got=[]
    for s in (0,1,2):
        p=f"results/Can/{m}_alex/seed_{s}/round_0/metrics.json"
        got.append(round(json.load(open(p))["success_rate"],4) if os.path.exists(p) else None)
    print(f"  {m:24s} Alex={got}  original={ref}")
print("unseen configuration (100 episodes, seed 0)")
for m, ref in (("fixed_policy",0.45),("soe",0.0),("failure_replay",0.36),("f2s",0.45),("unguided_latent_repair",0.0)):
    p=f"results/Can/{m}_alex/seed_0/unseen/metrics.json"
    v=round(json.load(open(p))["success_rate"],4) if os.path.exists(p) else None
    print(f"  {m:24s} Alex={v}  original={ref}")
PY
echo "=== DONE $(date) ==="
