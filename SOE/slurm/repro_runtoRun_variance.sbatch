#!/bin/bash
# F2S on Alex: quantify run-to-run variance of the evaluation itself.
# Two runs of fixed_policy at seed 0 from the same checkpoint already gave
# 73.3% and 83.3% (22/30 vs 25/30) with byte-identical initial states in all
# 30 episodes -- the divergence starts at timestep 0 at ~2.6e-5 in the action
# output (nondeterministic CUDA kernels in the diffusion policy) and compounds
# over the 400-step closed loop, flipping 3 episodes. This runs the SAME
# configuration 5 more times to measure that spread directly, because the
# main table's headline gap (F2S 74.4% vs Fixed Policy 72.2%) is 2.2 points
# and cannot be interpreted without it.
#   sbatch /anvme/workspace/b306dd11-f2s/F2S/SOE/slurm/repro_runtoRun_variance.sbatch
#SBATCH --job-name=f2s_variance
#SBATCH --partition=a40,a100
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --export=NONE
#SBATCH --output=/anvme/workspace/b306dd11-f2s/F2S/SOE/install_logs/alex_runtorun_variance_%j.log
unset SLURM_EXPORT_ENV
set -eo pipefail
source /anvme/workspace/b306dd11-f2s/F2S/SOE/env.sh
cd "$SOE_ROOT"
echo "=== $(date) | node $(hostname) | job $SLURM_JOB_ID | $(nvidia-smi --query-gpu=name --format=csv,noheader) ==="
CKPT="$SOE_ROOT/results/can/soe/seed_0/round_0/logs/soe_can_lowdim_baseline/2026-09-04-22-44-00/ckpt/policy_last.ckpt"
test -f "$CKPT"
for rep in 1 2 3 4 5; do
  out="results/Can/fixed_policy_variance/rep_${rep}/seed_0/round_0"
  echo; echo "=== repeat ${rep}/5, identical config: fixed_policy seed 0, 30 episodes ($(date +%H:%M:%S)) ==="
  python -u scripts/run_method.py --method fixed_policy --checkpoint "$CKPT" \
      --config configs/soe_can_lowdim_baseline.json --seed 0 --num_episodes 30 \
      --output_dir "$out"
  cat "$out/metrics.json"
done
echo; echo "=== run-to-run spread, same checkpoint / same seed / same 30 initial states ==="
python - <<'PY'
import json, os, statistics as st
vals=[]
for tag, p in ([("retrain","results/Can/fixed_policy_alex_retrain/seed_0/round_0/metrics.json"),
                ("main1","results/Can/fixed_policy_alex/seed_0/round_0/metrics.json")]
               + [(f"rep{r}", f"results/Can/fixed_policy_variance/rep_{r}/seed_0/round_0/metrics.json") for r in range(1,6)]):
    if os.path.exists(p):
        d=json.load(open(p)); v=d["success_rate"]; vals.append(v)
        print(f"  {tag:8s} {v*100:5.1f}%  ({round(v*30)}/30)  mean_len={d['mean_episode_length']:.1f}")
if len(vals)>1:
    print(f"\n  n={len(vals)}  mean={st.mean(vals)*100:.1f}%  std={st.stdev(vals)*100:.1f} points  "
          f"range=[{min(vals)*100:.1f}, {max(vals)*100:.1f}]")
    print(f"  reference: the three-SEED std reported in the paper is 1.6-1.9 points;")
    print(f"             the F2S vs Fixed Policy gap it is compared against is 2.2 points.")
PY
echo "=== DONE $(date) ==="
