#!/bin/bash
# F2S on Alex: reproduce the Lift cross-task pressure test (README_F2S.md
# "Pressure test: does the mechanism work at all on an easier task?" and the
# three follow-ups). Trains the Lift baseline (300 epochs, unmodified
# train_single_gpu.py) into a NEW timestamped dir, then runs the four
# diagnostics in their original order.
# The diagnostics hardcode results/lift/world_model_diag and
# results/lift/world_model_ensemble_diag and call ensure_fresh_dir on them,
# which would DELETE 9 tracked result files from the original machine, so the
# unmodified scripts run inside a symlink sandbox and outputs are copied back
# as results/lift/*_alex/.
#   sbatch /anvme/workspace/b306dd11-f2s/F2S/SOE/slurm/repro_lift.sbatch
#SBATCH --job-name=f2s_repro_lift
#SBATCH --partition=a40,a100
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --export=NONE
#SBATCH --output=/anvme/workspace/b306dd11-f2s/F2S/SOE/install_logs/alex_repro_lift_%j.log
unset SLURM_EXPORT_ENV
set -eo pipefail
source /anvme/workspace/b306dd11-f2s/F2S/SOE/env.sh
cd "$SOE_ROOT"
echo "=== $(date) | node $(hostname) | job $SLURM_JOB_ID | $(nvidia-smi --query-gpu=name --format=csv,noheader) ==="

echo; echo "=== [1/5] Lift baseline (reusing the checkpoint trained by job 4201269) ==="
if [ -n "$F2S_SKIP_LIFT_TRAIN" ]; then
  echo "  F2S_SKIP_LIFT_TRAIN set -- skipping the 300-epoch retrain"
else
  T0=$(date +%s)
  (cd src && python -u train_single_gpu.py --config ../configs/soe_lift_lowdim_baseline.json)
  echo "training wall time: $(( $(date +%s) - T0 )) s"
fi
LOGDIR="$SOE_ROOT/$(ls -td results/lift/soe/seed_0/round_0/logs/soe_lift_lowdim_baseline/*/ | head -1)"
export F2S_LIFT_CKPT="${LOGDIR}ckpt/policy_last.ckpt"
test -f "$F2S_LIFT_CKPT"; echo "Lift checkpoint: $F2S_LIFT_CKPT"
tail -2 "${LOGDIR}log.txt"

SB="$F2S_WS/repro_sandbox/lift"
rm -rf "$SB"; mkdir -p "$SB/results/lift/soe/seed_0/round_0"
ln -s "$SOE_ROOT/configs" "$SB/configs"
ln -s "$SOE_ROOT/simulation" "$SB/simulation"
ln -s "$SOE_ROOT/results/lift/soe/seed_0/round_0/logs" "$SB/results/lift/soe/seed_0/round_0/logs"

run_diag () {   # $1 = script basename, $2 = step label
  echo; echo "=== [$2] $1 ($(date +%H:%M:%S)) ==="
  (cd "$SB" && python -u "$SOE_ROOT/scripts/$1.py")
}
run_diag diagnose_lift_full_pipeline        "2/5"
run_diag diagnose_lift_skill_archiving      "3/5"
run_diag diagnose_lift_robust_cem           "4/5"
run_diag diagnose_lift_ensemble_uncertainty "5/5"

echo; echo "=== copying sandbox results back as *_alex ==="
for d in world_model_diag world_model_ensemble_diag; do
  src="$SB/results/lift/$d"
  if [ -d "$src" ]; then
    dst="$SOE_ROOT/results/lift/${d}_alex"
    rm -rf "$dst"; mkdir -p "$dst"; cp -r "$src"/* "$dst"/ && echo "  -> results/lift/${d}_alex"
  fi
done

echo; echo "=== Alex vs original machine ==="
cd "$SOE_ROOT"
python - <<'PY'
import json, os
pairs = [("world_model_diag","diagnostic_summary.json"),
         ("world_model_diag","skill_validation_summary.json"),
         ("world_model_diag","robust_cem_summary.json"),
         ("world_model_ensemble_diag","uncertainty_vs_offset.json"),
         ("world_model_diag","result.json"),
         ("world_model_ensemble_diag","result.json")]
for d, f in pairs:
    po, pn = f"results/lift/{d}/{f}", f"results/lift/{d}_alex/{f}"
    if not (os.path.exists(po) and os.path.exists(pn)):
        print(f"--- {d}/{f}: missing ({'orig' if not os.path.exists(po) else 'alex'})"); continue
    o, n = json.load(open(po)), json.load(open(pn))
    print(f"--- {d}/{f}")
    if isinstance(o, dict):
        for k in sorted(set(o) | set(n)):
            vo, vn = o.get(k), n.get(k)
            if isinstance(vo, (int, float, str, bool, type(None))) and isinstance(vn, (int, float, str, bool, type(None))):
                mark = "" if vo == vn else "   <-- differs"
                print(f"    {k:34s} orig={str(vo):>14s}  Alex={str(vn):>14s}{mark}")
PY
echo "=== DONE $(date) ==="
