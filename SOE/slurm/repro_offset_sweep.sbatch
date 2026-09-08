#!/bin/bash
# F2S on Alex: reproduce the per-state intervention-offset sweep -- the run that
# found the project's only 2 Day-19-validated Can skills (original machine:
# 4544 real candidate executions, 15 real successes, 2 skills archived).
#
# scripts/evaluate_candidate_ranking_per_state_offset_sweep.py hardcodes both its
# 10 input episode directories AND its output_dir, and calls ensure_fresh_dir on
# the latter -- running it in the repo would DELETE the original machine's
# tracked records.json (4544 rows, the capstone analysis's data source),
# skill_archive.json, single_mode_cluster_model.pkl, summary.json and
# validation_results.json. So the unmodified script is run inside a symlink
# sandbox outside the repo: inputs point at the Alex-reproduced directories,
# and the output dir is a real directory inside the sandbox. Results are copied
# back afterwards as results/can/candidate_ranking_per_state_offset_sweep_alex/.
#   sbatch /anvme/workspace/b306dd11-f2s/F2S/SOE/slurm/repro_offset_sweep.sbatch
#SBATCH --job-name=f2s_repro_sweep
#SBATCH --partition=a40,a100
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --export=NONE
#SBATCH --output=/anvme/workspace/b306dd11-f2s/F2S/SOE/install_logs/alex_repro_offset_sweep_%j.log
unset SLURM_EXPORT_ENV
set -eo pipefail
source /anvme/workspace/b306dd11-f2s/F2S/SOE/env.sh
echo "=== $(date) | node $(hostname) | job $SLURM_JOB_ID | $(nvidia-smi --query-gpu=name --format=csv,noheader) ==="

SB="$F2S_WS/repro_sandbox/offset_sweep"
rm -rf "$SB"
mkdir -p "$SB/results/can" "$SB/results/Can/fixed_policy/seed_0/round_0"
ln -s "$SOE_ROOT/configs" "$SB/configs"
ln -s "$SOE_ROOT/results/can/world_model_h20diag_alex" "$SB/results/can/world_model_h20diag"
for r in 0 1 2; do
  mkdir -p "$SB/results/can/f2s_final/seed_0/round_$r/eval" \
           "$SB/results/can/f2s_dev/seed_0/round_$r/eval" \
           "$SB/results/can/f2s_dev_cem/seed_2/round_$r/eval"
  ln -s "$SOE_ROOT/results/can/f2s_final_alex_random/seed_0/round_$r/eval/episodes" \
        "$SB/results/can/f2s_final/seed_0/round_$r/eval/episodes"
  ln -s "$SOE_ROOT/results/can/f2s_dev_alex/seed_0/round_$r/eval/episodes" \
        "$SB/results/can/f2s_dev/seed_0/round_$r/eval/episodes"
  ln -s "$SOE_ROOT/results/can/f2s_dev_cem_alex/seed_2/round_$r/eval/episodes" \
        "$SB/results/can/f2s_dev_cem/seed_2/round_$r/eval/episodes"
done
ln -s "$SOE_ROOT/results/Can/fixed_policy_alex/seed_0/round_0/episodes" \
      "$SB/results/Can/fixed_policy/seed_0/round_0/episodes"

echo "=== sandbox input episode counts ==="
for d in $(python3 -c "
import re,sys
src=open('$SOE_ROOT/scripts/evaluate_candidate_ranking_per_state_offset_sweep.py').read()
blk=re.search(r'ALL_EPISODE_DIRS = \[(.*?)\]', src, re.S).group(1)
print('\n'.join(re.findall(r'\"([^\"]+)\"', blk)))"); do
  printf "  %-52s %3d\n" "$d" "$(ls -1 "$SB/$d"/episode_*.json 2>/dev/null | wc -l)"
done

export F2S_CAN_CKPT="$SOE_ROOT/results/can/soe/seed_0/round_0/logs/soe_can_lowdim_baseline/2026-09-04-22-44-00/ckpt/policy_last.ckpt"
test -f "$F2S_CAN_CKPT"
echo; echo "=== running unmodified sweep script (original: 4544 executions, 15 successes, 2 skills) ==="
T0=$(date +%s)
cd "$SB"
python -u "$SOE_ROOT/scripts/evaluate_candidate_ranking_per_state_offset_sweep.py"
echo "sweep wall time: $(( $(date +%s) - T0 )) s"

OUT="$SOE_ROOT/results/can/candidate_ranking_per_state_offset_sweep_alex"
rm -rf "$OUT"; mkdir -p "$OUT"
cp -v "$SB/results/can/candidate_ranking_per_state_offset_sweep"/* "$OUT"/

echo; echo "=== Alex vs original machine ==="
cd "$SOE_ROOT"
python - <<'PY'
import json
o = json.load(open("results/can/candidate_ranking_per_state_offset_sweep/summary.json"))
n = json.load(open("results/can/candidate_ranking_per_state_offset_sweep_alex/summary.json"))
for k in sorted(set(o) | set(n)):
    mark = "" if o.get(k) == n.get(k) else "   <-- differs"
    print(f"  {k:32s} original={str(o.get(k)):>12s}  Alex={str(n.get(k)):>12s}{mark}")
for tag, p in (("original","candidate_ranking_per_state_offset_sweep"),
               ("Alex","candidate_ranking_per_state_offset_sweep_alex")):
    a = json.load(open(f"results/can/{p}/skill_archive.json"))
    sk = a.get("skills", a if isinstance(a, list) else [])
    print(f"  {tag} archived skills: {len(sk)}")
PY
echo "=== DONE $(date) ==="
