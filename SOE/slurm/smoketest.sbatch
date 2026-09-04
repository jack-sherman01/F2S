#!/bin/bash
# F2S on NHR@FAU Alex: GPU smoke test (re-runs the proposal's Day 1.3
# acceptance test on this cluster). Submit from anywhere:
#   sbatch /anvme/workspace/b306dd11-f2s/F2S/SOE/slurm/smoketest.sbatch
# Steps: GPU visible to torch -> 2-epoch DP training on 36 Can demos
# (configs/soe_can_lowdim_dev.json, unmodified SOE train_single_gpu.py)
# -> 2 real PickPlaceCan rollouts through SOE's own run.py (plain +
# exploration code paths) -> 2 episodes through the F2S evaluation harness
# (scripts/run_method.py --method fixed_policy) -> the simulator-backed
# F2S self-tests. Success rate is expected to be ~0 for a 2-epoch policy;
# this only proves the pipeline is wired correctly on this machine.
#SBATCH --job-name=f2s_smoketest
#SBATCH --partition=a40,a100
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --export=NONE
#SBATCH --output=/anvme/workspace/b306dd11-f2s/F2S/SOE/install_logs/alex_smoketest_%j.log
unset SLURM_EXPORT_ENV
set -eo pipefail
source /anvme/workspace/b306dd11-f2s/F2S/SOE/env.sh
cd "$SOE_ROOT"
echo "=== $(date) | node $(hostname) | job $SLURM_JOB_ID | partition $SLURM_JOB_PARTITION ==="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
python - <<'PY'
import torch
assert torch.cuda.is_available(), "torch cannot see the GPU"
x = torch.randn(2048, 2048, device="cuda"); y = (x @ x).sum().item()
print("torch", torch.__version__, "| GPU", torch.cuda.get_device_name(0), "| capability", torch.cuda.get_device_capability(0), "| matmul ok")
PY

echo "=== Day 1.3 (a): 2-epoch training, configs/soe_can_lowdim_dev.json ==="
(cd src && python train_single_gpu.py --config ../configs/soe_can_lowdim_dev.json)
LOGDIR=$(ls -td results/can/soe/seed_0/round_0/logs/soe_can_lowdim_dev/*/ | head -1)
CKPT=${LOGDIR}ckpt/policy_last.ckpt
echo "train log dir: $LOGDIR"; tail -4 "${LOGDIR}log.txt"; ls -la "${LOGDIR}ckpt"
test -f "$CKPT"

OUT=results/can/soe/seed_0/round_0/smoketest_alex_${SLURM_JOB_ID}
mkdir -p "$OUT"
echo "=== Day 1.3 (b): 2 rollouts via SOE's run.py ==="
(cd simulation && python run.py --agent "../$CKPT" --config ../configs/soe_can_lowdim_dev.json \
    --n_rollouts 2 --try_times 1 --seed 0 --dataset_path "../$OUT/rollouts.hdf5" --dataset_obs)
python - "$OUT/rollouts.hdf5" <<'PY'
import sys, h5py
f = h5py.File(sys.argv[1], "r"); demos = list(f["data"].keys())
d = f["data"][demos[0]]
print("rollouts.hdf5: %d episodes | actions %s | states %s" % (len(demos), d["actions"].shape, d["states"].shape))
PY

echo "=== F2S harness: run_method.py --method fixed_policy, 2 episodes ==="
python scripts/run_method.py --method fixed_policy --checkpoint "$CKPT" \
    --config configs/soe_can_lowdim_dev.json --seed 0 --num_episodes 2 \
    --output_dir "$OUT/run_method_fixed_policy"
cat "$OUT/run_method_fixed_policy/metrics.json"

echo "=== simulator-backed F2S self-tests (non-fatal) ==="
F2S_TEST_CKPT="$CKPT" python scripts/selftest_validator_perturbations.py || echo "selftest_validator_perturbations: FAILED (non-fatal, see above)"

echo "=== SMOKETEST PASSED $(date) ==="
