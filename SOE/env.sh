# F2S project environment setup -- NHR@FAU "Alex" GPU cluster (migrated
# here 2026-09-04 from the original /data/heng/F2S machine; the old file
# is kept at install_logs/env.sh.old_machine_data_heng for the record).
#
# Source this before any conda/pip/python command in this project:
#   source /anvme/workspace/b306dd11-f2s/F2S/SOE/env.sh
#
# Storage: on this cluster both $HOME (/home/hpc, 100GB quota) and $WORK
# (/home/atuin) are over quota for this user, so the conda env, package
# cache, pip cache, datasets, checkpoints and results all live inside the
# "f2s" hpc-workspace on /anvme (no block quota; check `ws_list f2s` for
# the expiry date and extend with `ws_extend f2s 30` before it runs out).
#
# GPUs: none on the login nodes. Anything that needs CUDA (policy training,
# rollouts, F2S evaluation) must go through Slurm, e.g.
#   sbatch slurm/smoketest.sbatch            # or
#   srun -p a40 --gres=gpu:a40:1 --time=01:00:00 --pty bash
# (a40 = sm_86, a100 = sm_80 -- both supported by the pinned torch 1.13
# cu117 wheel; do NOT use the rtxpro6k partition, Blackwell sm_120 is not.)

SOE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SOE_ROOT
export F2S_ROOT="$(dirname "$SOE_ROOT")"           # .../b306dd11-f2s/F2S
export F2S_WS="$(dirname "$F2S_ROOT")"             # /anvme/workspace/b306dd11-f2s
export F2S_ENV="$F2S_WS/conda/envs/f2s"

# Keep every cache off the (over-quota) home filesystems.
export CONDA_PKGS_DIRS="$F2S_WS/conda/pkgs"
export PIP_CACHE_DIR="$F2S_WS/.cache/pip"
export XDG_CACHE_HOME="$F2S_WS/.cache"
export TORCH_HOME="$F2S_WS/.cache/torch"
export TMPDIR="$F2S_WS/tmp"
export HF_HOME="$F2S_WS/.cache/huggingface"   # diffusers/huggingface_hub cache (~/.bashrc points HF_HOME at atuin, which is over quota)
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"     # ~/.bashrc also sets these two explicitly (they override HF_HOME)
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
mkdir -p "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$TORCH_HOME" "$TMPDIR" "$HF_HOME"

# Never pick up ~/.local site-packages from other projects.
export PYTHONNOUSERSITE=1

# Headless MuJoCo/robosuite. All F2S experiments use low-dim observations
# and never create a renderer, so no GL backend is needed at all;
# "disable" stops robosuite from importing a GLFW/X11 context at import
# time on display-less cluster nodes. (Set MUJOCO_GL=egl on a GPU node if
# you ever need offscreen video rendering.)
export MUJOCO_GL="${MUJOCO_GL:-disable}"

source /apps/python/3.12-miniforge/etc/profile.d/conda.sh
conda activate "$F2S_ENV"
