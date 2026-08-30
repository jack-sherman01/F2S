# F2S project environment setup.
# Source this before any conda/pip/python command in this project:
#   source /data/heng/F2S/SOE/env.sh
#
# Root filesystem (/) has very little free space on this machine, so all
# caches, conda envs, datasets, and checkpoints must live under /data.

export F2S_ROOT=/data/heng/F2S
export SOE_ROOT=/data/heng/F2S/SOE

# Keep pip/conda/HF/torch caches off the root filesystem.
export PIP_CACHE_DIR=/data/heng/.cache/pip
export XDG_CACHE_HOME=/data/heng/.cache
export TORCH_HOME=/data/heng/.cache/torch
export HF_HOME=/data/heng/.cache/huggingface
export CONDA_PKGS_DIRS=/data/heng/miniconda3/pkgs

source /data/heng/miniconda3/etc/profile.d/conda.sh
conda activate f2s
