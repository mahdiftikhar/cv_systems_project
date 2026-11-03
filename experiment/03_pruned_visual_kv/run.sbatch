#!/bin/bash
# Submit from the repository root: see README.md for the exact command.
#SBATCH --job-name=llava-prune
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.log
# evc26 had no usable GPU in job 942134; revisit after cluster support confirms repair.
#SBATCH --exclude=evc26

set -euo pipefail
: "${SLURM_JOB_ID:?Run this script with sbatch, not on a login node}"
: "${PYTHON_BIN:?Export an absolute PYTHON_BIN before submitting}"
trap 'status=$?; printf "batch_finished_utc=%s batch_exit_code=%s\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$status"' EXIT
printf 'batch_started_utc=%s slurm_job_id=%s experiment=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$SLURM_JOB_ID" "03_pruned_visual_kv"

# sbatch runs a copy of this file from its spool directory.
cd "$SLURM_SUBMIT_DIR"
[[ -f experiment/03_pruned_visual_kv/run.sh ]] || { echo "Submit from the repository root" >&2; exit 2; }
[[ "$PYTHON_BIN" == /* && -x "$PYTHON_BIN" ]] || { echo "Python is not executable: $PYTHON_BIN" >&2; exit 2; }

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
# Slurm owns CUDA_VISIBLE_DEVICES; do not replace its assigned GPU visibility.
# Dependencies and model/data caches must already be prepared. No installs here.
bash experiment/03_pruned_visual_kv/run.sh
