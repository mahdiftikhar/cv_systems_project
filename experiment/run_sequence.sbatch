#!/bin/bash
# Submit through experiment/submit_ampere.sh. Runs EXPERIMENTS (colon-separated
# directories) in order inside one allocation. Also works under srun.
#SBATCH --job-name=ampere-sequence
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
# A preempted job is requeued (the cluster default is JobRequeue=0) and reruns this script
# from the start, so every experiment gets a fresh run folder. Append keeps the log of
# each attempt in the same logs/<name>-<id>.log file.
#SBATCH --requeue
#SBATCH --open-mode=append

set -uo pipefail
: "${SLURM_JOB_ID:?Run this script under sbatch or srun}"
: "${REPO:?Submit through experiment/submit_ampere.sh}"
: "${EXPERIMENTS:?Set EXPERIMENTS to colon-separated experiment directories}"
trap 'status=$?; printf "batch_finished_utc=%s batch_exit_code=%s\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$status"' EXIT
printf 'batch_started_utc=%s slurm_job_id=%s slurm_restart_count=%s node=%s experiments=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$SLURM_JOB_ID" "${SLURM_RESTART_COUNT:-0}" "$(hostname)" "$EXPERIMENTS"

cd "$REPO"
source "$REPO/experiment/ampere_env.sh"
[[ "$PYTHON_BIN" == /* && -x "$PYTHON_BIN" ]] || { echo "Python is not executable: $PYTHON_BIN" >&2; exit 2; }

actual_gpu="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"
if [[ "$actual_gpu" != "$CLUSTER_EXPECTED_GPU" ]]; then
    echo "GPU mismatch: expected '$CLUSTER_EXPECTED_GPU' but node $(hostname) has '$actual_gpu'" >&2
    exit 3
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
# Slurm owns CUDA_VISIBLE_DEVICES; do not replace its assigned GPU visibility.

failed=0
IFS=: read -r -a exp_dirs <<< "$EXPERIMENTS"
for exp in "${exp_dirs[@]}"; do
    printf 'experiment_started_utc=%s dir=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$exp"
    bash "$exp/run.sh"
    rc=$?
    printf 'experiment_finished_utc=%s dir=%s exit_code=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$exp" "$rc"
    [[ $rc -eq 0 ]] || failed=1
done
exit "$failed"
