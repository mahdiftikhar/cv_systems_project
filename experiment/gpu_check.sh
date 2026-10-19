#!/usr/bin/env bash
# Usage: [SRUN_EXTRA="-w c1-8 --time=00:05:00"] bash experiment/gpu_check.sh CMD...
# Runs CMD on one pinned Ampere GPU via srun. Later options (SRUN_EXTRA) win.
set -euo pipefail
[[ $# -ge 1 ]] || { echo "Usage: bash experiment/gpu_check.sh CMD..." >&2; exit 2; }
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$here/ampere_env.sh"
clear_inherited_slurm
# shellcheck disable=SC2086
exec srun --ntasks=1 --partition="$CLUSTER_PARTITION" --gres="$CLUSTER_GPU_GRES" \
    --exclude="$CLUSTER_EXCLUDE" --cpus-per-task=4 --mem=32G --time=01:00:00 \
    ${SRUN_EXTRA:-} "$@"
