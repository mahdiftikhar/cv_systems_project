#!/usr/bin/env bash
# Run from a Newton login shell; this only submits, never evaluates locally.
set -euo pipefail
if [[ $# -ne 1 ]]; then
    echo "Usage: bash experiment/submit.sh /path/to/experiment_folder" >&2
    exit 2
fi
[[ -d "$1" ]] || { echo "Experiment folder does not exist: $1" >&2; exit 2; }
experiment_dir="$(cd -- "$1" && pwd)"
batch_file="$experiment_dir/run.sbatch"
[[ -f "$batch_file" && -r "$batch_file" ]] || { echo "Missing readable batch file: $batch_file" >&2; exit 2; }
repo_dir="$(cd -- "$experiment_dir/../.." && pwd)"
export PROJECT_ROOT="$repo_dir"
export REPO="$repo_dir"
export PYTHON_BIN="${PYTHON_BIN:-$repo_dir/.venv/bin/python}"
[[ "$PYTHON_BIN" == /* && -x "$PYTHON_BIN" ]] || { echo "Create the project .venv with uv or set an absolute PYTHON_BIN" >&2; exit 2; }
: "${NEWTON_ACCOUNT:?Export your Slurm account}"
: "${NEWTON_PARTITION:?Export your Slurm partition}"
: "${NEWTON_GPU_GRES:?Export the exact one-GPU GRES from sinfo}"
command -v sbatch >/dev/null
mkdir -p "$PROJECT_ROOT/logs"
cmd=(sbatch --account="$NEWTON_ACCOUNT" --partition="$NEWTON_PARTITION"
    --gres="$NEWTON_GPU_GRES" --export=ALL
    --output="$PROJECT_ROOT/logs/%x-%j.log"
    --error="$PROJECT_ROOT/logs/%x-%j.log"
    "$batch_file")
printf 'Submitting:'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"
