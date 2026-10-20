#!/usr/bin/env bash
# Submit one sbatch job that runs EXP_DIR/run.sh for each given experiment in order.
# Usage: PROMPT_STYLE=long|short [SBATCH_EXTRA="..."] bash experiment/submit_ampere.sh EXP_DIR [EXP_DIR...]
set -euo pipefail
[[ $# -ge 1 ]] || { echo "Usage: PROMPT_STYLE=long|short bash experiment/submit_ampere.sh EXP_DIR [EXP_DIR...]" >&2; exit 2; }
# The runners reject other styles too, but only after the job starts; fail before queueing.
if [[ "${PROMPT_STYLE:-}" != "long" && "${PROMPT_STYLE:-}" != "short" ]]; then
    echo "PROMPT_STYLE must be 'long' or 'short' (got '${PROMPT_STYLE:-}')." >&2
    exit 2
fi
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$here/.." && pwd)"
source "$here/ampere_env.sh"
[[ "$PYTHON_BIN" == /* && -x "$PYTHON_BIN" ]] || { echo "PYTHON_BIN is not executable: $PYTHON_BIN" >&2; exit 2; }
dirs=()
for d in "$@"; do
    [[ -d "$d" ]] || { echo "Experiment folder does not exist: $d" >&2; exit 2; }
    abs="$(cd -- "$d" && pwd)"
    [[ -f "$abs/run.sh" ]] || { echo "Missing run.sh in: $abs" >&2; exit 2; }
    dirs+=("$abs")
done
experiments="$(IFS=:; echo "${dirs[*]}")"
export REPO="$repo_dir" PROJECT_ROOT="$repo_dir" EXPERIMENTS="$experiments"
clear_inherited_slurm
mkdir -p "$repo_dir/logs"
cmd=(sbatch --partition="$CLUSTER_PARTITION" --gres="$CLUSTER_GPU_GRES"
    --exclude="$CLUSTER_EXCLUDE" --export=ALL
    --output="$repo_dir/logs/%x-%j.log" --error="$repo_dir/logs/%x-%j.log")
# shellcheck disable=SC2206
cmd+=(${SBATCH_EXTRA:-} "$here/run_sequence.sbatch")
# The job inherits these through --export=ALL, and each runner writes them to metadata.txt.
printf -v SUBMIT_COMMAND '%q ' "${cmd[@]}"
SUBMIT_COMMAND="${SUBMIT_COMMAND% }"
export SUBMIT_COMMAND SBATCH_EXTRA="${SBATCH_EXTRA:-}"
printf 'Submitting: %s\n' "$SUBMIT_COMMAND"
"${cmd[@]}"
