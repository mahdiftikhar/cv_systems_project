#!/usr/bin/env bash
# Use the prepared project .venv on a cluster compute node.
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# sbatch executes a copy of this file from its spool directory.
if [[ ! -f "$repo_dir/experiment/prepare_assets.py" ]]; then
    repo_dir="${SLURM_SUBMIT_DIR:-$repo_dir}"
fi
if [[ ! -f "$repo_dir/experiment/prepare_assets.py" ]]; then
    echo "Cannot find the repository; submit prepare.sh from the repo root." >&2
    exit 2
fi
python_bin="${PYTHON_BIN:-$repo_dir/.venv/bin/python}"
command -v "$python_bin" >/dev/null

case "${1:-}" in
    --help|-h) exec "$python_bin" "$repo_dir/experiment/prepare_assets.py" --help ;;
    --check-only) ;;
    *)
        if [[ -z "${SLURM_JOB_ID:-}" ]]; then
            echo "Run preparation inside a Slurm compute allocation (or submit prepare.sh with sbatch)." >&2
            echo "Use bash prepare.sh --check-only for an offline check without downloads." >&2
            exit 2
        fi
        ;;
esac

mkdir -p "$repo_dir/.preparation/runs"
run_dir="$(mktemp -d "$repo_dir/.preparation/runs/$(date -u +%Y%m%dT%H%M%SZ)_XXXXXX")"
echo "Preparation log: $run_dir/output.log"
set +e
"$python_bin" -u "$repo_dir/experiment/prepare_assets.py" --repo-dir "$repo_dir" "$@" 2>&1 | tee "$run_dir/output.log"
statuses=("${PIPESTATUS[@]}")
status="${statuses[0]}"
if [[ "$status" -eq 0 ]]; then status="${statuses[1]}"; fi
printf '%s\n' "$status" > "$run_dir/exit-code.txt"
exit "$status"
