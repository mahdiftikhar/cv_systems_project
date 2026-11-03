#!/usr/bin/env bash
# Run the keep-ratio sweep for both prompt styles so all runs share one GPU and allocation.
set -uo pipefail

experiment_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$experiment_dir/../.." && pwd)"
cd "$repo_dir"

if [[ $# -gt 0 ]]; then
    echo "Usage: [KEEP_PERCENTS=\"100 50 25 10 0\"] [ANSWER_STYLES=\"short long\"] [BENCH_DIR=/absolute/path] [SPEC_DIR=/absolute/path/or/hf-id] [PYTHON_BIN=python] $0" >&2
    exit 2
fi

python_bin="${PYTHON_BIN:-$repo_dir/.venv/bin/python}"
keep_percents="${KEEP_PERCENTS:-100 50 25 10 0}"
answer_styles="${ANSWER_STYLES:-short long}"
commit="$(git rev-parse HEAD)"
mkdir -p "$experiment_dir/runs"
sweep_dir="$(mktemp -d "$experiment_dir/runs/$(date -u +%Y%m%dT%H%M%SZ)_${commit:0:12}_XXXXXX")"
echo "Sweep artifacts: $sweep_dir"

# A failed ratio must not discard the others. The final exit code still reports it.
failed=0
for answer_style in $answer_styles; do
    style_dir="$sweep_dir/$answer_style"
    mkdir -p "$style_dir"
    bash "$experiment_dir/run_method.sh" autoregressive "$style_dir/autoregressive" "$answer_style" || failed=1
    for percent in $keep_percents; do
        ratio="$(awk -v percent="$percent" 'BEGIN { printf "%.2f", percent / 100 }')"
        label="$(printf 'keep_%03d' "$percent")"
        bash "$experiment_dir/run_method.sh" vispec "$style_dir/$label" "$answer_style" "$ratio" || failed=1
    done
    "$python_bin" -m vispec.evaluation.compare_pruning \
        --sweep-dir "$style_dir" \
        --output-dir "$style_dir/comparison" 2>&1 | tee "$style_dir/comparison.log"
    [[ "${PIPESTATUS[0]}" -eq 0 ]] || failed=1
done
echo "Sweep finished. failed=$failed Artifacts: $sweep_dir"
exit "$failed"
