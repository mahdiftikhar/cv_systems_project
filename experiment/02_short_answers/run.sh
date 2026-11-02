#!/usr/bin/env bash
# Run both methods one after the other so they share one GPU and allocation.
set -euo pipefail

experiment_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$experiment_dir/../.." && pwd)"
cd "$repo_dir"

if [[ $# -gt 0 ]]; then
    echo "Usage: [BENCH_DIR=/absolute/path] [SPEC_DIR=/absolute/path/or/hf-id] [PYTHON_BIN=python] [DEPTH=3] [TOP_K=8] [TOTAL_TOKEN=30] [NUM_Q=2] $0" >&2
    exit 2
fi

python_bin="${PYTHON_BIN:-$repo_dir/.venv/bin/python}"
commit="$(git rev-parse HEAD)"
mkdir -p "$experiment_dir/runs"
pair_dir="$(mktemp -d "$experiment_dir/runs/$(date -u +%Y%m%dT%H%M%SZ)_${commit:0:12}_XXXXXX")"
echo "Pair artifacts: $pair_dir"

bash "$experiment_dir/run_method.sh" autoregressive "$pair_dir/autoregressive"
bash "$experiment_dir/run_method.sh" vispec "$pair_dir/vispec"

# Agreement and paired ratios. Keep the answers if the comparison guards reject the pair.
original_report="$(ls -d "$pair_dir"/autoregressive/reports/*/)"
draft_report="$(ls -d "$pair_dir"/vispec/reports/*/)"
"$python_bin" -m vispec.evaluation.compare_metrics \
    --original "$original_report" \
    --draft "$draft_report" \
    --output-dir "$pair_dir/comparison" 2>&1 | tee "$pair_dir/comparison.log"
exit "${PIPESTATUS[0]}"
