#!/usr/bin/env bash
# Run inside a GPU allocation (normally via experiment/submit_ampere.sh).
# Requires PROMPT_STYLE=long|short. Every arm in KV_ARMS creates its own fresh run folder.
set -euo pipefail

experiment_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$experiment_dir/../.." && pwd)"
cd "$repo_dir"
source "$repo_dir/experiment/qwen_arm_runner.sh"

if [[ $# -gt 0 ]]; then
    qwen_arm_usage KV "[:vispec|:autoregressive]" "$0"
    exit 2
fi
qwen_require_style
sweep="kv"
arms="${KV_ARMS:-0.5:spatial}"
# Validate every arm before the first run folder is created.
qwen_validate_arms 1 "$arms"
qwen_run_arms
