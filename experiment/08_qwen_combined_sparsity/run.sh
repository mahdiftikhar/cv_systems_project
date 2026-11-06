#!/usr/bin/env bash
# Run inside a GPU allocation (normally via experiment/submit_ampere.sh).
# Requires PROMPT_STYLE=long|short. Every arm in COMBINED_ARMS creates its own fresh run folder.
# Each arm applies the same retention and policy to the target KV cache and to the draft adaptor input.
set -euo pipefail

experiment_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$experiment_dir/../.." && pwd)"
cd "$repo_dir"
source "$repo_dir/experiment/qwen_arm_runner.sh"

if [[ $# -gt 0 ]]; then
    qwen_arm_usage COMBINED "" "$0"
    exit 2
fi
qwen_require_style
sweep="combined"
arms="${COMBINED_ARMS:-0.5:spatial}"
# Validate every arm before the first run folder is created. No method suffix: ViSpec only.
qwen_validate_arms 0 "$arms"
qwen_run_arms
