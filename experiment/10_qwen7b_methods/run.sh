#!/usr/bin/env bash
# Run inside a GPU allocation (normally via experiment/submit_ampere.sh).
# Requires PROMPT_STYLE=long|short. Every arm in METHOD_ARMS creates its own fresh run folder.
# METHOD_ARMS is a space-separated list of <method>[:<tree>][:<order>]:
#   method  ar, vispec or hivis
#   tree    30-3-8 (default) or 60-5-10, written as total-depth-top-k; ar takes no tree
#   order   image-last (default) or image-first
set -euo pipefail

experiment_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$experiment_dir/../.." && pwd)"
cd "$repo_dir"
source "$repo_dir/experiment/qwen_arm_runner.sh"

if [[ $# -gt 0 ]]; then
    echo "Usage: PROMPT_STYLE={long|short} [METHOD_ARMS=\"ar vispec hivis\"] [BENCH_DIR=/absolute/path] [PYTHON_BIN=python] $0" >&2
    echo "  Each arm is <method>[:<tree>][:<order>], with method ar|vispec|hivis, tree 30-3-8|60-5-10, order image-last|image-first." >&2
    exit 2
fi
qwen_require_style
# 7B models and the 7B asset suite; each is overridable from the environment.
export QWEN_ASSET_SUITE="${QWEN_ASSET_SUITE:-qwen7b}"
export BASE_MODEL_PATH="${BASE_MODEL_PATH:-$repo_dir/models/Qwen2.5-VL-7B-Instruct}"
export SPEC_DIR="${SPEC_DIR:-$repo_dir/models/ViSpec-Qwen2.5-VL-7B-Instruct}"
export HIVIS_DIR="${HIVIS_DIR:-$repo_dir/models/HiViS-Qwen2.5-VL-7B-Instruct}"
export QWEN_BENCHMARKS="${QWEN_BENCHMARKS:-textvqa coco_caption}"
sweep="methods"
arms="${METHOD_ARMS-ar vispec hivis}"
# Validate every arm before the first run folder is created.
qwen_validate_method_arms "$arms"
qwen_run_arms
