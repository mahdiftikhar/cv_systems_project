#!/usr/bin/env bash
# Run inside a GPU allocation (normally via experiment/submit_ampere.sh).
# Runs AR, ViSpec and HiViS on Qwen2.5-VL-7B over the Re:Verse+ page windows (reverse_summary).
# Every arm in METHOD_ARMS creates its own fresh run folder, in the order given.
# METHOD_ARMS is a space-separated list of <method>[:<tree>]:
#   method  ar, vispec or hivis
#   tree    30-3-8 (default) or 60-5-10, written as total-depth-top-k; ar takes no tree
# The page windows have one fixed layout (pages first, then the instruction), so an arm takes
# no prompt order. PROMPT_STYLE is forced to long, the only style of reverse_summary.
set -euo pipefail

experiment_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$experiment_dir/../.." && pwd)"
cd "$repo_dir"
source "$repo_dir/experiment/qwen_arm_runner.sh"

if [[ $# -gt 0 ]]; then
    echo "Usage: [METHOD_ARMS=\"ar vispec hivis\"] [BENCH_DIR=/absolute/path] [PYTHON_BIN=python] $0" >&2
    echo "  Each arm is <method>[:<tree>], with method ar|vispec|hivis and tree 30-3-8|60-5-10." >&2
    exit 2
fi
# A job shared with other experiments may carry another PROMPT_STYLE for them, so say so and
# force long here.
if [[ "${PROMPT_STYLE:-long}" != "long" ]]; then
    echo "Experiment 11 runs the long prompt style only; overriding PROMPT_STYLE='${PROMPT_STYLE}' with 'long'." >&2
fi
export PROMPT_STYLE=long
qwen_require_style
# 7B models and the 7B asset suite; each is overridable from the environment.
export QWEN_ASSET_SUITE="${QWEN_ASSET_SUITE:-qwen7b}"
export BASE_MODEL_PATH="${BASE_MODEL_PATH:-$repo_dir/models/Qwen2.5-VL-7B-Instruct}"
export SPEC_DIR="${SPEC_DIR:-$repo_dir/models/ViSpec-Qwen2.5-VL-7B-Instruct}"
export HIVIS_DIR="${HIVIS_DIR:-$repo_dir/models/HiViS-Qwen2.5-VL-7B-Instruct}"
export QWEN_BENCHMARKS="${QWEN_BENCHMARKS:-reverse_summary}"
sweep="methods"
arms="${METHOD_ARMS-ar vispec hivis}"
# Validate every arm before the first run folder is created.
qwen_validate_method_arms "$arms"
for spec in $arms; do
    qwen_parse_method_arm "$spec" || exit 2
    if [[ "$arm_order" != "image-last" ]]; then
        echo "Invalid arm '$spec': the page windows have a fixed layout and take no prompt order." >&2
        exit 2
    fi
done
# The Re:Verse+ manifest is checked here once, before any arm; every arm checks the 7B suite.
"${PYTHON_BIN:-/home/aeternum/anaconda3/envs/vipsec/bin/python}" experiment/prepare_assets.py \
    --repo-dir "$repo_dir" --check-only --suite reverse
qwen_run_arms
