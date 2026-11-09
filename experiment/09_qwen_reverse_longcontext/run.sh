#!/usr/bin/env bash
# Run inside a GPU allocation (normally via experiment/submit_ampere.sh).
# Runs every Week 4 arm on the reverse_summary benchmark (Re:Verse+ manga windows) in one job:
# reference (ar, vispec), then kv, draft and combined sweeps. Every arm creates its own fresh run folder.
# PROMPT_STYLE is forced to long. QWEN_BENCHMARKS defaults to reverse_summary here.
# Override lists with REF_ARMS, KV_ARMS, DRAFT_ARMS, COMBINED_ARMS (an empty list is an error).
# Set KV_ARMS, DRAFT_ARMS or COMBINED_ARMS to "none" to skip that sweep, so the arms can be split
# across shorter jobs. The references always run first, so every job has its own.
set -euo pipefail

experiment_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$experiment_dir/../.." && pwd)"
cd "$repo_dir"
source "$repo_dir/experiment/qwen_arm_runner.sh"

if [[ $# -gt 0 ]]; then
    echo "Usage: [REF_ARMS=\"ar vispec\"] [KV_ARMS=\"<retention>:<policy>[:vispec|:autoregressive] ...|none\"] [DRAFT_ARMS=\"<retention>:<policy> ...|none\"] [COMBINED_ARMS=\"<retention>:<policy> ...|none\"] [BENCH_DIR=/absolute/path] [PYTHON_BIN=python] $0" >&2
    exit 2
fi
# Experiment 09 has one prompt style. A job shared with other experiments may carry another
# PROMPT_STYLE for them, so say so and force long here.
if [[ "${PROMPT_STYLE:-long}" != "long" ]]; then
    echo "Experiment 09 runs the long prompt style only; overriding PROMPT_STYLE='${PROMPT_STYLE}' with 'long'." >&2
fi
export PROMPT_STYLE=long
qwen_require_style
export QWEN_BENCHMARKS="${QWEN_BENCHMARKS:-reverse_summary}"

ref_arms="${REF_ARMS-ar vispec}"
kv_arms="${KV_ARMS-1.0:spatial 1.0:attention 0.75:spatial 0.5:spatial 0.5:attention 0.25:spatial 0.25:attention 0.5:spatial:autoregressive}"
draft_arms="${DRAFT_ARMS-0.5:spatial 0.25:spatial}"
combined_arms="${COMBINED_ARMS-0.5:spatial 0.25:spatial}"

# Validate every arm of every list before the first run folder is created.
qwen_validate_ref_arms "$ref_arms"
[[ "$kv_arms" == none ]] || qwen_validate_arms 1 "$kv_arms"
[[ "$draft_arms" == none ]] || qwen_validate_arms 0 "$draft_arms"
[[ "$combined_arms" == none ]] || qwen_validate_arms 0 "$combined_arms"

# A failed arm is recorded and the rest still run; exit 1 at the end if any sweep had a failure.
failed=0
sweep="reference"; arms="$ref_arms"; qwen_run_arms || failed=1
if [[ "$kv_arms" != none ]]; then sweep="kv"; arms="$kv_arms"; qwen_run_arms || failed=1; fi
if [[ "$draft_arms" != none ]]; then sweep="draft"; arms="$draft_arms"; qwen_run_arms || failed=1; fi
if [[ "$combined_arms" != none ]]; then sweep="combined"; arms="$combined_arms"; qwen_run_arms || failed=1; fi
exit "$failed"
