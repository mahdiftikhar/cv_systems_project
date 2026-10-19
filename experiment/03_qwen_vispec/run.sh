#!/usr/bin/env bash
# Run inside a GPU allocation (normally via experiment/submit_ampere.sh).
# Requires PROMPT_STYLE=long|short. Each invocation creates a fresh run folder.
# ==========================================
# step 01 -- Enable strict error handling and enter the repository root.
# ==========================================
set -euo pipefail

experiment_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$experiment_dir/../.." && pwd)"
cd "$repo_dir"

# ==========================================
# step 02 -- Reject positional arguments and require a valid prompt style.
# ==========================================
if [[ $# -gt 0 ]]; then
    echo "Usage: PROMPT_STYLE={long|short} [BENCH_DIR=/absolute/path] [PYTHON_BIN=python] [DEPTH=3] [TOP_K=8] [TOTAL_TOKEN=30] [NUM_Q=2] $0" >&2
    exit 2
fi
prompt_style="${PROMPT_STYLE:-}"
if [[ "$prompt_style" != "long" && "$prompt_style" != "short" ]]; then
    echo "PROMPT_STYLE must be 'long' or 'short' (got '${prompt_style}')." >&2
    exit 2
fi

# ==========================================
# step 03 -- Set model paths, data location, decoding settings, Python, and benchmark order.
# ==========================================
export HF_HOME="$repo_dir/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_ASSETS_CACHE="$HF_HOME/assets"
export HF_XET_CACHE="$HF_HOME/xet"
export PYTHONNOUSERSITE=1
base_model="${BASE_MODEL_PATH:-$repo_dir/models/Qwen2.5-VL-3B-Instruct}"
spec_dir="${SPEC_DIR:-$repo_dir/models/ViSpec-Qwen2.5-VL-3B-Instruct}"
bench_dir="${BENCH_DIR:-$repo_dir/bench_data}"
# Fixed greedy decoding; ignore TEMPERATURE.
temperature="0.0"
max_new_tokens=1024
num_samples=100
kv_capacity=16384
python_bin="${PYTHON_BIN:-/home/aeternum/anaconda3/envs/vipsec/bin/python}"
method="vispec"
method_dir="vispec_qwen"
depth="${DEPTH:-3}"
top_k="${TOP_K:-8}"
total_token="${TOTAL_TOKEN:-30}"
num_q="${NUM_Q:-2}"
benchmarks=(textvqa coco_caption)

# ==========================================
# step 04 -- Check Python and create a unique run folder with Git and script snapshots.
# ==========================================
command -v "$python_bin" >/dev/null
export PATH="$(dirname "$python_bin"):$PATH"
if [[ -f "$repo_dir/tools/downloaded/m4c_evaluator.py" ]]; then
    export TEXTVQA_EVALUATOR="${TEXTVQA_EVALUATOR:-$repo_dir/tools/downloaded/m4c_evaluator.py}"
fi
commit="$(git rev-parse HEAD)"
mkdir -p "$experiment_dir/runs"
run_dir="$(mktemp -d "$experiment_dir/runs/$(date -u +%Y%m%dT%H%M%SZ)_${commit:0:12}_${prompt_style}_XXXXXX")"
printf '%s\n' "$commit" > "$run_dir/commit.txt"
git status --short > "$run_dir/git-status.txt"
git diff HEAD --binary > "$run_dir/tracked-changes.patch"
cp "$experiment_dir/run.sh" "$run_dir/runner.sh"
cp "$repo_dir/experiment/capture_device.sh" "$run_dir/capture_device.sh"
cp "$repo_dir/experiment/check_cuda.sh" "$run_dir/check_cuda.sh"
if [[ -f "$repo_dir/.preparation/assets.json" ]]; then
    cp "$repo_dir/.preparation/assets.json" "$run_dir/prepared-assets.json"
fi

# ==========================================
# step 05 -- Start a logged subshell and register exit and interruption handlers.
# ==========================================
# Use a pipeline so logging also works on systems that restrict /dev/fd.
# Temporarily disable parent-shell errexit so pipeline failures can be recorded below.
set +e
(
set -e
finish() {
    local status=$?
    trap - EXIT
    # Report partial runs too; missing scores remain pending without blocking timing metrics.
    set +e
    report_cmd=("$python_bin" -m vispec.evaluation.report_metrics
        --run-dir "$run_dir" --bench-dir "$bench_dir" --evaluation-exit-code "$status")
    printf '\nReporting:'
    printf ' %q' "${report_cmd[@]}"
    printf '\n'
    "${report_cmd[@]}"
    report_status=$?
    printf 'report_exit_code=%s\n' "$report_status" >> "$run_dir/metadata.txt"
    if [[ "$status" -eq 0 && "$report_status" -ne 0 ]]; then
        status="$report_status"
    fi
    printf '%s\n' "$status" > "$run_dir/exit-code.txt"
    printf 'finished_utc=%s\nexit_code=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$status" >> "$run_dir/metadata.txt"
    echo "Run finished with exit code $status. Artifacts: $run_dir"
    exit "$status"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# ==========================================
# step 06 -- Record launch settings, host, scheduler job, and working-tree warning.
# ==========================================
{
    printf 'started_utc=%s\ncommit=%s\nhost=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$commit" "$(hostname)"
    printf 'base_model=%s\nspec_model=%s\nbench_dir=%s\ntemperature=%s\n' "$base_model" "$spec_dir" "$bench_dir" "$temperature"
    printf 'python=%s\nslurm_job_id=%s\ncuda_visible_devices=%s\n' "$(command -v "$python_bin")" "${SLURM_JOB_ID:-unset}" "${CUDA_VISIBLE_DEVICES:-unset}"
    printf 'method=vispec_speculative\ndepth=%s\ntop_k=%s\ntotal_token=%s\nnum_q=%s\n' "$depth" "$top_k" "$total_token" "$num_q"
    printf 'prompt_style=%s\nslim_prefill=true\nkv_capacity=%s\nnum_samples=%s\n' "$prompt_style" "$kv_capacity" "$num_samples"
    printf 'max_new_tokens=%s\n' "$max_new_tokens"
    printf 'benchmarks=%s\n' "${benchmarks[*]}"
} > "$run_dir/metadata.txt"
echo "Run artifacts: $run_dir"
cat "$run_dir/metadata.txt"
if [[ -s "$run_dir/git-status.txt" ]]; then
    echo "Working tree has changes; see git-status.txt and tracked-changes.patch."
fi

# ==========================================
# step 07 -- Capture packages and device details, check assets, then check CUDA.
# ==========================================
"$python_bin" --version
"$python_bin" "$repo_dir/experiment/capture_packages.py" > "$run_dir/packages.txt"
bash "$run_dir/capture_device.sh" "$python_bin" > "$run_dir/device.txt" 2>&1
echo "Launch device report: $run_dir/device.txt"
# A missing model, dataset, or scorer stops the run before any model loads.
"$python_bin" experiment/prepare_assets.py --repo-dir "$repo_dir" --check-only --suite qwen
bash "$run_dir/check_cuda.sh" "$python_bin"
export PYTHONUNBUFFERED=1

# ==========================================
# step 08 -- Build each evaluation command, running TextVQA, then COCO Captions.
# ==========================================
# Stop on the first failure so partial runs cannot appear successful.
for benchmark in "${benchmarks[@]}"; do
    cmd=("$python_bin" -m vispec.evaluation.evaluate
        --method "$method"
        --benchmark "$benchmark"
        --prompt-style "$prompt_style"
        --base-model-path "$base_model"
        --spec-model-path "$spec_dir"
        --data-dir "$bench_dir"
        --answer-file "$run_dir/results/${benchmark}_test/${method_dir}/test.jsonl"
        --model-id test
        --num-samples "$num_samples"
        --max-new-token "$max_new_tokens"
        --temperature "$temperature"
        --depth "$depth"
        --top-k "$top_k"
        --total-token "$total_token"
        --num-q "$num_q"
        --kv-capacity "$kv_capacity")

    # ==========================================
    # step 09 -- Print the exact command and run it; stop if evaluation fails.
    # ==========================================
    printf '\nRunning:'
    printf ' %q' "${cmd[@]}"
    printf '\n'
    "${cmd[@]}"
done

# ==========================================
# step 10 -- Stream combined stdout/stderr to the terminal and output.log.
# ==========================================
# This closes the subshell opened in step 05; its entire output passes through tee.
) 2>&1 | tee "$run_dir/output.log"
pipeline_status=("${PIPESTATUS[@]}")

# ==========================================
# step 11 -- Return evaluation failures first, otherwise record and return logging status.
# ==========================================
if [[ "${pipeline_status[0]}" -ne 0 ]]; then
    exit "${pipeline_status[0]}"
fi
# A logging failure also makes the run unsuccessful.
if [[ "${pipeline_status[1]}" -ne 0 ]]; then
    printf 'logging_exit_code=%s\n' "${pipeline_status[1]}" >> "$run_dir/metadata.txt"
fi
printf '%s\n' "${pipeline_status[1]}" > "$run_dir/exit-code.txt"
exit "${pipeline_status[1]}"
