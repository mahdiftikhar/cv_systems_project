#!/usr/bin/env bash
# Run inside a GPU allocation after activating the cluster Python environment.
# ==========================================
# step 01 -- Enable strict error handling and enter the repository root.
# ==========================================
set -euo pipefail

experiment_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$experiment_dir/../.." && pwd)"
cd "$repo_dir"

# ==========================================
# step 02 -- Reject positional arguments; configure this run with environment variables.
# ==========================================
if [[ $# -gt 0 ]]; then
    echo "Usage: [BENCH_DIR=/absolute/path] [SPEC_DIR=/absolute/path/or/hf-id] [TEMPERATURE=1.0] [PYTHON_BIN=python] $0" >&2
    exit 2
fi

# ==========================================
# step 03 -- Set model paths, data location, temperature, Python, and benchmark order.
# ==========================================
# Keep runtime downloads/caches inside the checkout, matching prepare.sh.
export HF_HOME="$repo_dir/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_ASSETS_CACHE="$HF_HOME/assets"
export HF_XET_CACHE="$HF_HOME/xet"
base_model="llava-hf/llava-v1.6-vicuna-7b-hf"
spec_default="JLKang/ViSpec-llava-v1.6-vicuna-7b-hf"
if [[ -f "$repo_dir/models/llava-v1.6-vicuna-7b-hf/config.json" ]]; then
    base_model="$repo_dir/models/llava-v1.6-vicuna-7b-hf"
fi
if [[ -f "$repo_dir/models/ViSpec-llava-v1.6-vicuna-7b-hf/config.json" ]]; then
    spec_default="$repo_dir/models/ViSpec-llava-v1.6-vicuna-7b-hf"
fi
base_model="${BASE_MODEL_PATH:-$base_model}"
spec_dir="${SPEC_DIR:-$spec_default}"
bench_dir="${BENCH_DIR:-$repo_dir/bench_data}"
if [[ -f "$repo_dir/tools/downloaded/m4c_evaluator.py" ]]; then
    export TEXTVQA_EVALUATOR="${TEXTVQA_EVALUATOR:-$repo_dir/tools/downloaded/m4c_evaluator.py}"
fi
if [[ -f "$bench_dir/gqa/testdev_balanced_questions.json" ]]; then
    export GQA_ANNOTATIONS="${GQA_ANNOTATIONS:-$bench_dir/gqa/testdev_balanced_questions.json}"
fi
temperature="${TEMPERATURE:-1.0}"
python_bin="${PYTHON_BIN:-$repo_dir/.venv/bin/python}"
benchmarks=(textvqa gqa mmvet)

# ==========================================
# step 04 -- Check Python and create a unique run folder with Git and script snapshots.
# ==========================================
command -v "$python_bin" >/dev/null
commit="$(git rev-parse HEAD)"
mkdir -p "$experiment_dir/runs"
run_dir="$(mktemp -d "$experiment_dir/runs/$(date -u +%Y%m%dT%H%M%SZ)_${commit:0:12}_XXXXXX")"
printf '%s\n' "$commit" > "$run_dir/commit.txt"
git status --short > "$run_dir/git-status.txt"
git diff HEAD --binary > "$run_dir/tracked-changes.patch"
cp "$experiment_dir/run.sh" "$run_dir/runner.sh"
cp "$repo_dir/experiment/capture_device.sh" "$run_dir/capture_device.sh"
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
    printf 'method=autoregressive\n'
    printf 'benchmarks=%s\n' "${benchmarks[*]}"
} > "$run_dir/metadata.txt"
echo "Run artifacts: $run_dir"
cat "$run_dir/metadata.txt"
if [[ -s "$run_dir/git-status.txt" ]]; then
    echo "Working tree has changes; see git-status.txt and tracked-changes.patch."
fi

# ==========================================
# step 07 -- Capture Python packages and device details before evaluation.
# ==========================================
"$python_bin" --version
"$python_bin" "$repo_dir/experiment/capture_packages.py" > "$run_dir/packages.txt"
bash "$run_dir/capture_device.sh" "$python_bin" > "$run_dir/device.txt" 2>&1
echo "Launch device report: $run_dir/device.txt"
export PYTHONUNBUFFERED=1

# ==========================================
# step 08 -- Build each evaluation command, running TextVQA, then GQA, then MM-Vet.
# ==========================================
# Selected modules and evaluation settings from the repository's baseline.sh.
# Run TextVQA first, then GQA and MM-Vet.
# Stop on the first failure so partial runs cannot appear successful.
for benchmark in "${benchmarks[@]}"; do
    cmd=("$python_bin" -m "vispec.evaluation.gen_baseline_answer_$benchmark"
        --base-model-path "$base_model"
        --spec-model-path "$spec_dir"
        --model-id test
        --bench-name "$run_dir/results/${benchmark}_test/baseline_llava"
        --temperature "$temperature")

    # ==========================================
    # step 09 -- Add the benchmark-specific data folder for GQA or TextVQA.
    # ==========================================
    case "$benchmark" in
        gqa) cmd+=(--data-folder "$bench_dir/gqa/") ;;
        textvqa) cmd+=(--data-folder "$bench_dir/textvqa") ;;
    esac

    # ==========================================
    # step 10 -- Print the exact command and run it; stop if evaluation fails.
    # ==========================================
    printf '\nRunning:'
    printf ' %q' "${cmd[@]}"
    printf '\n'
    "${cmd[@]}"
done

# ==========================================
# step 11 -- Stream combined stdout/stderr to the terminal and output.log.
# ==========================================
# This closes the subshell opened in step 05; its entire output passes through tee.
) 2>&1 | tee "$run_dir/output.log"
pipeline_status=("${PIPESTATUS[@]}")

# ==========================================
# step 12 -- Return evaluation failures first, otherwise record and return logging status.
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
