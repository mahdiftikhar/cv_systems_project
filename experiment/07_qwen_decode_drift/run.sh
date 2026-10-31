#!/usr/bin/env bash
# Run inside a GPU allocation (normally via experiment/submit_ampere.sh).
# Requires PROMPT_STYLE=long. Each invocation creates a fresh run folder.
# ==========================================
# step 01 -- Enable strict error handling and enter the repository root.
# ==========================================
set -euo pipefail

experiment_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$experiment_dir/../.." && pwd)"
cd "$repo_dir"

# ==========================================
# step 02 -- Reject positional arguments and require the long prompt style.
# ==========================================
if [[ $# -gt 0 ]]; then
    echo "Usage: PROMPT_STYLE=long [ANSWERS=/path/test.jsonl] [BENCH_DIR=/absolute/path] [PYTHON_BIN=python] $0" >&2
    exit 2
fi
prompt_style="${PROMPT_STYLE:-}"
# The focus case (TextVQA 35537, index 162) and the reference run are long-style only.
if [[ "$prompt_style" != "long" ]]; then
    echo "PROMPT_STYLE must be 'long' (got '${prompt_style}'); experiment 07 probes the long style only." >&2
    exit 2
fi

# ==========================================
# step 03 -- Set model paths, data, the reference answers, the focus case, and Python.
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
# Greedy decoding and the 1,024-token cap are fixed; the probe takes the fork's settings from
# the reference run's runtime.json. TEMPERATURE is ignored.
temperature="0.0"
max_new_tokens=1024
num_samples=100
benchmark=textvqa
python_bin="${PYTHON_BIN:-/home/aeternum/anaconda3/envs/vipsec/bin/python}"
# Week 2 long 02 compact reference of job 393336; its tokens equal Week 1's on 100/100.
reference_run="20261020T084113Z_66f492a5dc65_long_2P5sjB"
answers="${ANSWERS:-$repo_dir/experiment/02_qwen_autoregressive/runs/$reference_run/results/textvqa_test/autoregressive_qwen/test.jsonl}"
# Week 2's unexplained draft-only case: the fork emits "." (13), stock full recompute "**:" (95518).
focus="35537:162"
focus_tokens=(13 95518)
# A missing or empty reference must fail here, before a run folder exists; the 04 identity
# check recorded such a case as skipped and passed.
if [[ ! -f "$answers" || ! -s "$answers" ]]; then
    echo "Reference answers are missing or empty: $answers (set ANSWERS to the 02 test.jsonl)." >&2
    exit 2
fi
answers="$(cd -- "$(dirname -- "$answers")" && pwd)/$(basename -- "$answers")"

# ==========================================
# step 04 -- Check Python and create a unique run folder with Git and script snapshots.
# ==========================================
command -v "$python_bin" >/dev/null
export PATH="$(dirname "$python_bin"):$PATH"
commit="$(git rev-parse HEAD)"
mkdir -p "$experiment_dir/runs"
run_dir="$(mktemp -d "$experiment_dir/runs/$(date -u +%Y%m%dT%H%M%SZ)_${commit:0:12}_${prompt_style}_XXXXXX")"
printf '%s\n' "$commit" > "$run_dir/commit.txt"
git status --short > "$run_dir/git-status.txt"
git diff HEAD --binary > "$run_dir/tracked-changes.patch"
cp "$experiment_dir/run.sh" "$run_dir/runner.sh"
# The launch diff omits untracked files, so keep the probe source with the run.
cp "$repo_dir/vispec/evaluation/drift_probe.py" "$run_dir/drift_probe.py"
cp "$repo_dir/experiment/capture_device.sh" "$run_dir/capture_device.sh"
cp "$repo_dir/experiment/check_cuda.sh" "$run_dir/check_cuda.sh"
if [[ -f "$repo_dir/.preparation/assets_qwen.json" ]]; then
    cp "$repo_dir/.preparation/assets_qwen.json" "$run_dir/prepared-assets.json"
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
    # The probe writes no answers, so there is no report step; record the outcome only.
    set +e
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
# experiment/submit_ampere.sh exports the exact sbatch command and SBATCH_EXTRA (empty when
# no extra options were given); a direct run.sh or gpu_check.sh launch records "unset".
if [[ -n "${SUBMIT_COMMAND:-}" ]]; then
    submit_command="$SUBMIT_COMMAND"
    sbatch_extra="${SBATCH_EXTRA:-}"
else
    submit_command="unset"
    sbatch_extra="unset"
fi
{
    printf 'started_utc=%s\ncommit=%s\nhost=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$commit" "$(hostname)"
    printf 'base_model=%s\nspec_model=%s\nbench_dir=%s\ntemperature=%s\n' "$base_model" "$spec_dir" "$bench_dir" "$temperature"
    printf 'python=%s\nslurm_job_id=%s\ncuda_visible_devices=%s\n' "$(command -v "$python_bin")" "${SLURM_JOB_ID:-unset}" "${CUDA_VISIBLE_DEVICES:-unset}"
    # A requeued attempt keeps the job ID and increments the restart count.
    printf 'slurm_restart_count=%s\n' "${SLURM_RESTART_COUNT:-0}"
    printf 'submit_command=%s\nsbatch_extra=%s\n' "$submit_command" "$sbatch_extra"
    printf 'method=decode_drift_probe\nprompt_style=%s\nbenchmark=%s\n' "$prompt_style" "$benchmark"
    printf 'num_samples=%s\nmax_new_tokens=%s\n' "$num_samples" "$max_new_tokens"
    printf 'answers=%s\nanswers_sha256=%s\n' "$answers" "$(sha256sum "$answers" | cut -d' ' -f1)"
    printf 'focus=%s\nfocus_tokens=%s\n' "$focus" "${focus_tokens[*]}"
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
# step 08 -- Probe Part A (100 questions) and Part B (the focus case).
# ==========================================
cmd=("$python_bin" -m vispec.evaluation.drift_probe
    --answers "$answers"
    --data-dir "$bench_dir"
    --base-model-path "$base_model"
    --spec-model-path "$spec_dir"
    --num-samples "$num_samples"
    --prompt-style "$prompt_style"
    --focus "$focus"
    --focus-tokens "${focus_tokens[@]}"
    --output-dir "$run_dir/drift")
printf '\nRunning:'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"

# ==========================================
# step 09 -- Stream combined stdout/stderr to the terminal and output.log.
# ==========================================
# This closes the subshell opened in step 05; its entire output passes through tee.
) 2>&1 | tee "$run_dir/output.log"
pipeline_status=("${PIPESTATUS[@]}")

# ==========================================
# step 10 -- Return probe failures first, otherwise record and return logging status.
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
