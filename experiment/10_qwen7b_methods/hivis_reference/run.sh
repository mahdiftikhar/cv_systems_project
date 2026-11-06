#!/usr/bin/env bash
# Reference run of HiViS's own code, compared with our port. Run inside a GPU allocation
# (normally via experiment/submit_ampere.sh; PROMPT_STYLE is required by the submitter and only
# recorded here). Each invocation creates a fresh run folder under experiment/10_qwen7b_methods/runs/.
#
# Order: samples.json (ours) -> HiViS side (hivis_side.py in the separate HiViS environment,
# offline, without credentials, from inside the run folder) -> our side (hivis_parity ours) ->
# the verdict (hivis_parity compare, from the EXIT trap, so a partial run is reported too).
# Prerequisites: the environment from setup_env.sh and the read-only HiViS checkout.
# ==========================================
# step 01 -- Enable strict error handling and enter the repository root.
# ==========================================
set -euo pipefail

experiment_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
parent_dir="$(cd -- "$experiment_dir/.." && pwd)"
repo_dir="$(cd -- "$experiment_dir/../../.." && pwd)"
cd "$repo_dir"

# ==========================================
# step 02 -- Reject positional arguments.
# ==========================================
if [[ $# -gt 0 ]]; then
    echo "Usage: [BASE_MODEL_PATH=...] [HIVIS_DIR=...] [BENCH_DIR=...] [PYTHON_BIN=python] [HIVIS_ENV=...] [HIVIS_CHECKOUT=...] $0" >&2
    exit 2
fi

# ==========================================
# step 03 -- Set model paths, data location, both environments and the fixed settings.
# ==========================================
export HF_HOME="$repo_dir/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_ASSETS_CACHE="$HF_HOME/assets"
export HF_XET_CACHE="$HF_HOME/xet"
export PYTHONNOUSERSITE=1
base_model="${BASE_MODEL_PATH:-$repo_dir/models/Qwen2.5-VL-7B-Instruct}"
hivis_draft="${HIVIS_DIR:-$repo_dir/models/HiViS-Qwen2.5-VL-7B-Instruct}"
bench_dir="${BENCH_DIR:-$repo_dir/bench_data}"
python_bin="${PYTHON_BIN:-/home/aeternum/anaconda3/envs/vipsec/bin/python}"
cache_dir="$(cd -P -- "$repo_dir/.cache" && pwd)"
hivis_env="${HIVIS_ENV:-$cache_dir/envs/hivis-ref}"
hivis_python="$hivis_env/bin/python"
hivis_checkout="${HIVIS_CHECKOUT:-$cache_dir/ext/HiViS}"
freeze_file="${hivis_env}.freeze.txt"
hivis_pin="6db36b241eaf733ec313cbc9b86206c0f20762c9"
# Fixed: HiViS's own evaluation settings (ge_hivis_answer.py), greedy, fp16, 20 samples.
temperature="0.0"
max_new_tokens=500
num_samples=20
total_token=60
depth=5
top_k=10

# ==========================================
# step 04 -- Check both Pythons and create a unique run folder with Git and script snapshots.
# ==========================================
command -v "$python_bin" >/dev/null
[[ -x "$hivis_python" ]] || { echo "HiViS environment not found: $hivis_python (run setup_env.sh)" >&2; exit 2; }
[[ -f "$freeze_file" ]] || { echo "Freeze file not found: $freeze_file" >&2; exit 2; }
[[ -d "$hivis_checkout/hivis" ]] || { echo "HiViS checkout not found: $hivis_checkout" >&2; exit 2; }
export PATH="$(dirname "$python_bin"):$PATH"
commit="$(git rev-parse HEAD)"
mkdir -p "$parent_dir/runs"
run_dir="$(mktemp -d "$parent_dir/runs/$(date -u +%Y%m%dT%H%M%SZ)_${commit:0:12}_hivis_reference_XXXXXX")"
printf '%s\n' "$commit" > "$run_dir/commit.txt"
git status --short > "$run_dir/git-status.txt"
git diff HEAD --binary > "$run_dir/tracked-changes.patch"
cp "$experiment_dir/run.sh" "$run_dir/runner.sh"
cp "$experiment_dir/hivis_side.py" "$run_dir/hivis_side.py"
cp "$repo_dir/experiment/capture_device.sh" "$run_dir/capture_device.sh"
cp "$repo_dir/experiment/check_cuda.sh" "$run_dir/check_cuda.sh"
for suite in qwen qwen7b; do
    if [[ -f "$repo_dir/.preparation/assets_${suite}.json" ]]; then
        cp "$repo_dir/.preparation/assets_${suite}.json" "$run_dir/prepared-assets-${suite}.json"
    fi
done

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
    # The verdict is the report: it also runs after a failure and fails its gates then.
    set +e
    report_cmd=("$python_bin" -m vispec.evaluation.hivis_parity compare --run-dir "$run_dir")
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
if [[ -n "${SUBMIT_COMMAND:-}" ]]; then
    submit_command="$SUBMIT_COMMAND"
    sbatch_extra="${SBATCH_EXTRA:-}"
else
    submit_command="unset"
    sbatch_extra="unset"
fi
{
    printf 'started_utc=%s\ncommit=%s\nhost=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$commit" "$(hostname)"
    printf 'base_model=%s\nspec_model=%s\nbench_dir=%s\ntemperature=%s\n' "$base_model" "$hivis_draft" "$bench_dir" "$temperature"
    printf 'python=%s\nhivis_python=%s\nhivis_env=%s\nhivis_checkout=%s\nhivis_pin=%s\n' \
        "$(command -v "$python_bin")" "$hivis_python" "$hivis_env" "$hivis_checkout" "$hivis_pin"
    printf 'slurm_job_id=%s\ncuda_visible_devices=%s\n' "${SLURM_JOB_ID:-unset}" "${CUDA_VISIBLE_DEVICES:-unset}"
    # A requeued attempt keeps the job ID and increments the restart count.
    printf 'slurm_restart_count=%s\n' "${SLURM_RESTART_COUNT:-0}"
    printf 'submit_command=%s\nsbatch_extra=%s\n' "$submit_command" "$sbatch_extra"
    printf 'method=hivis_reference\ndraft_method=hivis\ndepth=%s\ntop_k=%s\ntotal_token=%s\n' "$depth" "$top_k" "$total_token"
    printf 'dtype=float16\nbenchmark=textvqa\nprompt=hivis_own\nnum_samples=%s\nmax_new_tokens=%s\n' "$num_samples" "$max_new_tokens"
    printf 'ours_kv_layout=fixed\nours_kv_capacity=16384\nours_slim_prefill=true\n'
    printf 'submit_prompt_style=%s\n' "${PROMPT_STYLE:-unset}"
} > "$run_dir/metadata.txt"
echo "Run artifacts: $run_dir"
cat "$run_dir/metadata.txt"
if [[ -s "$run_dir/git-status.txt" ]]; then
    echo "Working tree has changes; see git-status.txt and tracked-changes.patch."
fi

# ==========================================
# step 07 -- Capture packages and device details, check assets, then check CUDA in both
#            environments.
# ==========================================
"$python_bin" --version
"$python_bin" "$repo_dir/experiment/capture_packages.py" > "$run_dir/packages.txt"
bash "$run_dir/capture_device.sh" "$python_bin" > "$run_dir/device.txt" 2>&1
echo "Launch device report: $run_dir/device.txt"
# A missing model, dataset, or scorer stops the run before any model loads.
"$python_bin" experiment/prepare_assets.py --repo-dir "$repo_dir" --check-only --suite qwen
"$python_bin" experiment/prepare_assets.py --repo-dir "$repo_dir" --check-only --suite qwen7b
bash "$run_dir/check_cuda.sh" "$python_bin"
bash "$run_dir/check_cuda.sh" "$hivis_python"
export PYTHONUNBUFFERED=1

# ==========================================
# step 08 -- Record the HiViS checkout and environment; both must be unchanged.
# ==========================================
check_checkout() {
    local label="$1" head status writable lines=0
    head="$(git -C "$hivis_checkout" rev-parse HEAD)"
    status="$(git -C "$hivis_checkout" status --porcelain)"
    writable="$(find "$hivis_checkout" -perm /222 -print -quit)"
    if [[ -n "$status" ]]; then
        lines="$(printf '%s\n' "$status" | wc -l)"
    fi
    {
        printf '%s_utc=%s\n' "$label" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        printf '%s_path=%s\n%s_head=%s\n' "$label" "$hivis_checkout" "$label" "$head"
        # git status --porcelain must print nothing; any lines it prints follow this count.
        printf '%s_status_porcelain_lines=%s\n' "$label" "$lines"
        if [[ -n "$status" ]]; then
            printf '%s\n' "$status"
        fi
        printf '%s_first_writable=%s\n' "$label" "${writable:-none}"
    } >> "$run_dir/hivis-checkout.txt"
    if [[ "$head" != "$hivis_pin" || -n "$status" || -n "$writable" ]]; then
        echo "HiViS checkout check failed ($label); see hivis-checkout.txt" >&2
        return 1
    fi
}
check_checkout before
cp "$freeze_file" "$run_dir/hivis-env.freeze.txt"
PIP_CONFIG_FILE=/dev/null "$hivis_python" -m pip freeze --disable-pip-version-check > "$run_dir/hivis-env.packages.txt"
if ! diff -u "$run_dir/hivis-env.freeze.txt" "$run_dir/hivis-env.packages.txt"; then
    echo "The HiViS environment changed since setup_env.sh wrote its freeze file." >&2
    exit 1
fi

# ==========================================
# step 09 -- Write samples.json: the first 20 seed-42 TextVQA val questions.
# ==========================================
samples_cmd=("$python_bin" -m vispec.evaluation.hivis_parity samples
    --run-dir "$run_dir" --data-dir "$bench_dir" --num-samples "$num_samples")
printf '\nRunning:'
printf ' %q' "${samples_cmd[@]}"
printf '\n'
"${samples_cmd[@]}"

# ==========================================
# step 10 -- HiViS side: offline, no credentials, no writes outside the run folder.
# ==========================================
# HF_HOME is an empty folder inside the run folder, so no stored token or cache is read.
# The working directory is inside the run folder; -I keeps it and user site-packages off
# sys.path, -B writes no bytecode, and hivis_side.py puts only the checkout on sys.path.
hivis_work="$run_dir/hivis_work"
mkdir -p "$hivis_work/hf_home"
hivis_cmd=("$hivis_python" -I -B -u "$experiment_dir/hivis_side.py"
    --run-dir "$run_dir" --hivis-dir "$hivis_checkout"
    --base-model-path "$base_model" --ea-model-path "$hivis_draft")
printf '\nRunning (HiViS environment):'
printf ' %q' "${hivis_cmd[@]}"
printf '\n'
(
    cd "$hivis_work"
    unset HF_TOKEN HUGGING_FACE_HUB_TOKEN PYTHONPATH HF_HUB_CACHE HUGGINGFACE_HUB_CACHE \
        HF_DATASETS_CACHE HF_ASSETS_CACHE HF_XET_CACHE
    export HF_HOME="$hivis_work/hf_home" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
        HF_DATASETS_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
    "${hivis_cmd[@]}"
)
check_checkout after
find "$hivis_work/hf_home" -mindepth 1 > "$run_dir/hivis-hf-home-contents.txt"
echo "HiViS HF_HOME after the run: $(wc -l < "$run_dir/hivis-hf-home-contents.txt") entries (hivis-hf-home-contents.txt)"

# ==========================================
# step 11 -- Our side: drafter isolation, end-to-end tokens, then the near-tie margins.
# ==========================================
ours_cmd=("$python_bin" -m vispec.evaluation.hivis_parity ours
    --run-dir "$run_dir" --base-model-path "$base_model" --spec-model-path "$hivis_draft")
printf '\nRunning:'
printf ' %q' "${ours_cmd[@]}"
printf '\n'
"${ours_cmd[@]}"

# ==========================================
# step 12 -- Stream combined stdout/stderr to the terminal and output.log.
# ==========================================
# This closes the subshell opened in step 05; its entire output passes through tee.
) 2>&1 | tee "$run_dir/output.log"
pipeline_status=("${PIPESTATUS[@]}")

# ==========================================
# step 13 -- Return evaluation failures first, otherwise record and return logging status.
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
