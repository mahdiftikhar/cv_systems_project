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
    echo "Usage: PROMPT_STYLE={long|short} [BENCH_DIR=/absolute/path] [PYTHON_BIN=python] [IDENTITY_REFERENCE=/path/test.jsonl] $0" >&2
    exit 2
fi
prompt_style="${PROMPT_STYLE:-}"
if [[ "$prompt_style" != "long" && "$prompt_style" != "short" ]]; then
    echo "PROMPT_STYLE must be 'long' or 'short' (got '${prompt_style}')." >&2
    exit 2
fi

# ==========================================
# step 03 -- Set model paths, data location, decoding settings, Python, and profile order.
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
# Fixed greedy decoding and the Week 1 settings; ignore TEMPERATURE.
temperature="0.0"
max_new_tokens=1024
num_samples=3
kv_capacity=16384
depth=3
top_k=8
total_token=30
num_q=2
python_bin="${PYTHON_BIN:-/home/aeternum/anaconda3/envs/vipsec/bin/python}"
benchmarks=(textvqa coco_caption)
methods=(autoregressive vispec)
# The evaluate.py helpers were factored out for the profiler, so the same job re-runs the first
# 2 TextVQA questions with evaluate.py and compares their token IDs with a Week 1 03 run.
identity_samples=2
case "$prompt_style" in
    long) week1_run="20261019T054357Z_a06dd479abba_long_kDp191" ;;
    short) week1_run="20261019T063023Z_a06dd479abba_short_ogIM31" ;;
esac
identity_reference="${IDENTITY_REFERENCE:-$repo_dir/experiment/03_qwen_vispec/runs/$week1_run/results/textvqa_test/vispec_qwen/test.jsonl}"

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
# The launch diff omits untracked files, so keep the profiler source with the run.
cp "$repo_dir/vispec/evaluation/profile_generation.py" "$run_dir/profile_generation.py"
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
    # Profiling writes no answers, so there is no report step; record the outcome only.
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
    printf 'method=decode_profile\nprofile_methods=%s\n' "${methods[*]}"
    printf 'depth=%s\ntop_k=%s\ntotal_token=%s\nnum_q=%s\n' "$depth" "$top_k" "$total_token" "$num_q"
    printf 'prompt_style=%s\nslim_prefill=true\nkv_capacity=%s\nnum_samples=%s\n' "$prompt_style" "$kv_capacity" "$num_samples"
    printf 'max_new_tokens=%s\n' "$max_new_tokens"
    printf 'benchmarks=%s\n' "${benchmarks[*]}"
    printf 'identity_samples=%s\nidentity_reference=%s\n' "$identity_samples" "$identity_reference"
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
# step 08 -- Profile both methods on TextVQA, then on COCO Captions.
# ==========================================
# Stop on the first failure so partial runs cannot appear successful.
for benchmark in "${benchmarks[@]}"; do
    for method in "${methods[@]}"; do
        cmd=("$python_bin" -m vispec.evaluation.profile_generation
            --method "$method"
            --benchmark "$benchmark"
            --prompt-style "$prompt_style"
            --base-model-path "$base_model"
            --spec-model-path "$spec_dir"
            --data-dir "$bench_dir"
            --output-dir "$run_dir/profile/${method}_${benchmark}"
            --num-samples "$num_samples"
            --max-new-token "$max_new_tokens"
            --temperature "$temperature"
            --depth "$depth"
            --top-k "$top_k"
            --total-token "$total_token"
            --num-q "$num_q"
            --kv-capacity "$kv_capacity")
        printf '\nRunning:'
        printf ' %q' "${cmd[@]}"
        printf '\n'
        "${cmd[@]}"
    done
done

# ==========================================
# step 09 -- Re-run the first TextVQA questions with evaluate.py and compare token IDs.
# ==========================================
identity_answers="$run_dir/identity_check/results/textvqa_test/vispec_qwen/test.jsonl"
cmd=("$python_bin" -m vispec.evaluation.evaluate
    --method vispec
    --benchmark textvqa
    --prompt-style "$prompt_style"
    --base-model-path "$base_model"
    --spec-model-path "$spec_dir"
    --data-dir "$bench_dir"
    --answer-file "$identity_answers"
    --model-id test
    --num-samples "$identity_samples"
    --max-new-token "$max_new_tokens"
    --temperature "$temperature"
    --depth "$depth"
    --top-k "$top_k"
    --total-token "$total_token"
    --num-q "$num_q"
    --kv-capacity "$kv_capacity")
printf '\nRunning:'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"
"$python_bin" - "$identity_answers" "$identity_reference" "$run_dir/identity_check/identity_check.json" <<'PY'
import json
import sys
from pathlib import Path

answers, reference, out = (Path(p) for p in sys.argv[1:4])


def token_ids(path):
    rows = (json.loads(line) for line in path.read_text().splitlines() if line.strip())
    return {r["question_id"]: r["choices"][0]["measurement"][0]["output_token_ids"] for r in rows}


new = token_ids(answers)
result = {"answers": str(answers), "reference": str(reference), "questions": []}
if not reference.is_file():
    result["status"] = "skipped_missing_reference"
    print(f"Identity check skipped: reference {reference} is missing")
else:
    ref = token_ids(reference)
    for qid, ids in new.items():
        result["questions"].append({
            "question_id": qid,
            "output_tokens": len(ids),
            "reference_output_tokens": len(ref[qid]) if qid in ref else None,
            "identical": ref.get(qid) == ids,
        })
    same = all(q["identical"] for q in result["questions"])
    result["status"] = "identical" if same else "different"
    print(f"Identity check: {result['status']} for {len(new)} questions against {reference}")
out.write_text(json.dumps(result, indent=2) + "\n")
sys.exit(1 if result["status"] == "different" else 0)
PY

# ==========================================
# step 10 -- Stream combined stdout/stderr to the terminal and output.log.
# ==========================================
# This closes the subshell opened in step 05; its entire output passes through tee.
) 2>&1 | tee "$run_dir/output.log"
pipeline_status=("${PIPESTATUS[@]}")

# ==========================================
# step 11 -- Return profiling failures first, otherwise record and return logging status.
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
