#!/usr/bin/env bash
# Run inside a GPU allocation (normally via experiment/submit_ampere.sh).
# CHECK_SET=port       checks before the Re:Verse+ runs on Qwen2.5-VL-7B: HiViS single-image
#                      identity with experiment 10, fp32 exactness of HiViS and ViSpec-7B against
#                      AR on 3 page windows, and a bf16 smoke run of the three methods on 2 windows.
# CHECK_SET=diagnosis  diagnose_divergence for each line of DIAG_LIST
#                      (<comparison.json> <original run> <draft run> <base model path>), then
#                      offset_check for each line of ORACLE_LIST (the offset_check arguments
#                      without --output-dir). Both list files are copied into the run folder.
# Each check writes its own log under checks/. checks.json holds the command, exit code,
# pass or fail and key numbers of every check. The runner exits 1 if any check fails.
set -euo pipefail

experiment_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$experiment_dir/../../.." && pwd)"
cd "$repo_dir"

if [[ $# -gt 0 ]]; then
    echo "Usage: CHECK_SET={port|diagnosis} [DIAG_LIST=file] [ORACLE_LIST=file] [HIVIS_SINGLE_REF=run folder] [PYTHON_BIN=python] $0" >&2
    exit 2
fi
check_set="${CHECK_SET:-}"
if [[ "$check_set" != "port" && "$check_set" != "diagnosis" ]]; then
    echo "CHECK_SET must be 'port' or 'diagnosis' (got '${check_set}')." >&2
    exit 2
fi
if [[ "$check_set" == "diagnosis" ]]; then
    for var in DIAG_LIST ORACLE_LIST; do
        [[ -n "${!var:-}" && -f "${!var}" ]] || { echo "$var must name an existing file." >&2; exit 2; }
    done
fi

export HF_HOME="$repo_dir/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_ASSETS_CACHE="$HF_HOME/assets"
export HF_XET_CACHE="$HF_HOME/xet"
export PYTHONNOUSERSITE=1
base7b="${BASE_MODEL_PATH:-$repo_dir/models/Qwen2.5-VL-7B-Instruct}"
vispec7b="${SPEC_DIR:-$repo_dir/models/ViSpec-Qwen2.5-VL-7B-Instruct}"
hivis7b="${HIVIS_DIR:-$repo_dir/models/HiViS-Qwen2.5-VL-7B-Instruct}"
bench_dir="${BENCH_DIR:-$repo_dir/bench_data}"
# Experiment 10's HiViS arm on long TextVQA (job 412896, 30-3-8, image-last). Its outputs are
# identical in every job that ran this arm.
hivis_single_ref="${HIVIS_SINGLE_REF:-/home/aeternum/Research2/cv_systems_project/.claude/worktrees/hivis/experiment/10_qwen7b_methods/runs/20261121T220051Z_1a0f282bc280_long_hivis_image-last_1jGVHp}"
python_bin="${PYTHON_BIN:-/home/aeternum/anaconda3/envs/vipsec/bin/python}"
command -v "$python_bin" >/dev/null
export PATH="$(dirname "$python_bin"):$PATH"
if [[ -f "$repo_dir/tools/downloaded/m4c_evaluator.py" ]]; then
    export TEXTVQA_EVALUATOR="${TEXTVQA_EVALUATOR:-$repo_dir/tools/downloaded/m4c_evaluator.py}"
fi
if [[ "$check_set" == "port" ]]; then
    [[ -f "$hivis_single_ref/results/textvqa_test/hivis_qwen/test.jsonl" ]] || {
        echo "HIVIS_SINGLE_REF has no results/textvqa_test/hivis_qwen/test.jsonl: $hivis_single_ref" >&2; exit 2; }
fi

commit="$(git rev-parse HEAD)"
mkdir -p "$experiment_dir/../runs"
runs_dir="$(cd -- "$experiment_dir/../runs" && pwd)"
run_dir="$(mktemp -d "$runs_dir/$(date -u +%Y%m%dT%H%M%SZ)_${commit:0:12}_checks_${check_set}_XXXXXX")"
printf '%s\n' "$commit" > "$run_dir/commit.txt"
git status --short > "$run_dir/git-status.txt"
git diff HEAD --binary > "$run_dir/tracked-changes.patch"
cp "$experiment_dir/run.sh" "$run_dir/runner.sh"
cp "$repo_dir/experiment/capture_device.sh" "$run_dir/capture_device.sh"
cp "$repo_dir/experiment/check_cuda.sh" "$run_dir/check_cuda.sh"
for manifest in assets_qwen7b assets_reverse; do
    if [[ -f "$repo_dir/.preparation/$manifest.json" ]]; then
        cp "$repo_dir/.preparation/$manifest.json" "$run_dir/prepared-$manifest.json"
    fi
done
if [[ "$check_set" == "diagnosis" ]]; then
    cp "$DIAG_LIST" "$run_dir/diag-list.txt"
    cp "$ORACLE_LIST" "$run_dir/oracle-list.txt"
fi
mkdir -p "$run_dir/checks"

# Summarise every check into checks.json; returns 1 if any gate failed. Called from finish so a
# preempted or failed run still records the checks that finished.
summarise_checks() {
    "$python_bin" - "$run_dir" "$check_set" "$hivis_single_ref" <<'PY'
import glob, json, os, re, statistics, sys

run_dir, check_set, single_ref = sys.argv[1:4]
cmds = {}
for path in sorted(glob.glob(os.path.join(run_dir, "checks", "*.cmd.json"))):
    entry = json.load(open(path, encoding="utf-8"))
    cmds[entry["name"]] = entry


def log_text(name):
    path = os.path.join(run_dir, "checks", name + ".log")
    return open(path, encoding="utf-8", errors="replace").read() if os.path.exists(path) else ""


def rows_at(path):
    if not os.path.exists(path):
        return []
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def rows(name):
    return rows_at(os.path.join(run_dir, "results", name, "test.jsonl"))


def runtime(name):
    path = os.path.join(run_dir, "results", name, "test.runtime.json")
    return json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}


def accepted(name):
    return [a for r in rows(name) for a in r["choices"][0].get("acceptance_length", [])]


def mean(values):
    return statistics.mean(values) if values else None


def latency(name):
    rs = rows(name)
    return {
        "mean_wall_time_s": mean([r["choices"][0]["wall_time"][0] for r in rs]),
        "mean_ttft_s": mean([r["choices"][0]["ttft"][0] for r in rs]),
        "mean_new_tokens": mean([r["choices"][0]["new_tokens"][0] for r in rs]),
        "samples": len(rs),
    }


def tokens(name):
    return {r["question_id"]: r["choices"][0]["measurement"][0]["output_token_ids"] for r in rows(name)}


checks = []


def add(name, passed, numbers, rule, names):
    checks.append({
        "check": name,
        "rule": rule,
        "pass": bool(passed),
        "commands": [cmds[n]["command"] for n in names if n in cmds],
        "exit_codes": {n: cmds[n]["exit_code"] for n in names if n in cmds},
        "numbers": numbers,
    })


def exited(names):
    return all(n in cmds and cmds[n]["exit_code"] == 0 for n in names)


if check_set == "port":
    # 1. HiViS single-image identity with experiment 10.
    ref = {}
    for r in rows_at(os.path.join(single_ref, "results", "textvqa_test", "hivis_qwen", "test.jsonl")):
        m = r["choices"][0]["measurement"][0]
        ref[r["question_id"]] = (m["output_token_ids"], m["input_ids_sha256"])
    equal, misses = 0, []
    for r in rows("single_hivis"):
        m = r["choices"][0]["measurement"][0]
        got = (m["output_token_ids"], m["input_ids_sha256"])
        want = ref.get(r["question_id"])
        if want == got:
            equal += 1
        else:
            first = None
            if want:
                a, b = got[0], want[0]
                first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)) if a != b else None)
            misses.append({"question_id": r["question_id"], "ids_equal": bool(want) and want[1] == got[1], "first_token_mismatch": first})
    total = len(rows("single_hivis"))
    add("HiViS single-image identity", exited(["single_hivis"]) and equal == 5 and total == 5,
        {"equal": equal, "total": total, "misses": misses, "reference": single_ref},
        "5/5 HiViS TextVQA outputs equal experiment 10 in output_token_ids and input_ids_sha256", ["single_hivis"])

    # 2. fp32 exactness on page windows.
    fp32, ok = {}, True
    for name in ("fp32_hivis", "fp32_vispec"):
        path = os.path.join(run_dir, "checks", name, "offset_check.json")
        info = {}
        if os.path.exists(path):
            res = json.load(open(path, encoding="utf-8"))
            info = {"exact": res["summary"]["exact"], "questions": res["summary"]["questions"],
                    "first_mismatches": res["summary"]["first_mismatches"],
                    "settings": {k: res["settings"].get(k) for k in ("draft_method", "benchmark", "total_token", "depth", "top_k", "dtype")},
                    "gpu_uuid": res["environment"]["gpu"]["uuid"]}
        text = log_text(name)
        info["device_cuda0"] = bool(re.search(r"device=cuda:0\b", text))
        info["offload_or_meta_warning"] = bool(re.search(r"offload|meta device", text, re.I))
        fp32[name] = info
        ok = ok and info.get("exact") == 3 and info.get("questions") == 3 and info["device_cuda0"] and not info["offload_or_meta_warning"]
    add("fp32 exactness on page windows", exited(["fp32_hivis", "fp32_vispec"]) and ok, fp32,
        "offset_check exits 0 (AR == HiViS and AR == ViSpec-7B on 3/3 windows); model on cuda:0 with no offload",
        ["fp32_hivis", "fp32_vispec"])

    # 3. bf16 smoke on page windows.
    names = ("bf16_ar", "bf16_vispec", "bf16_hivis")
    rt = {}
    for n in names:
        r = runtime(n)
        rt[n] = {"draft_method": r.get("draft_method"), "base_model_revision": r.get("base_model_revision"),
                 "spec_model_revision": r.get("spec_model_revision"), "parameter_dtypes": r.get("parameter_dtypes"),
                 "gpu_uuid": [d["uuid"] for d in r.get("devices", [])]}
    base_rev, vis_rev, hv_rev = "cc594898137f460bfe9f0759e9844b3ce807cfb5", "e06635b54327df78a5eb82361791f2ccec9f41ab", "6dd8717d912a5626d6ad4e81b124c54bb8a08ef2"
    rt_ok = (
        rt["bf16_vispec"]["draft_method"] == "vispec" and rt["bf16_hivis"]["draft_method"] == "hivis"
        and all(rt[n]["base_model_revision"] == base_rev for n in rt)
        and rt["bf16_vispec"]["spec_model_revision"] == vis_rev and rt["bf16_hivis"]["spec_model_revision"] == hv_rev
        and all(rt[n]["parameter_dtypes"] == ["torch.bfloat16"] for n in rt)
    )
    nums = {"runtime": rt, "runtime_ok": rt_ok, "samples": {n: len(rows(n)) for n in names}}
    add("bf16 smoke on page windows", exited(list(names)) and rt_ok and all(len(rows(n)) == 2 for n in names), nums,
        "three runs exit 0 with 2/2 windows each; runtime.json shows draft_method, the pinned revisions and bf16",
        list(names))

    # Recorded, not gated.
    ar, vs, hv = tokens("bf16_ar"), tokens("bf16_vispec"), tokens("bf16_hivis")

    def agree(a, b):
        return {"equal": sum(1 for q in a if q in b and a[q] == b[q]), "total": len(a)}

    info = {"ar_vs_hivis_token_agreement": agree(ar, hv), "ar_vs_vispec_token_agreement": agree(ar, vs),
            "mean_accepted_draft_tokens": {n: mean(accepted(n)) for n in ("bf16_vispec", "bf16_hivis")},
            "latency": {n: latency(n) for n in names}}
    if all(info["latency"][n]["mean_wall_time_s"] for n in names):
        a = info["latency"]["bf16_ar"]["mean_wall_time_s"]
        info["speedup_vs_ar"] = {n: a / info["latency"][n]["mean_wall_time_s"] for n in ("bf16_vispec", "bf16_hivis")}
    checks.append({"check": "Recorded, not gated", "pass": True, "numbers": info})
else:
    for name, entry in cmds.items():
        passed, numbers, rule = entry["exit_code"] == 0, {}, "command exits 0"
        if name.startswith("diagnose_"):
            rule = "command exits 0 and diagnosis.json summary.gate_passes is true"
            path = os.path.join(run_dir, "checks", name, "diagnosis.json")
            if os.path.exists(path):
                summ = json.load(open(path, encoding="utf-8"))["summary"]
                numbers = {"gate_passes": summ["gate_passes"], "gate": summ["gate"],
                           "classes": summ["overall"]["classes"]}
                passed = passed and summ["gate_passes"] is True
            else:
                numbers, passed = {"diagnosis_json": "missing"}, False
        elif name.startswith("oracle_"):
            rule = "command exits 0 (offset_check exits 1 on a mismatch); settings recorded"
            path = os.path.join(run_dir, "checks", name, "offset_check.json")
            if os.path.exists(path):
                res = json.load(open(path, encoding="utf-8"))
                numbers = {"base_model_path": res["settings"].get("base_model_path"),
                           "draft_method": res["settings"].get("draft_method", "vispec"),
                           "benchmark": res["settings"].get("benchmark"),
                           "exact": res["summary"]["exact"], "questions": res["summary"]["questions"]}
            else:
                numbers, passed = {"offset_check_json": "missing"}, False
        add(name, passed, numbers, rule, [name])

result = {"check_set": check_set, "all_pass": bool(checks) and all(c["pass"] for c in checks), "checks": checks}
json.dump(result, open(os.path.join(run_dir, "checks.json"), "w", encoding="utf-8"), indent=2)
for c in checks:
    print(("PASS" if c["pass"] else "FAIL"), c["check"], json.dumps(c["numbers"])[:600])
sys.exit(0 if result["all_pass"] else 1)
PY
}

set +e
(
set -e
finish() {
    local status=$?
    trap - EXIT
    set +e
    summarise_checks
    local summary_status=$?
    if [[ "$status" -eq 0 ]]; then
        status=$summary_status
    fi
    printf 'finished_utc=%s\nexit_code=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$status" >> "$run_dir/metadata.txt"
    printf '%s\n' "$status" > "$run_dir/exit-code.txt"
    echo "Checks finished with exit code $status. Artifacts: $run_dir"
    exit "$status"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [[ -n "${SUBMIT_COMMAND:-}" ]]; then
    submit_command="$SUBMIT_COMMAND"; sbatch_extra="${SBATCH_EXTRA:-}"
else
    submit_command="unset"; sbatch_extra="unset"
fi
{
    printf 'started_utc=%s\ncommit=%s\nhost=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$commit" "$(hostname)"
    printf 'check_set=%s\nbench_dir=%s\n' "$check_set" "$bench_dir"
    printf 'base_model_7b=%s\nvispec_7b=%s\nhivis_7b=%s\n' "$base7b" "$vispec7b" "$hivis7b"
    printf 'hivis_single_ref=%s\n' "$hivis_single_ref"
    printf 'python=%s\nslurm_job_id=%s\ncuda_visible_devices=%s\n' "$(command -v "$python_bin")" "${SLURM_JOB_ID:-unset}" "${CUDA_VISIBLE_DEVICES:-unset}"
    printf 'slurm_restart_count=%s\n' "${SLURM_RESTART_COUNT:-0}"
    printf 'submit_command=%s\nsbatch_extra=%s\n' "$submit_command" "$sbatch_extra"
} > "$run_dir/metadata.txt"
echo "Run artifacts: $run_dir"
cat "$run_dir/metadata.txt"
if [[ -s "$run_dir/git-status.txt" ]]; then
    echo "Working tree has changes; see git-status.txt and tracked-changes.patch."
fi

"$python_bin" --version
"$python_bin" "$repo_dir/experiment/capture_packages.py" > "$run_dir/packages.txt"
bash "$run_dir/capture_device.sh" "$python_bin" > "$run_dir/device.txt" 2>&1
echo "Launch device report: $run_dir/device.txt"
"$python_bin" experiment/prepare_assets.py --repo-dir "$repo_dir" --check-only --suite qwen7b
"$python_bin" experiment/prepare_assets.py --repo-dir "$repo_dir" --check-only --suite reverse
bash "$run_dir/check_cuda.sh" "$python_bin"
export PYTHONUNBUFFERED=1

# run_check <name> <command...>: log to checks/<name>.log, record the command and exit code in
# checks/<name>.cmd.json, and keep going whatever the result.
run_check() {
    local name="$1"; shift
    local code
    printf '\n=== check %s ===\nRunning:' "$name"
    printf ' %q' "$@"
    printf '\n'
    local started; started="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    set +e
    "$@" 2>&1 | tee "$run_dir/checks/$name.log"
    code="${PIPESTATUS[0]}"
    set -e
    "$python_bin" - "$run_dir/checks/$name.cmd.json" "$name" "$code" "$started" "$@" <<'PY'
import json, sys
path, name, code, started, *cmd = sys.argv[1:]
json.dump({"name": name, "command": cmd, "exit_code": int(code), "started_utc": started}, open(path, "w", encoding="utf-8"), indent=2)
PY
    echo "check $name exit code $code"
}

evaluate_cmd() {  # <benchmark> <spec> <method> <name> <samples>
    local benchmark="$1" spec="$2" method="$3" name="$4" samples="$5"
    run_check "$name" "$python_bin" -m vispec.evaluation.evaluate \
        --method "$method" --benchmark "$benchmark" --prompt-style long \
        --base-model-path "$base7b" --spec-model-path "$spec" --data-dir "$bench_dir" \
        --answer-file "$run_dir/results/$name/test.jsonl" --model-id test \
        --num-samples "$samples" --max-new-token 1024 --temperature 0.0 \
        --depth 3 --top-k 8 --total-token 30 --num-q 2 --prompt-order image-last \
        --kv-capacity 16384 --kv-layout compact
}

offset_cmd() {  # <name> <draft-method> <spec>
    run_check "$1" "$python_bin" -m vispec.evaluation.offset_check \
        --draft-method "$2" --kv-retention 1.0 --kv-policy spatial \
        --benchmark reverse_summary --prompt-style long \
        --num-samples 3 --dtype float32 \
        --base-model-path "$base7b" --spec-model-path "$3" --data-dir "$bench_dir" \
        --total-token 30 --depth 3 --top-k 8 \
        --output-dir "$run_dir/checks/$1"
}

if [[ "$check_set" == "port" ]]; then
    # 1. HiViS on one image is unchanged: 5 long TextVQA questions against experiment 10.
    evaluate_cmd textvqa "$hivis7b" hivis single_hivis 5
    # 2. fp32 exactness on the first 3 page windows.
    offset_cmd fp32_hivis hivis "$hivis7b"
    offset_cmd fp32_vispec vispec "$vispec7b"
    # 3. bf16 smoke on the first 2 page windows.
    evaluate_cmd reverse_summary "$vispec7b" autoregressive bf16_ar 2
    evaluate_cmd reverse_summary "$vispec7b" vispec bf16_vispec 2
    evaluate_cmd reverse_summary "$hivis7b" hivis bf16_hivis 2
else
    n=0
    while read -r comparison original draft base_path; do
        [[ -z "${comparison:-}" || "$comparison" == \#* ]] && continue
        n=$((n + 1))
        run_check "diagnose_$n" "$python_bin" -m vispec.evaluation.diagnose_divergence \
            --comparison "$comparison" --original-run "$original" --draft-run "$draft" \
            --data-dir "$bench_dir" --base-model-path "$base_path" \
            --output-dir "$run_dir/checks/diagnose_$n"
    done < "$DIAG_LIST"
    n=0
    while read -r -a oracle_args; do
        [[ ${#oracle_args[@]} -eq 0 || "${oracle_args[0]}" == \#* ]] && continue
        n=$((n + 1))
        run_check "oracle_$n" "$python_bin" -m vispec.evaluation.offset_check \
            "${oracle_args[@]}" --output-dir "$run_dir/checks/oracle_$n"
    done < "$ORACLE_LIST"
fi

) 2>&1 | tee "$run_dir/output.log"
pipeline_status=("${PIPESTATUS[@]}")
if [[ "${pipeline_status[0]}" -ne 0 ]]; then
    exit "${pipeline_status[0]}"
fi
if [[ "${pipeline_status[1]}" -ne 0 ]]; then
    printf 'logging_exit_code=%s\n' "${pipeline_status[1]}" >> "$run_dir/metadata.txt"
fi
printf '%s\n' "${pipeline_status[1]}" > "$run_dir/exit-code.txt"
exit "${pipeline_status[1]}"
