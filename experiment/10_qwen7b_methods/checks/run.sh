#!/usr/bin/env bash
# Run inside a GPU allocation (normally via experiment/submit_ampere.sh).
# CHECK_SET=port       port checks for the Qwen2.5-VL-7B methods (3B identity, fp32 losslessness,
#                      bf16 smoke, large tree, position ablation).
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
    echo "Usage: CHECK_SET={port|diagnosis} [DIAG_LIST=file] [ORACLE_LIST=file] [PYTHON_BIN=python] $0" >&2
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
base3b="$repo_dir/models/Qwen2.5-VL-3B-Instruct"
spec3b="$repo_dir/models/ViSpec-Qwen2.5-VL-3B-Instruct"
base7b="${BASE_MODEL_PATH:-$repo_dir/models/Qwen2.5-VL-7B-Instruct}"
vispec7b="${SPEC_DIR:-$repo_dir/models/ViSpec-Qwen2.5-VL-7B-Instruct}"
hivis7b="${HIVIS_DIR:-$repo_dir/models/HiViS-Qwen2.5-VL-7B-Instruct}"
bench_dir="${BENCH_DIR:-$repo_dir/bench_data}"
week2_root="/home/aeternum/Research2/cv_systems_project/experiment"
week2_ar="$week2_root/02_qwen_autoregressive/runs/20261020T084113Z_66f492a5dc65_long_2P5sjB"
week2_vispec="$week2_root/03_qwen_vispec/runs/20261020T090922Z_66f492a5dc65_long_wDElc3"
python_bin="${PYTHON_BIN:-/home/aeternum/anaconda3/envs/vipsec/bin/python}"
command -v "$python_bin" >/dev/null
export PATH="$(dirname "$python_bin"):$PATH"
if [[ -f "$repo_dir/tools/downloaded/m4c_evaluator.py" ]]; then
    export TEXTVQA_EVALUATOR="${TEXTVQA_EVALUATOR:-$repo_dir/tools/downloaded/m4c_evaluator.py}"
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
for manifest in assets_qwen assets_qwen7b; do
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
    "$python_bin" - "$run_dir" "$check_set" "$week2_ar" "$week2_vispec" <<'PY'
import glob, json, os, re, statistics, sys

run_dir, check_set, week2_ar, week2_vispec = sys.argv[1:5]
cmds = {}
for path in sorted(glob.glob(os.path.join(run_dir, "checks", "*.cmd.json"))):
    entry = json.load(open(path))
    cmds[entry["name"]] = entry


def log_text(name):
    path = os.path.join(run_dir, "checks", name + ".log")
    return open(path, errors="replace").read() if os.path.exists(path) else ""


def rows(name):
    path = os.path.join(run_dir, "results", name, "test.jsonl")
    if not os.path.exists(path):
        return []
    return [json.loads(line) for line in open(path) if line.strip()]


def runtime(name):
    path = os.path.join(run_dir, "results", name, "test.runtime.json")
    return json.load(open(path)) if os.path.exists(path) else {}


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
    # 1. 3B identity.
    ident = {}
    for name, week2 in (("id3b_ar", week2_ar), ("id3b_vispec", week2_vispec)):
        ref_dir = glob.glob(os.path.join(week2, "results", "textvqa_test", "*"))[0]
        ref = {}
        for line in open(os.path.join(ref_dir, "test.jsonl")):
            r = json.loads(line)
            m = r["choices"][0]["measurement"][0]
            ref[r["question_id"]] = (m["output_token_ids"], m["input_ids_sha256"])
        equal, misses = 0, []
        for r in rows(name):
            m = r["choices"][0]["measurement"][0]
            got = (m["output_token_ids"], m["input_ids_sha256"])
            if ref.get(r["question_id"]) == got:
                equal += 1
            else:
                want = ref.get(r["question_id"])
                first = None
                if want:
                    a, b = got[0], want[0]
                    first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)) if a != b else None)
                misses.append({"question_id": r["question_id"], "ids_equal": bool(want) and want[1] == got[1], "first_token_mismatch": first})
        ident[name] = {"equal": equal, "total": len(rows(name)), "misses": misses}
    add("3B identity", exited(["id3b_ar", "id3b_vispec"]) and all(v["equal"] == 5 and v["total"] == 5 for v in ident.values()),
        ident, "5/5 AR and 5/5 ViSpec equal Week 2 in output_token_ids and input_ids_sha256", ["id3b_ar", "id3b_vispec"])

    # 2. fp32 losslessness.
    fp32 = {}
    ok = True
    for name in ("fp32_image_last", "fp32_image_first"):
        path = os.path.join(run_dir, "checks", name, "offset_check.json")
        info = {}
        if os.path.exists(path):
            res = json.load(open(path))
            info = {"exact": res["summary"]["exact"], "questions": res["summary"]["questions"],
                    "first_mismatches": res["summary"]["first_mismatches"],
                    "settings": {k: res["settings"][k] for k in ("draft_method", "prompt_order", "total_token", "depth", "top_k", "dtype")},
                    "gpu_uuid": res["environment"]["gpu"]["uuid"]}
        text = log_text(name)
        info["device_cuda0"] = bool(re.search(r"device=cuda:0\b", text))
        info["offload_or_meta_warning"] = bool(re.search(r"offload|meta device", text, re.I))
        fp32[name] = info
        ok = ok and info.get("exact") == 5 and info.get("questions") == 5 and info["device_cuda0"] and not info["offload_or_meta_warning"]
    add("fp32 losslessness", exited(["fp32_image_last", "fp32_image_first"]) and ok, fp32,
        "offset_check exits 0 (AR == HiViS on 5/5) in both runs; model on cuda:0 with no offload", ["fp32_image_last", "fp32_image_first"])

    # 3. bf16 smoke.
    acc = {n: accepted(n) for n in ("bf16_vispec", "bf16_hivis")}
    nums = {n: {"mean_accepted_draft_tokens": mean(v), "iterations": len(v)} for n, v in acc.items()}
    vlog = log_text("bf16_vispec") + log_text("bf16_ar")
    nums["vispec7b_load_key_lines"] = len(re.findall(r"missing_keys|unexpected_keys", vlog))
    rt = {}
    for n, want in (("bf16_ar", None), ("bf16_vispec", "vispec"), ("bf16_hivis", "hivis")):
        r = runtime(n)
        g = r.get("generation_args", {})
        rt[n] = {"draft_method": r.get("draft_method"), "base_model_revision": r.get("base_model_revision"),
                 "spec_model_revision": r.get("spec_model_revision"), "parameter_dtypes": r.get("parameter_dtypes"),
                 "gpu_uuid": [d["uuid"] for d in r.get("devices", [])]}
    nums["runtime"] = rt
    base_rev, vis_rev, hv_rev = "cc594898137f460bfe9f0759e9844b3ce807cfb5", "e06635b54327df78a5eb82361791f2ccec9f41ab", "6dd8717d912a5626d6ad4e81b124c54bb8a08ef2"
    rt_ok = (
        rt["bf16_vispec"]["draft_method"] == "vispec" and rt["bf16_hivis"]["draft_method"] == "hivis"
        and all(rt[n]["base_model_revision"] == base_rev for n in rt)
        and rt["bf16_vispec"]["spec_model_revision"] == vis_rev and rt["bf16_hivis"]["spec_model_revision"] == hv_rev
        and all(rt[n]["parameter_dtypes"] == ["torch.bfloat16"] for n in rt)
    )
    nums["runtime_ok"] = rt_ok
    add("bf16 smoke",
        exited(["bf16_ar", "bf16_vispec", "bf16_hivis"]) and all((mean(v) or 0) >= 1.5 for v in acc.values())
        and nums["vispec7b_load_key_lines"] == 0 and rt_ok,
        nums, "three runs exit 0; HiViS and ViSpec-7B each average >= 1.5 accepted draft tokens per iteration; no missing_keys/unexpected_keys line in the ViSpec-7B log; runtime.json shows draft_method, both revisions and bf16",
        ["bf16_ar", "bf16_vispec", "bf16_hivis"])

    # 4. Large tree.
    big = {n: {"mean_accepted_draft_tokens": mean(accepted(n)), "iterations": len(accepted(n)), **latency(n)} for n in ("large_vispec", "large_hivis")}
    add("Large tree", exited(["large_vispec", "large_hivis"]) and all(v["samples"] == 2 for v in big.values()),
        big, "both complete; acceptance recorded", ["large_vispec", "large_hivis"])

    # 5. Position ablation.
    normal, pruned = mean(acc["bf16_hivis"]), mean(accepted("ablation_hivis_pruned"))
    add("Position ablation",
        exited(["ablation_hivis_pruned"]) and normal is not None and pruned is not None and pruned < normal
        and "position_mode = pruned" in log_text("ablation_hivis_pruned"),
        {"normal_mean": normal, "pruned_mean": pruned, "pruned_samples": len(rows("ablation_hivis_pruned")),
         "pruned_iterations": len(accepted("ablation_hivis_pruned"))},
        "mean acceptance below the normal HiViS mean on the same 5 samples", ["ablation_hivis_pruned"])

    # Recorded, not gated.
    ar, vs, hv = tokens("bf16_ar"), tokens("bf16_vispec"), tokens("bf16_hivis")
    def agree(a, b):
        return {"equal": sum(1 for q in a if q in b and a[q] == b[q]), "total": len(a)}
    info = {"ar_vs_hivis_token_agreement": agree(ar, hv), "ar_vs_vispec_token_agreement": agree(ar, vs),
            "latency": {n: latency(n) for n in ("bf16_ar", "bf16_vispec", "bf16_hivis", "large_vispec", "large_hivis", "ablation_hivis_pruned")}}
    if all(info["latency"][n]["mean_wall_time_s"] for n in ("bf16_ar", "bf16_vispec", "bf16_hivis")):
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
                summ = json.load(open(path))["summary"]
                numbers = {"gate_passes": summ["gate_passes"], "gate": summ["gate"],
                           "classes": summ["overall"]["classes"]}
                passed = passed and summ["gate_passes"] is True
            else:
                numbers, passed = {"diagnosis_json": "missing"}, False
        elif name.startswith("oracle_"):
            rule = "command exits 0 (offset_check exits 1 on a mismatch); settings recorded"
            path = os.path.join(run_dir, "checks", name, "offset_check.json")
            if os.path.exists(path):
                res = json.load(open(path))
                numbers = {"base_model_path": res["settings"].get("base_model_path"),
                           "draft_method": res["settings"].get("draft_method", "vispec"),
                           "prompt_order": res["settings"].get("prompt_order", "image-last"),
                           "exact": res["summary"]["exact"], "questions": res["summary"]["questions"]}
            else:
                numbers, passed = {"offset_check_json": "missing"}, False
        add(name, passed, numbers, rule, [name])

result = {"check_set": check_set, "all_pass": bool(checks) and all(c["pass"] for c in checks), "checks": checks}
json.dump(result, open(os.path.join(run_dir, "checks.json"), "w"), indent=2)
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
"$python_bin" experiment/prepare_assets.py --repo-dir "$repo_dir" --check-only --suite qwen
"$python_bin" experiment/prepare_assets.py --repo-dir "$repo_dir" --check-only --suite qwen7b
bash "$run_dir/check_cuda.sh" "$python_bin"
export PYTHONUNBUFFERED=1

# run_check <name> <command...>: log to checks/<name>.log, record the command and exit code in
# checks/<name>.json, and keep going whatever the result.
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
json.dump({"name": name, "command": cmd, "exit_code": int(code), "started_utc": started}, open(path, "w"), indent=2)
PY
    echo "check $name exit code $code"
}

evaluate_cmd() {  # <base> <spec> <method> <name> <samples> <total> <depth> <topk>
    local base="$1" spec="$2" method="$3" name="$4" samples="$5" total="$6" depth="$7" topk="$8"
    run_check "$name" "$python_bin" -m vispec.evaluation.evaluate \
        --method "$method" --benchmark textvqa --prompt-style long \
        --base-model-path "$base" --spec-model-path "$spec" --data-dir "$bench_dir" \
        --answer-file "$run_dir/results/$name/test.jsonl" --model-id test \
        --num-samples "$samples" --max-new-token 1024 --temperature 0.0 \
        --depth "$depth" --top-k "$topk" --total-token "$total" --num-q 2 \
        --kv-capacity 16384 --kv-layout compact
}

offset_cmd() {  # <name> <draft-method> <total> <depth> <topk> <order>
    run_check "$1" "$python_bin" -m vispec.evaluation.offset_check \
        --draft-method "$2" --kv-retention 1.0 --kv-policy spatial \
        --num-samples 5 --dtype float32 --prompt-order "$6" \
        --base-model-path "$base7b" --spec-model-path "$hivis7b" --data-dir "$bench_dir" \
        --total-token "$3" --depth "$4" --top-k "$5" \
        --output-dir "$run_dir/checks/$1"
}

if [[ "$check_set" == "port" ]]; then
    # 1. 3B identity against the Week 2 runs.
    evaluate_cmd "$base3b" "$spec3b" autoregressive id3b_ar 5 30 3 8
    evaluate_cmd "$base3b" "$spec3b" vispec id3b_vispec 5 30 3 8
    # 2. fp32 losslessness: image-last 30/3/8, then image-first 60/5/10.
    offset_cmd fp32_image_last hivis 30 3 8 image-last
    offset_cmd fp32_image_first hivis 60 5 10 image-first
    # 3. bf16 smoke on 7B.
    evaluate_cmd "$base7b" "$vispec7b" autoregressive bf16_ar 5 30 3 8
    evaluate_cmd "$base7b" "$vispec7b" vispec bf16_vispec 5 30 3 8
    evaluate_cmd "$base7b" "$hivis7b" hivis bf16_hivis 5 30 3 8
    # 4. Large tree on 2 samples.
    evaluate_cmd "$base7b" "$vispec7b" vispec large_vispec 2 60 5 10
    evaluate_cmd "$base7b" "$hivis7b" hivis large_hivis 2 60 5 10
    # 5. Position ablation: the same HiViS samples with the pruned position mode.
    run_check ablation_hivis_pruned "$python_bin" - "$base7b" "$hivis7b" "$bench_dir" \
        "$run_dir/results/ablation_hivis_pruned/test.jsonl" <<'PY'
import sys
from vispec.model.cnets_hivis import HiViSDraft
from vispec.evaluation import evaluate

base, spec, data, answer = sys.argv[1:5]
HiViSDraft.position_mode = "pruned"
print("HiViSDraft.position_mode =", HiViSDraft.position_mode, flush=True)
evaluate.run(evaluate.parse_args([
    "--method", "hivis", "--benchmark", "textvqa", "--prompt-style", "long",
    "--base-model-path", base, "--spec-model-path", spec, "--data-dir", data,
    "--answer-file", answer, "--model-id", "test",
    "--num-samples", "5", "--max-new-token", "1024", "--temperature", "0.0",
    "--depth", "3", "--top-k", "8", "--total-token", "30", "--num-q", "2",
    "--kv-capacity", "16384", "--kv-layout", "compact",
]))
PY
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
