#!/usr/bin/env bash
# Shared by the run.sh of experiments 05_qwen_draft_pruning, 06_qwen_kv_eviction, 08_qwen_combined_sparsity,
# 09_qwen_reverse_longcontext and 10_qwen7b_methods.
# Source it; do not run it. The caller sets: experiment_dir, repo_dir, sweep (reference|draft|kv|combined|methods),
# arms (space-separated <retention>:<policy>[:<method>]; reference sweeps take `ar` and `vispec`), prompt_style, and has done
# `set -euo pipefail` and `cd "$repo_dir"`.
# The methods sweep (experiment 10) takes arms `<method>[:<tree>][:<order>]`; see qwen_validate_method_arms.
# Each arm gets its own run folder with the full launch evidence and its own EXIT report.
# Failure: a failed arm records a failure and the remaining arms still run; a failed benchmark stops that
# arm's remaining benchmarks; the runner exits 1 if any arm failed; run_sequence.sbatch then continues
# to the next experiment.

qwen_arm_usage() {
    echo "Usage: PROMPT_STYLE={long|short} ${1}_ARMS=\"<retention>:<policy>${2} ...\" [BENCH_DIR=/absolute/path] [PYTHON_BIN=python] $3" >&2
}

qwen_require_style() {
    if [[ "${PROMPT_STYLE:-}" != "long" && "${PROMPT_STYLE:-}" != "short" ]]; then
        echo "PROMPT_STYLE must be 'long' or 'short' (got '${PROMPT_STYLE:-}')." >&2
        exit 2
    fi
    prompt_style="$PROMPT_STYLE"
}

# Validate the arms of a reference sweep (ar, vispec) before any run folder exists. Usage: qwen_validate_ref_arms <arms>
qwen_validate_ref_arms() {
    local list="$1" spec
    if [[ -z "${list// /}" ]]; then
        echo "The arm list is empty." >&2
        exit 2
    fi
    for spec in $list; do
        case "$spec" in ar|vispec) ;; *)
            echo "Invalid reference arm '$spec': must be ar or vispec." >&2; exit 2 ;;
        esac
    done
}

# Split a methods-sweep arm `<method>[:<tree>][:<order>]` into globals arm_method, arm_tree and arm_order, with the
# defaults filled in, and set arm_tree_given=1 when the tree was written out. Return 1 with a message on any
# invalid arm. Fields are told apart by value, so each may appear once. Usage: qwen_parse_method_arm <arm>
qwen_parse_method_arm() {
    local spec="$1" field
    local -a fields
    arm_method=""; arm_tree="30-3-8"; arm_order="image-last"; arm_tree_given=0
    local tree_seen=0 order_seen=0
    IFS=: read -r -a fields <<< "$spec"
    if [[ "$spec" == *: || ${#fields[@]} -eq 0 ]]; then
        echo "Invalid arm '$spec': empty field." >&2; return 1
    fi
    case "${fields[0]}" in ar|vispec|hivis) arm_method="${fields[0]}" ;; *)
        echo "Invalid arm '$spec': method must be ar, vispec or hivis." >&2; return 1 ;;
    esac
    for field in "${fields[@]:1}"; do
        case "$field" in
            30-3-8|60-5-10)
                if [[ $tree_seen -eq 1 ]]; then echo "Invalid arm '$spec': tree given twice." >&2; return 1; fi
                tree_seen=1; arm_tree="$field"; arm_tree_given=1 ;;
            image-last|image-first)
                if [[ $order_seen -eq 1 ]]; then echo "Invalid arm '$spec': prompt order given twice." >&2; return 1; fi
                order_seen=1; arm_order="$field" ;;
            *) echo "Invalid arm '$spec': '$field' is not a tree (30-3-8 or 60-5-10) or a prompt order (image-last or image-first)." >&2; return 1 ;;
        esac
    done
    if [[ "$arm_method" == "ar" && $tree_seen -eq 1 ]]; then
        echo "Invalid arm '$spec': ar takes no tree." >&2; return 1
    fi
    [[ "$arm_method" == "ar" ]] && arm_tree="none"
    return 0
}

# Validate the arms of a methods sweep before any run folder exists, including duplicates once the defaults are
# filled in. Usage: qwen_validate_method_arms <arms>
qwen_validate_method_arms() {
    local list="$1" spec seen=" " ar_orders=" " spec_orders=" " order
    if [[ -z "${list// /}" ]]; then
        echo "The arm list is empty." >&2
        exit 2
    fi
    for spec in $list; do
        qwen_parse_method_arm "$spec" || exit 2
        if [[ "$seen" == *" ${arm_method}/${arm_tree}/${arm_order} "* ]]; then
            echo "Invalid arm '$spec': duplicates an earlier arm (${arm_method}, tree ${arm_tree}, ${arm_order})." >&2; exit 2
        fi
        seen+="${arm_method}/${arm_tree}/${arm_order} "
        if [[ "$arm_method" == "ar" ]]; then
            ar_orders+="${arm_order} "
        else
            spec_orders+="${arm_order} "
        fi
    done
    # Every prompt order that has a speculative arm needs an ar arm of its own to compare with.
    for order in image-last image-first; do
        if [[ "$spec_orders" == *" ${order} "* && "$ar_orders" != *" ${order} "* ]]; then
            echo "Invalid arm list: a speculative arm uses ${order} but there is no ar arm with ${order} to compare it with." >&2
            exit 2
        fi
    done
}

# Validate every arm before any run folder exists. Usage: qwen_validate_arms <allow-method:0|1> <arms>
qwen_validate_arms() {
    local allow_method="$1" list="$2" spec ret policy method extra
    if [[ -z "${list// /}" ]]; then
        echo "The arm list is empty." >&2
        exit 2
    fi
    for spec in $list; do
        IFS=: read -r ret policy method extra <<< "$spec"
        case "$ret" in 0.25|0.5|0.75|1.0) ;; *)
            echo "Invalid arm '$spec': retention must be one of 0.25 0.5 0.75 1.0." >&2; exit 2 ;;
        esac
        case "$policy" in spatial|attention) ;; *)
            echo "Invalid arm '$spec': policy must be spatial or attention." >&2; exit 2 ;;
        esac
        if [[ -n "$extra" ]]; then
            echo "Invalid arm '$spec': too many fields." >&2; exit 2
        fi
        if [[ "$allow_method" == "1" ]]; then
            case "$method" in ""|vispec|autoregressive) ;; *)
                echo "Invalid arm '$spec': method must be vispec or autoregressive." >&2; exit 2 ;;
            esac
        elif [[ -n "$method" ]]; then
            echo "Invalid arm '$spec': this experiment takes no method suffix." >&2; exit 2
        fi
    done
}

# Run every arm in order; return 1 if any arm failed. Usage: qwen_run_arms
qwen_run_arms() {
    local spec rc failed=0
    for spec in $arms; do
        set +e
        qwen_run_arm "$spec"
        rc=$?
        set -e
        [[ $rc -eq 0 ]] || failed=1
    done
    return "$failed"
}

# One arm = one run folder. Mirrors the 03 runner steps 03-11.
qwen_run_arm() {
    set +e
    local spec="$1" ret="" policy="" method_field="" method suffix="" folder_arm
    local arm_method arm_tree arm_order arm_tree_given
    local m_depth="" m_top_k="" m_total="" draft_method="none"
    if [[ "$sweep" == "methods" ]]; then
        qwen_parse_method_arm "$spec" || return 2
        folder_arm="${arm_method}"
        [[ $arm_tree_given -eq 1 ]] && folder_arm+="_t${arm_tree}"
        folder_arm+="_${arm_order}"
        case "$arm_method" in
            ar) method="autoregressive" ;;
            vispec) method="vispec"; draft_method="vispec" ;;
            hivis) method="hivis"; draft_method="hivis" ;;
        esac
        [[ "$arm_method" != "ar" ]] && IFS=- read -r m_total m_depth m_top_k <<< "$arm_tree"
    else
        IFS=: read -r ret policy method_field <<< "$spec"
        method="${method_field:-vispec}"
        [[ -n "$method_field" ]] && suffix="_${method_field}"
        folder_arm="r${ret}_${policy}${suffix}"
    fi
    if [[ "$sweep" == "reference" ]]; then
        # A reference arm is `ar` or `vispec`: no sparsity flags, folder name ..._ref_<ar|vispec>_XXXXXX.
        case "$spec" in
            ar) method="autoregressive" ;;
            vispec) method="vispec" ;;
            *) echo "Invalid reference arm '$spec': must be ar or vispec." >&2; return 2 ;;
        esac
        folder_arm="ref_${spec}"
    fi

    export HF_HOME="$repo_dir/.cache/huggingface"
    export HF_HUB_CACHE="$HF_HOME/hub"
    export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
    export HF_DATASETS_CACHE="$HF_HOME/datasets"
    export HF_ASSETS_CACHE="$HF_HOME/assets"
    export HF_XET_CACHE="$HF_HOME/xet"
    export PYTHONNOUSERSITE=1
    local base_model="${BASE_MODEL_PATH:-$repo_dir/models/Qwen2.5-VL-3B-Instruct}"
    local spec_dir="${SPEC_DIR:-$repo_dir/models/ViSpec-Qwen2.5-VL-3B-Instruct}"
    local asset_suite="${QWEN_ASSET_SUITE:-qwen}"
    if [[ "$method" == "hivis" ]]; then
        spec_dir="${HIVIS_DIR:-$repo_dir/models/HiViS-Qwen2.5-VL-7B-Instruct}"
    fi
    local bench_dir="${BENCH_DIR:-$repo_dir/bench_data}"
    local temperature="0.0" max_new_tokens=1024 num_samples=100 kv_capacity=16384
    local python_bin="${PYTHON_BIN:-/home/aeternum/anaconda3/envs/vipsec/bin/python}"
    local depth="${DEPTH:-3}" top_k="${TOP_K:-8}" total_token="${TOTAL_TOKEN:-30}" num_q="${NUM_Q:-2}"
    if [[ "$sweep" == "methods" && -n "$m_total" ]]; then
        depth="$m_depth"; top_k="$m_top_k"; total_token="$m_total"; num_q="2"
    fi
    local qwen_benchmarks="${QWEN_BENCHMARKS:-textvqa coco_caption}" benchmarks=()
    read -r -a benchmarks <<< "$qwen_benchmarks"
    local method_dir meta_method
    if [[ "$method" == "autoregressive" ]]; then
        method_dir="autoregressive_qwen"; meta_method="autoregressive"
    elif [[ "$method" == "hivis" ]]; then
        method_dir="hivis_qwen"; meta_method="hivis_speculative"
    else
        method_dir="vispec_qwen"; meta_method="vispec_speculative"
    fi
    # Sparsity settings this arm applies; "none" in metadata means the flag is omitted.
    local kv_retention="none" kv_policy="none" draft_retention="none" draft_policy="none"
    case "$sweep" in
        reference|methods) ;;
        draft) draft_retention="$ret"; draft_policy="$policy" ;;
        kv) kv_retention="$ret"; kv_policy="$policy" ;;
        combined)
            kv_retention="$ret"; kv_policy="$policy"
            draft_retention="$ret"; draft_policy="$policy" ;;
        *) echo "Unknown sweep type '$sweep' (expected reference, draft, kv, combined or methods)." >&2; exit 2 ;;
    esac

    command -v "$python_bin" >/dev/null || return 1
    export PATH="$(dirname "$python_bin"):$PATH"
    if [[ -f "$repo_dir/tools/downloaded/m4c_evaluator.py" ]]; then
        export TEXTVQA_EVALUATOR="${TEXTVQA_EVALUATOR:-$repo_dir/tools/downloaded/m4c_evaluator.py}"
    fi
    local commit run_dir
    commit="$(git rev-parse HEAD)"
    mkdir -p "$experiment_dir/runs"
    run_dir="$(mktemp -d "$experiment_dir/runs/$(date -u +%Y%m%dT%H%M%SZ)_${commit:0:12}_${prompt_style}_${folder_arm}_XXXXXX")" || return 1
    printf '%s\n' "$commit" > "$run_dir/commit.txt"
    git status --short > "$run_dir/git-status.txt"
    git diff HEAD --binary > "$run_dir/tracked-changes.patch"
    cp "$experiment_dir/run.sh" "$run_dir/runner.sh"
    cp "$repo_dir/experiment/qwen_arm_runner.sh" "$run_dir/qwen_arm_runner.sh"
    cp "$repo_dir/experiment/capture_device.sh" "$run_dir/capture_device.sh"
    cp "$repo_dir/experiment/check_cuda.sh" "$run_dir/check_cuda.sh"
    if [[ -f "$repo_dir/.preparation/assets_${asset_suite}.json" ]]; then
        cp "$repo_dir/.preparation/assets_${asset_suite}.json" "$run_dir/prepared-assets.json"
    fi

    # Use a pipeline so logging also works on systems that restrict /dev/fd.
    (
    set -e
    finish() {
        local status=$?
        trap - EXIT
        # Report partial runs too; missing scores remain pending without blocking timing metrics.
        set +e
        local report_cmd=("$python_bin" -m vispec.evaluation.report_metrics
            --run-dir "$run_dir" --bench-dir "$bench_dir" --evaluation-exit-code "$status")
        printf '\nReporting:'
        printf ' %q' "${report_cmd[@]}"
        printf '\n'
        "${report_cmd[@]}"
        local report_status=$?
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

    local submit_command sbatch_extra
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
        printf 'slurm_restart_count=%s\n' "${SLURM_RESTART_COUNT:-0}"
        printf 'submit_command=%s\nsbatch_extra=%s\n' "$submit_command" "$sbatch_extra"
        if [[ "$method" == "autoregressive" ]]; then
            printf 'method=%s\n' "$meta_method"
        elif [[ "$sweep" == "methods" && "$method" == "hivis" ]]; then
            printf 'method=%s\ndepth=%s\ntop_k=%s\ntotal_token=%s\n' "$meta_method" "$depth" "$top_k" "$total_token"
        else
            printf 'method=%s\ndepth=%s\ntop_k=%s\ntotal_token=%s\nnum_q=%s\n' "$meta_method" "$depth" "$top_k" "$total_token" "$num_q"
        fi
        if [[ "$sweep" == "methods" ]]; then
            printf 'prompt_order=%s\ndraft_method=%s\nasset_suite=%s\n' "$arm_order" "$draft_method" "$asset_suite"
        fi
        printf 'prompt_style=%s\nslim_prefill=true\nkv_capacity=%s\nnum_samples=%s\n' "$prompt_style" "$kv_capacity" "$num_samples"
        printf 'max_new_tokens=%s\n' "$max_new_tokens"
        printf 'benchmarks=%s\n' "${benchmarks[*]}"
        printf 'qwen_benchmarks=%s\n' "$qwen_benchmarks"
        printf 'kv_layout=compact\nkv_retention=%s\nkv_policy=%s\ndraft_retention=%s\ndraft_policy=%s\n' \
            "$kv_retention" "$kv_policy" "$draft_retention" "$draft_policy"
        printf 'arm=%s\nsweep=%s\narm_list=%s\n' "$spec" "$sweep" "$arms"
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
    "$python_bin" experiment/prepare_assets.py --repo-dir "$repo_dir" --check-only --suite "$asset_suite"
    bash "$run_dir/check_cuda.sh" "$python_bin"
    export PYTHONUNBUFFERED=1

    # A failed benchmark stops this arm's remaining benchmarks (set -e in this subshell); qwen_run_arms
    # still runs the other arms.
    local benchmark
    for benchmark in "${benchmarks[@]}"; do
        local cmd=("$python_bin" -m vispec.evaluation.evaluate
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
            --temperature "$temperature")
        if [[ "$method" == "vispec" ]]; then
            cmd+=(--depth "$depth" --top-k "$top_k" --total-token "$total_token" --num-q "$num_q")
        elif [[ "$method" == "hivis" ]]; then
            cmd+=(--depth "$depth" --top-k "$top_k" --total-token "$total_token")
        fi
        if [[ "$sweep" == "methods" ]]; then
            cmd+=(--prompt-order "$arm_order")
        fi
        cmd+=(--kv-capacity "$kv_capacity" --kv-layout compact)
        case "$sweep" in
            reference|methods) ;;
            draft) cmd+=(--draft-retention "$ret" --draft-policy "$policy") ;;
            kv) cmd+=(--kv-retention "$ret" --kv-policy "$policy") ;;
            combined) cmd+=(--kv-retention "$ret" --kv-policy "$policy"
                --draft-retention "$ret" --draft-policy "$policy") ;;
            *) echo "Unknown sweep type '$sweep' (expected reference, draft, kv, combined or methods)." >&2; exit 2 ;;
        esac
        printf '\nRunning:'
        printf ' %q' "${cmd[@]}"
        printf '\n'
        "${cmd[@]}"
    done
    ) 2>&1 | tee "$run_dir/output.log"
    local pipeline_status=("${PIPESTATUS[@]}")

    if [[ "${pipeline_status[0]}" -ne 0 ]]; then
        return "${pipeline_status[0]}"
    fi
    if [[ "${pipeline_status[1]}" -ne 0 ]]; then
        printf 'logging_exit_code=%s\n' "${pipeline_status[1]}" >> "$run_dir/metadata.txt"
    fi
    printf '%s\n' "${pipeline_status[1]}" > "$run_dir/exit-code.txt"
    return "${pipeline_status[1]}"
}
