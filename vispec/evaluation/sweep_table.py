"""Build one retention-sweep table per Slurm job from per-run reports.

Runs are grouped by (prompt_style, slurm_job_id). Every ratio is computed inside one group,
so runs from different cards or jobs are never mixed. Reads saved reports only; no GPU.
"""

import argparse
import csv
import json
import re
import statistics
from pathlib import Path

from .compare_metrics import (
    DEFAULT_PROMPT_ORDER,
    SPARSITY_SETTINGS,
    answer_agreement_rates,
    check_same_slurm_job,
    generation_setting,
    load_report,
    prompt_order_of,
    require_same,
    runtime_signature,
    sample_key,
    token_agreement,
)
from .report_metrics import fmt, metadata, memory_summary

# Run settings that every run of one job group must share.
SHARED_SETTINGS = (
    "base_model", "spec_model", "temperature", "kv_layout", "slim_prefill", "kv_capacity", "num_samples", "max_new_tokens",
)

# The shared settings of the methods layout. The draft path is checked apart: AR and ViSpec
# share one (the AR runs record the ViSpec draft they load and do not use), HiViS has its own.
METHODS_SHARED_SETTINGS = tuple(k for k in SHARED_SETTINGS if k != "spec_model")
DEFAULT_TREE = ("3", "8", "30")  # depth, top_k, total_token as the runners record them
LAYOUTS = ("sweep", "methods")

METHODS_COLUMNS = (
    "prompt_style", "slurm_job_id", "method", "tree", "prompt_order", "role", "benchmark", "n",
    "latency_mean_s", "ttft_mean_s", "tpot_mean_s", "throughput_tokens_per_s", "output_tokens_mean",
    "acceptance_mean", "output_tokens_per_iteration_mean", "decode_peak_mean_gib",
    "post_compaction_peak_gib", "quality", "token_matched_vs_ar", "token_checked_vs_ar",
    "token_agreement_vs_ar", "answer_agreement_vs_ar", "speedup_vs_ar", "speedup_vs_vispec",
    "speedup_vs_tree_30_3_8", "note",
)

METHODS_COLUMN_NOTES = {
    "tree": "total tokens, depth and top-k of the draft tree as total-depth-top_k; none for autoregressive",
    "role": "reference for the autoregressive run, primary for a speculative run at 30-3-8 with image-last, secondary for any other speculative run",
    "acceptance_mean": "accepted draft tokens per verification iteration, mean over turns",
    "output_tokens_per_iteration_mean": "accepted draft tokens plus the target token per verification iteration, mean over turns",
    "speedup_vs_ar": "AR mean latency / row mean latency, same job and prompt order; above 1 means the row is faster",
    "speedup_vs_vispec": "ViSpec mean latency / row mean latency at the same tree, same job and prompt order (HiViS rows); above 1 means the row is faster",
    "speedup_vs_tree_30_3_8": "the same method at 30-3-8 mean latency / row mean latency, same job and prompt order (60-5-10 rows); above 1 means the row is faster",
    "token_agreement_vs_ar": "share of turns whose generated token IDs through first EOS equal the AR run's",
    "answer_agreement_vs_ar": "share of turns whose whitespace-normalised decoded text equals the AR run's",
    "note": "per-row warning, for example a missing reference or sample sets that differ from it (affected cells are empty)",
}

COLUMNS = (
    "prompt_style", "slurm_job_id", "sweep", "method", "role", "arm", "benchmark",
    "kv_retention", "kv_policy", "draft_retention", "draft_policy", "n",
    "latency_mean_s", "ttft_mean_s", "tpot_mean_s", "throughput_tokens_per_s",
    "acceptance_mean", "decode_peak_mean_gib", "decode_peak_max_gib", "live_kv_mean_mib",
    "quality", "kv_image_tokens_mean", "kv_min_kept_per_image",
    "draft_min_kept_per_image", "post_compaction_peak_gib",
    "token_matched_vs_vispec", "token_checked_vs_vispec", "token_agreement_vs_vispec",
    "answer_agreement_vs_vispec", "token_matched_vs_ar", "token_checked_vs_ar",
    "token_agreement_vs_ar", "speedup_vs_ar", "latency_speedup_vs_vispec_ref",
    "latency_speedup_vs_samepolicy_1_0", "decode_peak_arm_over_samepolicy_1_0",
    "live_kv_arm_over_samepolicy_1_0", "post_compaction_peak_arm_over_samepolicy_1_0",
    "quality_delta_vs_vispec_ref", "bertscore_f1", "bertscore_f1_delta_vs_vispec_ref", "note",
)

COLUMN_NOTES = {
    "speedup_vs_ar": "AR reference mean latency / arm mean latency; above 1 means the arm is faster",
    "latency_speedup_vs_vispec_ref": "ViSpec reference mean latency / arm mean latency; above 1 means the arm is faster",
    "latency_speedup_vs_samepolicy_1_0": "same-policy 1.0 arm mean latency / arm mean latency; above 1 means the arm is faster",
    "decode_peak_arm_over_samepolicy_1_0": "arm mean decode peak / same-policy 1.0 arm mean decode peak; above 1 means more memory",
    "live_kv_arm_over_samepolicy_1_0": "arm mean live KV / same-policy 1.0 arm mean live KV; above 1 means more memory",
    "post_compaction_peak_arm_over_samepolicy_1_0": "arm mean post-compaction peak / same-policy 1.0 arm mean post-compaction peak; above 1 means more memory; empty when either run lacks the field",
    "bertscore_f1": "reverse_summary only: mean BERTScore F1 over windows, 0-100; truncated at 512 word pieces, does not reflect later pages",
    "bertscore_f1_delta_vs_vispec_ref": "reverse_summary only: arm BERTScore F1 minus ViSpec reference BERTScore F1, points; truncated at 512 word pieces, does not reflect later pages; the primary quality delta stays ROUGE-1",
    "quality_delta_vs_vispec_ref": "arm quality minus ViSpec reference quality, percentage points",
    "note": "per-row warning, for example sample sets that differ from a reference (affected cells are empty)",
}


def newest_report(run_dir):
    reports = sorted(
        path for path in (run_dir / "reports").glob("*/summary.json")
    ) if (run_dir / "reports").is_dir() else []
    return reports[-1].parent if reports else None


def device_uuids(run_dir):
    path = run_dir / "device.txt"
    if not path.exists():
        return set()
    return set(re.findall(r"GPU UUID\s*:\s*(GPU-\S+)", path.read_text()))


def settings_of(report, benchmark):
    return {key: generation_setting(report, benchmark, key) for key in SPARSITY_SETTINGS}


def sparse_key(settings):
    return tuple(settings[key] for key in SPARSITY_SETTINGS)


def sweep_name(settings):
    kv = settings["kv_retention"] is not None or settings["kv_policy"] is not None
    draft = settings["draft_retention"] is not None or settings["draft_policy"] is not None
    return {(False, False): "reference", (True, False): "kv", (False, True): "draft",
            (True, True): "combined"}[(kv, draft)]


def load_run(run_dir):
    """Return a completed run, or (None, reason) when it has no usable final report."""
    run_dir = Path(run_dir).resolve()
    meta = metadata(run_dir)
    if not meta:
        return None, f"{run_dir}: no metadata.txt"
    folder = newest_report(run_dir)
    if folder is None:
        return None, f"{run_dir}: no report"
    report = load_report(folder)
    if meta.get("exit_code", "").strip() != "0" or report["summary"].get("evaluation_exit_code") != 0:
        return None, f"{run_dir}: did not complete (exit_code={meta.get('exit_code')!r})"
    benchmarks = [
        name for name, bench in report["summary"]["benchmarks"].items()
        if bench.get("completed_turns")
    ]
    if not benchmarks:
        return None, f"{run_dir}: report has no completed benchmark"
    settings = {name: settings_of(report, name) for name in benchmarks}
    if len({sparse_key(value) for value in settings.values()}) != 1:
        raise ValueError(f"Benchmarks of one run use different sparsity settings: {run_dir}")
    for name in benchmarks:
        if report["summary"].get("runtime", {}).get(name) is None:
            raise ValueError(f"Missing runtime entry for {name}: {run_dir}")
    orders = {prompt_order_of(report, name) for name in benchmarks}
    if len(orders) != 1:
        raise ValueError(f"Benchmarks of one run use different prompt orders: {run_dir}")
    return {
        "run_dir": run_dir, "meta": meta, "report": report, "benchmarks": benchmarks,
        "settings": settings[benchmarks[0]],
        "method": meta.get("method"),
        "prompt_order": orders.pop(),
        "maps": {
            name: {sample_key(r): r for r in report["rows"] if r["benchmark"] == name}
            for name in benchmarks
        },
    }, None


def check_group(runs, methods_layout=False):
    """Refuse a group that mixes cards, allocations, prompts or decoding settings."""
    reports = [run["report"] for run in runs]
    check_same_slurm_job(reports)
    for setting in METHODS_SHARED_SETTINGS if methods_layout else SHARED_SETTINGS:
        require_same([run["meta"].get(setting) for run in runs], f"{setting} (metadata)")
    if methods_layout:
        # AR and ViSpec share one draft path; HiViS has its own.
        for names in (("autoregressive", "vispec_speculative"), ("hivis_speculative",)):
            require_same(
                [r["meta"].get("spec_model") for r in runs if r["method"] in names],
                "spec_model (metadata)",
            )
    require_same([run["meta"].get("prompt_style") for run in runs], "prompt_style (metadata)")
    # The autoregressive runs record no draft-tree settings; compare the ViSpec runs only.
    # The methods layout runs several trees on purpose; it keys them apart instead.
    for setting in () if methods_layout else ("depth", "top_k", "total_token", "num_q"):
        require_same(
            [r["meta"].get(setting) for r in runs if r["method"] == "vispec_speculative"],
            f"{setting} (metadata)",
        )
    uuids = set()
    for run in runs:
        uuids |= device_uuids(run["run_dir"])
    if len(uuids) > 1:
        raise ValueError(f"One job group spans several GPU UUIDs in device.txt: {sorted(uuids)}")
    benchmarks = sorted({name for run in runs for name in run["benchmarks"]})
    for name in benchmarks:
        signatures = []
        for run in runs:
            if name not in run["benchmarks"]:
                continue
            signature = runtime_signature(run["report"], name)
            if signature is None:
                raise ValueError(f"Missing runtime hardware/precision evidence: {run['run_dir']}")
            signatures.append(signature)
        if any(signature != signatures[0] for signature in signatures):
            raise ValueError(
                f"Hardware/allocation/precision mismatch in {name}; one group needs one card"
            )
        runtime_uuids = {d.get("uuid") for s in signatures for d in s["devices"]}
        runtime_uuids = {f"GPU-{u}" for u in runtime_uuids if u}
        if uuids and runtime_uuids and runtime_uuids != uuids:
            raise ValueError(f"device.txt and runtime GPU UUIDs disagree for {name}")
        prompts = {}
        for run in runs:
            for key, row in run["maps"].get(name, {}).items():
                digest = row["measurement"].get("input_ids_sha256")
                if not digest:
                    raise ValueError(f"Missing prompt-token fingerprint: {run['run_dir']}")
                prompt_key = (name, run["prompt_order"], key) if methods_layout else (name, key)
                if prompts.setdefault(prompt_key, digest) != digest:
                    raise ValueError(f"Prompt mismatch for {name} {key}: {run['run_dir']}")


def quality_mean(rows, keys):
    scores = [rows[key].get("score") for key in keys]
    if not keys or any(score is None for score in scores):
        return None
    return 100 * statistics.mean(scores)


def measurement_values(run, benchmark, field):
    return [
        r["measurement"][field] for r in run["maps"][benchmark].values()
        if r["measurement"].get(field) is not None
    ]


def agreement(ref, run, benchmark):
    """Token and decoded-text agreement of run against ref, with the compare_metrics logic."""
    if ref is None or benchmark not in ref["benchmarks"]:
        return None
    a, b = ref["maps"][benchmark], run["maps"][benchmark]
    common = set(a) & set(b)
    if set(a) != set(b):
        return None
    tokens = token_agreement([a], [b], common)
    text = answer_agreement_rates([a], [b], common)
    return {
        "matched": tokens["matched_pairs"], "checked": tokens["checked_pairs"],
        "rate": tokens["sequence_agreement_rate"],
        "text": text[0] if text else None,
    }


def ratio(numerator, denominator):
    return numerator / denominator if numerator is not None and denominator else None


def policies_of(settings):
    return tuple(settings[key] for key in ("kv_policy", "draft_policy"))


def at_full_retention(settings):
    return all(
        settings[key] in (None, 1.0) for key in ("kv_retention", "draft_retention")
    ) and any(settings[key] == 1.0 for key in ("kv_retention", "draft_retention"))


def build_group(style, job, runs, notes):
    check_group(runs)
    by_arm = {}
    for run in runs:
        key = (run["method"], sparse_key(run["settings"]))
        if key in by_arm:
            raise ValueError(
                "Two completed folders for one arm: "
                f"{by_arm[key]['run_dir']} and {run['run_dir']}"
            )
        by_arm[key] = run
    references = {"autoregressive": None, "vispec_speculative": None}
    for (method, settings), run in by_arm.items():
        if method in references and all(value is None for value in settings):
            references[method] = run
    ar, vispec = references["autoregressive"], references["vispec_speculative"]
    if ar is None:
        notes.append(f"{style}/{job}: no autoregressive reference; speedup and AR agreement are null")
    if vispec is None:
        notes.append(f"{style}/{job}: no ViSpec reference; ViSpec-referenced columns are null")
    rows = []
    for run in sorted(by_arm.values(), key=lambda r: (r["method"], r["meta"].get("sweep", ""),
                                                     str(sparse_key(r["settings"])))):
        settings = run["settings"]
        sweep = sweep_name(settings)
        role = "reference" if sweep == "reference" else "arm"
        same_policy = next(
            (
                other for other in by_arm.values()
                if other["method"] == run["method"] and sweep != "reference"
                and sweep_name(other["settings"]) == sweep
                and policies_of(other["settings"]) == policies_of(settings)
                and at_full_retention(other["settings"])
                and all(
                    other["settings"][k] == 1.0
                    for k in ("kv_retention", "draft_retention")
                    if settings[k] is not None
                )
            ),
            None,
        )
        for benchmark in run["benchmarks"]:
            bench = run["report"]["summary"]["benchmarks"][benchmark]
            memory = bench.get("memory") or {}

            def stat(block, field, name="mean"):
                return (block.get(field) or {}).get(name)

            def mean_of(field):
                return (bench.get(field) or {}).get("mean")

            def peers(other):
                return other is not None and benchmark in other["benchmarks"]

            def mean_latency(other):
                return (
                    other["report"]["summary"]["benchmarks"][benchmark]["wall_time_s"]["mean"]
                    if peers(other) else None
                )

            def mem(other, field):
                return stat(other["report"]["summary"]["benchmarks"][benchmark].get("memory") or {},
                            field) if peers(other) else None

            def same_keys(other):
                return peers(other) and set(other["maps"][benchmark]) == set(run["maps"][benchmark])

            mismatches = [
                f"sample sets differ vs {label}: {len(run['maps'][benchmark])} vs "
                f"{len(other['maps'][benchmark])}"
                for label, other in (
                    ("vispec ref", vispec), ("ar ref", ar), ("same-policy 1.0", same_policy)
                )
                if peers(other) and not same_keys(other)
            ]
            vs_vispec = agreement(vispec, run, benchmark)
            vs_ar = agreement(ar, run, benchmark)
            quality = quality_mean(run["maps"][benchmark], set(run["maps"][benchmark]))
            ref_quality = (
                quality_mean(vispec["maps"][benchmark], set(vispec["maps"][benchmark]))
                if same_keys(vispec) else None
            )
            quality_comparable = same_keys(vispec) and len({
                (r.get("score_method"), json.dumps(r.get("score_protocol"), sort_keys=True))
                for m in (vispec["maps"][benchmark], run["maps"][benchmark]) for r in m.values()
            }) == 1
            bert = (bench.get("quality") or {}).get("bertscore_f1") if benchmark == "reverse_summary" else None
            ref_bert = (
                (vispec["report"]["summary"]["benchmarks"][benchmark].get("quality") or {}).get("bertscore_f1")
                if benchmark == "reverse_summary" and same_keys(vispec) else None
            )
            latency = mean_of("wall_time_s")
            kv_min = measurement_values(run, benchmark, "kv_min_kept_per_image")
            draft_min = measurement_values(run, benchmark, "draft_min_kept_per_image")
            kv_images = measurement_values(run, benchmark, "kv_image_tokens")
            sp = same_policy if same_keys(same_policy) else None
            rows.append({
                "prompt_style": style, "slurm_job_id": job,
                "sweep": sweep, "method": run["method"], "role": role,
                "arm": run["meta"].get("arm") or ("reference" if role == "reference" else None),
                "benchmark": benchmark,
                "kv_retention": settings["kv_retention"], "kv_policy": settings["kv_policy"],
                "draft_retention": settings["draft_retention"],
                "draft_policy": settings["draft_policy"],
                "n": bench["completed_turns"],
                "latency_mean_s": latency, "ttft_mean_s": mean_of("ttft_s"),
                "tpot_mean_s": mean_of("time_per_output_token_s"),
                "throughput_tokens_per_s": bench.get("throughput_tokens_per_second"),
                "acceptance_mean": mean_of("accepted_draft_tokens_per_iteration"),
                "decode_peak_mean_gib": stat(memory, "decode_peak_allocated_gib"),
                "decode_peak_max_gib": stat(memory, "decode_peak_allocated_gib", "max"),
                "live_kv_mean_mib": stat(memory, "kv_live_mib"),
                "quality": (bench.get("quality") or {}).get("score_percent"),
                "kv_image_tokens_mean": statistics.mean(kv_images) if kv_images else None,
                "kv_min_kept_per_image": min(kv_min) if kv_min else None,
                "draft_min_kept_per_image": min(draft_min) if draft_min else None,
                "post_compaction_peak_gib": stat(memory, "post_compaction_peak_allocated_gib"),
                "bertscore_f1": bert,
                "bertscore_f1_delta_vs_vispec_ref": (
                    bert - ref_bert if bert is not None and ref_bert is not None else None
                ),
                "post_compaction_peak_arm_over_samepolicy_1_0": (
                    ratio(stat(memory, "post_compaction_peak_allocated_gib"),
                          mem(sp, "post_compaction_peak_allocated_gib")) if sp else None
                ),
                "token_matched_vs_vispec": vs_vispec and vs_vispec["matched"],
                "token_checked_vs_vispec": vs_vispec and vs_vispec["checked"],
                "token_agreement_vs_vispec": vs_vispec and vs_vispec["rate"],
                "answer_agreement_vs_vispec": vs_vispec and vs_vispec["text"],
                "token_matched_vs_ar": vs_ar and vs_ar["matched"],
                "token_checked_vs_ar": vs_ar and vs_ar["checked"],
                "token_agreement_vs_ar": vs_ar and vs_ar["rate"],
                "speedup_vs_ar": ratio(mean_latency(ar), latency) if same_keys(ar) else None,
                "latency_speedup_vs_vispec_ref": (
                    ratio(mean_latency(vispec), latency) if same_keys(vispec) else None
                ),
                "latency_speedup_vs_samepolicy_1_0": (
                    ratio(mean_latency(sp), latency) if sp else None
                ),
                "decode_peak_arm_over_samepolicy_1_0": (
                    ratio(stat(memory, "decode_peak_allocated_gib"),
                          mem(sp, "decode_peak_allocated_gib")) if sp else None
                ),
                "live_kv_arm_over_samepolicy_1_0": (
                    ratio(stat(memory, "kv_live_mib"), mem(sp, "kv_live_mib")) if sp else None
                ),
                "quality_delta_vs_vispec_ref": (
                    quality - ref_quality
                    if quality is not None and ref_quality is not None and quality_comparable
                    else None
                ),
                "note": "; ".join(mismatches) or None,
                "run_dir": str(run["run_dir"]),
            })
    for row in rows:
        if row["note"]:
            notes.append(f"{style}/{job} {row['method']} {row['arm']} {row['benchmark']}: {row['note']}")
    return rows


def tree_of(meta):
    """The draft tree as total-depth-top_k, or None for a run that records no tree."""
    values = [meta.get(key) for key in ("total_token", "depth", "top_k")]
    return "-".join(str(v).strip() for v in values) if all(v is not None for v in values) else None


def refuse_for_sweep_layout(run):
    """The sweep layout cannot hold HiViS, another tree or another prompt order."""
    meta = run["meta"]
    reasons = []
    if run["method"] == "hivis_speculative":
        reasons.append("it is a HiViS run")
    tree = tuple(str(meta.get(k)).strip() for k in ("depth", "top_k", "total_token"))
    if (
        run["method"] != "autoregressive"
        and all(meta.get(k) is not None for k in ("depth", "top_k", "total_token"))
        and tree != DEFAULT_TREE
    ):
        reasons.append(f"its tree is {tree_of(meta)}, not 30-3-8")
    if run["prompt_order"] != DEFAULT_PROMPT_ORDER:
        reasons.append(f"its prompt order is {run['prompt_order']}, not {DEFAULT_PROMPT_ORDER}")
    if reasons:
        raise ValueError(
            f"The sweep layout cannot hold {run['run_dir']}: " + "; ".join(reasons)
            + ". Use --layout methods."
        )


def role_of(run):
    if run["method"] == "autoregressive":
        return "reference"
    primary = tree_of(run["meta"]) == "30-3-8" and run["prompt_order"] == DEFAULT_PROMPT_ORDER
    return "primary" if primary else "secondary"


def build_methods_group(style, job, runs, notes):
    check_group(runs, methods_layout=True)
    for run in runs:
        if any(value is not None for value in sparse_key(run["settings"])):
            raise ValueError(
                f"The methods layout refuses runs with sparsity settings: {run['run_dir']}"
            )
    by_key = {}
    for run in runs:
        if run["method"] not in ("autoregressive", "vispec_speculative", "hivis_speculative"):
            raise ValueError(f"Unknown method {run['method']!r}: {run['run_dir']}")
        tree = None if run["method"] == "autoregressive" else tree_of(run["meta"])
        if run["method"] != "autoregressive" and tree is None:
            raise ValueError(f"Speculative run without a recorded tree: {run['run_dir']}")
        key = (run["method"], tree, run["prompt_order"])
        if key in by_key:
            raise ValueError(
                "Two completed folders for one method, tree and prompt order: "
                f"{by_key[key]['run_dir']} and {run['run_dir']}"
            )
        by_key[key] = run
    method_rank = {"autoregressive": 0, "vispec_speculative": 1, "hivis_speculative": 2}
    rows = []
    for (method, tree, order), run in sorted(
        by_key.items(),
        key=lambda kv: (kv[0][2] != DEFAULT_PROMPT_ORDER, kv[0][2], kv[0][1] or "",
                        method_rank[kv[0][0]]),
    ):
        ar = by_key.get(("autoregressive", None, order))
        vispec = by_key.get(("vispec_speculative", tree, order)) if method == "hivis_speculative" else None
        base_tree = by_key.get((method, "30-3-8", order)) if tree == "60-5-10" else None
        for benchmark in run["benchmarks"]:
            bench = run["report"]["summary"]["benchmarks"][benchmark]
            memory = bench.get("memory") or {}

            def mean_of(field, bench=bench):
                return (bench.get(field) or {}).get("mean")

            def peers(other, benchmark=benchmark):
                return other is not None and benchmark in other["benchmarks"]

            def same_keys(other, benchmark=benchmark, run=run):
                return peers(other) and set(other["maps"][benchmark]) == set(run["maps"][benchmark])

            def mean_latency(other, benchmark=benchmark):
                return other["report"]["summary"]["benchmarks"][benchmark]["wall_time_s"]["mean"]

            latency = mean_of("wall_time_s")
            problems = []
            for label, other, needed in (
                ("AR run", ar, True),
                ("ViSpec run at the same tree", vispec, method == "hivis_speculative"),
                ("30-3-8 run of the same method", base_tree, tree == "60-5-10"),
            ):
                if not needed:
                    continue
                if not peers(other):
                    problems.append(f"no {label} for {benchmark} with prompt order {order}")
                elif not same_keys(other):
                    problems.append(
                        f"sample sets differ vs {label}: {len(run['maps'][benchmark])} vs "
                        f"{len(other['maps'][benchmark])}"
                    )
            vs_ar = agreement(ar, run, benchmark)
            row = {
                "prompt_style": style, "slurm_job_id": job, "method": method,
                "tree": tree or "none", "prompt_order": order, "role": role_of(run),
                "benchmark": benchmark, "n": bench["completed_turns"],
                "latency_mean_s": latency, "ttft_mean_s": mean_of("ttft_s"),
                "tpot_mean_s": mean_of("time_per_output_token_s"),
                "throughput_tokens_per_s": bench.get("throughput_tokens_per_second"),
                "output_tokens_mean": mean_of("output_tokens"),
                "acceptance_mean": mean_of("accepted_draft_tokens_per_iteration"),
                "output_tokens_per_iteration_mean": mean_of("output_tokens_per_verification_iteration"),
                "decode_peak_mean_gib": (memory.get("decode_peak_allocated_gib") or {}).get("mean"),
                "post_compaction_peak_gib": (memory.get("post_compaction_peak_allocated_gib") or {}).get("mean"),
                "quality": (bench.get("quality") or {}).get("score_percent"),
                "token_matched_vs_ar": vs_ar and vs_ar["matched"],
                "token_checked_vs_ar": vs_ar and vs_ar["checked"],
                "token_agreement_vs_ar": vs_ar and vs_ar["rate"],
                "answer_agreement_vs_ar": vs_ar and vs_ar["text"],
                "speedup_vs_ar": ratio(mean_latency(ar), latency) if same_keys(ar) else None,
                "speedup_vs_vispec": (
                    ratio(mean_latency(vispec), latency) if same_keys(vispec) else None
                ),
                "speedup_vs_tree_30_3_8": (
                    ratio(mean_latency(base_tree), latency) if same_keys(base_tree) else None
                ),
                "note": "; ".join(problems) or None,
                "run_dir": str(run["run_dir"]),
            }
            rows.append(row)
            if row["note"]:
                notes.append(f"{style}/{job} {method} {tree or 'none'} {order} {benchmark}: {row['note']}")
    return rows


def build(run_dirs, layout="sweep"):
    notes, skipped, groups = [], [], {}
    for run_dir in run_dirs:
        run, reason = load_run(run_dir)
        if run is None:
            skipped.append(reason)
            continue
        if layout == "sweep":
            refuse_for_sweep_layout(run)
        job = (run["meta"].get("slurm_job_id") or "").strip()
        style = run["meta"].get("prompt_style")
        if not job or job == "unset" or not style:
            raise ValueError(f"Run lacks slurm_job_id or prompt_style: {run['run_dir']}")
        groups.setdefault((style, job), []).append(run)
    rows = []
    for (style, job), runs in sorted(groups.items()):
        rows += (build_methods_group if layout == "methods" else build_group)(style, job, runs, notes)
    return rows, notes, skipped


def cell(value, digits=4):
    if value is None:
        return "Pending"
    if isinstance(value, float):
        return fmt(value) if digits == 4 else f"{value:.{digits}f}"
    return str(value)


def methods_cell(row, column):
    """A cell of the methods table: n/a where the column cannot apply, Pending where data is missing."""
    value = row[column]
    if value is None:
        if row["method"] == "autoregressive" and column in (
            "acceptance_mean", "output_tokens_per_iteration_mean",
            "speedup_vs_vispec", "speedup_vs_tree_30_3_8",
        ):
            return "n/a"
        if column == "speedup_vs_vispec" and row["method"] != "hivis_speculative":
            return "n/a"
        if column == "speedup_vs_tree_30_3_8" and row["tree"] != "60-5-10":
            return "n/a"
    return cell(value)


def write_methods_markdown(rows, notes, skipped, output_dir):
    lines = [
        "# Method comparison table", "",
        "- One table per (prompt style, Slurm job, benchmark). Every ratio stays inside its job"
        " and uses the same prompt order.",
        "- `speedup_vs_*` columns are reference mean latency / row mean latency (above 1 means the row"
        " is faster). `speedup_vs_ar` uses the AR run, `speedup_vs_vispec` the ViSpec run at the same"
        " tree, `speedup_vs_tree_30_3_8` the same method at 30-3-8.",
        "- `n/a` means the column does not apply to the row; `Pending` means the value is absent or"
        " its reference is missing.", "",
    ]
    shown = tuple(c for c in METHODS_COLUMNS if c not in ("prompt_style", "slurm_job_id", "benchmark", "note"))
    seen = []
    for r in rows:
        key = (r["prompt_style"], r["slurm_job_id"], r["benchmark"])
        if key in seen:
            continue
        seen.append(key)
        lines += [
            f"## {r['prompt_style']} job {r['slurm_job_id']} {r['benchmark']}", "",
            "| " + " | ".join(shown) + " |", "| " + " | ".join("---" for _ in shown) + " |",
        ]
        for other in rows:
            if (other["prompt_style"], other["slurm_job_id"], other["benchmark"]) == key:
                lines.append("| " + " | ".join(methods_cell(other, c) for c in shown) + " |")
        lines.append("")
    if notes or skipped:
        lines += ["## Notes", ""]
        lines += [f"- {note}" for note in notes]
        lines += [f"- Skipped: {reason}" for reason in skipped]
        lines.append("")
    (output_dir / "sweep.md").write_text("\n".join(lines))


def write_outputs(rows, notes, skipped, output_dir, layout="sweep"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    methods_layout = layout == "methods"
    fields = list(METHODS_COLUMNS if methods_layout else COLUMNS) + ["run_dir"]
    (output_dir / "sweep.json").write_text(json.dumps(
        {"columns": fields, "column_notes": METHODS_COLUMN_NOTES if methods_layout else COLUMN_NOTES,
         "rows": rows, "notes": notes,
         "skipped_runs": skipped},
        indent=2, allow_nan=False) + "\n")
    with (output_dir / "sweep.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: ("" if v is None else v) for k, v in r.items()} for r in rows)
    if methods_layout:
        write_methods_markdown(rows, notes, skipped, output_dir)
        return output_dir
    lines = [
        "# Retention sweep table", "",
        "- One table per (prompt style, Slurm job). Every ratio stays inside its job.",
        "- `*_speedup_vs_*` columns are reference latency / arm latency (above 1 means the arm is faster);"
        " `*_arm_over_*` columns are arm / reference memory (above 1 means more memory).",
        "- Speedup is the AR reference latency over the arm latency. Agreement is exact generated"
        " token IDs through first EOS (token) or whitespace-normalised decoded text (text).",
        "- `Pending` means the value is absent or its reference is missing.", "",
    ]
    shown = (
        "sweep", "method", "arm", "n", "latency_mean_s", "ttft_mean_s", "tpot_mean_s",
        "throughput_tokens_per_s", "acceptance_mean", "decode_peak_mean_gib",
        "decode_peak_max_gib", "live_kv_mean_mib", "quality", "kv_min_kept_per_image",
        "draft_min_kept_per_image", "post_compaction_peak_gib",
        "token_agreement_vs_vispec", "answer_agreement_vs_vispec", "token_agreement_vs_ar",
        "speedup_vs_ar", "latency_speedup_vs_vispec_ref", "latency_speedup_vs_samepolicy_1_0",
        "decode_peak_arm_over_samepolicy_1_0", "live_kv_arm_over_samepolicy_1_0",
        "post_compaction_peak_arm_over_samepolicy_1_0", "quality_delta_vs_vispec_ref",
    )
    seen = []
    for r in rows:
        key = (r["prompt_style"], r["slurm_job_id"], r["benchmark"])
        if key in seen:
            continue
        seen.append(key)
        lines += [
            f"## {r['prompt_style']} job {r['slurm_job_id']} {r['benchmark']}", "",
            "| " + " | ".join(shown) + " |", "| " + " | ".join("---" for _ in shown) + " |",
        ]
        for other in rows:
            if (other["prompt_style"], other["slurm_job_id"], other["benchmark"]) == key:
                values = dict(other)
                values["arm"] = other["arm"] or "reference"
                lines.append("| " + " | ".join(cell(values[c]) for c in shown) + " |")
        lines.append("")
    if notes or skipped:
        lines += ["## Notes", ""]
        lines += [f"- {note}" for note in notes]
        lines += [f"- Skipped: {reason}" for reason in skipped]
        lines.append("")
    (output_dir / "sweep.md").write_text("\n".join(lines))
    return output_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dirs", nargs="+", required=True, type=Path,
                        help="Run folders; each needs metadata.txt and a report")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="New folder, for example experiment/sweeps/<UTC>_<label>")
    parser.add_argument("--layout", choices=LAYOUTS, default="sweep",
                        help="sweep: the retention sweep of ViSpec and AR runs at the default tree "
                        "and prompt order; methods: one row per method, tree, prompt order and benchmark")
    args = parser.parse_args()
    rows, notes, skipped = build(args.run_dirs, args.layout)
    print(f"Sweep table: {write_outputs(rows, notes, skipped, args.output_dir, args.layout) / 'sweep.md'}")


if __name__ == "__main__":
    main()
