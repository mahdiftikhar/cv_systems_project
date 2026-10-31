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
    SPARSITY_SETTINGS,
    answer_agreement_rates,
    check_same_slurm_job,
    generation_setting,
    load_report,
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
    "live_kv_arm_over_samepolicy_1_0", "quality_delta_vs_vispec_ref", "note",
)

COLUMN_NOTES = {
    "speedup_vs_ar": "AR reference mean latency / arm mean latency; above 1 means the arm is faster",
    "latency_speedup_vs_vispec_ref": "ViSpec reference mean latency / arm mean latency; above 1 means the arm is faster",
    "latency_speedup_vs_samepolicy_1_0": "same-policy 1.0 arm mean latency / arm mean latency; above 1 means the arm is faster",
    "decode_peak_arm_over_samepolicy_1_0": "arm mean decode peak / same-policy 1.0 arm mean decode peak; above 1 means more memory",
    "live_kv_arm_over_samepolicy_1_0": "arm mean live KV / same-policy 1.0 arm mean live KV; above 1 means more memory",
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
    return {
        "run_dir": run_dir, "meta": meta, "report": report, "benchmarks": benchmarks,
        "settings": settings[benchmarks[0]],
        "method": meta.get("method"),
        "maps": {
            name: {sample_key(r): r for r in report["rows"] if r["benchmark"] == name}
            for name in benchmarks
        },
    }, None


def check_group(runs):
    """Refuse a group that mixes cards, allocations, prompts or decoding settings."""
    reports = [run["report"] for run in runs]
    check_same_slurm_job(reports)
    for setting in SHARED_SETTINGS:
        require_same([run["meta"].get(setting) for run in runs], f"{setting} (metadata)")
    require_same([run["meta"].get("prompt_style") for run in runs], "prompt_style (metadata)")
    # The autoregressive runs record no draft-tree settings; compare the ViSpec runs only.
    for setting in ("depth", "top_k", "total_token", "num_q"):
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
                if prompts.setdefault((name, key), digest) != digest:
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


def build(run_dirs):
    notes, skipped, groups = [], [], {}
    for run_dir in run_dirs:
        run, reason = load_run(run_dir)
        if run is None:
            skipped.append(reason)
            continue
        job = (run["meta"].get("slurm_job_id") or "").strip()
        style = run["meta"].get("prompt_style")
        if not job or job == "unset" or not style:
            raise ValueError(f"Run lacks slurm_job_id or prompt_style: {run['run_dir']}")
        groups.setdefault((style, job), []).append(run)
    rows = []
    for (style, job), runs in sorted(groups.items()):
        rows += build_group(style, job, runs, notes)
    return rows, notes, skipped


def cell(value, digits=4):
    if value is None:
        return "Pending"
    if isinstance(value, float):
        return fmt(value) if digits == 4 else f"{value:.{digits}f}"
    return str(value)


def write_outputs(rows, notes, skipped, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    fields = list(COLUMNS) + ["run_dir"]
    (output_dir / "sweep.json").write_text(json.dumps(
        {"columns": fields, "column_notes": COLUMN_NOTES, "rows": rows, "notes": notes,
         "skipped_runs": skipped},
        indent=2, allow_nan=False) + "\n")
    with (output_dir / "sweep.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: ("" if v is None else v) for k, v in r.items()} for r in rows)
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
        "quality_delta_vs_vispec_ref",
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
    args = parser.parse_args()
    rows, notes, skipped = build(args.run_dirs)
    print(f"Sweep table: {write_outputs(rows, notes, skipped, args.output_dir) / 'sweep.md'}")


if __name__ == "__main__":
    main()
