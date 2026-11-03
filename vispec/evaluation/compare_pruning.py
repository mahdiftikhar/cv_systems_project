"""Compare one KV-pruning sweep with unpruned ViSpec and autoregressive decoding."""

import argparse
import json
import statistics
from pathlib import Path

from .compare_metrics import load_report, runtime_signature, sample_key
from .report_metrics import BENCHMARKS, fmt


def latest_report(run_dir):
    reports = sorted((run_dir / "reports").glob("*/summary.json"))
    if not reports:
        raise ValueError(f"No report found in {run_dir}")
    return load_report(reports[-1].parent)


def output_ids(row):
    """Raw generated IDs through first EOS inclusive, or None when not recorded."""
    ids = row["measurement"].get("output_token_ids")
    eos = row["measurement"].get("eos_token_id")
    if ids is not None and eos is not None and eos in ids:
        return ids[: ids.index(eos) + 1]
    return ids


def matching_prefix(a, b):
    for index, (left, right) in enumerate(zip(a, b)):
        if left != right:
            return index
    return min(len(a), len(b))


def mean(values):
    values = [value for value in values if value is not None]
    return statistics.mean(values) if values else None


def compare_to_reference(rows, reference, autoregressive):
    """Measure one run on the questions it shares with the unpruned ViSpec run."""
    common = sorted(set(rows) & set(reference))
    if not common:
        return {"questions": 0}
    for key in common:
        hashes = {
            rows[key]["measurement"].get("input_ids_sha256"),
            reference[key]["measurement"].get("input_ids_sha256"),
        }
        if len(hashes) > 1:
            raise ValueError(f"Prompt mismatch for {key}; compare matching prompts")
    selected = [rows[key] for key in common]
    baseline = [reference[key] for key in common]
    pairs = [(output_ids(a), output_ids(b)) for a, b in zip(selected, baseline)]
    pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
    latency = mean([row["wall_time_s"] for row in selected])
    reference_latency = mean([row["wall_time_s"] for row in baseline])
    shared = [key for key in common if key in autoregressive]
    shared_latency = mean([rows[key]["wall_time_s"] for key in shared])
    autoregressive_latency = mean([autoregressive[key]["wall_time_s"] for key in shared])
    scored = [
        (row["score"], base["score"])
        for row, base in zip(selected, baseline)
        if row.get("score") is not None and base.get("score") is not None
    ]
    accepted = [length for row in selected for length in row["accepted_draft_tokens"]]
    pruning = [row["measurement"].get("kv_pruning") for row in selected]
    known = [item for item in pruning if item]
    return {
        "questions": len(common),
        "mean_latency_s": latency,
        "mean_ttft_s": mean([row["ttft_s"] for row in selected]),
        "speedup_vs_unpruned": reference_latency / latency if latency else None,
        "speedup_vs_autoregressive": autoregressive_latency / shared_latency
        if shared and shared_latency
        else None,
        "accepted_draft_tokens_per_step": mean(accepted),
        "mean_output_tokens": mean([row["output_tokens"] for row in selected]),
        "mean_image_positions": mean([item["image_positions"] for item in known]),
        "mean_kept_image_positions": mean(
            [item["kept_image_positions"] for item in known]
        ),
        "mean_logical_kv_removed_gib": mean(
            [
                item["removed_positions"] * item["bytes_per_position"] / 2**30
                for item in known
            ]
        ),
        "token_pairs_checked": len(pairs),
        "identical_token_sequences": sum(a == b for a, b in pairs),
        "matching_prefix_percent": 100
        * sum(matching_prefix(a, b) for a, b in pairs)
        / sum(len(b) for _, b in pairs)
        if pairs
        else None,
        "identical_answer_text": sum(
            " ".join(row["answer"].split()) == " ".join(base["answer"].split())
            for row, base in zip(selected, baseline)
        ),
        "scored_questions": len(scored),
        "quality_percent": 100 * mean([a for a, _ in scored]) if scored else None,
        "quality_delta_points": 100
        * (mean([a for a, _ in scored]) - mean([b for _, b in scored]))
        if scored
        else None,
        "truncated_answers": sum(
            row["measurement"].get("truncated") is True for row in selected
        ),
    }


def table_row(label, row):
    if not row["questions"]:
        return f"| {label} | 0 |" + " Pending |" * 13
    cells = [
        label,
        row["questions"],
        fmt(row["mean_kept_image_positions"]),
        fmt(row["mean_logical_kv_removed_gib"]),
        fmt(row["mean_latency_s"]),
        fmt(row["speedup_vs_unpruned"]),
        fmt(row["speedup_vs_autoregressive"]),
        fmt(row["accepted_draft_tokens_per_step"]),
        fmt(row["mean_output_tokens"]),
        f"{row['identical_token_sequences']}/{row['token_pairs_checked']}",
        fmt(row["matching_prefix_percent"]),
        f"{row['identical_answer_text']}/{row['questions']}",
        fmt(row["quality_percent"]),
        fmt(row["quality_delta_points"]),
        row["truncated_answers"],
    ]
    return "| " + " | ".join(str(cell) for cell in cells) + " |"


def compare(sweep_dir, output_dir):
    autoregressive = latest_report(sweep_dir / "autoregressive")
    runs = {}
    for run_dir in sorted(sweep_dir.glob("keep_*"), reverse=True):
        report = latest_report(run_dir)
        runs[float(report["summary"]["metadata"]["kv_keep_ratio"])] = report
    if 1.0 not in runs:
        raise ValueError("The sweep needs an unpruned keep_100 run as its reference")
    all_reports = [autoregressive, *runs.values()]
    for setting in ("base_model", "temperature", "answer_style", "max_new_tokens"):
        if len({r["summary"]["metadata"].get(setting) for r in all_reports}) > 1:
            raise ValueError(f"Cannot compare different {setting} settings")

    def by_key(report, benchmark):
        return {sample_key(r): r for r in report["rows"] if r["benchmark"] == benchmark}

    result = {"sweep_dir": str(sweep_dir), "benchmarks": {}}
    lines = [
        "# KV pruning sweep",
        "",
        f"Answer style: {autoregressive['summary']['metadata'].get('answer_style')}.",
        "",
        "Agreement columns compare each row with unpruned ViSpec (keep 100%), through",
        "first EOS inclusive. The autoregressive row shows the difference that already",
        "exists between the methods. Logical KV removed counts positions dropped from the",
        "cache; the cache tensor stays preallocated, so peak GPU memory does not fall.",
        "",
    ]
    for benchmark in BENCHMARKS:
        reference = by_key(runs[1.0], benchmark)
        baseline = by_key(autoregressive, benchmark)
        if not reference and not baseline:
            continue
        signatures = {
            json.dumps(runtime_signature(report, benchmark), sort_keys=True)
            for report in all_reports
            if runtime_signature(report, benchmark) is not None
        }
        if len(signatures) > 1:
            raise ValueError(f"Hardware or allocation mismatch for {benchmark}")
        rows = {"autoregressive": compare_to_reference(baseline, reference, baseline)}
        for ratio, report in sorted(runs.items(), reverse=True):
            rows[str(ratio)] = compare_to_reference(
                by_key(report, benchmark), reference, baseline
            )
        result["benchmarks"][benchmark] = rows
        lines += [
            f"## {benchmark}",
            "",
            "| Keep | Questions | Image positions kept | Logical KV removed (GiB) | Mean latency (s) | Speedup vs keep 100% | Speedup vs autoregressive | Accepted draft tokens/step | Output tokens | Identical token sequences | Matching prefix (%) | Identical answer text | Quality (%) | Quality Δ (points) | Truncated |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            table_row("Autoregressive", rows["autoregressive"]),
            *[
                table_row(f"{100 * ratio:g}%", rows[str(ratio)])
                for ratio in sorted(runs, reverse=True)
            ],
            "",
        ]
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "comparison.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    (output_dir / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    print(f"Comparison: {compare(args.sweep_dir, args.output_dir) / 'comparison.md'}")


if __name__ == "__main__":
    main()
