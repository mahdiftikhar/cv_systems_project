"""Compare report directories, with variation across repeated cluster runs."""

import argparse
import json
import statistics
from pathlib import Path

from .report_metrics import BENCHMARKS, distribution, fmt
from .scoring import evidence


def load_report(path):
    path = Path(path).resolve()
    summary = json.loads((path / "summary.json").read_text())
    rows = [
        json.loads(line)
        for line in (path / "samples.jsonl").read_text().splitlines()
        if line.strip()
    ]
    if summary.get("schema_version") != 1:
        raise ValueError(f"Unsupported report version: {path}")
    return {"path": path, "summary": summary, "rows": rows}


def sample_key(row):
    return (row["question_id"], row["choice"], row["turn"])


def runtime_signature(report, benchmark):
    runtime = report["summary"].get("runtime", {}).get(benchmark)
    if not runtime or not runtime.get("devices") or not runtime.get("parameter_dtypes"):
        return None
    return {
        key: runtime.get(key)
        for key in (
            "devices",
            "parameter_dtypes",
            "allocation",
            "cpu_affinity",
            "base_model_revision",
        )
    }



def token_agreement(original_maps, draft_maps, common):
    """Compare raw generated IDs through first EOS, including EOS itself."""
    checked = matched = 0
    missing = 0
    mismatches = []
    for original_run, original in enumerate(original_maps):
        for draft_run, draft in enumerate(draft_maps):
            for key in sorted(common):
                sequences = []
                for row in (original[key], draft[key]):
                    measurement = row["measurement"]
                    ids = measurement.get("output_token_ids")
                    eos = measurement.get("eos_token_id")
                    if ids is None:
                        break
                    if eos is not None and eos in ids:
                        ids = ids[: ids.index(eos) + 1]
                    sequences.append(ids)
                if len(sequences) != 2:
                    missing += 1
                    continue
                checked += 1
                a, b = sequences
                if a == b:
                    matched += 1
                    continue
                position = next(
                    (i for i in range(min(len(a), len(b))) if a[i] != b[i]),
                    min(len(a), len(b)),
                )
                mismatches.append({
                    "original_run_index": original_run,
                    "draft_run_index": draft_run,
                    "question_id": key[0], "choice": key[1], "turn": key[2],
                    "first_mismatch_index": position,
                    "original_token_id": a[position] if position < len(a) else None,
                    "draft_token_id": b[position] if position < len(b) else None,
                    "original_length": len(a), "draft_length": len(b),
                })
    return {
        "checked_pairs": checked, "matched_pairs": matched,
        "missing_pairs": missing,
        "sequence_agreement_rate": matched / checked if checked else None,
        "mismatches": mismatches,
    }


def compare(original_paths, draft_paths, output_dir):
    original = [load_report(path) for path in original_paths]
    draft = [load_report(path) for path in draft_paths]
    all_reports = original + draft
    paths = [str(report["path"]) for report in all_reports]
    if len(set(paths)) != len(paths):
        raise ValueError("Repeated report paths would count the same run twice")
    run_paths = [report["summary"]["run_dir"] for report in all_reports]
    if len(set(run_paths)) != len(run_paths):
        raise ValueError("Multiple reports from one run are not independent repeats")
    for group, method in [(original, "autoregressive"), (draft, "vispec_speculative")]:
        for report in group:
            if report["summary"]["metadata"].get("method") != method:
                raise ValueError(f"Wrong method for comparison group: {report['path']}")
        for setting in ("depth", "top_k", "total_token", "num_q"):
            values = {report["summary"]["metadata"].get(setting) for report in group}
            if len(values) > 1:
                raise ValueError(f"Different {setting} settings are not repeated runs")
    for setting in ("base_model", "spec_model", "temperature"):
        values = {report["summary"]["metadata"].get(setting) for report in all_reports}
        if len(values) > 1:
            raise ValueError(f"Cannot compare different {setting} settings")
    styles = {report["summary"]["metadata"].get("prompt_style") for report in all_reports}
    if len(styles) > 1:
        raise ValueError(
            "Cannot compare different prompt_style values: "
            + ", ".join(repr(style) for style in sorted(styles, key=repr))
        )
    result = {
        "original_reports": [str(report["path"]) for report in original],
        "draft_reports": [str(report["path"]) for report in draft],
        "benchmarks": {},
        "sources": [
            evidence(report["path"] / "summary.json") for report in all_reports
        ],
    }
    for benchmark in BENCHMARKS:
        signatures = [runtime_signature(report, benchmark) for report in all_reports]
        known_signatures = [
            signature for signature in signatures if signature is not None
        ]
        if known_signatures and any(
            signature != known_signatures[0] for signature in known_signatures
        ):
            raise ValueError(
                f"Hardware/allocation/precision mismatch for {benchmark}; keep separate comparisons"
            )
        maps = [
            {
                sample_key(row): row
                for row in report["rows"]
                if row["benchmark"] == benchmark
            }
            for report in all_reports
        ]
        common = set.intersection(*(set(rows) for rows in maps))
        issues = []
        if not common:
            issues.append("No matching completed turns")
        if any(set(rows) != common for rows in maps):
            issues.append(
                "Question/choice/turn sets differ; ratios use their intersection"
            )
        if any(signature is None for signature in signatures):
            issues.append("Missing runtime hardware/precision evidence")
        for report in all_reports:
            summary = report["summary"]
            if summary.get("evaluation_exit_code") != 0:
                issues.append("At least one evaluation did not complete successfully")
            bench = summary["benchmarks"][benchmark]
            if (
                bench["missing_question_ids"] is None
                or bench["missing_question_ids"]
                or bench["unexpected_question_ids"]
            ):
                issues.append("Incomplete or unknown expected question coverage")
        matched_rows = [rows[key] for rows in maps for key in common]
        if any(not row["measurement"].get("input_ids_sha256") for row in matched_rows):
            issues.append("Missing prompt-token fingerprints")
        if any(
            len(
                {
                    rows[key]["measurement"].get("input_ids_sha256")
                    for rows in maps
                    if rows[key]["measurement"].get("input_ids_sha256")
                }
            )
            > 1
            for key in common
        ):
            raise ValueError(
                f"Prompt mismatch for {benchmark}; compare matching prompts"
            )
        for group_maps in (maps[: len(original)], maps[len(original) :]):
            for key in common:
                limits = [
                    rows[key]["measurement"].get("generation_limits")
                    for rows in group_maps
                ]
                if any(limit != limits[0] for limit in limits):
                    raise ValueError(
                        f"Different generation limits are not repeated runs: {benchmark}"
                    )
        if any(row["wall_time_s"] is None for row in matched_rows):
            issues.append("Missing generation latencies among matched turns")
        if any(row["measurement"].get("stop_reason") != "eos" for row in matched_rows):
            issues.append("Truncated or unknown stop reasons among matched turns")
        if any(
            row["measurement"].get("token_count_matches") is False
            for row in matched_rows
        ):
            issues.append("Generation counters disagree with returned token IDs")
        if any(
            not report["summary"]["metadata"].get("base_model")
            or not report["summary"]["metadata"].get("temperature")
            for report in all_reports
        ):
            issues.append("Missing model or temperature settings")
        groups = []
        for group_maps in (maps[: len(original)], maps[len(original) :]):
            means, ttfts, rates, lengths, qualities = [], [], [], [], []
            for rows in group_maps:
                selected = [rows[key] for key in common]
                times = [
                    row["wall_time_s"]
                    for row in selected
                    if row["wall_time_s"] is not None
                ]
                first = [row["ttft_s"] for row in selected if row["ttft_s"] is not None]
                paired = [
                    row
                    for row in selected
                    if row["output_tokens"] is not None and row["wall_time_s"]
                ]
                scores = [
                    row["score"] for row in selected if row.get("score") is not None
                ]
                means.append(
                    statistics.mean(times)
                    if times and len(times) == len(selected)
                    else None
                )
                ttfts.append(
                    statistics.mean(first)
                    if first and len(first) == len(selected)
                    else None
                )
                rates.append(
                    sum(row["output_tokens"] for row in paired)
                    / sum(row["wall_time_s"] for row in paired)
                    if paired
                    else None
                )
                lengths.append(
                    statistics.mean(row["output_tokens"] for row in paired)
                    if paired
                    else None
                )
                qualities.append(100 * statistics.mean(scores) if scores else None)
            groups.append(
                {
                    "run_count": len(group_maps),
                    "run_mean_latency_s": distribution(means),
                    "run_mean_ttft_s": distribution(ttfts),
                    "run_throughput_tokens_per_second": distribution(rates),
                    "run_mean_output_tokens": distribution(lengths),
                    "run_quality_percent": distribution(qualities),
                }
            )

        def ratio(field, groups=groups):
            numerator = groups[0][field]["mean"]
            denominator = groups[1][field]["mean"]
            return (
                numerator / denominator
                if numerator is not None and denominator
                else None
            )

        agreement = []
        for original_map in maps[: len(original)]:
            for draft_map in maps[len(original) :]:
                if common:
                    agreement.append(
                        sum(
                            " ".join(original_map[key]["answer"].split())
                            == " ".join(draft_map[key]["answer"].split())
                            for key in common
                        )
                        / len(common)
                    )
        tokens = token_agreement(maps[: len(original)], maps[len(original) :], common)
        greedy = all(
            float(report["summary"]["metadata"].get("temperature", "nan")) == 0
            for report in all_reports
        )
        caps = [(row["measurement"].get("generation_limits") or {}).get("max_new_tokens")
                for row in matched_rows]
        if not caps or any(cap is None for cap in caps):
            issues.append("Missing output-token limit evidence")
        elif len(set(caps)) != 1:
            issues.append("Output-token limits differ across methods or runs")
        if greedy and (tokens["missing_pairs"] or not tokens["checked_pairs"]):
            issues.append("Missing generated token IDs for greedy correctness validation")
        if greedy and tokens["mismatches"]:
            issues.append("Greedy token sequences differ; inspect first mismatches")
        quality_sets = [
            {key for key in common if rows[key].get("score") is not None}
            for rows in maps
        ]
        quality_methods = {
            rows[key].get("score_method")
            for rows in maps
            for key in common
            if rows[key].get("score") is not None
        }
        quality_a = groups[0]["run_quality_percent"]["mean"]
        quality_b = groups[1]["run_quality_percent"]["mean"]
        quality_protocols_match = all(
            all(
                rows[key].get("score_protocol") == maps[0][key].get("score_protocol")
                for rows in maps
            )
            for key in common
        )
        quality_comparable = (
            bool(common)
            and all(keys == common for keys in quality_sets)
            and len(quality_methods) == 1
            and quality_protocols_match
        )
        result["benchmarks"][benchmark] = {
            "matched_turns_per_run": len(common),
            "original": groups[0],
            "draft": groups[1],
            "latency_ratio_original_over_draft": ratio("run_mean_latency_s"),
            "controlled_latency_speedup": ratio("run_mean_latency_s")
            if not issues
            else None,
            "ttft_ratio_original_over_draft": ratio("run_mean_ttft_s"),
            "answer_agreement_rate": statistics.mean(agreement) if agreement else None,
            "token_agreement": tokens,
            "greedy_correctness_validated": (
                tokens["checked_pairs"] > 0 and tokens["missing_pairs"] == 0
                and not tokens["mismatches"] and not issues
            ) if greedy else None,
            "quality_delta_percentage_points": quality_b - quality_a
            if quality_comparable
            else None,
            "warnings": sorted(set(issues)),
        }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "comparison.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    lines = [
        "# Original vs ViSpec",
        "",
        "- Ratios compare means of per-run means on matching completed turns.",
        "- Repeated-run standard deviations are in `comparison.json`; one run cannot estimate variability.",
        "- Answer agreement ignores whitespace differences; it is not an accuracy score.",
        "- Token agreement compares IDs through first EOS inclusive; indices are zero-based. Missing IDs stay pending.",
        "- At nonzero temperature, token agreement does not validate sampling correctness.",
        "- Controlled speedup stays pending when coverage, settings, or stop reasons are unsuitable.",
        "",
        "| Benchmark | Matched turns/run | Original mean (s) | Draft mean (s) | Latency ratio | Controlled speedup | Agreement | Quality Δ (pp) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for benchmark, row in result["benchmarks"].items():
        lines.append(
            f"| {benchmark} | {row['matched_turns_per_run']} | {fmt(row['original']['run_mean_latency_s']['mean'])} | "
            f"{fmt(row['draft']['run_mean_latency_s']['mean'])} | {fmt(row['latency_ratio_original_over_draft'])} | "
            f"{fmt(row['controlled_latency_speedup'])} | {fmt(row['answer_agreement_rate'])} | {fmt(row['quality_delta_percentage_points'])} |"
        )
    lines += [
        "",
        "| Benchmark | Token pairs checked | Matching sequences | Missing pairs | Token agreement | Greedy validated |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    mismatch_lines = []
    for benchmark, row in result["benchmarks"].items():
        tokens = row["token_agreement"]
        lines.append(
            f"| {benchmark} | {tokens['checked_pairs']} | {tokens['matched_pairs']} | "
            f"{tokens['missing_pairs']} | {fmt(tokens['sequence_agreement_rate'])} | "
            f"{row['greedy_correctness_validated']} |"
        )
        for mismatch in tokens["mismatches"]:
            mismatch_lines.append(
                f"- {benchmark}, question {mismatch['question_id']}, choice {mismatch['choice']}, "
                f"turn {mismatch['turn']}, original run {mismatch['original_run_index']}, "
                f"draft run {mismatch['draft_run_index']}: first mismatch at output index "
                f"{mismatch['first_mismatch_index']}; token IDs "
                f"{mismatch['original_token_id']} vs {mismatch['draft_token_id']} "
                "(null means sequence ended)."
            )
    lines += [
        "",
        *mismatch_lines,
        "",
        *[
            f"- **{benchmark}:** " + "; ".join(row["warnings"])
            for benchmark, row in result["benchmarks"].items()
            if row["warnings"]
        ],
        "",
    ]
    (output_dir / "comparison.md").write_text("\n".join(lines))
    return output_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--original",
        nargs="+",
        required=True,
        type=Path,
        help="Report directories from independent original runs",
    )
    parser.add_argument(
        "--draft",
        nargs="+",
        required=True,
        type=Path,
        help="Report directories from independent ViSpec runs",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    print(
        f"Comparison: {compare(args.original, args.draft, args.output_dir) / 'comparison.md'}"
    )


if __name__ == "__main__":
    main()
