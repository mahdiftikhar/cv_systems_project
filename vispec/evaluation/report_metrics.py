"""Summarize saved runs and export benchmark predictions without loading models."""

import argparse
import json
import math
import os
import statistics
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .scoring import Scorer, evidence

BENCHMARKS = ("textvqa", "gqa", "mmvet", "coco_caption")


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = math.floor(position)
    high = math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def distribution(values):
    values = [value for value in values if value is not None]
    if any(
        not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0
        for value in values
    ):
        raise ValueError("Metrics must be finite nonnegative numbers")
    if not values:
        return {"count": 0, "mean": None, "median": None, "p95": None, "stdev": None}
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p95": percentile(values, 0.95),
        "stdev": statistics.stdev(values) if len(values) > 1 else None,
    }


def metadata(run_dir):
    path = run_dir / "metadata.txt"
    if not path.exists():
        return {}
    return dict(
        line.split("=", 1) for line in path.read_text().splitlines() if "=" in line
    )


def at(choice, key, turn):
    values = choice.get(key, [])
    return values[turn] if turn < len(values) else None


def samples_from_file(path, benchmark):
    seen = set()
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        question_id = str(record["question_id"])
        for choice in record["choices"]:
            for turn, answer in enumerate(choice["turns"]):
                key = (question_id, choice["index"], turn)
                if key in seen:
                    raise ValueError(f"Duplicate answer key in {path}: {key}")
                seen.add(key)
                measurement = at(choice, "measurement", turn) or {}
                valid_count = measurement.get("schema_version") == 1
                tokens = measurement.get("output_tokens") if valid_count else None
                latency = at(choice, "wall_time", turn)
                ttft = at(choice, "ttft", turn)
                if latency is not None and latency <= 0:
                    raise ValueError(f"Nonpositive generation time in {path}: {key}")
                if (
                    ttft is not None
                    and latency is not None
                    and not 0 <= ttft <= latency
                ):
                    raise ValueError(f"TTFT outside generation time in {path}: {key}")
                if tokens is not None and (not isinstance(tokens, int) or tokens < 0):
                    raise ValueError(f"Invalid output token count in {path}: {key}")
                accepted = choice.get("acceptance_length", [])
                rows.append(
                    {
                        "benchmark": benchmark,
                        "question_id": question_id,
                        "choice": choice["index"],
                        "turn": turn,
                        "answer": answer,
                        "reference_answers": record.get("reference_answers", []),
                        "wall_time_s": latency,
                        "ttft_s": ttft,
                        "decode_time_s": at(choice, "decode_time", turn),
                        "output_tokens": tokens,
                        "tokens_per_second": tokens / latency
                        if tokens is not None and latency
                        else None,
                        "time_per_output_token_s": (latency - ttft) / (tokens - 1)
                        if tokens is not None
                        and tokens > 1
                        and ttft is not None
                        and latency
                        else None,
                        "accepted_draft_tokens": accepted if turn == 0 else [],
                        "measurement": measurement,
                    }
                )
    return rows


def summarize(rows, expected_ids=None):
    latency = distribution([row["wall_time_s"] for row in rows])
    ttft = distribution([row["ttft_s"] for row in rows])
    paired = [
        row for row in rows if row["output_tokens"] is not None and row["wall_time_s"]
    ]
    accepted = [length for row in rows for length in row["accepted_draft_tokens"]]
    known_stops = [
        row for row in rows if row["measurement"].get("truncated") is not None
    ]
    memory = {}
    for row in rows:
        for device, values in row["measurement"].get("gpu_memory", {}).items():
            peaks = memory.setdefault(
                device, {"peak_allocated_bytes": 0, "peak_reserved_bytes": 0}
            )
            for field in peaks:
                peaks[field] = max(peaks[field], values[field])
    completed_ids = {row["question_id"] for row in rows}
    expected = set(expected_ids) if expected_ids is not None else None
    score_rows = [row for row in rows if row.get("score") is not None]
    reasons = {}
    for row in rows:
        reason = row["measurement"].get("stop_reason") or "unknown"
        reasons[reason] = reasons.get(reason, 0) + 1
    return {
        "completed_questions": len(
            {(row["benchmark"], row["question_id"]) for row in rows}
        ),
        "completed_turns": len(rows),
        "expected_questions": len(expected) if expected is not None else None,
        "missing_question_ids": sorted(expected - completed_ids)
        if expected is not None
        else None,
        "unexpected_question_ids": sorted(completed_ids - expected)
        if expected is not None
        else None,
        "wall_time_s": latency,
        "ttft_s": ttft,
        "missing_latency_turns": len(rows) - latency["count"],
        "missing_ttft_turns": len(rows) - ttft["count"],
        "decode_time_s": distribution([row["decode_time_s"] for row in rows]),
        "output_tokens": distribution([row["output_tokens"] for row in rows]),
        "answer_tokens": distribution(
            [row["measurement"].get("answer_tokens") for row in rows]
        ),
        "throughput_tokens_per_second": sum(row["output_tokens"] for row in paired)
        / sum(row["wall_time_s"] for row in paired)
        if paired
        else None,
        "throughput_turns": len(paired),
        "per_answer_tokens_per_second": distribution(
            [row["tokens_per_second"] for row in rows]
        ),
        "time_per_output_token_s": distribution(
            [row["time_per_output_token_s"] for row in rows]
        ),
        "accepted_draft_tokens_per_iteration": distribution(accepted),
        "output_tokens_per_verification_iteration": distribution(
            [value + 1 for value in accepted]
        ),
        "stop_reasons": reasons,
        "known_stop_turns": len(known_stops),
        "truncation_rate": sum(row["measurement"]["truncated"] for row in known_stops)
        / len(known_stops)
        if known_stops
        else None,
        "post_eos_tokens": sum(
            row["measurement"]["post_eos_tokens"]
            for row in rows
            if "post_eos_tokens" in row["measurement"]
        )
        if any("post_eos_tokens" in row["measurement"] for row in rows)
        else None,
        "post_eos_counted_turns": sum(
            "post_eos_tokens" in row["measurement"] for row in rows
        ),
        "token_counter_checked_turns": sum(
            "token_count_matches" in row["measurement"] for row in rows
        ),
        "token_counter_mismatch_turns": sum(
            row["measurement"].get("token_count_matches") is False for row in rows
        ),
        "gpu_memory_by_visible_device": memory,
        "quality": {
            "scored_questions": len(score_rows),
            "eligible_questions": sum(
                row["choice"] == 0 and row["turn"] == 0 for row in rows
            ),
            "score_percent": 100 * statistics.mean(row["score"] for row in score_rows)
            if score_rows
            else None,
            "status": "complete"
            if score_rows
            and len(score_rows)
            == sum(row["choice"] == 0 and row["turn"] == 0 for row in rows)
            else "partial"
            if score_rows
            else "pending",
            "methods": sorted(
                {row["score_method"] for row in rows if row.get("score_method")}
            ),
        },
    }


def fmt(value):
    return "Pending" if value is None else f"{value:.4f}"


def create_report(run_dir, output_dir=None, scorer=None, evaluation_exit_code=None):
    run_dir = Path(run_dir).resolve()
    if not run_dir.is_dir():
        raise ValueError(f"Run directory does not exist: {run_dir}")
    scorer = scorer or Scorer()
    if output_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_dir = run_dir / "reports" / f"{stamp}_{uuid.uuid4().hex[:8]}"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    sources = []
    all_rows = []
    benches = {}
    runtimes = {}
    predictions = {}
    coco_scores = {}
    for benchmark in BENCHMARKS:
        files = list((run_dir / "results").glob(f"{benchmark}_test/*/*.jsonl"))
        manifests = list(
            (run_dir / "results").glob(f"{benchmark}_test/*/*.manifest.json")
        )
        if len(files) > 1 or len(manifests) > 1:
            raise ValueError(f"Ambiguous result files for {benchmark}")
        rows = samples_from_file(files[0], benchmark) if files else []
        expected_ids = None
        if manifests:
            manifest = json.loads(manifests[0].read_text())
            expected_ids = [item["question_id"] for item in manifest["questions"]]
            sources.append(evidence(manifests[0]))
        if files:
            sources.append(evidence(files[0]))
            runtime_path = files[0].with_suffix(".runtime.json")
            if runtime_path.exists():
                runtimes[benchmark] = json.loads(runtime_path.read_text())
                sources.append(evidence(runtime_path))
        first_answers = {}
        corpus = None
        if benchmark == "coco_caption":
            first = [r for r in rows if r["choice"] == 0 and r["turn"] == 0]
            corpus = scorer.score_corpus(
                benchmark,
                {r["question_id"]: r["answer"] for r in first},
                {r["question_id"]: r["reference_answers"] for r in first},
            )
            coco_scores[benchmark] = {
                k: corpus[k] for k in ("score", "scored", "method", "pending_reason")
            }
        for row in rows:
            row["score"] = None
            if corpus is not None:
                if row["choice"] == 0 and row["turn"] == 0:
                    row["score"] = corpus["per_sample"].get(row["question_id"])
                    row["score_method"] = (
                        corpus["method"]
                        if row["score"] is not None
                        else corpus["pending_reason"]
                    )
                    row["score_protocol"] = scorer.protocol(benchmark, row["question_id"])
                    first_answers[row["question_id"]] = row["answer"]
            elif row["choice"] == 0 and row["turn"] == 0:
                row["score"], row["score_method"] = scorer.score(
                    benchmark,
                    row["question_id"],
                    row["answer"],
                    row["reference_answers"],
                )
                row["score_protocol"] = scorer.protocol(benchmark, row["question_id"])
                first_answers[row["question_id"]] = row["answer"]
        predictions[benchmark] = first_answers
        benches[benchmark] = summarize(rows, expected_ids)
        all_rows.extend(rows)
    # Export formats accepted by benchmark tooling. No answer extraction heuristics.
    (output_dir / "mmvet_predictions.json").write_text(
        json.dumps(predictions["mmvet"], indent=2) + "\n"
    )
    (output_dir / "gqa_predictions.json").write_text(
        json.dumps(
            [
                {"questionId": key, "prediction": value}
                for key, value in predictions["gqa"].items()
            ],
            indent=2,
        )
        + "\n"
    )
    (output_dir / "textvqa_predictions.json").write_text(
        json.dumps(
            [
                {"question_id": key, "answer": value}
                for key, value in predictions["textvqa"].items()
            ],
            indent=2,
        )
        + "\n"
    )
    (output_dir / "coco_caption_predictions.json").write_text(
        json.dumps(
            [
                {"image_id": key, "caption": value}
                for key, value in predictions["coco_caption"].items()
            ],
            indent=2,
        )
        + "\n"
    )
    exit_path = run_dir / "exit-code.txt"
    if evaluation_exit_code is None and exit_path.exists():
        evaluation_exit_code = int(exit_path.read_text().strip())
    overall = summarize(all_rows)
    # Benchmark scores use different protocols; do not average them across the suite.
    overall.pop("quality")
    summary = {
        "schema_version": 1,
        "run_dir": str(run_dir),
        "evaluation_exit_code": evaluation_exit_code,
        "metadata": metadata(run_dir),
        "runtime": runtimes,
        "sources": sources,
        "scoring_sources": scorer.sources,
        "benchmarks": benches,
        "coco_caption_cider": coco_scores.get("coco_caption"),
        "overall": overall,
        "definitions": {
            "wall_time_s": "generation call; excludes loading, prompt preparation and reporting",
            "ttft_s": "generation entry to first authoritative target token ready; not streamed delivery",
            "output_tokens": "returned token IDs, including special tokens and post-EOS tokens; not retokenized text",
            "throughput": "sum(output_tokens) / sum(wall_time_s) on turns with both measurements",
            "time_per_output_token_s": "(wall_time_s - ttft_s) / (output_tokens - 1), only for output_tokens > 1",
            "percentile": "linear interpolation at (n - 1) * quantile",
            "memory": "per-visible-device PyTorch allocator peaks during each call; includes resident model/KV cache; excludes non-PyTorch allocations",
            "quality": "choice 0, turn 0 only; subset score, no averaging across benchmarks",
            "coco_caption_quality": "CIDEr-D x100 (pycocoevalcap 1.2, PTB tokenizer, IDF from the evaluated subset's references); needs Java",
        },
        "warnings": [
            "Generation limits differ between original and draft; inspect stop reasons before interpreting speedups.",
            "Inherited TextVQA prompts request OCR explanations; strict answer scoring may penalize verbose responses.",
            "Memory is measured after warmup; initial cache/model allocation peaks are excluded.",
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    with (output_dir / "samples.jsonl").open("w") as stream:
        for row in all_rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    lines = [
        "# Run metrics",
        "",
        f"- Run: `{run_dir}`",
        f"- Evaluation exit code: `{evaluation_exit_code}`",
        "- Units: seconds, tokens/second, and quality score percent.",
        "- Quality applies to completed questions in the evaluated subset.",
        f"- Device report: [device.txt]({os.path.relpath(run_dir / 'device.txt', output_dir)})",
        "",
        "| Benchmark | Questions | Mean latency | Median | p95 | Mean TTFT | TTFT p95 | Tokens/s | Truncated | Quality (%) | Scored |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for benchmark, bench in benches.items():
        quality = bench["quality"]
        lines.append(
            f"| {benchmark} | {bench['completed_questions']}/{bench['expected_questions'] if bench['expected_questions'] is not None else '?'} | "
            f"{fmt(bench['wall_time_s']['mean'])} | {fmt(bench['wall_time_s']['median'])} | {fmt(bench['wall_time_s']['p95'])} | "
            f"{fmt(bench['ttft_s']['mean'])} | {fmt(bench['ttft_s']['p95'])} | {fmt(bench['throughput_tokens_per_second'])} | "
            f"{fmt(bench['truncation_rate'])} | {fmt(quality['score_percent'])} | {quality['scored_questions']}/{quality['eligible_questions']} |"
        )
    lines += [
        "",
        "- Mean/median/p95 TTFT, decode time, TPOT, output lengths, acceptance, and per-device peak memory are in `summary.json`.",
        "- Missing measurements remain null; summary distributions include their sample counts.",
        "- Peak bytes are reported separately for each visible GPU; peaks are not summed across devices.",
        "- Quality scoring status and missing prerequisites are listed below.",
        "",
    ]
    for benchmark, bench in benches.items():
        lines.append(
            f"- **{benchmark}:** {bench['quality']['status']}; "
            + "; ".join(bench["quality"]["methods"])
        )
    cider = coco_scores.get("coco_caption")
    if cider:
        lines.append(
            "- **coco_caption CIDEr-D (x100):** "
            + (
                f"{100 * cider['score']:.2f} on {cider['scored']} scored captions ({cider['method']})"
                if cider["score"] is not None
                else f"Pending, {cider['pending_reason']}"
            )
        )
    lines += [
        "",
        "| Benchmark | Visible GPU | Peak allocated (GiB) | Peak reserved (GiB) |",
        "| --- | --- | --- | --- |",
    ]
    for benchmark, bench in benches.items():
        for device, memory in bench["gpu_memory_by_visible_device"].items():
            lines.append(
                f"| {benchmark} | {device} | {memory['peak_allocated_bytes'] / 2**30:.3f} | {memory['peak_reserved_bytes'] / 2**30:.3f} |"
            )
    lines.append("")
    for benchmark, runtime in runtimes.items():
        gpus = ", ".join(
            f"{gpu['index']}: {gpu['name']} ({gpu['total_memory_bytes'] / 2**30:.1f} GiB)"
            if "total_memory_bytes" in gpu
            else gpu["name"]
            for gpu in runtime["devices"]
        )
        lines.append(
            f"- **{benchmark} hardware:** {gpus}; precision {runtime['parameter_dtypes']}."
        )
        lines.append(
            f"- **{benchmark} allocation:** `{runtime['allocation']}`; CPU affinity: `{runtime.get('cpu_affinity')}`."
        )
    lines += ["", *["- " + warning for warning in summary["warnings"]], ""]
    (output_dir / "summary.md").write_text("\n".join(lines))
    return output_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--bench-dir", type=Path)
    parser.add_argument(
        "--textvqa-evaluator", default=os.environ.get("TEXTVQA_EVALUATOR")
    )
    parser.add_argument(
        "--textvqa-annotations", default=os.environ.get("TEXTVQA_ANNOTATIONS")
    )
    parser.add_argument("--gqa-annotations", default=os.environ.get("GQA_ANNOTATIONS"))
    parser.add_argument("--mmvet-grades", default=os.environ.get("MMVET_GRADES"))
    parser.add_argument(
        "--mmvet-graded-predictions", default=os.environ.get("MMVET_GRADED_PREDICTIONS")
    )
    parser.add_argument("--evaluation-exit-code", type=int)
    args = parser.parse_args()
    if args.bench_dir and not args.textvqa_annotations:
        candidate = args.bench_dir / "textvqa/TextVQA_0.5.1_val.json"
        if candidate.exists():
            args.textvqa_annotations = str(candidate)
    scorer = Scorer(
        args.textvqa_evaluator,
        args.textvqa_annotations,
        args.gqa_annotations,
        args.mmvet_grades,
        args.mmvet_graded_predictions,
    )
    report = create_report(
        args.run_dir, args.output_dir, scorer, args.evaluation_exit_code
    )
    print(f"Metrics report: {report / 'summary.md'}")


if __name__ == "__main__":
    main()
