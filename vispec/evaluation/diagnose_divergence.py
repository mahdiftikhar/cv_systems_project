"""Diagnose the first greedy divergence of each autoregressive/ViSpec answer pair.

Greedy ViSpec should reproduce autoregressive greedy decoding token for token, but in
bf16 a different kernel shape (one-token decode vs tree verification) can flip a
near-tied argmax. For every mismatching turn in a ``compare_metrics`` comparison, this
tool rebuilds the exact prompt, appends the common output prefix (the autoregressive
IDs before the first differing index) and runs one full forward pass of the stock
``transformers.Qwen2_5_VLForConditionalGeneration``, which is independent of the
repository's KV-cache fork. It records the top-2 next-token logits and their margin at
the last position, and repeats the pass in fp32 for up to ``--fp32-limit`` cases.

A case is ``near_tie`` when the bf16 top-2 margin is at most 0.5 logits, otherwise
``unexplained`` (a possible bug).

Usage:
python -m vispec.evaluation.diagnose_divergence --comparison CMP/comparison.json \
    --original-run RUN02 --draft-run RUN03 --data-dir bench_data \
    --base-model-path models/Qwen2.5-VL-3B-Instruct --output-dir OUT [--fp32-limit 50]
"""

import argparse
import gc
import hashlib
import json
import os
import socket
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

from .benchmarks import build_inputs, load_processor, load_samples
from .report_metrics import BENCHMARKS, metadata
from .scoring import evidence

NEAR_TIE_MARGIN = 0.5
DEVICE = "cuda:0"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--comparison", required=True, type=Path)
    parser.add_argument("--original-run", required=True, type=Path)
    parser.add_argument("--draft-run", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--base-model-path", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--fp32-limit", type=int, default=50)
    args = parser.parse_args(argv)
    if args.fp32_limit < 0:
        parser.error("--fp32-limit must be nonnegative")
    return args


def fingerprint(input_ids):
    """Same prompt fingerprint as ``GenerationMeasurement``."""
    return hashlib.sha256(json.dumps(input_ids.tolist()).encode()).hexdigest()


def qid_key(question_id):
    return (0, int(question_id), "") if question_id.isdigit() else (1, 0, question_id)


def bench_key(benchmark):
    return BENCHMARKS.index(benchmark) if benchmark in BENCHMARKS else len(BENCHMARKS)


def result_file(run_dir, benchmark, pattern):
    files = sorted((run_dir / "results").glob(f"{benchmark}_test/*/{pattern}"))
    if len(files) != 1:
        raise ValueError(
            f"Expected one {pattern} for {benchmark} in {run_dir}, found {len(files)}"
        )
    return files[0]


def load_records(run_dir, benchmark):
    path = result_file(run_dir, benchmark, "*.jsonl")
    records = {}
    for line in path.read_text().splitlines():
        if line.strip():
            record = json.loads(line)
            records[str(record["question_id"])] = record
    return path, records


def turn_measurement(record, choice, turn):
    for entry in record["choices"]:
        if entry["index"] == choice:
            return entry["measurement"][turn]
    raise KeyError(f"choice {choice} missing for question {record['question_id']}")


def report_run_dir(report_path):
    summary = json.loads((Path(report_path) / "summary.json").read_text())
    return Path(summary["run_dir"]).resolve()


def collect_cases(comparison, original_run, draft_run):
    """Mismatches whose report indices point at the two given run folders."""
    original_runs = [report_run_dir(p) for p in comparison["original_reports"]]
    draft_runs = [report_run_dir(p) for p in comparison["draft_reports"]]
    cases, skipped = [], 0
    for benchmark, row in comparison["benchmarks"].items():
        for mismatch in row["token_agreement"]["mismatches"]:
            if (
                original_runs[mismatch["original_run_index"]] != original_run
                or draft_runs[mismatch["draft_run_index"]] != draft_run
            ):
                skipped += 1
                continue
            cases.append({"benchmark": benchmark, **mismatch})
    if skipped and not cases:
        raise ValueError(
            "No comparison mismatch belongs to the given --original-run/--draft-run pair"
        )
    cases.sort(
        key=lambda c: (
            bench_key(c["benchmark"]),
            qid_key(str(c["question_id"])),
            c["choice"],
            c["turn"],
            c["original_run_index"],
            c["draft_run_index"],
        )
    )
    return cases, skipped


def attach_ids(cases, original_run, draft_run):
    """Add output IDs and prompt fingerprints from both runs; check the comparison."""
    loaded = {}
    for case in cases:
        benchmark = case["benchmark"]
        if benchmark not in loaded:
            loaded[benchmark] = (
                load_records(original_run, benchmark),
                load_records(draft_run, benchmark),
            )
        (original_path, original), (draft_path, draft) = loaded[benchmark]
        qid = str(case["question_id"])
        a = turn_measurement(original[qid], case["choice"], case["turn"])
        b = turn_measurement(draft[qid], case["choice"], case["turn"])
        ar_ids, draft_ids = a["output_token_ids"], b["output_token_ids"]
        index = case["first_mismatch_index"]
        ar_token = ar_ids[index] if index < len(ar_ids) else None
        vispec_token = draft_ids[index] if index < len(draft_ids) else None
        if (
            ar_ids[:index] != draft_ids[:index]
            or ar_token != case["original_token_id"]
            or vispec_token != case["draft_token_id"]
        ):
            raise ValueError(
                f"comparison.json does not match the run outputs for {benchmark} "
                f"question {qid}; pass the runs the comparison was made from"
            )
        case.update(
            prefix_ids=ar_ids[:index],
            ar_token_id=ar_token,
            vispec_token_id=vispec_token,
            original_input_ids_sha256=a.get("input_ids_sha256"),
            draft_input_ids_sha256=b.get("input_ids_sha256"),
            original_results=str(original_path),
            draft_results=str(draft_path),
        )
    return {
        benchmark: [str(pair[0][0]), str(pair[1][0])] for benchmark, pair in loaded.items()
    }


def subset_size(run_dir, benchmark, run_metadata):
    try:
        manifest = json.loads(
            result_file(run_dir, benchmark, "*.manifest.json").read_text()
        )
        return len(manifest["questions"])
    except ValueError:
        return int(run_metadata.get("num_samples", 100))


def load_model(path, dtype):
    from transformers import Qwen2_5_VLForConditionalGeneration

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        path, torch_dtype=dtype, low_cpu_mem_usage=True, device_map=DEVICE
    )
    model.eval()
    return model


@torch.inference_mode()
def last_logits(model, inputs, prefix_ids):
    input_ids = inputs["input_ids"]
    if prefix_ids:
        prefix = torch.tensor([prefix_ids], dtype=input_ids.dtype, device=input_ids.device)
        input_ids = torch.cat([input_ids, prefix], dim=1)
    extras = {k: v for k, v in inputs.items() if k not in ("input_ids", "attention_mask")}
    outputs = model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        use_cache=False,
        **extras,
    )
    return outputs.logits[0, -1].float()


def agrees(token_id, case):
    if token_id == case["ar_token_id"]:
        return "ar"
    if token_id == case["vispec_token_id"]:
        return "vispec"
    return "neither"


def inspect(logits, case, tokenizer):
    # torch.argmax returns the first maximal index, the decoders' tie-break; topk's
    # order among exactly tied logits differs, so the top-2 are argmax and the best rest.
    first = int(logits.argmax())
    rest = logits.clone()
    rest[first] = float("-inf")
    second = int(rest.argmax())
    ids = [first, second]
    values = [logits[i].item() for i in ids]
    tied = (logits == logits[first]).nonzero().flatten().tolist()

    def token_logit(token_id):
        return None if token_id is None else logits[token_id].item()

    return {
        "top2": [
            {"token_id": i, "text": tokenizer.decode([i]), "logit": v}
            for i, v in zip(ids, values)
        ],
        "margin": values[0] - values[1],
        "argmax_token_id": first,
        "tokens_tied_at_max": len(tied),
        "tied_token_ids": tied[:10],
        "agrees": agrees(first, case),
        "ar_token_logit": token_logit(case["ar_token_id"]),
        "vispec_token_logit": token_logit(case["vispec_token_id"]),
    }


def run_pass(model, cases, processor, samples, prompt_style, label, check_prompt):
    mismatched = []
    for number, case in enumerate(cases, 1):
        sample = samples[case["benchmark"]][str(case["question_id"])]
        inputs = build_inputs(processor, case["benchmark"], sample, prompt_style, DEVICE)
        if check_prompt:
            rebuilt = fingerprint(inputs["input_ids"])
            recorded = {
                case["original_input_ids_sha256"],
                case["draft_input_ids_sha256"],
            }
            case["rebuilt_input_ids_sha256"] = rebuilt
            case["prompt_fingerprint_matches"] = recorded == {rebuilt}
            if not case["prompt_fingerprint_matches"]:
                mismatched.append(case)
                print(
                    f"WARNING: PROMPT FINGERPRINT MISMATCH for {case['benchmark']} "
                    f"question {case['question_id']}: rebuilt {rebuilt}, recorded "
                    f"{sorted(recorded, key=str)}",
                    file=sys.stderr,
                    flush=True,
                )
        logits = last_logits(model, inputs, case["prefix_ids"])
        case[label] = inspect(logits, case, processor.tokenizer)
        print(
            f"[{label} {number}/{len(cases)}] {case['benchmark']} {case['question_id']} "
            f"index {case['first_mismatch_index']}: margin {case[label]['margin']:.4f}, "
            f"argmax agrees with {case[label]['agrees']}",
            flush=True,
        )
        del inputs, logits
    return mismatched


def margin_stats(values):
    values = [v for v in values if v is not None]
    if not values:
        return {"count": 0, "min": None, "median": None, "max": None}
    return {
        "count": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
    }


def counts(values, keys):
    return {key: sum(value == key for value in values) for key in keys}


def environment(model_path):
    from transformers import AutoConfig
    import transformers

    prop = torch.cuda.get_device_properties(0)
    return {
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu": {
            "name": prop.name,
            "uuid": str(getattr(prop, "uuid", "unavailable")),
            "total_memory_bytes": prop.total_memory,
            "compute_capability": f"{prop.major}.{prop.minor}",
        },
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "base_model_revision": getattr(
            AutoConfig.from_pretrained(model_path), "_commit_hash", None
        ),
    }


def fmt(value, digits=4):
    return "n/a" if value is None else f"{value:.{digits}f}"


def cell(text):
    return "`" + json.dumps(text, ensure_ascii=False).replace("|", "\\|").replace("`", "'") + "`"


def token_cell(entry):
    return f"{cell(entry['text'])} ({entry['token_id']}, {fmt(entry['logit'], 3)})"


def token_ref(case, key):
    token_id = case[f"{key}_token_id"]
    return "(ended)" if token_id is None else f"{cell(case[f'{key}_token_text'])} ({token_id})"


def write_markdown(path, result):
    summary = result["summary"]
    lines = ["# Greedy divergence diagnosis", ""]
    if summary["prompt_fingerprint_mismatches"]:
        lines += [
            f"**WARNING: {summary['prompt_fingerprint_mismatches']} case(s) rebuilt a prompt "
            "whose input-ID fingerprint differs from the recorded one. Those cases do not "
            "diagnose the recorded generations.**",
            "",
        ]
    lines += [
        f"Comparison: `{result['inputs']['comparison']['path']}`. Original run: "
        f"`{result['inputs']['original_run']}`. Draft run: `{result['inputs']['draft_run']}`. "
        f"Prompt style: `{result['inputs']['prompt_style']}`.",
        "",
        "For every turn whose greedy autoregressive (AR) and ViSpec token sequences differ, "
        "the prompt is rebuilt, the shared output prefix (AR IDs before the first differing "
        "zero-based index) is appended, and one full forward pass of the stock "
        f"`transformers.Qwen2_5_VLForConditionalGeneration` "
        f"(attention `{result['settings']['attn_implementation']}`, no KV cache, not the "
        "repository's KV fork) gives the next-token logits at the last position. The "
        "margin is the top-1 minus top-2 logit in raw logit units. A case is `near_tie` "
        f"when the bf16 margin is at most {NEAR_TIE_MARGIN}, otherwise `unexplained`. "
        "`bf16 agrees` / `fp32 agrees` say whether that pass's argmax equals the AR token, "
        "the ViSpec token, or neither. The fp32 pass (TF32 disabled) runs for at most "
        f"{result['settings']['fp32_limit']} cases in sorted order; other cases show "
        "`not_run`.",
        "",
        f"Hardware: {result['environment']['gpu']['name']} on `{result['environment']['hostname']}`, "
        f"Slurm job `{result['environment']['slurm_job_id']}`, torch "
        f"{result['environment']['torch']}, transformers {result['environment']['transformers']}.",
        "",
        "## Counts",
        "",
        "| Benchmark | Cases | near_tie | unexplained | fp32 agrees ar | fp32 agrees vispec | fp32 agrees neither | fp32 not run |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for benchmark, row in [*summary["per_benchmark"].items(), ("all", summary["overall"])]:
        fp32 = row["fp32_agrees"]
        lines.append(
            f"| {benchmark} | {row['cases']} | {row['classes']['near_tie']} | "
            f"{row['classes']['unexplained']} | {fp32['ar']} | {fp32['vispec']} | "
            f"{fp32['neither']} | {fp32['not_run']} |"
        )
    lines += [
        "",
        "## Margin distribution (logits)",
        "",
        "| Pass | Cases | Min | Median | Max |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for label in ("bf16", "fp32"):
        stats = summary["overall"][f"{label}_margin"]
        lines.append(
            f"| {label} | {stats['count']} | {fmt(stats['min'])} | "
            f"{fmt(stats['median'])} | {fmt(stats['max'])} |"
        )
    lines += [
        "",
        "## Cases",
        "",
        "Tokens show the decoded text, token ID and logit. Indices are zero-based output positions. "
        "Top-1 is `torch.argmax` (first maximal index, the decoders' tie-break); top-2 is the "
        "best remaining token. `bf16 ties` counts tokens whose bf16 logit equals the maximum.",
        "",
        "| Benchmark | Question | Choice/turn | Runs (orig/draft) | First diff | AR token | ViSpec token | bf16 top-1 | bf16 top-2 | bf16 margin | bf16 ties | bf16 agrees | fp32 margin | fp32 agrees | Class | Prompt match |",
        "| --- | --- | --- | --- | ---: | --- | --- | --- | --- | ---: | ---: | --- | ---: | --- | --- | --- |",
    ]
    for case in result["cases"]:
        bf16, fp32 = case["bf16"], case.get("fp32")
        lines.append(
            f"| {case['benchmark']} | {case['question_id']} | {case['choice']}/{case['turn']} | "
            f"{case['original_run_index']}/{case['draft_run_index']} | {case['first_mismatch_index']} | "
            f"{token_ref(case, 'ar')} | {token_ref(case, 'vispec')} | {token_cell(bf16['top2'][0])} | "
            f"{token_cell(bf16['top2'][1])} | {fmt(bf16['margin'])} | "
            f"{bf16['tokens_tied_at_max']} | {bf16['agrees']} | "
            f"{fmt(fp32['margin']) if fp32 else 'n/a'} | {case['fp32_agrees']} | "
            f"{case['class']} | {case['prompt_fingerprint_matches']} |"
        )
    lines.append("")
    path.write_text("\n".join(lines))


def main(argv=None):
    args = parse_args(argv)
    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    original_run = args.original_run.resolve()
    draft_run = args.draft_run.resolve()
    comparison = json.loads(args.comparison.read_text())
    original_meta, draft_meta = metadata(original_run), metadata(draft_run)
    if original_meta.get("method") != "autoregressive":
        raise ValueError(f"{original_run} is not an autoregressive run")
    if draft_meta.get("method") != "vispec_speculative":
        raise ValueError(f"{draft_run} is not a ViSpec run")
    prompt_style = original_meta.get("prompt_style")
    if not prompt_style or prompt_style != draft_meta.get("prompt_style"):
        raise ValueError(
            f"prompt_style differs or is missing: {prompt_style!r} vs "
            f"{draft_meta.get('prompt_style')!r}"
        )

    cases, skipped = collect_cases(comparison, original_run, draft_run)
    result_files = attach_ids(cases, original_run, draft_run)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    print(f"{len(cases)} mismatching turns to diagnose ({skipped} from other runs skipped)")

    samples = {}
    for benchmark in {case["benchmark"] for case in cases}:
        size = subset_size(original_run, benchmark, original_meta)
        samples[benchmark] = {
            s.question_id: s for s in load_samples(benchmark, args.data_dir, size)
        }
        missing = {str(c["question_id"]) for c in cases if c["benchmark"] == benchmark}
        missing -= set(samples[benchmark])
        if missing:
            raise ValueError(f"{benchmark} questions not in the subset: {sorted(missing)}")

    processor = load_processor(args.base_model_path)
    for case in cases:
        for key in ("ar", "vispec"):
            token_id = case[f"{key}_token_id"]
            case[f"{key}_token_text"] = (
                None if token_id is None else processor.tokenizer.decode([token_id])
            )
    fp32_cases = cases[: args.fp32_limit]
    attn = None
    mismatched = []
    if cases:
        model = load_model(args.base_model_path, torch.bfloat16)
        attn = model.config._attn_implementation
        mismatched = run_pass(
            model, cases, processor, samples, prompt_style, "bf16", check_prompt=True
        )
        del model
        gc.collect()
        torch.cuda.empty_cache()
    if fp32_cases:
        # A true fp32 reference: no TF32 in matmuls or convolutions (vision patch embed).
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        model = load_model(args.base_model_path, torch.float32)
        run_pass(
            model, fp32_cases, processor, samples, prompt_style, "fp32", check_prompt=False
        )
        del model

    for case in cases:
        case["class"] = (
            "near_tie" if case["bf16"]["margin"] <= NEAR_TIE_MARGIN else "unexplained"
        )
        case["fp32_agrees"] = case["fp32"]["agrees"] if "fp32" in case else "not_run"
        case["prefix_length"] = len(case.pop("prefix_ids"))

    def group(rows):
        return {
            "cases": len(rows),
            "classes": counts([r["class"] for r in rows], ("near_tie", "unexplained")),
            "bf16_agrees": counts(
                [r["bf16"]["agrees"] for r in rows], ("ar", "vispec", "neither")
            ),
            "fp32_agrees": counts(
                [r["fp32_agrees"] for r in rows], ("ar", "vispec", "neither", "not_run")
            ),
            "bf16_margin": margin_stats([r["bf16"]["margin"] for r in rows]),
            "fp32_margin": margin_stats([r.get("fp32", {}).get("margin") for r in rows]),
        }

    overall = group(cases)
    result = {
        "schema_version": 1,
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "command": [
            sys.executable,
            "-m",
            "vispec.evaluation.diagnose_divergence",
            *(sys.argv[1:] if argv is None else argv),
        ],
        "inputs": {
            "comparison": evidence(args.comparison),
            "original_run": str(original_run),
            "draft_run": str(draft_run),
            "result_files": result_files,
            "data_dir": str(args.data_dir.resolve()),
            "base_model_path": str(Path(args.base_model_path).resolve()),
            "run_base_models": [original_meta.get("base_model"), draft_meta.get("base_model")],
            "prompt_style": prompt_style,
        },
        "settings": {
            "model_class": "transformers.Qwen2_5_VLForConditionalGeneration",
            "attn_implementation": attn,
            "passes": ["bfloat16", "float32 (TF32 disabled)"],
            "near_tie_max_margin": NEAR_TIE_MARGIN,
            "fp32_limit": args.fp32_limit,
            "case_order": "benchmark order, then question ID, choice, turn, run indices",
        },
        "environment": environment(args.base_model_path),
        "definitions": {
            "first_mismatch_index": "zero-based output position of the first differing token (from comparison.json)",
            "margin": "top-1 minus top-2 next-token logit at the last position, raw logit units",
            "class": f"near_tie if bf16 margin <= {NEAR_TIE_MARGIN}, otherwise unexplained",
            "agrees": "whether that pass's argmax equals the AR token, the ViSpec token, or neither",
            "argmax": "torch.argmax, the first maximal index (decoder tie-break); top-2 is the best remaining token",
            "tokens_tied_at_max": "number of vocabulary entries whose logit equals the argmax logit",
        },
        "summary": {
            "overall": overall,
            "per_benchmark": {
                benchmark: group([c for c in cases if c["benchmark"] == benchmark])
                for benchmark in sorted({c["benchmark"] for c in cases}, key=bench_key)
            },
            "skipped_other_run_mismatches": skipped,
            "prompt_fingerprint_mismatches": len(mismatched),
        },
        "cases": cases,
    }
    (args.output_dir / "diagnosis.json").write_text(
        json.dumps(result, indent=2, allow_nan=False, ensure_ascii=False) + "\n"
    )
    write_markdown(args.output_dir / "diagnosis.md", result)
    print(
        f"Diagnosis: {args.output_dir / 'diagnosis.md'}; classes {overall['classes']}; "
        f"fp32 agrees {overall['fp32_agrees']}"
    )
    if mismatched:
        print(
            f"WARNING: {len(mismatched)} PROMPT FINGERPRINT MISMATCH(ES); see diagnosis.md",
            file=sys.stderr,
        )
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
