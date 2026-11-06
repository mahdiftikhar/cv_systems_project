"""fp32 oracle for the compact KV layout: AR and ViSpec must agree under one eviction setting.

Greedy ViSpec commits only tokens that the target model itself picks during tree
verification, so an autoregressive run and a ViSpec run with the same target-side KV
eviction must return the same tokens, whatever the draft proposes. Draft-side pruning
changes only the proposals, so the agreement must also hold with ``--draft-retention``. In
bf16, different kernel shapes (one-token decode against tree verification) can flip a
near-tied argmax. In fp32 with TF32 off those flips go away, so an fp32 mismatch points at
the cache bookkeeping: ``kv_offset``, decode positions or the kept rows. The stock
``transformers`` model cannot reproduce eviction, so this check is the only oracle for a
large ``kv_offset`` (the evicted prompt tokens).

The tool loads one ``SpecModel`` with ``--kv-layout compact`` and runs, for each sample,
``autoregressive_generate`` and ``specgenerate`` with the same ``SparsityConfig``. It reports
exact agreement of the raw output IDs per question (through EOS), the first differing
zero-based index when they differ, and the eviction counts of both runs. It writes
``offset_check.json`` and ``offset_check.md`` to a new ``--output-dir`` and exits 0 when
every question agrees, 1 otherwise.

Usage:
python -m vispec.evaluation.offset_check --kv-retention 0.25 --kv-policy spatial \
    [--draft-retention 0.25 --draft-policy spatial] --num-samples 5 --dtype float32 \
    --output-dir OUT
"""

import argparse
import json
import os
import socket
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

from ..model.autoregressive import autoregressive_generate
from ..model.sparsity import POLICIES, RETENTIONS, SparsityConfig
from ..model.spec_model_ours import SpecModel
from .benchmarks import (
    BENCHMARK_NAMES,
    PROMPT_STYLES,
    build_inputs,
    check_prompt_style,
    draft_position_budget,
    load_processor,
    load_samples,
)
from .diagnose_divergence import fingerprint, git_commit

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}
REPO_DIR = Path(__file__).resolve().parents[2]
# Eviction counts that both methods must report identically for one prompt.
KV_FIELDS = (
    "kv_prompt_tokens",
    "kv_kept_prompt_tokens",
    "kv_evicted_tokens",
    "kv_image_tokens",
    "kv_min_kept_per_image",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--kv-retention", type=float, choices=RETENTIONS, required=True)
    parser.add_argument("--kv-policy", choices=POLICIES, required=True)
    parser.add_argument(
        "--draft-retention",
        type=float,
        choices=RETENTIONS,
        default=None,
        help="Prune the ViSpec draft's image-adaptor input as well (default 1.0 once "
        "--draft-policy is given).",
    )
    parser.add_argument(
        "--draft-policy",
        choices=POLICIES,
        default=None,
        help="Draft pruning policy (default spatial once --draft-retention is given).",
    )
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--dtype", choices=tuple(DTYPES), default="float32")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--benchmark",
        choices=BENCHMARK_NAMES,
        default="textvqa",
        help="Benchmark whose first --num-samples prompts are checked (default textvqa). "
        "reverse_summary uses its Re:Verse page windows (long style only), at most 55.",
    )
    parser.add_argument("--prompt-style", choices=PROMPT_STYLES, default="long")
    parser.add_argument("--base-model-path", default="models/Qwen2.5-VL-3B-Instruct")
    parser.add_argument(
        "--spec-model-path", default="models/ViSpec-Qwen2.5-VL-3B-Instruct"
    )
    parser.add_argument("--data-dir", type=Path, default=Path("bench_data"))
    parser.add_argument("--max-new-token", type=int, default=1024)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--total-token", type=int, default=30)
    parser.add_argument("--num-q", type=int, default=2)
    parser.add_argument("--kv-capacity", type=int, default=16384)
    args = parser.parse_args(argv)
    if args.num_samples < 1:
        parser.error("--num-samples must be positive")
    if args.max_new_token < 1:
        parser.error("--max-new-token must be positive")
    try:
        check_prompt_style(args.benchmark, args.prompt_style)
    except ValueError as error:
        parser.error(str(error))
    if args.output_dir.exists():
        parser.error(f"--output-dir {args.output_dir} already exists; use a new folder")
    return args


def sparsity_configs(args):
    """(kv, draft) SparsityConfigs; draft is None unless a draft flag is given."""
    kv = SparsityConfig(retention=args.kv_retention, policy=args.kv_policy)
    if args.draft_retention is None and args.draft_policy is None:
        return kv, None
    draft = SparsityConfig(
        retention=1.0 if args.draft_retention is None else args.draft_retention,
        policy="spatial" if args.draft_policy is None else args.draft_policy,
    )
    return kv, draft


def load_model(args):
    """One compact-layout SpecModel in ``--dtype``, with the evaluator's tree settings."""
    model = SpecModel.from_pretrained(
        base_model_path=args.base_model_path,
        spec_model_path=args.spec_model_path,
        total_token=args.total_token,
        depth=args.depth,
        top_k=args.top_k,
        num_q=args.num_q,
        slim_prefill=True,
        kv_capacity=args.kv_capacity,
        kv_layout="compact",
        torch_dtype=DTYPES[args.dtype],
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    model.eval()
    return model


def first_mismatch(a, b):
    """First differing zero-based index, None when equal; a length difference counts."""
    if a == b:
        return None
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


def eviction_counts(stats):
    eviction = stats.get("kv_eviction") or {}
    return {field: eviction.get(field) for field in KV_FIELDS}


def check_sample(model, inputs, max_new_token, kv, draft):
    """Run both methods on one prompt; return the per-question row."""
    input_len = inputs["input_ids"].shape[1]
    ar_stats = {}
    torch.manual_seed(0)
    ar_ids, _, _ = autoregressive_generate(
        model, inputs, max_new_token, generation_stats=ar_stats, kv_sparsity=kv
    )
    vs_stats = {}
    torch.manual_seed(0)
    vs_ids, _, _, acceptance = model.specgenerate(
        **inputs,
        temperature=0.0,
        max_new_tokens=max_new_token,
        log=True,
        return_acceptance_len=True,
        generation_stats=vs_stats,
        kv_sparsity=kv,
        draft_sparsity=draft,
    )
    ar = ar_ids[0, input_len:].tolist()
    vs = vs_ids[0, input_len:].tolist()
    index = first_mismatch(ar, vs)
    ar_kv, vs_kv = eviction_counts(ar_stats), eviction_counts(vs_stats)
    row = {
        "input_ids_sha256": fingerprint(inputs["input_ids"]),
        "input_tokens": input_len,
        "exact": index is None,
        "first_mismatch_index": index,
        "ar_token_at_mismatch": None if index is None or index >= len(ar) else ar[index],
        "vispec_token_at_mismatch": None if index is None or index >= len(vs) else vs[index],
        "ar_tokens": len(ar),
        "vispec_tokens": len(vs),
        "ar_stop_reason": ar_stats.get("stop_reason"),
        "vispec_stop_reason": vs_stats.get("stop_reason"),
        "vispec_iterations": vs_stats.get("iterations"),
        "vispec_mean_acceptance": statistics.mean(acceptance) if acceptance else None,
        # Physical cache length = token count - kv_offset during decoding.
        "kv_offset": vs_kv["kv_evicted_tokens"],
        "kv_counts_match": ar_kv == vs_kv,
        "kv": vs_kv,
        "ar_kv": ar_kv,
        "draft_pruning": vs_stats.get("draft_pruning"),
        "ar_output_token_ids": ar,
        "vispec_output_token_ids": vs,
    }
    return row


def environment():
    import transformers

    prop = torch.cuda.get_device_properties(0)
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPO_DIR, capture_output=True, text=True
    )
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
        "git_commit": git_commit(),
        "git_status_porcelain": status.stdout.splitlines() if status.returncode == 0 else None,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }


def write_markdown(path, result):
    settings, summary = result["settings"], result["summary"]
    draft = settings["draft_sparsity"]
    lines = [
        "# Offset check: AR vs ViSpec under one eviction setting",
        "",
        f"- dtype: `{settings['dtype']}`; KV eviction: {settings['kv_sparsity']['retention']} "
        f"`{settings['kv_sparsity']['policy']}`; draft pruning: "
        + (f"{draft['retention']} `{draft['policy']}`" if draft else "off"),
        f"- {settings['benchmark']} `{settings['prompt_style']}`, {summary['questions']} questions, "
        f"`max_new_tokens` {settings['max_new_tokens']}, compact layout",
        f"- Host `{result['environment']['hostname']}`, job `{result['environment']['slurm_job_id']}`, "
        f"GPU {result['environment']['gpu']['name']} `{result['environment']['gpu']['uuid']}`",
        f"- **AR == ViSpec: {summary['exact']}/{summary['questions']}**",
        "",
        "| Question | Input | Image tokens | Kept prompt | kv_offset | Min kept/image | Draft min kept/image | AR tokens | ViSpec tokens | Exact | First mismatch | KV counts match |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |",
    ]
    for row in result["questions"]:
        kv = row["kv"]
        pruning = row["draft_pruning"] or {}
        lines.append(
            f"| {row['question_id']} | {row['input_tokens']} | {kv['kv_image_tokens']} | "
            f"{kv['kv_kept_prompt_tokens']} | {row['kv_offset']} | {kv['kv_min_kept_per_image']} | "
            f"{pruning.get('draft_min_kept_per_image', 'n/a')} | {row['ar_tokens']} | "
            f"{row['vispec_tokens']} | {row['exact']} | "
            f"{'-' if row['first_mismatch_index'] is None else row['first_mismatch_index']} | "
            f"{row['kv_counts_match']} |"
        )
    lines.append("")
    path.write_text("\n".join(lines))


@torch.inference_mode()
def run(args):
    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    kv, draft = sparsity_configs(args)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    if args.dtype == "float32":
        # A true fp32 reference: no TF32 in matmuls or convolutions (vision patch embed).
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
    # The processor comes first: reverse_summary checks each window's prompt length
    # against the draft's position budget while loading its samples.
    processor = load_processor(args.base_model_path)
    samples = load_samples(
        args.benchmark,
        args.data_dir,
        args.num_samples,
        processor=processor,
        position_budget=draft_position_budget(
            args.max_new_token, args.depth, args.spec_model_path
        ),
    )
    model = load_model(args)
    if processor.tokenizer.eos_token_id != model.tokenizer.eos_token_id:
        raise RuntimeError(
            f"processor EOS {processor.tokenizer.eos_token_id} != model EOS "
            f"{model.tokenizer.eos_token_id}"
        )
    device = str(model.base_model.device)
    print(
        f"offset_check dtype={args.dtype} kv={kv} draft={draft} benchmark={args.benchmark} "
        f"prompt_style={args.prompt_style} samples={len(samples)} device={device}",
        flush=True,
    )
    rows = []
    for number, sample in enumerate(samples, 1):
        inputs = build_inputs(
            processor, args.benchmark, sample, args.prompt_style, device=device
        )
        row = {"question_id": sample.question_id}
        row.update(check_sample(model, inputs, args.max_new_token, kv, draft))
        rows.append(row)
        print(
            f"[{number}/{len(samples)}] {sample.question_id}: exact={row['exact']} "
            f"first_mismatch={row['first_mismatch_index']} kv_offset={row['kv_offset']} "
            f"ar_tokens={row['ar_tokens']} vispec_tokens={row['vispec_tokens']} "
            f"kv_counts_match={row['kv_counts_match']}",
            flush=True,
        )
    offsets = [row["kv_offset"] for row in rows if row["kv_offset"] is not None]
    summary = {
        "questions": len(rows),
        "exact": sum(row["exact"] for row in rows),
        "all_exact": all(row["exact"] for row in rows),
        "first_mismatches": [row["first_mismatch_index"] for row in rows],
        "kv_counts_match": all(row["kv_counts_match"] for row in rows),
        "kv_offset_min": min(offsets) if offsets else None,
        "kv_offset_max": max(offsets) if offsets else None,
    }
    result = {
        "schema_version": 1,
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "command": [sys.executable, "-m", "vispec.evaluation.offset_check", *sys.argv[1:]],
        "settings": {
            "dtype": args.dtype,
            "kv_layout": "compact",
            "kv_sparsity": {"retention": kv.retention, "policy": kv.policy},
            "draft_sparsity": (
                None if draft is None else {"retention": draft.retention, "policy": draft.policy}
            ),
            "benchmark": args.benchmark,
            "prompt_style": args.prompt_style,
            "num_samples": args.num_samples,
            "max_new_tokens": args.max_new_token,
            "temperature": 0.0,
            "depth": args.depth,
            "top_k": args.top_k,
            "total_token": args.total_token,
            "num_q": args.num_q,
            "kv_capacity": args.kv_capacity,
            "slim_prefill": True,
            "base_model_path": str(Path(args.base_model_path).resolve()),
            "spec_model_path": str(Path(args.spec_model_path).resolve()),
            "data_dir": str(args.data_dir.resolve()),
        },
        "environment": environment(),
        "definitions": {
            "exact": "AR and ViSpec raw output token IDs are equal, through EOS",
            "first_mismatch_index": "zero-based output index of the first differing token; "
            "the shorter length when one output is a prefix of the other",
            "kv_offset": "prompt tokens evicted from the target KV cache (token count minus "
            "physical cache length during decoding)",
            "kv_counts_match": "AR and ViSpec report the same eviction counts for the prompt",
        },
        "summary": summary,
        "questions": rows,
    }
    (args.output_dir / "offset_check.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    write_markdown(args.output_dir / "offset_check.md", result)
    print(
        f"AR == ViSpec on {summary['exact']}/{summary['questions']}; first mismatches "
        f"{summary['first_mismatches']}; kv_offset {summary['kv_offset_min']}.."
        f"{summary['kv_offset_max']}; wrote {args.output_dir / 'offset_check.md'}",
        flush=True,
    )
    return 0 if summary["all_exact"] and summary["kv_counts_match"] else 1


def main(argv=None):
    return run(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
