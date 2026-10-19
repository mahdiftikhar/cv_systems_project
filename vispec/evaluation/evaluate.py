"""Unified Qwen2.5-VL evaluator for standard (autoregressive) decoding and ViSpec.

One ``spec_model_ours.SpecModel`` is loaded for either method, so both share weights, the
KV-cache fork and the preallocated KV buffer. Answer records use the schema written by
``gen_spec_answer_textvqa.get_model_answers``, so ``report_metrics`` and ``compare_metrics``
read them unchanged.

Usage:
python -m vispec.evaluation.evaluate --method vispec --benchmark textvqa --prompt-style long \
    --base-model-path models/Qwen2.5-VL-3B-Instruct \
    --spec-model-path models/ViSpec-Qwen2.5-VL-3B-Instruct \
    --data-dir bench_data --model-id test --answer-file out/textvqa.jsonl
"""

import argparse
import json
import time
from pathlib import Path

import shortuuid
import torch
from tqdm import tqdm

from ..model.autoregressive import autoregressive_generate
from ..model.spec_model_ours import SpecModel
from .benchmarks import (
    BENCHMARK_NAMES,
    PROMPT_STYLES,
    build_inputs,
    load_processor,
    load_samples,
)
from .measurement import (
    GenerationMeasurement,
    kv_bytes_per_token,
    write_manifest,
    write_runtime,
)

METHODS = ("autoregressive", "vispec")
WARMUPS = 3


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen2.5-VL with autoregressive or ViSpec decoding."
    )
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--benchmark", choices=BENCHMARK_NAMES, required=True)
    parser.add_argument("--prompt-style", choices=PROMPT_STYLES, required=True)
    parser.add_argument("--base-model-path", required=True)
    parser.add_argument("--spec-model-path", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--answer-file", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument(
        "--max-new-token",
        type=int,
        default=1024,
        help="Maximum number of new tokens per answer.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--total-token", type=int, default=30)
    parser.add_argument("--num-q", type=int, default=2)
    parser.add_argument(
        "--kv-capacity",
        type=int,
        default=16384,
        help="Token slots of the preallocated target KV cache.",
    )
    parser.add_argument(
        "--no-slim-prefill",
        action="store_true",
        help="Keep full prefill logits and per-layer hidden states (slimming is on by default).",
    )
    args = parser.parse_args(argv)
    if args.method == "autoregressive" and args.temperature > 1e-5:
        parser.error("--method autoregressive is greedy only; use --temperature 0.0")
    if args.num_samples < 1:
        parser.error("--num-samples must be positive")
    return args


def answer_text(output_ids, tokenizer):
    """Output IDs up to the first EOS, decoded without special tokens and stripped."""
    eos = tokenizer.eos_token_id
    cut = output_ids[: output_ids.index(eos)] if eos in output_ids else output_ids
    return tokenizer.decode(cut, skip_special_tokens=True).strip()


def generate(model, args, model_inputs, on_first_token=None, generation_stats=None):
    """Run one generation; returns (ids [1, L], new_token, idx, acceptance, decode_time)."""
    if args.method == "vispec":
        return model.specgenerate(
            **model_inputs,
            temperature=args.temperature,
            max_new_tokens=args.max_new_token,
            log=True,
            return_acceptance_len=True,
            on_first_token=on_first_token,
            generation_stats=generation_stats,
            return_decode_time=True,
        )
    stats = {} if generation_stats is None else generation_stats
    output_ids, new_token, steps = autoregressive_generate(
        model,
        model_inputs,
        args.max_new_token,
        on_first_token=on_first_token,
        generation_stats=stats,
    )
    return output_ids, new_token, steps, [], stats["decode_time"]


def reorg_answer_file(answer_file):
    """Sort by question id and de-duplicate, as gen_spec_answer_textvqa does."""
    answers = {}
    with open(answer_file, "r") as fin:
        for line in fin:
            answers[json.loads(line)["question_id"]] = line
    with open(answer_file, "w") as fout:
        for qid in sorted(answers):
            fout.write(answers[qid])


@torch.inference_mode()
def run(args):
    answer_file = Path(args.answer_file)
    if answer_file.exists():
        raise FileExistsError(
            f"{answer_file} already exists; write each run to a fresh answer file"
        )
    answer_file.parent.mkdir(parents=True, exist_ok=True)

    samples = load_samples(args.benchmark, Path(args.data_dir), args.num_samples)
    write_manifest(
        answer_file,
        [
            {
                "qid": s.question_id,
                "question_id": s.question_id,
                "answers": list(s.references),
            }
            for s in samples
        ],
        args.benchmark,
    )
    processor = load_processor(args.base_model_path)

    model = SpecModel.from_pretrained(
        base_model_path=args.base_model_path,
        spec_model_path=args.spec_model_path,
        total_token=args.total_token,
        depth=args.depth,
        top_k=args.top_k,
        num_q=args.num_q,
        slim_prefill=not args.no_slim_prefill,
        kv_capacity=args.kv_capacity,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    model.eval()
    # Decode with the processor's tokenizer (the one that built the prompt); generation
    # stops on SpecModel's tokenizer EOS, so the two must agree.
    tokenizer = processor.tokenizer
    if tokenizer.eos_token_id != model.tokenizer.eos_token_id:
        raise RuntimeError(
            f"processor EOS {tokenizer.eos_token_id} != model EOS "
            f"{model.tokenizer.eos_token_id}"
        )

    write_runtime(answer_file, model)
    runtime_path = answer_file.with_suffix(".runtime.json")
    runtime = json.loads(runtime_path.read_text())
    runtime.update(
        prompt_style=args.prompt_style,
        slim_prefill=model.slim_prefill,
        kv_capacity=model.kv_capacity,
        method=args.method,
        benchmark=args.benchmark,
    )
    runtime_path.write_text(json.dumps(runtime, indent=2) + "\n")

    kv_bpt = kv_bytes_per_token(model.base_model.config, model.base_model.dtype)
    device = str(model.base_model.device)
    print(
        f"method={args.method} benchmark={args.benchmark} prompt_style={args.prompt_style} "
        f"slim_prefill={model.slim_prefill} kv_capacity={model.kv_capacity} "
        f"samples={len(samples)} device={device}"
    )

    warmup_inputs = build_inputs(
        processor, args.benchmark, samples[0], args.prompt_style, device=device
    )
    for _ in range(WARMUPS):
        torch.manual_seed(0)
        generate(model, args, warmup_inputs)
    print("Warmup done")

    for sample in tqdm(samples):
        # Image decoding and tokenization stay outside the timed region.
        model_inputs = build_inputs(
            processor, args.benchmark, sample, args.prompt_style, device=device
        )
        input_len = model_inputs["input_ids"].shape[1]
        torch.manual_seed(0)
        measurement = GenerationMeasurement(
            model_inputs, phase_memory=True, kv_bytes_per_token=kv_bpt
        )
        generation_stats = {}
        output_ids, new_token, idx, acceptance, decode_time = generate(
            model,
            args,
            model_inputs,
            on_first_token=measurement.mark,
            generation_stats=generation_stats,
        )
        wall_time = measurement.elapsed()
        kv_tokens = int(model.current_length_data[0])
        output_ids = output_ids[0][input_len:]
        record = measurement.finish(
            output_ids, tokenizer, new_token, generation_stats, kv_tokens=kv_tokens
        )
        choice = {
            "index": 0,
            "turns": [answer_text(record["output_token_ids"], tokenizer)],
            "idxs": [int(idx)],
            "new_tokens": [record["output_tokens"]],
            "wall_time": [wall_time],
            "ttft": [measurement.seconds],
            "measurement": [record],
            "decode_time": [decode_time],
            "acceptance_length": [int(a) for a in acceptance],
        }
        with open(answer_file, "a") as fout:
            fout.write(
                json.dumps(
                    {
                        "question_id": sample.question_id,
                        "reference_answers": list(sample.references),
                        "answer_id": shortuuid.uuid(),
                        "model_id": args.model_id,
                        "choices": [choice],
                        "tstamp": time.time(),
                    }
                )
                + "\n"
            )

    reorg_answer_file(answer_file)
    print(f"Wrote {len(samples)} answers to {answer_file}")


if __name__ == "__main__":
    run(parse_args())
