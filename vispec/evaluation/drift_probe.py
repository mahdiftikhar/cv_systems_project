"""Probe drift between the KV-cache fork's cached greedy decode and the stock HF model.

Week 2's draft-only diagnosis failed on one case: on TextVQA question 35537 at output index
162 the fork's greedy decode emitted "." (token 13), while the stock
``transformers.Qwen2_5_VLForConditionalGeneration``, teacher-forced with a full recompute on the
same prefix, prefers "**:" (token 95518). This probe asks whether that is generic bf16 drift
between cached (incremental) decoding and full recompute, or a deviation of the repository's
KV-cache fork.

Part A compares, for every question, the fork's saved greedy AR tokens (``--answers``) with stock
HF greedy ``generate`` (bf16, the stock attention implementation, the fork's stop rule). For each
disagreement it records the first differing index and the stock teacher-forced (full recompute,
no KV cache) top-2 logits there, in bf16 and in fp32 with TF32 off.

Part B records, at the focus question and index, the logits of the fork's token and an
alternative token under:

- (i) fork incremental AR in bf16 (a re-run of ``autoregressive_generate`` with the reference
  run's settings; a forward hook on the base model captures the step's logits);
- (ii) stock incremental ``generate`` with ``output_scores``, bf16;
- (ii-tf) stock incremental decode in bf16, teacher-forced on the fork's prefix (a manual
  ``DynamicCache`` loop that feeds the same inputs ``generate`` feeds). It stands in for (ii)
  when ``generate``'s own tokens leave the fork's prefix before the focus index;
- (iii) stock full recompute, bf16;
- (iv) stock full recompute, fp32 (TF32 off).

The two stop conditions (Week 3 plan, Task 1 Step 6) are evaluated and written out:

- (a) in Part B, (ii) prefers the alternative token while (i) prefers the fork token;
- (b) in Part A, a first mismatch whose stock bf16 teacher-forced margin is above 0.5.

Usage:
python -m vispec.evaluation.drift_probe --answers RUN02/results/textvqa_test/autoregressive_qwen/test.jsonl \
    --data-dir bench_data --base-model-path models/Qwen2.5-VL-3B-Instruct --num-samples 100 \
    --focus 35537:162 --focus-tokens 13 95518 --output-dir OUT
"""

import argparse
import gc
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from . import evaluate as fork_eval
from .benchmarks import build_inputs, load_processor, load_samples
from .diagnose_divergence import (
    DEVICE,
    cell,
    environment,
    fingerprint,
    fmt,
    git_commit,
    inspect as top2,
    last_logits,
    load_model as load_stock_model,
    qid_key,
    turn_measurement,
)
from .scoring import evidence

# Stop condition (b): the 0.5-logit near-tie threshold of the Week 2 diagnosis, unchanged.
MARGIN_THRESHOLD = 0.5
SPARSITY_KEYS = ("kv_retention", "kv_policy", "draft_retention", "draft_policy")


def parse_focus(text):
    try:
        qid, index = text.rsplit(":", 1)
        index = int(index)
    except ValueError:
        raise argparse.ArgumentTypeError(f"--focus must be QUESTION_ID:INDEX, got {text!r}")
    if not qid or index < 0:
        raise argparse.ArgumentTypeError(f"--focus must be QUESTION_ID:INDEX, got {text!r}")
    return qid, index


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--answers",
        required=True,
        type=Path,
        help="The fork's AR answers (an evaluate.py test.jsonl with its .runtime.json beside it).",
    )
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--base-model-path", required=True)
    parser.add_argument(
        "--spec-model-path",
        default=None,
        help="Draft checkpoint the fork's SpecModel loads (unused by AR decoding). Default: "
        "the spec_model_path recorded in the answers' runtime.json.",
    )
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument(
        "--question-ids",
        nargs="+",
        default=None,
        help="Restrict Part A to these questions (default: every question of the subset).",
    )
    parser.add_argument(
        "--focus",
        required=True,
        type=parse_focus,
        help="QUESTION_ID:INDEX for Part B, INDEX being a zero-based output position.",
    )
    parser.add_argument(
        "--focus-tokens",
        nargs=2,
        type=int,
        default=None,
        metavar=("FORK_TOKEN", "ALT_TOKEN"),
        help="The fork's token at the focus index and the alternative token. Default: the "
        "saved fork token and the stock bf16 full-recompute choice among the other tokens.",
    )
    parser.add_argument(
        "--prompt-style",
        default=None,
        help="Expected prompt style; must equal the answers' runtime.json prompt_style.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    # Fail before anything loads: an absent or empty answer file must never pass silently.
    if not args.answers.is_file():
        parser.error(f"--answers {args.answers} does not exist or is not a file")
    if args.answers.stat().st_size == 0:
        parser.error(f"--answers {args.answers} is empty")
    if args.num_samples < 1:
        parser.error("--num-samples must be positive")
    if args.output_dir.exists():
        parser.error(f"--output-dir {args.output_dir} already exists; use a new folder")
    return args


def load_answers(path):
    records = {}
    for line in path.read_text().splitlines():
        if line.strip():
            record = json.loads(line)
            records[str(record["question_id"])] = record
    if not records:
        raise ValueError(f"--answers {path} holds no answer records")
    return records


def load_reference_runtime(answers_path):
    """The answers' runtime.json; refuses non-AR runs and any sparsity setting."""
    path = answers_path.with_suffix(".runtime.json")
    if not path.is_file():
        raise ValueError(f"Missing {path}; --answers must be an evaluate.py answer file")
    runtime = json.loads(path.read_text())
    gen = runtime.get("generation_args") or {}
    if gen.get("method") != "autoregressive":
        raise ValueError(f"{answers_path} is not an autoregressive run (method {gen.get('method')!r})")
    # Absent and null are the same setting (ruling R9); the probe needs the plain reference.
    active = {k: gen.get(k) for k in SPARSITY_KEYS if gen.get(k) is not None}
    if active:
        raise ValueError(f"{answers_path} ran with sparsity settings {active}; use a reference run")
    return path, runtime, gen


def fork_arguments(gen, base_model_path, spec_model_path, data_dir):
    """evaluate.py arguments that reproduce the reference run's AR decode (re-validated)."""
    argv = [
        "--method", "autoregressive",
        "--benchmark", gen["benchmark"],
        "--prompt-style", gen["prompt_style"],
        "--base-model-path", base_model_path,
        "--spec-model-path", spec_model_path,
        "--data-dir", str(data_dir),
        "--answer-file", "(not written by drift_probe)",
        "--model-id", "drift_probe",
        "--num-samples", str(gen["num_samples"]),
        "--max-new-token", str(gen["max_new_token"]),
        "--temperature", str(gen["temperature"]),
        "--depth", str(gen["depth"]),
        "--top-k", str(gen["top_k"]),
        "--total-token", str(gen["total_token"]),
        "--num-q", str(gen["num_q"]),
        "--kv-capacity", str(gen["kv_capacity"]),
        "--kv-layout", gen.get("kv_layout") or "fixed",
    ]
    if gen.get("no_slim_prefill"):
        argv.append("--no-slim-prefill")
    return fork_eval.parse_args(argv)


def release_memory():
    """Return freed CUDA memory; call after ``del`` of the last references."""
    gc.collect()
    torch.cuda.empty_cache()


def relabel(entry, first, second):
    """diagnose_divergence.inspect output with its ar/vispec labels renamed."""
    names = {"ar": first, "vispec": second, "neither": "neither"}
    entry = dict(entry)
    entry["agrees"] = names[entry["agrees"]]
    entry[f"{first}_token_logit"] = entry.pop("ar_token_logit")
    entry[f"{second}_token_logit"] = entry.pop("vispec_token_logit")
    return entry


def pair_view(logits, fork_token, alt_token, tokenizer):
    """Top-2 and both focus tokens' logits; ``preferred`` uses the decoders' tie-break."""
    entry = relabel(
        top2(logits, {"ar_token_id": fork_token, "vispec_token_id": alt_token}, tokenizer),
        "fork",
        "alt",
    )
    fork_logit, alt_logit = logits[fork_token].item(), logits[alt_token].item()
    # torch.argmax keeps the first maximal index, so an exact tie goes to the lower ID.
    fork_wins = fork_logit > alt_logit or (fork_logit == alt_logit and fork_token < alt_token)
    entry.update(
        preferred_token_id=fork_token if fork_wins else alt_token,
        alt_minus_fork_logit=alt_logit - fork_logit,
        top5=[
            {"token_id": int(i), "text": tokenizer.decode([int(i)]), "logit": logits[i].item()}
            for i in torch.topk(logits, 5).indices
        ],
    )
    return entry


def first_mismatch(a, b):
    for index, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return index
    return None if len(a) == len(b) else min(len(a), len(b))


def stop_reason(tokens, eos, max_new_tokens):
    if tokens and tokens[-1] == eos:
        return "eos"
    return "token_limit" if len(tokens) >= max_new_tokens else "other"


@torch.inference_mode()
def fork_focus(fork_args, sample, saved_ids, saved_sha, focus_index):
    """(i): re-run the fork's AR decode; capture the logits that produce output ``focus_index``.

    Forward call k of the base model produces output token k: call 0 is the prefill, and each
    later call feeds the previous token. A hook counts calls; autoregressive.py is not edited.
    """
    processor, model, tokenizer = fork_eval.load_model(fork_args)
    device = str(model.base_model.device)
    inputs = build_inputs(
        processor, fork_args.benchmark, sample, fork_args.prompt_style, device=device
    )
    rebuilt = fingerprint(inputs["input_ids"])
    if rebuilt != saved_sha:
        raise ValueError(f"Fork prompt fingerprint {rebuilt} != recorded {saved_sha}")
    input_len = inputs["input_ids"].shape[1]
    calls, captured = [0], {}

    def hook(module, hook_args, output):
        logits = output.logits if hasattr(output, "logits") else output[0]
        if calls[0] == focus_index:
            captured["logits"] = logits[0, -1].float().cpu()
        calls[0] += 1

    handle = model.base_model.register_forward_hook(hook)
    try:
        # Seeded as evaluate.py seeds each answer; greedy decoding draws no random numbers.
        torch.manual_seed(0)
        generation_stats = {}
        started = time.perf_counter()
        output_ids, _, _, _, _ = fork_eval.generate(
            model, fork_args, inputs, generation_stats=generation_stats
        )
        seconds = time.perf_counter() - started
    finally:
        handle.remove()
    tokens = output_ids[0][input_len:].tolist()
    info = {
        "attn_implementation": getattr(model.base_model.config, "_attn_implementation", None),
        "kv_layout": model.kv_layout,
        "slim_prefill": model.slim_prefill,
        "kv_capacity": model.kv_capacity,
        "prompt_tokens": input_len,
        "rerun_output_tokens": len(tokens),
        "rerun_stop_reason": generation_stats.get("stop_reason"),
        "rerun_seconds": seconds,
        "base_model_forward_calls": calls[0],
        "forward_calls_equal_output_tokens": calls[0] == len(tokens),
        "saved_output_tokens": len(saved_ids),
        "rerun_reproduces_saved_through_focus": tokens[: focus_index + 1]
        == saved_ids[: focus_index + 1],
        "rerun_reproduces_saved_entirely": tokens == saved_ids,
        "rerun_first_difference_from_saved": first_mismatch(tokens, saved_ids),
    }
    logits = captured.get("logits")
    if logits is not None:
        info["captured_argmax_equals_rerun_token"] = int(logits.argmax()) == tokens[focus_index]
    del model, inputs, output_ids
    release_memory()
    return logits, info, tokenizer


@torch.inference_mode()
def stock_generate(model, inputs, gen_kwargs, capture_index=None):
    """Stock greedy generate; returns (tokens, focus scores, focus raw logits)."""
    extra = {}
    if capture_index is not None:
        extra = {"output_scores": True, "output_logits": True}
    out = model.generate(**inputs, **gen_kwargs, return_dict_in_generate=True, **extra)
    tokens = out.sequences[0, inputs["input_ids"].shape[1] :].tolist()
    scores = raw = None
    if capture_index is not None and capture_index < len(out.scores):
        # Only the focus step is kept; the rest of the 1,024-step tuples is dropped here.
        scores = out.scores[capture_index][0].float().cpu()
        raw = out.logits[capture_index][0].float().cpu()
    del out
    return tokens, scores, raw


@torch.inference_mode()
def stock_cached_logits(model, inputs, prefix_ids):
    """(ii-tf): stock incremental decode teacher-forced on ``prefix_ids``.

    Feeds what ``generate`` feeds: a prefill with ``cache_position`` 0..L-1 into a
    ``DynamicCache``, then one token per step with the full 2D attention mask and its cache
    position, so positions come from the cached ``rope_deltas``. Returns the float logits that
    predict output index ``len(prefix_ids)``.
    """
    from transformers import DynamicCache

    input_ids = inputs["input_ids"]
    device = input_ids.device
    length = input_ids.shape[1]
    mask = inputs["attention_mask"]
    extras = {k: v for k, v in inputs.items() if k not in ("input_ids", "attention_mask")}
    cache = DynamicCache()
    outputs = model(
        input_ids=input_ids,
        attention_mask=mask,
        past_key_values=cache,
        use_cache=True,
        cache_position=torch.arange(length, device=device),
        **extras,
    )
    logits = outputs.logits[0, -1].float()
    for step, token in enumerate(prefix_ids):
        mask = torch.cat([mask, mask.new_ones((1, 1))], dim=1)
        outputs = model(
            input_ids=torch.tensor([[token]], dtype=input_ids.dtype, device=device),
            attention_mask=mask,
            past_key_values=cache,
            use_cache=True,
            cache_position=torch.tensor([length + step], device=device),
            image_grid_thw=inputs.get("image_grid_thw"),
        )
        logits = outputs.logits[0, -1].float()
    del cache, outputs
    return logits.cpu()


def max_abs_diff(a, b):
    return None if a is None or b is None else (a - b).abs().max().item()


def stats(values):
    values = [v for v in values if v is not None]
    if not values:
        return {"count": 0, "min": None, "median": None, "max": None}
    return {
        "count": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
    }


def token_ref(token_id, text):
    return "(ended)" if token_id is None else f"{cell(text)} ({token_id})"


def top_ref(entry):
    return f"{cell(entry['text'])} ({entry['token_id']}, {fmt(entry['logit'], 3)})"


def write_markdown(path, result):
    a, b = result["part_a"], result["part_b"]
    cond = result["stop_conditions"]
    env = result["environment"]
    fork_tok, alt_tok = b["fork_token_id"], b["alt_token_id"]
    lines = [
        "# Cached-decode drift probe",
        "",
        f"Answers: `{result['inputs']['answers']['path']}` (sha256 "
        f"`{result['inputs']['answers']['sha256'][:16]}...`). Benchmark `{result['inputs']['benchmark']}`, "
        f"prompt style `{result['inputs']['prompt_style']}`, {a['questions']} questions.",
        "",
        f"Hardware: {env['gpu']['name']} (`{env['gpu']['uuid']}`) on `{env['hostname']}`, "
        f"Slurm job `{env['slurm_job_id']}`, torch {env['torch']}, transformers "
        f"{env['transformers']}. Code commit `{result['git_commit']}`.",
        "",
        "## Stop conditions",
        "",
        f"- **(a) {'HOLDS' if cond['a']['holds'] else 'does not hold'}.** {cond['a']['rule']} "
        f"(i) prefers {cond['a']['fork_preferred_token_id']}; "
        f"({cond['a']['stock_incremental_source']}) prefers {cond['a']['stock_incremental_preferred_token_id']}.",
        f"- **(b) {'HOLDS' if cond['b']['holds'] else 'does not hold'}.** {cond['b']['rule']} "
        f"{len(cond['b']['questions'])} first mismatch(es) above {MARGIN_THRESHOLD}"
        + (f": {', '.join(cond['b']['questions'])}." if cond["b"]["questions"] else "."),
        "",
        f"**Verdict: {result['verdict']}**",
        "",
        "## Part A: fork AR tokens against stock greedy `generate`",
        "",
        f"Exact agreement: **{a['exact_agreement']}/{a['questions']}**. A mismatch row gives the "
        "first differing zero-based output index, both tokens, and the stock teacher-forced "
        "(full recompute, no KV cache) logits on the shared prefix. `agrees` names the decoder "
        "whose token is the teacher-forced argmax.",
        "",
    ]
    rows = [q for q in a["per_question"] if q["first_mismatch_index"] is not None]
    if rows:
        lines += [
            "| Question | Fork / stock tokens | First mismatch | Fork token | Stock token | bf16 top-1 | bf16 top-2 | bf16 margin | bf16 agrees | fp32 margin | fp32 agrees |",
            "| --- | --- | ---: | --- | --- | --- | --- | ---: | --- | ---: | --- |",
        ]
        for q in rows:
            bf, fp = q["teacher_forced_bf16"], q["teacher_forced_fp32"]
            lines.append(
                f"| {q['question_id']} | {q['fork_output_tokens']} / {q['stock_output_tokens']} | "
                f"{q['first_mismatch_index']} | {token_ref(q['fork_token_id'], q['fork_token_text'])} | "
                f"{token_ref(q['stock_token_id'], q['stock_token_text'])} | {top_ref(bf['top2'][0])} | "
                f"{top_ref(bf['top2'][1])} | {fmt(bf['margin'])} | {bf['agrees']} | "
                f"{fmt(fp['margin']) if fp else 'n/a'} | {fp['agrees'] if fp else 'n/a'} |"
            )
        m = a["bf16_margin"]
        lines += [
            "",
            f"bf16 teacher-forced margin over {m['count']} mismatches: min {fmt(m['min'])}, "
            f"median {fmt(m['median'])}, max {fmt(m['max'])} logits.",
        ]
    else:
        lines.append("No mismatches.")
    fr = b["fork_rerun"]
    lines += [
        "",
        f"## Part B: question {b['question_id']}, output index {b['index']}",
        "",
        f"Prefix: {b['index']} fork tokens. Fork re-run reproduces the saved tokens through the "
        f"focus index: **{fr['rerun_reproduces_saved_through_focus']}** (entirely: "
        f"{fr['rerun_reproduces_saved_entirely']}). Stock `generate` follows the fork's prefix up "
        f"to the focus index: **{b['stock_generate_prefix_matches_fork']}**.",
        "",
        f"| Condition | Logit of {fork_tok} {cell(b['fork_token_text'])} | Logit of {alt_tok} "
        f"{cell(b['alt_token_text'])} | {alt_tok} minus {fork_tok} | Preferred of the pair | Argmax | Top-2 margin |",
        "| --- | ---: | ---: | ---: | --- | --- | ---: |",
    ]
    for key, label in b["condition_labels"].items():
        entry = b["conditions"].get(key)
        if entry is None:
            lines.append(f"| {label} | n/a | n/a | n/a | n/a | n/a | n/a |")
            continue
        lines.append(
            f"| {label} | {fmt(entry['fork_token_logit'])} | {fmt(entry['alt_token_logit'])} | "
            f"{fmt(entry['alt_minus_fork_logit'])} | {entry['preferred_token_id']} | "
            f"{entry['argmax_token_id']} | {fmt(entry['margin'])} |"
        )
    lines += [
        "",
        "Largest absolute logit difference over the vocabulary between two conditions:",
        "",
        "| Pair | Max abs logit difference |",
        "| --- | ---: |",
    ]
    for pair, value in b["pairwise_max_abs_logit_diff"].items():
        lines.append(f"| {pair} | {fmt(value)} |")
    lines.append("")
    path.write_text("\n".join(lines))


def main(argv=None):
    args = parse_args(argv)
    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    focus_qid, focus_index = args.focus
    answers_path = args.answers.resolve()
    answers = load_answers(answers_path)
    runtime_path, runtime, gen = load_reference_runtime(answers_path)
    benchmark, prompt_style = gen["benchmark"], gen["prompt_style"]
    if args.prompt_style is not None and args.prompt_style != prompt_style:
        raise ValueError(
            f"--prompt-style {args.prompt_style} but the answers ran {prompt_style}"
        )
    max_new_tokens = int(gen["max_new_token"])
    spec_model_path = args.spec_model_path or gen.get("spec_model_path")
    if not spec_model_path:
        raise ValueError("No --spec-model-path given and none recorded in runtime.json")

    samples = {s.question_id: s for s in load_samples(benchmark, args.data_dir, args.num_samples)}
    if set(samples) != set(answers):
        raise ValueError(
            f"The {len(samples)}-question subset and the {len(answers)} answers differ: "
            f"{len(set(samples) - set(answers))} subset questions unanswered, "
            f"{len(set(answers) - set(samples))} answers outside the subset"
        )
    scope = sorted(set(args.question_ids or samples), key=qid_key)
    unknown = [q for q in scope + [focus_qid] if q not in samples]
    if unknown:
        raise ValueError(f"Questions not in the subset: {unknown}")
    saved = {}
    for qid in set(scope) | {focus_qid}:
        m = turn_measurement(answers[qid], 0, 0)
        saved[qid] = (m["output_token_ids"], m.get("input_ids_sha256"), m.get("stop_reason"))
    focus_ids = saved[focus_qid][0]
    if focus_index >= len(focus_ids):
        raise ValueError(
            f"--focus index {focus_index} is past question {focus_qid}'s "
            f"{len(focus_ids)} saved tokens"
        )
    if args.focus_tokens and args.focus_tokens[0] != focus_ids[focus_index]:
        raise ValueError(
            f"--focus-tokens FORK_TOKEN {args.focus_tokens[0]} != saved fork token "
            f"{focus_ids[focus_index]} at index {focus_index}"
        )
    if args.focus_tokens and args.focus_tokens[0] == args.focus_tokens[1]:
        raise ValueError("--focus-tokens needs two different tokens")
    fork_args = fork_arguments(gen, args.base_model_path, spec_model_path, args.data_dir)

    # Every prompt must rebuild to its recorded fingerprint before any model loads.
    processor = load_processor(args.base_model_path)
    tokenizer = processor.tokenizer
    mismatched = []
    for qid in sorted(set(scope) | {focus_qid}, key=qid_key):
        inputs = build_inputs(processor, benchmark, samples[qid], prompt_style, device="cpu")
        if fingerprint(inputs["input_ids"]) != saved[qid][1]:
            mismatched.append(qid)
    if mismatched:
        raise ValueError(f"Prompt fingerprint mismatch for questions {mismatched}; no probe was run")
    print(f"Prompt fingerprints match for {len(set(scope) | {focus_qid})} questions", flush=True)
    args.output_dir.mkdir(parents=True)
    eos = tokenizer.eos_token_id
    timings = {}

    # (i) Fork incremental AR, bf16.
    t0 = time.perf_counter()
    fork_logits, fork_rerun, fork_tokenizer = fork_focus(
        fork_args, samples[focus_qid], focus_ids, saved[focus_qid][1], focus_index
    )
    if fork_tokenizer.eos_token_id != eos:
        raise RuntimeError(f"Fork EOS {fork_tokenizer.eos_token_id} != processor EOS {eos}")
    timings["fork_phase_seconds"] = time.perf_counter() - t0
    print(f"(i) fork re-run: {json.dumps(fork_rerun)}", flush=True)

    # Stock bf16: Part A, then (ii), (ii-tf) and (iii) for the focus question.
    t0 = time.perf_counter()
    model = load_stock_model(args.base_model_path, torch.bfloat16)
    stock_attn = {
        "text": model.config._attn_implementation,
        "vision": getattr(model.config.vision_config, "_attn_implementation", None),
    }
    if stock_attn["text"] != "sdpa":
        raise RuntimeError(f"Stock model loaded with attention {stock_attn}, not sdpa")
    # The model's generation_config samples with repetition_penalty 1.05 and stops on two
    # EOS IDs; greedy decoding with the fork's single stop token makes the tokens comparable.
    gen_kwargs = {
        "do_sample": False,
        "max_new_tokens": max_new_tokens,
        "temperature": None,
        "repetition_penalty": 1.0,
        "eos_token_id": eos,
        "pad_token_id": tokenizer.pad_token_id,
    }
    model_generation_config = model.generation_config.to_dict()

    # Part B's stock bf16 conditions run before Part A, so a failure shows within minutes.
    inputs = build_inputs(processor, benchmark, samples[focus_qid], prompt_style, DEVICE)
    start = time.perf_counter()
    focus_stock_tokens, focus_scores, focus_raw = stock_generate(
        model, inputs, gen_kwargs, focus_index
    )
    focus_seconds = time.perf_counter() - start
    prefix = focus_ids[:focus_index]
    stock_prefix_matches = focus_stock_tokens[:focus_index] == prefix
    tf_logits = stock_cached_logits(model, inputs, prefix)
    full_bf16 = last_logits(model, inputs, prefix).cpu()
    del inputs
    print(
        f"(ii)/(ii-tf)/(iii) done: stock generate follows the fork prefix {stock_prefix_matches}",
        flush=True,
    )

    per_question = []
    for number, qid in enumerate(scope, 1):
        inputs = build_inputs(processor, benchmark, samples[qid], prompt_style, DEVICE)
        fork_ids, _, fork_stop = saved[qid]
        if qid == focus_qid:
            # Same call; output_scores does not change the tokens.
            stock_ids, seconds = focus_stock_tokens, focus_seconds
        else:
            start = time.perf_counter()
            stock_ids, _, _ = stock_generate(model, inputs, gen_kwargs)
            seconds = time.perf_counter() - start
        index = first_mismatch(fork_ids, stock_ids)
        row = {
            "question_id": qid,
            "prompt_tokens": inputs["input_ids"].shape[1],
            "fork_output_tokens": len(fork_ids),
            "stock_output_tokens": len(stock_ids),
            "fork_stop_reason": fork_stop,
            "stock_stop_reason": stop_reason(stock_ids, eos, max_new_tokens),
            "exact_agreement": index is None,
            "first_mismatch_index": index,
            "fork_token_id": None,
            "stock_token_id": None,
            "fork_token_text": None,
            "stock_token_text": None,
            "teacher_forced_bf16": None,
            "teacher_forced_fp32": None,
            "stock_generate_seconds": seconds,
        }
        if index is not None:
            fork_tok = fork_ids[index] if index < len(fork_ids) else None
            stock_tok = stock_ids[index] if index < len(stock_ids) else None
            logits = last_logits(model, inputs, fork_ids[:index]).cpu()
            row.update(
                fork_token_id=fork_tok,
                stock_token_id=stock_tok,
                fork_token_text=None if fork_tok is None else tokenizer.decode([fork_tok]),
                stock_token_text=None if stock_tok is None else tokenizer.decode([stock_tok]),
                teacher_forced_bf16=relabel(
                    top2(logits, {"ar_token_id": fork_tok, "vispec_token_id": stock_tok}, tokenizer),
                    "fork",
                    "stock",
                ),
            )
        per_question.append(row)
        print(
            f"[A {number}/{len(scope)}] {qid}: fork {len(fork_ids)} / stock {len(stock_ids)} "
            f"tokens, first mismatch {index}"
            + (
                f", bf16 margin {row['teacher_forced_bf16']['margin']:.4f}"
                if index is not None
                else ""
            )
            + f" ({seconds:.1f} s)",
            flush=True,
        )
        del inputs
    timings["part_a_stock_generate_seconds"] = sum(q["stock_generate_seconds"] for q in per_question)
    del model
    release_memory()
    timings["stock_bf16_phase_seconds"] = time.perf_counter() - t0

    # Stock fp32 with TF32 off (as diagnose_divergence's fp32 pass): (iv) and Part A mismatches.
    t0 = time.perf_counter()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    model = load_stock_model(args.base_model_path, torch.float32)
    for row in per_question:
        index = row["first_mismatch_index"]
        if index is None:
            continue
        inputs = build_inputs(processor, benchmark, samples[row["question_id"]], prompt_style, DEVICE)
        logits = last_logits(model, inputs, saved[row["question_id"]][0][:index]).cpu()
        row["teacher_forced_fp32"] = relabel(
            top2(
                logits,
                {"ar_token_id": row["fork_token_id"], "vispec_token_id": row["stock_token_id"]},
                tokenizer,
            ),
            "fork",
            "stock",
        )
        del inputs
    inputs = build_inputs(processor, benchmark, samples[focus_qid], prompt_style, DEVICE)
    full_fp32 = last_logits(model, inputs, prefix).cpu()
    del model, inputs
    release_memory()
    timings["fp32_phase_seconds"] = time.perf_counter() - t0

    # Part B. The alternative token defaults to the stock bf16 full-recompute choice.
    fork_tok = focus_ids[focus_index]
    if args.focus_tokens:
        alt_tok = args.focus_tokens[1]
    else:
        ranked = torch.topk(full_bf16, 2).indices.tolist()
        alt_tok = ranked[0] if ranked[0] != fork_tok else ranked[1]
    vectors = {
        "i": fork_logits,
        "ii": focus_scores if stock_prefix_matches else None,
        "ii_tf": tf_logits,
        "iii": full_bf16,
        "iv": full_fp32,
    }
    labels = {
        "i": "(i) fork incremental AR, bf16",
        "ii": "(ii) stock incremental generate, bf16",
        "ii_tf": "(ii-tf) stock incremental, teacher-forced on the fork prefix, bf16",
        "iii": "(iii) stock full recompute, bf16",
        "iv": "(iv) stock full recompute, fp32",
    }
    conditions = {
        key: None if v is None else pair_view(v, fork_tok, alt_tok, tokenizer)
        for key, v in vectors.items()
    }
    pairs = [("i", "ii"), ("i", "ii_tf"), ("ii", "ii_tf"), ("i", "iii"), ("ii_tf", "iii"), ("iii", "iv"), ("i", "iv")]
    pairwise = {f"{a} vs {b}": max_abs_diff(vectors[a], vectors[b]) for a, b in pairs}

    # Stop condition (a): (ii) when generate followed the fork's prefix, else (ii-tf).
    source = "ii" if conditions["ii"] is not None else "ii_tf"
    fork_pref = conditions["i"]["preferred_token_id"] if conditions["i"] else None
    stock_pref = conditions[source]["preferred_token_id"]
    cond_a = {
        "rule": f"(ii) stock incremental prefers {alt_tok} while (i) the fork prefers {fork_tok}.",
        "stock_incremental_source": source,
        "fork_preferred_token_id": fork_pref,
        "stock_incremental_preferred_token_id": stock_pref,
        "holds": fork_pref == fork_tok and stock_pref == alt_tok,
    }
    mismatches = [q for q in per_question if q["first_mismatch_index"] is not None]
    above = [q["question_id"] for q in mismatches if q["teacher_forced_bf16"]["margin"] > MARGIN_THRESHOLD]
    cond_b = {
        "rule": (
            "Fork and stock greedy disagree at a first mismatch whose stock bf16 "
            f"teacher-forced margin is above {MARGIN_THRESHOLD}."
        ),
        "threshold": MARGIN_THRESHOLD,
        "questions": above,
        "holds": bool(above),
    }
    stop = cond_a["holds"] or cond_b["holds"]
    verdict = (
        "STOP: a stop condition holds, which points to a deviation specific to the fork"
        if stop
        else "neither stop condition holds: generic bf16 drift in cached decoding"
    )

    result = {
        "schema_version": 1,
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": git_commit(),
        "command": [sys.executable, "-m", "vispec.evaluation.drift_probe", *(sys.argv[1:] if argv is None else argv)],
        "inputs": {
            "answers": evidence(answers_path),
            "answers_runtime": evidence(runtime_path),
            "reference_run_base_model": gen.get("base_model_path"),
            "reference_run_spec_model": gen.get("spec_model_path"),
            "base_model_path": str(Path(args.base_model_path).resolve()),
            "spec_model_path": str(Path(spec_model_path).resolve()),
            "data_dir": str(args.data_dir.resolve()),
            "benchmark": benchmark,
            "prompt_style": prompt_style,
            "num_samples": args.num_samples,
            "question_ids": args.question_ids,
            "focus": {"question_id": focus_qid, "index": focus_index},
        },
        "settings": {
            "max_new_tokens": max_new_tokens,
            "margin_threshold": MARGIN_THRESHOLD,
            "fork_generation_args": vars(fork_args),
            "reference_attn_implementation": runtime.get("attn_implementation"),
            "stock_attn_implementation": stock_attn,
            "stock_generate_kwargs": gen_kwargs,
            "stock_model_generation_config": model_generation_config,
            "fp32": "float32 weights, TF32 off for matmul and cuDNN, matmul precision highest",
        },
        "environment": environment(args.base_model_path),
        "timings": timings,
        "definitions": {
            "first_mismatch_index": "zero-based output position of the first differing token; the shorter length when one sequence is a prefix of the other",
            "margin": "top-1 minus top-2 next-token logit, raw logit units",
            "teacher_forced": "stock model, one forward pass over prompt + shared prefix with use_cache=False (full recompute)",
            "agrees": "which decoder's token is the teacher-forced argmax: fork, stock or neither",
            "preferred_token_id": "of the two focus tokens, the one with the higher logit (a tie goes to the lower ID, as torch.argmax does)",
            "condition_a": "evaluated on (ii) when stock generate's own tokens equal the fork's before the focus index, otherwise on (ii-tf)",
        },
        "part_a": {
            "questions": len(per_question),
            "exact_agreement": sum(q["exact_agreement"] for q in per_question),
            "mismatches": len(mismatches),
            "bf16_margin": stats([q["teacher_forced_bf16"]["margin"] for q in mismatches]),
            "fp32_margin": stats(
                [q["teacher_forced_fp32"]["margin"] for q in mismatches if q["teacher_forced_fp32"]]
            ),
            "bf16_agrees": {
                k: sum(q["teacher_forced_bf16"]["agrees"] == k for q in mismatches)
                for k in ("fork", "stock", "neither")
            },
            "fp32_agrees": {
                k: sum(q["teacher_forced_fp32"]["agrees"] == k for q in mismatches)
                for k in ("fork", "stock", "neither")
            },
            "per_question": per_question,
        },
        "part_b": {
            "question_id": focus_qid,
            "index": focus_index,
            "prefix_length": len(prefix),
            "fork_token_id": fork_tok,
            "fork_token_text": tokenizer.decode([fork_tok]),
            "alt_token_id": alt_tok,
            "alt_token_text": tokenizer.decode([alt_tok]),
            "alt_token_source": "--focus-tokens" if args.focus_tokens else "stock bf16 full recompute",
            "fork_rerun": fork_rerun,
            "stock_generate_prefix_matches_fork": stock_prefix_matches,
            "stock_generate_first_difference_from_fork": first_mismatch(focus_stock_tokens, focus_ids),
            "stock_generate_scores_equal_raw_logits": None
            if focus_scores is None
            else bool(torch.equal(focus_scores, focus_raw)),
            "condition_labels": labels,
            "conditions": conditions,
            "pairwise_max_abs_logit_diff": pairwise,
        },
        "stop_conditions": {"a": cond_a, "b": cond_b, "stop": stop},
        "verdict": verdict,
    }
    (args.output_dir / "drift.json").write_text(
        json.dumps(result, indent=2, allow_nan=False, ensure_ascii=False) + "\n"
    )
    write_markdown(args.output_dir / "drift.md", result)
    print(
        f"Drift probe: {args.output_dir / 'drift.md'}; Part A {result['part_a']['exact_agreement']}/"
        f"{len(per_question)} exact; condition (a) {cond_a['holds']}, (b) {cond_b['holds']}; {verdict}",
        flush=True,
    )
    if not fork_rerun["rerun_reproduces_saved_through_focus"]:
        print(
            "ERROR: the fork re-run does not reproduce the saved tokens through the focus index, "
            "so (i) is not on the saved prefix; see drift.json part_b.fork_rerun",
            file=sys.stderr,
        )
        return 4
    return 0


if __name__ == "__main__":
    sys.exit(main())
