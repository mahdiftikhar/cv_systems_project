"""Parity check of our HiViS port against HiViS's own code, on the same 20 TextVQA prompts.

The reference runner (``experiment/10_qwen7b_methods/hivis_reference/run.sh``) calls the
subcommands in this order, all in the ``vipsec`` environment, around one run of
``hivis_side.py`` in the separate HiViS environment:

``samples --run-dir R --data-dir D``
    Writes ``R/samples.json``: the first 20 questions of our seed-42 TextVQA val subset
    (``load_samples("textvqa", D, 20)``, the first 20 of the 100 used in every run), each with
    its question, absolute image path and image sha256.

``ours --run-dir R --base-model-path B --spec-model-path S``
    Our side, on the tensors HiViS's side saved (``bundle/``, ``first_call/``):

    - ``SpecModel`` with ``draft_method="hivis"``, 60/5/10, fp16, slim prefill, the fixed
      layout with 16,384 KV slots.
    - Drafter isolation: the adapter's first call on HiViS's recorded full hidden states,
      embeds, prompt IDs and first token, with our target's head, in fp16 and again in fp32.
    - End to end: ``specgenerate`` and ``autoregressive_generate``, greedy, 500 new tokens. The
      pixel values are cast to bf16 first, as HiViS's ``eagenerate`` does; the vision tower then
      casts them to fp16. Both of our paths get the same pixel values.
    - Margins: where our HiViS tokens first differ from HiViS's, the stock model's fp16 top-2
      margin at that index (``diagnose_divergence.load_model`` and ``last_logits``), loaded
      after ``SpecModel`` is freed.

    Writes ``ours_first_call/``, ``ours_side.jsonl`` and ``ours_side_env.json``.

``compare --run-dir R``
    The verdict: ``parity.json`` and ``parity.md`` (in ``R`` unless ``--output-dir`` is given),
    exit 1 if any gate fails. The gates: inputs, prune and positions (exact integers), first
    tree with the drafter isolated, tokens end to end, and acceptance.

Only safetensors and JSON files pass between the two environments; nothing is unpickled.
"""

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[2]

BENCHMARK = "textvqa"
NUM_SAMPLES = 20
TOTAL_TOKEN = 60
DEPTH = 5
TOP_K = 10
THRESHOLD = 1.0
MAX_NEW_TOKENS = 500
KV_CAPACITY = 16384
TEMPERATURE = 0.0
# The same rule as diagnose_divergence.NEAR_TIE_MARGIN (checked when `ours` imports it).
NEAR_TIE_MARGIN = 0.5
# "At least 18 of 20 identical": at most this many samples may differ, each a near-tie.
MAX_TOKEN_MISMATCHES = 2
ACCEPTANCE_TOLERANCE = 0.03
# HiViS's own number for Qwen2.5-VL-7B TextVQA at 60/5/10 (arXiv 2509.23928 v3, Table 2), on a
# different sample set and prompt; reported for context only.
PAPER_ACCEPTANCE = 3.95
TREE_KEYS = ("draft_tokens", "retrieve_indices", "tree_mask", "tree_position_ids")
GATES = ("inputs", "prune_positions", "first_tree", "tokens", "acceptance")

UNAVOIDABLE_DIFFERENCES = [
    (
        "Target code and kernels",
        "HiViS runs its Transformers 4.54 model files on torch 2.6, and we run our 4.51.3 fork "
        "on torch 2.7; the attention dispatch and vision attention differ too. The prefill "
        "hidden states can therefore differ in the low bits. That is why the token gate "
        "tolerates near-ties, and why the drafter is also checked in isolation on HiViS's own "
        "hidden states.",
    ),
    (
        "Drafter kernels",
        "Both sides run the same vendored drafter code, but on torch 2.6 and torch 2.7. An "
        "isolated first-tree mismatch is not accepted by default: it fails the gate. It is "
        "accepted only if the same sample's isolated drafter in fp32 gives equal trees on both "
        "sides, which makes it a kernel rounding tie; this report then lists it.",
    ),
    (
        "Stop rule",
        "HiViS can overshoot max_new_tokens by up to one tree, because it stops at "
        "new_token > 500, while ours clamps at 500. Hence the cut at 500, and leaving out each "
        "sample's last iteration from the acceptance means.",
    ),
    (
        "KV cache and prefill",
        "The target cache has 128,000 slots in HiViS's code and 16,384 here. Here the prefill "
        "is slim; there the prefill logits are full. Neither changes a value that is used.",
    ),
]


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ids_sha256(input_ids):
    """Prompt fingerprint of a [1, L] tensor, as GenerationMeasurement computes it."""
    return hashlib.sha256(json.dumps(input_ids.tolist()).encode()).hexdigest()


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_new(path, text):
    """Write a file that must not exist yet."""
    with open(path, "x", encoding="utf-8") as handle:
        handle.write(text)


def through_eos(ids, eos, cap=MAX_NEW_TOKENS):
    """Output IDs through the first EOS, cut at ``cap``."""
    if eos is not None and eos in ids:
        ids = ids[: ids.index(eos) + 1]
    return list(ids[:cap])


def first_difference(a, b):
    """First index where the two lists differ, or None if they are equal."""
    for index, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return index
    return None if len(a) == len(b) else min(len(a), len(b))


def pooled_mean(lists):
    """Accepted draft tokens per iteration over all iterations, last iteration of each left out."""
    values = [v for lst in lists for v in lst[:-1]]
    return (sum(values) / len(values) if values else None), len(values)


def git_commit():
    try:
        return subprocess.run(
            ["git", "-C", str(REPO_DIR), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


# ---------------------------------------------------------------------------------------------
# samples
# ---------------------------------------------------------------------------------------------


def cmd_samples(args):
    from .benchmarks import load_samples

    run_dir = args.run_dir.resolve()
    data_dir = args.data_dir.resolve()
    samples = load_samples(BENCHMARK, data_dir, args.num_samples)
    rows = []
    for sample in samples:
        if len(sample.image_paths) != 1:
            raise SystemExit(f"{sample.question_id}: expected one image")
        image = Path(sample.image_paths[0]).resolve()
        rows.append({
            "question_id": sample.question_id,
            "question": sample.question,
            "image_path": str(image),
            "image_sha256": file_sha256(image),
        })
    payload = {
        "benchmark": BENCHMARK,
        "split": "val",
        "selection": (
            "load_samples('textvqa', data_dir, n): TextVQA 0.5.1 val, "
            "Dataset.shuffle(seed=42), first n rows, in evaluation order"
        ),
        "num_samples": len(rows),
        "data_dir": str(data_dir),
        "created_utc": utc_now(),
        "samples": rows,
    }
    write_new(run_dir / "samples.json", json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {len(rows)} samples to {run_dir / 'samples.json'}")
    return 0


# ---------------------------------------------------------------------------------------------
# ours
# ---------------------------------------------------------------------------------------------


def target_head(model):
    """The head our initialize_tree passes to the drafter."""
    base = model.base_model
    return base.language_model.lm_head if hasattr(base, "language_model") else base.lm_head


def cpu_tensor(value):
    import torch

    if not torch.is_tensor(value):
        value = torch.tensor(value)
    return value.detach().to("cpu").contiguous().clone()


def isolated_first_call(layer, head, first_call, device, dtype, image_token_id, prefix=""):
    """The adapter's first call on HiViS's recorded inputs, as our initialize_tree makes it."""
    import torch

    prompt_ids = first_call["prune.in.input_ids"].to(device)
    token = first_call["topk.in.input_ids"][:, -1:].to(device)
    hidden = first_call["prune.in.hidden_states"].to(device=device, dtype=dtype)
    embeds = first_call["prune.in.embeds"].to(device=device, dtype=dtype)
    layer.reset_kv()
    with torch.inference_mode():
        out = layer.topK_genrate(
            hidden,
            torch.cat([prompt_ids, token], dim=1),
            head,
            None,
            inputs_embeds=embeds,
            image_mask=prompt_ids == image_token_id,
        )
    result = {f"{prefix}{name}": cpu_tensor(value) for name, value in zip(TREE_KEYS, out)}
    result[f"{prefix}kept_ids"] = cpu_tensor(layer.pruned_prompt_ids)
    result[f"{prefix}kept_positions"] = cpu_tensor(layer.prompt_positions)
    result[f"{prefix}add"] = torch.tensor([int(layer.add)], dtype=torch.long)
    if int(layer.prompt_len) != prompt_ids.shape[1]:
        raise RuntimeError(f"adapter prompt_len {layer.prompt_len} != {prompt_ids.shape[1]}")
    return result


def fp32_drafter(model, spec_model_path, device):
    """Our HiViS drafter in fp32, from the checkpoint's own fp32 weights."""
    import torch
    from safetensors.torch import load_file

    from ..model.cnets_hivis import load_hivis_draft

    state_dict = load_file(os.path.join(spec_model_path, "model.safetensors"))
    layer = load_hivis_draft(
        os.path.join(spec_model_path, "config.json"),
        state_dict,
        TOTAL_TOKEN,
        DEPTH,
        TOP_K,
        THRESHOLD,
        model.config,
    )
    layer.to(torch.float32).to(device)
    layer.init_tree()
    layer.diff_device = False
    layer.eval()
    return layer


def fp32_head(head, device):
    import torch

    copy = torch.nn.Linear(
        head.weight.shape[1], head.weight.shape[0], bias=False, device=device,
        dtype=torch.float32,
    )
    with torch.no_grad():
        copy.weight.copy_(head.weight.float())
    return copy


def top2(logits):
    """argmax and the best other token (diagnose_divergence.inspect's rule)."""
    first = int(logits.argmax())
    rest = logits.clone()
    rest[first] = float("-inf")
    second = int(rest.argmax())
    return first, second


def cmd_ours(args):
    import gc

    import torch
    import transformers
    from safetensors.torch import load_file, save_file

    from ..model.autoregressive import autoregressive_generate
    from ..model.spec_model_ours import SpecModel
    from . import diagnose_divergence

    if diagnose_divergence.NEAR_TIE_MARGIN != NEAR_TIE_MARGIN:
        raise RuntimeError("NEAR_TIE_MARGIN differs from diagnose_divergence")
    run_dir = args.run_dir.resolve()
    out_jsonl = run_dir / "ours_side.jsonl"
    out_env = run_dir / "ours_side_env.json"
    iso_dir = run_dir / "ours_first_call"
    for path in (out_jsonl, out_env, iso_dir):
        if path.exists():
            raise SystemExit(f"Refusing to overwrite {path}")
    samples = json.loads((run_dir / "samples.json").read_text())["samples"]
    hivis = {r["question_id"]: r for r in read_jsonl(run_dir / "hivis_side.jsonl")}
    missing = [s["question_id"] for s in samples if s["question_id"] not in hivis]
    if missing:
        raise SystemExit(f"HiViS side has no record for {missing}")
    iso_dir.mkdir()
    device = diagnose_divergence.DEVICE

    model = SpecModel.from_pretrained(
        base_model_path=args.base_model_path,
        spec_model_path=args.spec_model_path,
        draft_method="hivis",
        total_token=TOTAL_TOKEN,
        depth=DEPTH,
        top_k=TOP_K,
        torch_dtype=torch.float16,
        slim_prefill=True,
        kv_layout="fixed",
        kv_capacity=KV_CAPACITY,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    model.eval()
    layer = model.spec_layer
    head = target_head(model)
    image_token_id = int(model.config.image_token_id)
    eos = model.tokenizer.eos_token_id
    layer32 = fp32_drafter(model, args.spec_model_path, device)
    head32 = fp32_head(head, device)

    env = {
        "started_utc": utc_now(),
        "commit": git_commit(),
        "host": socket.gethostname(),
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "base_model_path": args.base_model_path,
        "spec_model_path": args.spec_model_path,
        "draft_method": getattr(model, "draft_method", None),
        "draft_class": type(layer).__module__ + "." + type(layer).__name__,
        "dtype": str(model.base_model.dtype),
        "draft_dtype": str(next(layer.parameters()).dtype),
        "attn_implementation": getattr(model.base_model.config, "_attn_implementation", None),
        "eos_token_id": eos,
        "image_token_id": image_token_id,
        "settings": {
            "total_token": TOTAL_TOKEN, "depth": DEPTH, "top_k": TOP_K, "threshold": THRESHOLD,
            "temperature": TEMPERATURE, "max_new_tokens": MAX_NEW_TOKENS,
            "slim_prefill": True, "kv_layout": "fixed", "kv_capacity": KV_CAPACITY,
            "torch_dtype": "float16", "device_map": "auto",
            "pixel_values": "bundle fp32 -> bf16 (as eagenerate) -> fp16 in the vision tower",
        },
        "stock_model": "transformers.Qwen2_5_VLForConditionalGeneration, fp16 (margins only)",
    }
    out_env.write_text(json.dumps(env, indent=2) + "\n")

    records = []
    for index, sample in enumerate(samples):
        qid = sample["question_id"]
        theirs = hivis[qid]
        bundle_path = run_dir / theirs["bundle"]
        first_call_path = run_dir / theirs["first_call"]
        bundle = load_file(str(bundle_path))
        first_call = load_file(str(first_call_path))
        input_ids = bundle["input_ids"].to(device)
        pixel_values = bundle["pixel_values"].to(device).to(torch.bfloat16)
        grid = bundle["image_grid_thw"].to(device)
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "pixel_values": pixel_values,
            "image_grid_thw": grid,
        }

        isolated = isolated_first_call(
            layer, head, first_call, device, torch.float16, image_token_id
        )
        isolated.update(isolated_first_call(
            layer32, head32, first_call, device, torch.float32, image_token_id, prefix="fp32."
        ))
        iso_path = iso_dir / f"{qid}.safetensors"
        save_file(isolated, str(iso_path))

        stats = {}
        torch.cuda.synchronize()
        start = time.time()
        output_ids, new_token, idx, acceptance = model.specgenerate(
            **model_inputs,
            temperature=TEMPERATURE,
            max_new_tokens=MAX_NEW_TOKENS,
            log=True,
            return_acceptance_len=True,
            generation_stats=stats,
        )
        torch.cuda.synchronize()
        spec_time = time.time() - start

        ar_stats = {}
        torch.cuda.synchronize()
        start = time.time()
        ar_ids, ar_new_token, ar_steps = autoregressive_generate(
            model, model_inputs, MAX_NEW_TOKENS, generation_stats=ar_stats
        )
        torch.cuda.synchronize()
        ar_time = time.time() - start

        prompt_len = int(input_ids.shape[1])
        ours_ids = output_ids[0, prompt_len:].tolist()
        their_ids = theirs["hivis"]["output_ids"]
        ours_cut = through_eos(ours_ids, eos)
        their_cut = through_eos(their_ids, theirs.get("eos_token_id"))
        diff = first_difference(ours_cut, their_cut)
        records.append({
            "question_id": qid,
            "index": index,
            "prompt_len": prompt_len,
            "input_ids_sha256": ids_sha256(bundle["input_ids"]),
            "bundle": theirs["bundle"],
            "bundle_sha256": file_sha256(bundle_path),
            "first_call": theirs["first_call"],
            "first_call_sha256": file_sha256(first_call_path),
            "eos_token_id": eos,
            "isolation": {
                "file": f"ours_first_call/{qid}.safetensors",
                "sha256": file_sha256(iso_path),
                "kept_len": int(isolated["kept_ids"].numel()),
                "add": int(isolated["add"][0]),
                "fp32_add": int(isolated["fp32.add"][0]),
            },
            "hivis": {
                "output_ids": ours_ids,
                "new_token": int(new_token),
                "idx": int(idx),
                "acceptance_len": [int(a) for a in acceptance],
                "stop_reason": stats.get("stop_reason"),
                "wall_time": spec_time,
            },
            "ar": {
                "output_ids": ar_ids[0, prompt_len:].tolist(),
                "new_token": int(ar_new_token),
                "steps": int(ar_steps),
                "stop_reason": ar_stats.get("stop_reason"),
                "wall_time": ar_time,
            },
            "divergence": {
                "index": diff,
                "compared_lengths": [len(ours_cut), len(their_cut)],
                "ours_token_id": None if diff is None or diff >= len(ours_cut) else ours_cut[diff],
                "hivis_token_id": None if diff is None or diff >= len(their_cut) else their_cut[diff],
                "status": "identical" if diff is None else "pending",
                "margin": None,
            },
        })
        mean, count = pooled_mean([records[-1]["hivis"]["acceptance_len"]])
        print(
            f"[{index + 1}/{len(samples)}] {qid}: prompt {prompt_len}, kept "
            f"{records[-1]['isolation']['kept_len']}, add {records[-1]['isolation']['add']}, "
            f"ours {len(ours_ids)} tokens, first difference from HiViS: {diff}, "
            f"mean accepted {mean}",
            flush=True,
        )
    # Generation results first, so that a failure in the margin pass keeps them.
    out_jsonl.write_text("".join(json.dumps(r) + "\n" for r in records))

    pending = [r for r in records if r["divergence"]["status"] == "pending"]
    del model, layer, layer32, head, head32
    gc.collect()
    torch.cuda.empty_cache()
    if pending:
        stock = diagnose_divergence.load_model(args.base_model_path, torch.float16)
        env["stock_attn_implementation"] = getattr(stock.config, "_attn_implementation", None)
        for record in pending:
            bundle = load_file(str(run_dir / record["bundle"]))
            inputs = {
                "input_ids": bundle["input_ids"].to(device),
                "pixel_values": bundle["pixel_values"].to(device).to(torch.bfloat16),
                "image_grid_thw": bundle["image_grid_thw"].to(device),
            }
            div = record["divergence"]
            prefix = through_eos(record["hivis"]["output_ids"], eos)[: div["index"]]
            logits = diagnose_divergence.last_logits(stock, inputs, prefix)
            first, second = top2(logits)

            def token_logit(token_id):
                return None if token_id is None else logits[token_id].item()

            div["margin"] = {
                "top2": [
                    {"token_id": first, "logit": logits[first].item()},
                    {"token_id": second, "logit": logits[second].item()},
                ],
                "margin": logits[first].item() - logits[second].item(),
                "argmax_token_id": first,
                "ours_token_logit": token_logit(div["ours_token_id"]),
                "hivis_token_logit": token_logit(div["hivis_token_id"]),
                "dtype": "float16",
            }
            div["status"] = "measured"
            print(
                f"{record['question_id']}: first difference at {div['index']}, stock fp16 margin "
                f"{div['margin']['margin']:.4f}",
                flush=True,
            )
        del stock
        gc.collect()
        torch.cuda.empty_cache()
    tmp = out_jsonl.with_suffix(".jsonl.tmp")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in records))
    os.replace(tmp, out_jsonl)
    env["finished_utc"] = utc_now()
    out_env.write_text(json.dumps(env, indent=2) + "\n")
    print(f"Our side done: {out_jsonl}")
    return 0


# ---------------------------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------------------------


def load_optional(path, loader):
    try:
        return loader(path), None
    except FileNotFoundError:
        return None, f"missing {path.name}"
    except Exception as error:  # a corrupt file fails its gate instead of the report
        return None, f"cannot read {path.name}: {type(error).__name__}: {error}"


def tensors_or_none(path):
    from safetensors.torch import load_file

    try:
        return load_file(str(path))
    except Exception:
        return None


def same_tensor(a, b):
    import torch

    return (
        a is not None and b is not None and a.dtype == b.dtype
        and tuple(a.shape) == tuple(b.shape) and bool(torch.equal(a, b))
    )


def same_values(a, b):
    return a is not None and b is not None and a.reshape(-1).tolist() == b.reshape(-1).tolist()


def gate(name, check, passed, detail, **extra):
    return {"gate": name, "check": check, "passed": bool(passed), "detail": detail, **extra}


def compare(run_dir, expected):
    """Evaluate every gate. Returns the parity.json payload."""
    samples_doc, samples_error = load_optional(run_dir / "samples.json", lambda p: json.loads(p.read_text()))
    hivis_rows, hivis_error = load_optional(run_dir / "hivis_side.jsonl", read_jsonl)
    ours_rows, ours_error = load_optional(run_dir / "ours_side.jsonl", read_jsonl)
    hivis_env, _ = load_optional(run_dir / "hivis_side_env.json", lambda p: json.loads(p.read_text()))
    ours_env, _ = load_optional(run_dir / "ours_side_env.json", lambda p: json.loads(p.read_text()))
    samples = (samples_doc or {}).get("samples", [])
    qids = [s["question_id"] for s in samples]
    hivis = {r["question_id"]: r for r in hivis_rows or []}
    ours = {r["question_id"]: r for r in ours_rows or []}

    # Inputs: digests, order, count, and the skip and length rules.
    problems = [e for e in (samples_error, hivis_error, ours_error) if e]
    if len(samples) != expected:
        problems.append(f"samples.json has {len(samples)} samples, expected {expected}")
    if hivis_rows is not None and [r["question_id"] for r in hivis_rows] != qids:
        problems.append("hivis_side.jsonl question IDs differ from samples.json (order, skips or extras)")
    if ours_rows is not None and [r["question_id"] for r in ours_rows] != qids:
        problems.append("ours_side.jsonl question IDs differ from samples.json (order, skips or extras)")
    per_sample = {}
    first_calls = {}
    isolations = {}
    for sample in samples:
        qid = sample["question_id"]
        row = per_sample.setdefault(qid, {"question_id": qid})
        theirs, mine = hivis.get(qid), ours.get(qid)
        if theirs is None or mine is None:
            problems.append(f"{qid}: missing on the {'HiViS' if theirs is None else 'our'} side")
            continue
        row.update(prompt_len=theirs.get("prompt_len"), kept_len=theirs.get("kept_len"), add=theirs.get("add"))
        if theirs.get("image_sha256") != sample["image_sha256"]:
            problems.append(f"{qid}: image digest differs from samples.json")
        if not (theirs.get("prompt_limit_ok") and theirs.get("length_budget_ok")):
            problems.append(f"{qid}: prompt length rule not met")
        for kind in ("bundle", "first_call"):
            path = run_dir / theirs[kind]
            actual = file_sha256(path) if path.is_file() else None
            if actual is None:
                problems.append(f"{qid}: {theirs[kind]} is missing")
            elif not (actual == theirs[f"{kind}_sha256"] == mine.get(f"{kind}_sha256")):
                problems.append(f"{qid}: {kind} digest differs between the file and the two sides")
        if theirs.get("input_ids_sha256") != mine.get("input_ids_sha256"):
            problems.append(f"{qid}: input_ids_sha256 differs between the sides")
        if theirs.get("prompt_len") != mine.get("prompt_len"):
            problems.append(f"{qid}: prompt length differs between the sides")
        bundle = tensors_or_none(run_dir / theirs["bundle"])
        first_call = tensors_or_none(run_dir / theirs["first_call"])
        first_calls[qid] = first_call
        if bundle is None or first_call is None:
            problems.append(f"{qid}: bundle or first-call file cannot be read")
        elif not same_values(bundle["input_ids"], first_call["prune.in.input_ids"]):
            problems.append(f"{qid}: the first draft call's prompt IDs differ from the bundle")
        iso_file = (mine.get("isolation") or {}).get("file")
        iso = tensors_or_none(run_dir / iso_file) if iso_file else None
        if iso is not None and file_sha256(run_dir / iso_file) != mine["isolation"].get("sha256"):
            problems.append(f"{qid}: {iso_file} digest differs from ours_side.jsonl")
            iso = None
        isolations[qid] = iso
    inputs_gate = gate(
        "inputs", "bundle and sample digests; all samples present, none skipped",
        not problems, "; ".join(problems) if problems else f"{len(samples)}/{expected} samples, digests equal",
    )

    # Prune and positions: exact integers, same prompt.
    prune_ok = 0
    prune_problems = []
    for qid in qids:
        fc, iso = first_calls.get(qid), isolations.get(qid)
        row = per_sample[qid]
        if fc is None or iso is None:
            row["prune_equal"] = None
            prune_problems.append(f"{qid}: records missing")
            continue
        checks = {
            "kept_ids": same_values(fc["prune.out.input_ids"], iso["kept_ids"]),
            "kept_positions": same_values(fc["prune.out.position_ids"], iso["kept_positions"]),
            "add": same_values(fc["topk.add"], iso["add"]),
        }
        # HiViS-internal consistency: its first draft call got the pruned values.
        consistent = same_values(fc["topk.in.position_ids"], fc["prune.out.position_ids"]) and (
            fc["topk.in.input_ids"][:, :-1].reshape(-1).tolist()
            == fc["prune.out.input_ids"].reshape(-1).tolist()
        )
        row["prune_equal"] = all(checks.values())
        row["prune_checks"] = checks
        row["hivis_first_call_consistent"] = consistent
        if row["prune_equal"] and consistent:
            prune_ok += 1
        else:
            failed = [k for k, v in checks.items() if not v] + ([] if consistent else ["hivis_internal"])
            prune_problems.append(f"{qid}: {', '.join(failed)} differ")
    prune_gate = gate(
        "prune_positions", "kept IDs, kept positions and add equal HiViS's recorded values",
        prune_ok == expected == len(qids), f"{prune_ok}/{expected} equal"
        + ("; " + "; ".join(prune_problems) if prune_problems else ""),
    )

    # First tree, drafter isolated: fp16 equal, or accepted when fp32 is equal on both sides.
    tree_equal = 0
    tree_accepted = []
    tree_problems = []
    for qid in qids:
        fc, iso = first_calls.get(qid), isolations.get(qid)
        row = per_sample[qid]
        if fc is None or iso is None:
            row["tree"] = None
            tree_problems.append(f"{qid}: records missing")
            continue
        fp16 = {k: same_tensor(fc[f"topk.out.{k}"], iso[k]) for k in TREE_KEYS}
        fp32 = {k: same_tensor(fc.get(f"fp32.{k}"), iso.get(f"fp32.{k}")) for k in TREE_KEYS}
        if all(fp16.values()):
            row["tree"] = "equal"
            tree_equal += 1
        elif all(fp32.values()):
            row["tree"] = "accepted_fp32_equal"
            tree_accepted.append(qid)
        else:
            row["tree"] = "differs"
            tree_problems.append(
                f"{qid}: fp16 {[k for k, v in fp16.items() if not v]} and "
                f"fp32 {[k for k, v in fp32.items() if not v]} differ"
            )
        row["tree_fp16"] = fp16
        row["tree_fp32"] = fp32
    tree_gate = gate(
        "first_tree", "draft_tokens, retrieve_indices, tree_mask and tree_position_ids equal, drafter isolated, fp16",
        tree_equal + len(tree_accepted) == expected == len(qids) and not tree_problems,
        f"{tree_equal}/{expected} equal in fp16"
        + (f"; accepted as kernel rounding ties (fp32 trees equal): {tree_accepted}" if tree_accepted else "")
        + ("; " + "; ".join(tree_problems) if tree_problems else ""),
        accepted=tree_accepted,
    )

    # Tokens end to end: through the first EOS, cut at 500; mismatches must be near-ties.
    identical = []
    near_ties = []
    token_problems = []
    for qid in qids:
        theirs, mine = hivis.get(qid), ours.get(qid)
        row = per_sample[qid]
        if theirs is None or mine is None:
            token_problems.append(f"{qid}: missing")
            continue
        if theirs.get("eos_token_id") != mine.get("eos_token_id"):
            token_problems.append(f"{qid}: EOS IDs differ")
        their_cut = through_eos(theirs["hivis"]["output_ids"], theirs.get("eos_token_id"))
        ours_cut = through_eos(mine["hivis"]["output_ids"], mine.get("eos_token_id"))
        diff = first_difference(ours_cut, their_cut)
        row["tokens"] = {"identical": diff is None, "first_difference": diff,
                         "lengths": [len(ours_cut), len(their_cut)]}
        if diff is None:
            identical.append(qid)
            continue
        div = mine.get("divergence") or {}
        margin = (div.get("margin") or {}).get("margin")
        row["tokens"]["margin"] = margin
        if div.get("index") != diff:
            token_problems.append(f"{qid}: recorded divergence index {div.get('index')} != {diff}")
        elif margin is None:
            token_problems.append(f"{qid}: no margin recorded at index {diff}")
        elif margin <= NEAR_TIE_MARGIN:
            near_ties.append(qid)
            row["tokens"]["near_tie"] = True
        else:
            row["tokens"]["near_tie"] = False
            token_problems.append(f"{qid}: margin {margin:.4f} > {NEAR_TIE_MARGIN} at index {diff}")
    mismatches = len(qids) - len(identical)
    tokens_gate = gate(
        "tokens", f"output IDs through the first EOS, cut at {MAX_NEW_TOKENS}; at most "
        f"{MAX_TOKEN_MISMATCHES} differ, each a near-tie (stock fp16 top-2 margin <= {NEAR_TIE_MARGIN})",
        len(qids) == expected and mismatches <= MAX_TOKEN_MISMATCHES and not token_problems
        and len(near_ties) == mismatches,
        f"{len(identical)}/{expected} identical, {mismatches} differ, {len(near_ties)} of them near-ties"
        + ("; " + "; ".join(token_problems) if token_problems else ""),
        identical=identical, near_ties=near_ties,
    )

    # Acceptance over the samples with identical tokens, last iteration left out on both sides.
    hivis_mean, hivis_count = pooled_mean([hivis[q]["hivis"]["accept_lengths"] for q in identical])
    ours_mean, ours_count = pooled_mean([ours[q]["hivis"]["acceptance_len"] for q in identical])
    relative = (
        abs(ours_mean - hivis_mean) / hivis_mean
        if hivis_mean not in (None, 0) and ours_mean is not None else None
    )
    acceptance_gate = gate(
        "acceptance", f"mean accepted draft tokens per iteration over the identical samples, "
        f"last iteration left out; within {ACCEPTANCE_TOLERANCE:.0%} relative",
        relative is not None and relative <= ACCEPTANCE_TOLERANCE,
        f"HiViS {fmt(hivis_mean)} ({hivis_count} iterations), ours {fmt(ours_mean)} "
        f"({ours_count} iterations), relative difference {fmt(relative)}"
        if relative is not None else "no identical sample with more than one iteration",
        hivis_mean=hivis_mean, ours_mean=ours_mean, relative_difference=relative,
        samples=len(identical),
    )

    # Reported, not gated.
    def self_agreement(rows, key_spec, key_ar, eos_key="eos_token_id"):
        result = {"identical": 0, "differ": {}}
        for qid in qids:
            r = rows.get(qid)
            if r is None:
                continue
            a = through_eos(r[key_spec]["output_ids"], r.get(eos_key))
            b = through_eos(r[key_ar]["output_ids"], r.get(eos_key))
            d = first_difference(a, b)
            if d is None:
                result["identical"] += 1
            else:
                result["differ"][qid] = d
        return result

    all_hivis_mean, all_hivis_count = pooled_mean([hivis[q]["hivis"]["accept_lengths"] for q in qids if q in hivis])
    all_ours_mean, all_ours_count = pooled_mean([ours[q]["hivis"]["acceptance_len"] for q in qids if q in ours])
    reported = {
        "hivis_speculative_vs_hivis_autoregressive": self_agreement(hivis, "hivis", "ar"),
        "ours_hivis_vs_ours_autoregressive": self_agreement(ours, "hivis", "ar"),
        "hivis_side_mean_acceptance_all_samples": all_hivis_mean,
        "hivis_side_iterations": all_hivis_count,
        "ours_mean_acceptance_all_samples": all_ours_mean,
        "ours_iterations": all_ours_count,
        "paper_acceptance_60_5_10": PAPER_ACCEPTANCE,
        "note": (
            "HiViS's eagenerate against its naivegenerate: its AR path does not round the pixel "
            "values through bf16, so its vision inputs differ slightly. The paper's 3.95 used "
            "different samples (the first 80 TextVQA test rows) and its own prompt."
        ),
    }
    gates = [inputs_gate, prune_gate, tree_gate, tokens_gate, acceptance_gate]
    return {
        "run_dir": str(run_dir),
        "created_utc": utc_now(),
        "commit": git_commit(),
        "expected_samples": expected,
        "settings": {
            "total_token": TOTAL_TOKEN, "depth": DEPTH, "top_k": TOP_K,
            "max_new_tokens": MAX_NEW_TOKENS, "near_tie_margin": NEAR_TIE_MARGIN,
            "max_token_mismatches": MAX_TOKEN_MISMATCHES,
            "acceptance_tolerance": ACCEPTANCE_TOLERANCE,
        },
        "passed": all(g["passed"] for g in gates),
        "gates": gates,
        "reported": reported,
        "samples": [per_sample[q] for q in qids],
        "environments": {
            "hivis": {k: (hivis_env or {}).get(k) for k in (
                "torch_version", "transformers_version", "processor_class",
                "image_processor_class", "attn_implementation", "dtype", "gpu", "kv_cache_slots")},
            "ours": {k: (ours_env or {}).get(k) for k in (
                "torch", "transformers", "draft_class", "attn_implementation", "dtype", "gpu")},
        },
        "unavoidable_differences": [{"name": n, "text": t} for n, t in UNAVOIDABLE_DIFFERENCES],
    }


def fmt(value, digits=4):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def verdict(flag):
    return "n/a" if flag is None else ("equal" if flag else "differ")


def cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def markdown(result):
    lines = [
        "# HiViS reference parity",
        "",
        f"- Run: `{result['run_dir']}`",
        f"- Created: {result['created_utc']}, commit `{result['commit']}`",
        f"- Verdict: **{'PASS' if result['passed'] else 'FAIL'}**",
        f"- Settings: tree {TOTAL_TOKEN}/{DEPTH}/{TOP_K}, greedy, fp16, at most "
        f"{MAX_NEW_TOKENS} new tokens, {result['expected_samples']} TextVQA val samples "
        "with HiViS's own prompt",
        "",
        "## Gates",
        "",
        "| Gate | Check | Result | Detail |",
        "| --- | --- | --- | --- |",
    ]
    for g in result["gates"]:
        lines.append(
            f"| {g['gate']} | {cell(g['check'])} | {'pass' if g['passed'] else 'FAIL'} | {cell(g['detail'])} |"
        )
    lines += [
        "",
        "## Samples",
        "",
        "| Question | Prompt | Kept | add | Prune and positions | First tree | Tokens | First difference | Margin |",
        "| --- | ---: | ---: | ---: | --- | --- | --- | ---: | ---: |",
    ]
    for s in result["samples"]:
        tokens = s.get("tokens") or {}
        lines.append(
            f"| {s['question_id']} | {fmt(s.get('prompt_len'))} | {fmt(s.get('kept_len'))} | "
            f"{fmt(s.get('add'))} | {verdict(s.get('prune_equal'))} | {fmt(s.get('tree'))} | "
            f"{'identical' if tokens.get('identical') else ('differ' if tokens else 'n/a')} | "
            f"{fmt(tokens.get('first_difference'))} | {fmt(tokens.get('margin'))} |"
        )
    rep = result["reported"]
    hv, ov = rep["hivis_speculative_vs_hivis_autoregressive"], rep["ours_hivis_vs_ours_autoregressive"]
    lines += [
        "",
        "## Reported, not gated",
        "",
        f"- HiViS's `eagenerate` against its `naivegenerate`: {hv['identical']} identical; "
        f"first differences {hv['differ'] or 'none'}.",
        f"- Our fp16 HiViS against our fp16 AR: {ov['identical']} identical; first differences "
        f"{ov['differ'] or 'none'}.",
        f"- Mean accepted draft tokens per iteration over all samples, last iteration left out: "
        f"HiViS side {fmt(rep['hivis_side_mean_acceptance_all_samples'])} "
        f"({rep['hivis_side_iterations']} iterations), ours "
        f"{fmt(rep['ours_mean_acceptance_all_samples'])} ({rep['ours_iterations']} iterations). "
        f"The paper reports {PAPER_ACCEPTANCE} at 60/5/10 on other samples.",
        f"- {rep['note']}",
        "",
        "## Environments",
        "",
        f"- HiViS side: `{json.dumps(result['environments']['hivis'])}`",
        f"- Our side: `{json.dumps(result['environments']['ours'])}`",
        "",
        "## Unavoidable differences",
        "",
    ]
    for item in result["unavoidable_differences"]:
        lines.append(f"- **{item['name']}.** {item['text']}")
    accepted = next(g for g in result["gates"] if g["gate"] == "first_tree").get("accepted") or []
    if accepted:
        lines.append(
            f"- **Accepted first-tree differences.** {', '.join(accepted)}: the fp16 trees "
            "differ, and the fp32 trees are equal on both sides."
        )
    lines.append("")
    return "\n".join(lines)


def cmd_compare(args):
    run_dir = args.run_dir.resolve()
    output_dir = (args.output_dir or run_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = [output_dir / "parity.json", output_dir / "parity.md"]
    for path in targets:
        if path.exists():
            raise SystemExit(f"Refusing to overwrite {path}; pass a new --output-dir")
    result = compare(run_dir, args.expected_samples)
    write_new(targets[0], json.dumps(result, indent=2) + "\n")
    write_new(targets[1], markdown(result))
    for g in result["gates"]:
        print(f"{g['gate']}: {'pass' if g['passed'] else 'FAIL'} ({g['detail']})")
    print(f"Verdict: {'PASS' if result['passed'] else 'FAIL'}. Wrote {targets[0]} and {targets[1]}")
    return 0 if result["passed"] else 1


def build_parser():
    parser = argparse.ArgumentParser(
        prog="python -m vispec.evaluation.hivis_parity",
        description=__doc__.split("\n\n")[0],
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("samples", help="write samples.json (first 20 seed-42 TextVQA val questions)")
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--data-dir", required=True, type=Path)
    p.add_argument("--num-samples", type=int, default=NUM_SAMPLES)
    p.set_defaults(func=cmd_samples)

    p = sub.add_parser("ours", help="our side: drafter isolation, end-to-end tokens, margins")
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--base-model-path", required=True)
    p.add_argument("--spec-model-path", required=True)
    p.set_defaults(func=cmd_ours)

    p = sub.add_parser("compare", help="the verdict: parity.json and parity.md, exit 1 on a failed gate")
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--output-dir", type=Path, help="default: the run folder")
    p.add_argument("--expected-samples", type=int, default=NUM_SAMPLES)
    p.set_defaults(func=cmd_compare)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if getattr(args, "num_samples", 1) < 1 or getattr(args, "expected_samples", 1) < 1:
        raise SystemExit("sample counts must be positive")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
