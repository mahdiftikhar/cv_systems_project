"""HiViS side of the reference run: HiViS's own code on our first 20 TextVQA samples.

This script runs only in the separate HiViS environment (Python 3.9, torch 2.6.0+cu124,
Transformers 4.54.0), started by run.sh from inside the run folder, offline and without
credentials. It imports nothing from this repository. It puts the read-only HiViS checkout
(``--hivis-dir``) on ``sys.path`` and calls HiViS's own ``EaModel``, ``prepare_inputs``,
``eagenerate`` and ``naivegenerate``. Only HiViS's data loader is replaced: the rows come from
our ``samples.json`` (seed-42 TextVQA val subset), in the row shape ``_load_textvqa`` produces.

Per sample, in evaluation order:

1. ``prepare_inputs(model, ds, i, "textvqa", truncation=False)``: HiViS's own prompt (no
   system message, ``[image, question + OCR suffix]``, processor defaults). The prompt must
   fit HiViS's skip rule (at most 4,000 tokens) and the ``max_length`` stop of ``eagenerate``.
   ``input_ids``, ``pixel_values`` (as the processor gives them, before any cast) and
   ``image_grid_thw`` go to ``bundle/<qid>.safetensors``.
2. ``eagenerate(temperature=0.0, max_new_tokens=500, log=True)``. Two recording wrappers are
   active only around the first draft call: ``hivis.model.utils_hivis.prune_image_tokens`` and
   ``model.ea_layer.topK_genrate`` (its arguments, its four outputs and ``ea_layer.add``).
   The same first call is then rerun in fp32 (the draft checkpoint's own fp32 weights and the
   target head upcast to fp32, on the recorded inputs upcast to fp32). The records go to
   ``first_call/<qid>.safetensors``.
3. ``naivegenerate(temperature=0.0, max_new_tokens=500, log=True)``: HiViS's own AR tokens.

Outputs (in ``--run-dir`` only): ``bundle/``, ``first_call/``, ``hivis_side.jsonl`` and
``hivis_side_env.json``. Only safetensors and JSON files are written; nothing is pickled.
"""

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

TOTAL_TOKEN = 60
DEPTH = 5
TOP_K = 10
THRESHOLD = 1.0
MAX_NEW_TOKENS = 500
TEMPERATURE = 0.0
SEED = 42
WARMUP_COUNT = 3
# ge_hivis_answer.get_model_answers skips prompts longer than this; here no sample may be skipped.
PROMPT_LIMIT = 4000
# eagenerate's default max_length; its loop stops once the sequence is longer than
# max_length - ea_layer.total_tokens - 10 (4027 at total_token 60).
EAGENERATE_MAX_LENGTH = 4096
EXPECTED_MAX_LENGTH_STOP = 4027
HIVIS_PIN = "6db36b241eaf733ec313cbc9b86206c0f20762c9"
SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ids_sha256(input_ids):
    """Prompt fingerprint of a [1, L] tensor, as our GenerationMeasurement computes it."""
    return hashlib.sha256(json.dumps(input_ids.tolist()).encode()).hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--hivis-dir", required=True, type=Path, help="read-only HiViS checkout")
    parser.add_argument("--base-model-path", required=True)
    parser.add_argument("--ea-model-path", required=True)
    return parser.parse_args()


def import_hivis(hivis_dir):
    """Put the checkout first on sys.path and import the HiViS modules used here."""
    hivis_dir = str(hivis_dir.resolve())
    sys.path.insert(0, hivis_dir)
    import hivis.evaluation.benchmark_data as benchmark_data
    import hivis.model.cnets_hivis as cnets_hivis
    import hivis.model.configs as configs
    import hivis.model.model_hivis as model_hivis
    import hivis.model.utils_hivis as utils_hivis

    modules = {
        m.__name__: m.__file__
        for m in (benchmark_data, cnets_hivis, configs, model_hivis, utils_hivis)
    }
    for name, path in modules.items():
        if not str(path).startswith(hivis_dir + os.sep):
            raise SystemExit(f"{name} was imported from {path}, not from {hivis_dir}")
    return benchmark_data, cnets_hivis, configs, model_hivis, utils_hivis, modules


def setup_seed(seed):
    """The same steps as hivis/evaluation/ge_hivis_answer.setup_seed."""
    import numpy as np
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def load_samples(run_dir):
    """samples.json rows, with each image's sha256 checked before any model loads."""
    payload = json.loads((run_dir / "samples.json").read_text())
    rows = payload["samples"]
    for row in rows:
        if not SAFE_ID.match(row["question_id"]):
            raise SystemExit(f"Unexpected question id {row['question_id']!r}")
        actual = file_sha256(row["image_path"])
        if actual != row["image_sha256"]:
            raise SystemExit(
                f"Image digest mismatch for {row['question_id']}: {actual} != {row['image_sha256']}"
            )
    return payload, rows


def build_dataset(rows):
    """The row shape HiViS's _load_textvqa produces: question_id, question and an Image column."""
    from datasets import Dataset, Image

    records = [
        {"question_id": r["question_id"], "question": r["question"], "image": r["image_path"]}
        for r in rows
    ]
    return Dataset.from_list(records).cast_column("image", Image())


def cpu(tensor):
    return tensor.detach().to("cpu").contiguous().clone()


class FirstCallRecorder:
    """Records the first prune_image_tokens and topK_genrate calls of one eagenerate.

    The wrappers are installed on entry and removed on exit; every call passes through to the
    original function with its arguments and return values unchanged.
    """

    def __init__(self, utils_module, ea_layer):
        self.utils = utils_module
        self.ea_layer = ea_layer
        self.tensors = {}
        self.info = {"prune_calls": 0, "topk_calls": 0}

    def __enter__(self):
        original_prune = self.utils.prune_image_tokens
        original_topk = self.ea_layer.topK_genrate
        prune_signature = inspect.signature(original_prune)
        topk_signature = inspect.signature(original_topk)
        self._original_prune = original_prune

        def prune(*args, **kwargs):
            first = self.info["prune_calls"] == 0
            self.info["prune_calls"] += 1
            if not first:
                return original_prune(*args, **kwargs)
            bound = prune_signature.bind(*args, **kwargs)
            bound.apply_defaults()
            a = bound.arguments
            self.tensors["prune.in.input_ids"] = cpu(a["input_ids"])
            self.tensors["prune.in.embeds"] = cpu(a["embeds"])
            self.tensors["prune.in.hidden_states"] = cpu(a["hidden_states"])
            self.tensors["prune.in.text_position_ids"] = cpu(a["text_position_ids"])
            self.info["prune.image_token_id"] = int(a["image_token_id"])
            result = original_prune(*args, **kwargs)
            ids, embeds, states, positions = result
            self.tensors["prune.out.input_ids"] = cpu(ids)
            self.tensors["prune.out.position_ids"] = cpu(positions)
            return result

        def topk(*args, **kwargs):
            first = self.info["topk_calls"] == 0
            self.info["topk_calls"] += 1
            if not first:
                return original_topk(*args, **kwargs)
            bound = topk_signature.bind(*args, **kwargs)
            bound.apply_defaults()
            a = bound.arguments
            self.tensors["topk.in.hidden_states"] = cpu(a["hidden_states"])
            self.tensors["topk.in.input_ids"] = cpu(a["input_ids"])
            self.tensors["topk.in.input_embeds"] = cpu(a["input_embeds"])
            self.tensors["topk.in.position_ids"] = cpu(a["position_ids"])
            head = a["head"]
            self.info["topk.logits_processor_is_none"] = a["logits_processor"] is None
            self.info["topk.head"] = {
                "class": type(head).__name__,
                "weight_shape": list(head.weight.shape),
                "weight_dtype": str(head.weight.dtype),
            }
            result = original_topk(*args, **kwargs)
            draft_tokens, retrieve_indices, tree_mask, tree_position_ids = result
            self.tensors["topk.out.draft_tokens"] = cpu(draft_tokens)
            self.tensors["topk.out.retrieve_indices"] = cpu(retrieve_indices)
            self.tensors["topk.out.tree_mask"] = cpu(tree_mask)
            self.tensors["topk.out.tree_position_ids"] = cpu(tree_position_ids)
            self.info["topk.add"] = int(self.ea_layer.add)
            return result

        self.utils.prune_image_tokens = prune
        # An instance attribute shadows the bound method until __exit__ deletes it.
        self.ea_layer.topK_genrate = topk
        return self

    def __exit__(self, *exc):
        self.utils.prune_image_tokens = self._original_prune
        del self.ea_layer.topK_genrate
        return False

    def check(self, qid):
        required = (
            "prune.in.input_ids", "prune.in.embeds", "prune.in.hidden_states",
            "prune.out.input_ids", "prune.out.position_ids", "topk.in.hidden_states",
            "topk.in.input_ids", "topk.in.input_embeds", "topk.in.position_ids",
            "topk.out.draft_tokens", "topk.out.retrieve_indices", "topk.out.tree_mask",
            "topk.out.tree_position_ids",
        )
        missing = [key for key in required if key not in self.tensors]
        if missing or self.info["prune_calls"] != 1:
            raise SystemExit(
                f"First-call records incomplete for {qid}: missing {missing}, "
                f"prune calls {self.info['prune_calls']}"
            )


def load_fp32_drafter(cnets_hivis, configs, ea_model_path, device):
    """A second HiViS drafter in fp32, from the checkpoint's own fp32 weights."""
    import torch
    from safetensors.torch import load_file

    config_path = os.path.join(ea_model_path, "config.json")
    config = configs.EConfig.from_pretrained(config_path)
    with open(config_path) as handle:
        bias = json.load(handle).get("bias", True)
    state_dict = load_file(os.path.join(ea_model_path, "model.safetensors"))
    drafter = cnets_hivis.Model(
        config,
        bias=bias,
        total_tokens=TOTAL_TOKEN,
        depth=DEPTH,
        top_k=TOP_K,
        threshold=THRESHOLD,
        residual_count=state_dict["residual"].shape[0],
    )
    drafter.load_state_dict(state_dict, strict=True)
    drafter.to(torch.float32).to(device)
    drafter.init_tree()
    drafter.diff_device = False
    drafter.eval()
    return drafter


def fp32_head(lm_head, device):
    import torch

    head = torch.nn.Linear(
        lm_head.weight.shape[1], lm_head.weight.shape[0], bias=False,
        device=device, dtype=torch.float32,
    )
    with torch.no_grad():
        head.weight.copy_(lm_head.weight.float())
    return head


def fp32_first_call(drafter, head, tensors, device):
    """Rerun the recorded first draft call in fp32 on the recorded inputs."""
    drafter.reset_kv()
    out = drafter.topK_genrate(
        tensors["topk.in.hidden_states"].to(device).float(),
        tensors["topk.in.input_ids"].to(device),
        head,
        None,
        input_embeds=tensors["topk.in.input_embeds"].to(device).float(),
        position_ids=tensors["topk.in.position_ids"].to(device),
    )
    names = ("draft_tokens", "retrieve_indices", "tree_mask", "tree_position_ids")
    result = {f"fp32.{name}": cpu(value) for name, value in zip(names, out)}
    return result, int(drafter.add)


def collapsed_prompt(tokenizer, ids, image_token_id):
    """Decoded prompt with each run of image pads shown as one marker."""
    parts, run = [], 0
    text_ids = []
    for token in ids:
        if token == image_token_id:
            if text_ids:
                parts.append(tokenizer.decode(text_ids, skip_special_tokens=False))
                text_ids = []
            run += 1
            continue
        if run:
            parts.append(f"<|image_pad|>x{run}")
            run = 0
        text_ids.append(token)
    if run:
        parts.append(f"<|image_pad|>x{run}")
    if text_ids:
        parts.append(tokenizer.decode(text_ids, skip_special_tokens=False))
    return "".join(parts)


def attn_implementations(embed_model):
    config = embed_model.config
    found = {"model": getattr(config, "_attn_implementation", None)}
    for name in ("text_config", "vision_config"):
        sub = getattr(config, name, None)
        if sub is not None:
            found[name] = getattr(sub, "_attn_implementation", None)
    return found


def distribution_versions(names):
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def main():
    args = parse_args()
    run_dir = args.run_dir.resolve()
    started = utc_now()
    out_jsonl = run_dir / "hivis_side.jsonl"
    out_env = run_dir / "hivis_side_env.json"
    bundle_dir = run_dir / "bundle"
    first_call_dir = run_dir / "first_call"
    for path in (out_jsonl, out_env, bundle_dir, first_call_dir):
        if path.exists():
            raise SystemExit(f"Refusing to overwrite {path}")
    if Path.cwd().resolve() == args.hivis_dir.resolve() or str(Path.cwd().resolve()).startswith(
        str(args.hivis_dir.resolve()) + os.sep
    ):
        raise SystemExit("The working directory must not be inside the HiViS checkout")

    payload, rows = load_samples(run_dir)
    benchmark_data, cnets_hivis, configs, model_hivis, utils_hivis, modules = import_hivis(
        args.hivis_dir
    )

    import torch
    import transformers
    from safetensors.torch import save_file

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available in the HiViS environment")
    bundle_dir.mkdir()
    first_call_dir.mkdir()
    dataset = build_dataset(rows)

    # ge_hivis_answer.main seeds before loading; get_model_answers runs in inference mode.
    setup_seed(SEED)
    with torch.inference_mode():
        model = model_hivis.EaModel.from_pretrained(
            base_model_path=args.base_model_path,
            ea_model_path=args.ea_model_path,
            total_token=TOTAL_TOKEN,
            depth=DEPTH,
            top_k=TOP_K,
            draft_method="hivis",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto",
        )
        tokenizer = model.get_tokenizer()
        model.eval()
        device = model.base_model.device
        max_length_stop = EAGENERATE_MAX_LENGTH - model.ea_layer.total_tokens - 10
        if max_length_stop != EXPECTED_MAX_LENGTH_STOP:
            raise SystemExit(f"Unexpected eagenerate max_length stop {max_length_stop}")
        image_token_id = int(model.embed_model.config.image_token_id)

        env = {
            "started_utc": started,
            "python": sys.version,
            "executable": sys.executable,
            "cwd": os.getcwd(),
            "versions": distribution_versions(
                ["torch", "torchvision", "transformers", "accelerate", "datasets",
                 "safetensors", "tokenizers", "pillow", "numpy", "huggingface-hub"]
            ),
            "torch_version": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "transformers_version": transformers.__version__,
            "hivis_dir": str(args.hivis_dir.resolve()),
            "hivis_pin": HIVIS_PIN,
            "hivis_modules": modules,
            "base_model_path": args.base_model_path,
            "ea_model_path": args.ea_model_path,
            "processor_class": type(model.processor).__name__,
            "image_processor_class": type(model.processor.image_processor).__name__,
            "processor_tokenizer_class": type(model.processor.tokenizer).__name__,
            "tokenizer_class": type(tokenizer).__name__,
            "eos_token_id": tokenizer.eos_token_id,
            "image_token_id": image_token_id,
            "attn_implementation": attn_implementations(model.embed_model),
            "dtype": str(model.base_model.dtype),
            "draft_dtype": str(next(model.ea_layer.parameters()).dtype),
            "draft_class": type(model.ea_layer).__module__ + "." + type(model.ea_layer).__name__,
            "draft": {
                "total_tokens": int(model.ea_layer.total_tokens),
                "depth": int(model.ea_layer.depth),
                "top_k": int(model.ea_layer.top_k),
                "residual_count": int(model.ea_layer.residual.shape[0]),
            },
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0),
            "hf_device_map": getattr(model.embed_model, "hf_device_map", None),
            "settings": {
                "total_token": TOTAL_TOKEN, "depth": DEPTH, "top_k": TOP_K,
                "threshold": THRESHOLD, "temperature": TEMPERATURE,
                "max_new_tokens": MAX_NEW_TOKENS, "seed": SEED, "warmup_count": WARMUP_COUNT,
                "prompt_limit": PROMPT_LIMIT, "max_length_stop": max_length_stop,
                "torch_dtype": "float16", "device_map": "auto", "low_cpu_mem_usage": True,
            },
            "environment": {
                key: os.environ.get(key)
                for key in ("HF_HOME", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
                            "HF_DATASETS_OFFLINE", "PYTHONNOUSERSITE",
                            "PYTHONDONTWRITEBYTECODE", "CUDA_VISIBLE_DEVICES")
            },
            "token_variables_set": [
                key for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN") if os.environ.get(key)
            ],
            "num_samples": len(rows),
            "sample_source": {k: v for k, v in payload.items() if k != "samples"},
        }
        print(json.dumps({k: env[k] for k in ("processor_class", "image_processor_class",
                                              "attn_implementation", "dtype", "gpu")}))
        # Written now and again at the end, so a run that stops midway still has it.
        out_env.write_text(json.dumps(env, indent=2) + "\n")

        # Warm-up as get_model_answers does: indices 0-2, truncation=True, seed 0 before each.
        for warmup_index in range(min(WARMUP_COUNT, len(dataset))):
            torch.manual_seed(0)
            inputs = benchmark_data.prepare_inputs(
                model, dataset, warmup_index, "textvqa", truncation=True
            )
            model.eagenerate(
                inputs, temperature=TEMPERATURE, max_new_tokens=MAX_NEW_TOKENS, log=True
            )
        print("Warmup done", flush=True)
        env["kv_cache_slots"] = int(model.past_key_values_data[0].shape[3])

        drafter32 = load_fp32_drafter(cnets_hivis, configs, args.ea_model_path, device)
        head32 = fp32_head(model.embed_model.lm_head, device)

        for index, row in enumerate(rows):
            qid = row["question_id"]
            inputs = benchmark_data.prepare_inputs(
                model, dataset, index, "textvqa", truncation=False
            )
            input_ids = inputs.input_ids
            prompt_len = int(input_ids.shape[1])
            if prompt_len > PROMPT_LIMIT:
                raise SystemExit(f"{qid}: prompt of {prompt_len} tokens exceeds HiViS's skip rule")
            budget = prompt_len + MAX_NEW_TOKENS + TOTAL_TOKEN + 10
            if not budget < max_length_stop:
                raise SystemExit(f"{qid}: {budget} >= eagenerate max_length stop {max_length_stop}")
            if index == 0:
                env["example_prompt"] = collapsed_prompt(
                    tokenizer, input_ids[0].tolist(), image_token_id
                )

            bundle_path = bundle_dir / f"{qid}.safetensors"
            save_file(
                {
                    "input_ids": cpu(input_ids),
                    "pixel_values": cpu(inputs.pixel_values),
                    "image_grid_thw": cpu(inputs.image_grid_thw),
                },
                str(bundle_path),
            )

            recorder = FirstCallRecorder(utils_hivis, model.ea_layer)
            torch.cuda.synchronize()
            start = time.time()
            with recorder:
                output_ids, new_token, idx, accept_lengths = model.eagenerate(
                    inputs, temperature=TEMPERATURE, max_new_tokens=MAX_NEW_TOKENS, log=True
                )
            torch.cuda.synchronize()
            spec_time = time.time() - start
            recorder.check(qid)

            tensors = dict(recorder.tensors)
            tensors["topk.add"] = torch.tensor([recorder.info["topk.add"]], dtype=torch.long)
            fp32_tensors, fp32_add = fp32_first_call(drafter32, head32, tensors, device)
            tensors.update(fp32_tensors)
            tensors["fp32.add"] = torch.tensor([fp32_add], dtype=torch.long)
            first_call_path = first_call_dir / f"{qid}.safetensors"
            save_file(tensors, str(first_call_path))

            torch.cuda.synchronize()
            start = time.time()
            ar_ids, ar_new_token, ar_idx, ar_init_time = model.naivegenerate(
                inputs, temperature=TEMPERATURE, max_new_tokens=MAX_NEW_TOKENS, log=True
            )
            torch.cuda.synchronize()
            ar_time = time.time() - start

            record = {
                "question_id": qid,
                "index": index,
                "image_path": row["image_path"],
                "image_sha256": file_sha256(row["image_path"]),
                "prompt_len": prompt_len,
                "input_ids_sha256": ids_sha256(input_ids),
                "image_pad_count": int((input_ids == image_token_id).sum()),
                "prompt_limit_ok": prompt_len <= PROMPT_LIMIT,
                "length_budget": budget,
                "length_budget_ok": budget < max_length_stop,
                "bundle": f"bundle/{qid}.safetensors",
                "bundle_sha256": file_sha256(bundle_path),
                "first_call": f"first_call/{qid}.safetensors",
                "first_call_sha256": file_sha256(first_call_path),
                "first_call_info": recorder.info,
                "kept_len": int(tensors["prune.out.input_ids"].shape[1]),
                "add": recorder.info["topk.add"],
                "fp32_add": fp32_add,
                "eos_token_id": tokenizer.eos_token_id,
                "hivis": {
                    "output_ids": output_ids[0, prompt_len:].tolist(),
                    "new_token": int(new_token),
                    "idx": int(idx),
                    "accept_lengths": [int(a) for a in accept_lengths],
                    "wall_time": spec_time,
                },
                "ar": {
                    "output_ids": ar_ids[0, prompt_len:].tolist(),
                    "new_token": int(ar_new_token),
                    "idx": int(ar_idx),
                    "init_time": float(ar_init_time),
                    "wall_time": ar_time,
                },
            }
            with open(out_jsonl, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            mean = sum(accept_lengths) / max(len(accept_lengths), 1)
            print(
                f"[{index + 1}/{len(rows)}] {qid}: prompt {prompt_len}, kept {record['kept_len']}, "
                f"add {record['add']}, HiViS {len(record['hivis']['output_ids'])} tokens in "
                f"{len(accept_lengths)} iterations (mean accepted {mean:.3f}), "
                f"AR {len(record['ar']['output_ids'])} tokens",
                flush=True,
            )

    env["finished_utc"] = utc_now()
    out_env.write_text(json.dumps(env, indent=2) + "\n")
    print(f"HiViS side done: {out_jsonl}")


if __name__ == "__main__":
    main()
