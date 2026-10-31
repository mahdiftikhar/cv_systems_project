"""Generation measurements; probes outside elapsed time do not load any models."""

import hashlib
import json
import os
import time
from pathlib import Path

import torch

from .timing import FirstTokenTimer

# Target KV eviction fields, added to a record only when eviction ran. The last two were
# added in Week 3; kv_min_kept_per_image is null for a prompt without image tokens.
EVICTION_FIELDS = (
    "kv_prompt_tokens",
    "kv_kept_prompt_tokens",
    "kv_evicted_tokens",
    "kv_score_window",
    "kv_image_tokens",
    "kv_min_kept_per_image",
)
# Draft-side visual pruning fields (image tokens summed over the prompt's images, and the
# fewest kept for one image), added to a record only when draft pruning ran.
DRAFT_PRUNING_FIELDS = (
    "draft_image_tokens",
    "draft_kept_image_tokens",
    "draft_min_kept_per_image",
)


def _count(value):
    return None if value is None else int(value)


def kv_bytes_per_token(config, dtype) -> int:
    """Bytes of key plus value cache one token occupies across all decoder layers."""
    return (
        config.num_hidden_layers
        * 2
        * config.num_key_value_heads
        * (config.hidden_size // config.num_attention_heads)
        * dtype.itemsize
    )


class GenerationMeasurement(FirstTokenTimer):
    """Measure a generation call after prompt preparation, excluding warmups.

    With ``phase_memory`` the first ``mark`` (the first output token, after the target
    prefill) records each device's allocated-memory peak as the prefill peak and resets the
    peak counters, so ``finish`` can report the remaining (decode) peak separately. In the
    compact KV layout ``mark_compaction`` (right after the target cache is compacted) splits
    the decode phase once more, for the post-compaction peak.
    """

    def __init__(
        self,
        model_inputs=None,
        phase_memory: bool = False,
        kv_bytes_per_token: int | None = None,
    ):
        self.phase_memory = bool(phase_memory)
        self.kv_bytes_per_token = kv_bytes_per_token
        self.prefill_peak_allocated = None
        self.prefill_peak_reserved = None
        self.compaction_peak_allocated = None
        self.compaction_peak_reserved = None
        self.input_fingerprint = None
        self.input_tokens = None
        if model_inputs is not None:
            ids = model_inputs["input_ids"].tolist()
            self.input_tokens = len(ids[0])
            self.input_fingerprint = hashlib.sha256(
                json.dumps(ids).encode()
            ).hexdigest()
        self.devices = list(range(torch.cuda.device_count()))
        for device in self.devices:
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        self.initial_memory = {
            str(device): torch.cuda.memory_allocated(device) for device in self.devices
        }
        super().__init__()

    def mark(self, token):
        first = self.seconds is None
        super().mark(token)
        if self.phase_memory and first:
            # Read after the first-token time is taken, so TTFT excludes these probes.
            self.prefill_peak_allocated = {
                str(device): torch.cuda.max_memory_allocated(device)
                for device in self.devices
            }
            self.prefill_peak_reserved = {
                str(device): torch.cuda.max_memory_reserved(device)
                for device in self.devices
            }
            for device in self.devices:
                torch.cuda.reset_peak_memory_stats(device)

    def mark_compaction(self):
        """Split the decode-phase peak at the target KV compaction (``on_kv_compacted``).

        Keeps each device's allocated and reserved peaks since ``mark`` (they include the
        compaction transient), then resets the peak counters, so ``finish`` reads the
        post-compaction peak on its own. ``finish`` folds the kept peaks back into
        ``decode_peak_allocated_bytes`` and the whole-call peaks, which therefore keep their
        meaning. The allocator counters are host-side, so no synchronize is needed. Does
        nothing without ``phase_memory`` or before the first ``mark``.
        """
        if not self.phase_memory or self.prefill_peak_allocated is None:
            return
        allocated = {
            str(device): torch.cuda.max_memory_allocated(device) for device in self.devices
        }
        reserved = {
            str(device): torch.cuda.max_memory_reserved(device) for device in self.devices
        }
        if self.compaction_peak_allocated is not None:
            # A second call: the earlier window still belongs to the decode phase.
            for device in allocated:
                allocated[device] = max(allocated[device], self.compaction_peak_allocated[device])
                reserved[device] = max(reserved[device], self.compaction_peak_reserved[device])
        self.compaction_peak_allocated = allocated
        self.compaction_peak_reserved = reserved
        for device in self.devices:
            torch.cuda.reset_peak_memory_stats(device)

    def elapsed(self):
        for device in self.devices:
            torch.cuda.synchronize(device)
        self.duration = time.perf_counter() - self.started
        return self.duration

    def finish(
        self,
        output_ids,
        tokenizer,
        reported_tokens,
        stop,
        kv_tokens: int | None = None,
        kv_eviction: dict | None = None,
        draft_pruning: dict | None = None,
    ):
        """Read counters after elapsed() so reporting overhead is excluded.

        ``kv_tokens`` is the target KV cache length at finish. The phase and live-KV fields
        are added only when the caller opts in (``phase_memory``, ``kv_bytes_per_token`` or
        ``kv_tokens``); otherwise the record is unchanged. ``kv_live_bytes`` covers every
        layer, so on a multi-GPU split each device entry repeats the whole-model figure.
        After ``mark_compaction`` each device entry also has
        ``post_compaction_peak_allocated_bytes``, the peak from the compaction to the end of
        the call; ``decode_peak_allocated_bytes`` stays the peak of the whole decode phase.
        ``kv_eviction`` (target KV eviction runs only) adds the top-level fields
        ``kv_prompt_tokens``, ``kv_kept_prompt_tokens``, ``kv_evicted_tokens``,
        ``kv_score_window`` (prompt tokens after the last image token; the attention policy
        scores over them), ``kv_image_tokens`` (image tokens before eviction) and
        ``kv_min_kept_per_image`` (fewest kept for one image). ``draft_pruning`` (draft
        pruning runs only) then adds ``draft_image_tokens`` and ``draft_kept_image_tokens``,
        the image tokens the draft's image adaptor had and saw, summed over the prompt's
        images, and ``draft_min_kept_per_image``. Without either the record is unchanged.
        """
        ids = output_ids.tolist()
        eos = tokenizer.eos_token_id
        first_eos = ids.index(eos) if eos is not None and eos in ids else None
        special = set(tokenizer.all_special_ids)
        visible_ids = ids if first_eos is None else ids[:first_eos]
        memory = {
            str(device): {
                "initial_allocated_bytes": self.initial_memory[str(device)],
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            }
            for device in self.devices
        }
        if self.phase_memory:
            for device, entry in memory.items():
                if self.prefill_peak_allocated is None:
                    # mark() never ran, so the counters were never split by phase.
                    entry["prefill_peak_allocated_bytes"] = None
                    entry["decode_peak_allocated_bytes"] = None
                    continue
                prefill = self.prefill_peak_allocated[device]
                decode = entry["peak_allocated_bytes"]
                reserved = entry["peak_reserved_bytes"]
                post_compaction = None
                if self.compaction_peak_allocated is not None:
                    # The counters were reset at the compaction: what they hold now is the
                    # post-compaction peak, and the decode peak is the larger of both windows.
                    post_compaction = decode
                    decode = max(self.compaction_peak_allocated[device], post_compaction)
                    reserved = max(self.compaction_peak_reserved[device], reserved)
                entry["prefill_peak_allocated_bytes"] = prefill
                entry["decode_peak_allocated_bytes"] = decode
                if post_compaction is not None:
                    entry["post_compaction_peak_allocated_bytes"] = post_compaction
                # Whole-generation peaks, as without phase_memory.
                entry["peak_allocated_bytes"] = max(prefill, decode)
                entry["peak_reserved_bytes"] = max(self.prefill_peak_reserved[device], reserved)
        if (
            self.phase_memory
            or self.kv_bytes_per_token is not None
            or kv_tokens is not None
        ):
            live = (
                None
                if kv_tokens is None or self.kv_bytes_per_token is None
                else int(kv_tokens) * int(self.kv_bytes_per_token)
            )
            for entry in memory.values():
                entry["kv_tokens"] = None if kv_tokens is None else int(kv_tokens)
                entry["kv_live_bytes"] = live
        record = {
            "schema_version": 1,
            "input_ids_sha256": self.input_fingerprint,
            "input_ids_tokens": self.input_tokens,
            "output_token_ids": ids,
            "eos_token_id": eos,
            "output_tokens": len(ids),
            "reported_new_tokens": int(reported_tokens),
            "token_count_matches": len(ids) == int(reported_tokens),
            "answer_tokens": sum(token not in special for token in visible_ids),
            "post_eos_tokens": 0 if first_eos is None else len(ids) - first_eos - 1,
            "stop_reason": stop.get("stop_reason"),
            "truncated": stop.get("stop_reason") in {"token_limit", "iteration_limit"}
            if stop
            else None,
            "generation_limits": stop.get("generation_limits"),
            "iterations": stop.get("iterations"),
            "gpu_memory": memory,
        }
        if kv_eviction is not None:
            for field in EVICTION_FIELDS:
                record[field] = _count(kv_eviction[field])
        if draft_pruning is not None:
            for field in DRAFT_PRUNING_FIELDS:
                record[field] = _count(draft_pruning[field])
        return record


def references(row):
    value = row.get("answers", row.get("answer", row.get("label")))
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return value
    return []


def write_manifest(answer_file, data, benchmark):
    questions = [
        {
            "question_id": str(
                row[
                    {
                        "textvqa": "qid",
                        "gqa": "question_id",
                        "mmvet": "id",
                        "coco_caption": "question_id",
                    }[benchmark]
                ]
            ),
            "reference_answers": references(row),
        }
        for row in data
    ]
    path = Path(answer_file).with_suffix(".manifest.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"benchmark": benchmark, "questions": questions}, indent=2) + "\n"
    )


def write_runtime(answer_file, model):
    devices = []
    for device in range(torch.cuda.device_count()):
        prop = torch.cuda.get_device_properties(device)
        devices.append(
            {
                "index": device,
                "name": prop.name,
                "uuid": str(getattr(prop, "uuid", "unavailable")),
                "total_memory_bytes": prop.total_memory,
            }
        )
    allocation_keys = (
        "CUDA_VISIBLE_DEVICES",
        "SLURM_CPUS_PER_TASK",
        "SLURM_CPUS_ON_NODE",
        "SLURM_MEM_PER_NODE",
        "SLURM_MEM_PER_CPU",
        "SLURM_GPUS",
        "SLURM_GPUS_PER_TASK",
        "SLURM_GPUS_ON_NODE",
    )
    payload = {
        "devices": devices,
        "parameter_dtypes": sorted({str(param.dtype) for param in model.parameters()}),
        "allocation": {key: os.environ.get(key) for key in allocation_keys},
        "cpu_affinity": sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
        "base_model_revision": getattr(model.base_model.config, "_commit_hash", None),
    }
    Path(answer_file).with_suffix(".runtime.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
