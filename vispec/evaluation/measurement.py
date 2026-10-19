"""Generation measurements; probes outside elapsed time do not load any models."""

import hashlib
import json
import os
import time
from pathlib import Path

import torch

from .timing import FirstTokenTimer


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
    peak counters, so ``finish`` can report the remaining (decode) peak separately.
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

    def elapsed(self):
        for device in self.devices:
            torch.cuda.synchronize(device)
        self.duration = time.perf_counter() - self.started
        return self.duration

    def finish(
        self, output_ids, tokenizer, reported_tokens, stop, kv_tokens: int | None = None
    ):
        """Read counters after elapsed() so reporting overhead is excluded.

        ``kv_tokens`` is the target KV cache length at finish. The phase and live-KV fields
        are added only when the caller opts in (``phase_memory``, ``kv_bytes_per_token`` or
        ``kv_tokens``); otherwise the record is unchanged. ``kv_live_bytes`` covers every
        layer, so on a multi-GPU split each device entry repeats the whole-model figure.
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
                entry["prefill_peak_allocated_bytes"] = prefill
                entry["decode_peak_allocated_bytes"] = decode
                # Whole-generation peaks, as without phase_memory.
                entry["peak_allocated_bytes"] = max(prefill, decode)
                entry["peak_reserved_bytes"] = max(
                    self.prefill_peak_reserved[device], entry["peak_reserved_bytes"]
                )
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
        return {
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
