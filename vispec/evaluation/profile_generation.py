"""Profile where Qwen2.5-VL decode time goes, for autoregressive and ViSpec decoding.

The run has two phases. First, before any profiler starts in the process, each sample runs 3
untimed warmups and then one unprofiled run timed exactly as ``evaluate.py`` times its answers.
Once CUPTI is initialized it keeps adding host time to every kernel launch after the profiler
stops, so the timed runs must come first. Second, each sample runs 3 untimed warmups and then a
fourth run under ``torch.profiler`` (CPU and CUDA activities, ``record_shapes=False``), then a
fifth, unprofiled run with CUPTI still initialized. That fifth run repeats the same GPU work with
more host time per launch, so it shows how step time responds to host cost. The profiled run is
split into ``record_function`` ranges:

- ``prefill``: from the start of generation to the first target token (TTFT);
- ``draft_init`` (ViSpec only): from the first token to the end of ``initialize_tree``, which
  holds the draft prefill and the first draft tree;
- ``decode``: the decode loop. For ViSpec it starts where ``specgenerate`` starts its
  ``decode_time`` clock, so it holds exactly the verification iterations.

The device is synchronized at every range boundary, so a GPU activity belongs to the range in
which it starts. Kernel durations come from CUPTI and are barely affected by the profiler, but
host-side times are inflated by it, so ratios (GPU-busy fraction, launches per step) are the
primary evidence. ``estimated_unprofiled_gpu_busy_fraction`` divides the profiled decode kernel
time by the unprofiled decode time of the same sample.

Usage:
python -m vispec.evaluation.profile_generation --method autoregressive --benchmark textvqa \
    --prompt-style long --base-model-path models/Qwen2.5-VL-3B-Instruct \
    --spec-model-path models/ViSpec-Qwen2.5-VL-3B-Instruct --data-dir bench_data \
    --num-samples 3 --output-dir out/profile/autoregressive_textvqa
"""

import argparse
import contextlib
import gzip
import json
import shutil
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch.autograd import DeviceType
from torch.profiler import ProfilerActivity, profile, record_function

from ..model import spec_model_ours
from .benchmarks import build_inputs, load_samples
from .evaluate import (
    WARMUPS,
    add_generation_arguments,
    add_model_arguments,
    check_arguments,
    generate,
    load_model,
    warm_up,
    write_run_runtime,
)
from .measurement import GenerationMeasurement, kv_bytes_per_token

SUMMARY_NAME = "profile_summary.json"
TRACE_NAME = "trace_sample1.json.gz"
PHASES = ("prefill", "draft_init", "decode")
TOP_N = 15
LAUNCH_CALLS = {
    "cudaLaunchKernel",
    "cudaLaunchKernelExC",
    "cudaLaunchCooperativeKernel",
    "cuLaunchKernel",
    "cuLaunchKernelEx",
}
SYNC_CALLS = {"cudaStreamSynchronize", "cudaDeviceSynchronize", "cudaEventSynchronize"}
COPY_CALLS = {"cudaMemcpyAsync", "cudaMemcpy"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Profile Qwen2.5-VL decoding with torch.profiler (CPU and CUDA)."
    )
    add_model_arguments(parser)
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Fresh folder for profile_summary.json and the sample-1 Chrome trace.",
    )
    add_generation_arguments(parser)
    parser.set_defaults(num_samples=3)
    args = parser.parse_args(argv)
    check_arguments(parser, args)
    return args


class PhaseRanges:
    """Back-to-back ``record_function`` ranges with a device sync at each boundary."""

    def __init__(self):
        self.current = None

    def switch(self, name):
        torch.cuda.synchronize()
        if self.current is not None:
            self.current.__exit__(None, None, None)
            self.current = None
        if name is not None:
            self.current = record_function(name)
            self.current.__enter__()


@contextlib.contextmanager
def decode_starts_after_tree_init(phases):
    """Open the ``decode`` range when ``initialize_tree`` returns inside ``specgenerate``.

    ``specgenerate`` looks ``initialize_tree`` up in its module globals, so a temporary
    wrapper marks the boundary without editing the model code. The original is restored.
    """
    original = spec_model_ours.initialize_tree

    def wrapped(*args, **kwargs):
        result = original(*args, **kwargs)
        phases.switch("decode")
        return result

    spec_model_ours.initialize_tree = wrapped
    try:
        yield
    finally:
        spec_model_ours.initialize_tree = original


def decode_steps(method, idx, generation_stats):
    """Decode forward passes (AR) or verification iterations (ViSpec) inside ``decode``."""
    if method == "vispec":
        return int(generation_stats["iterations"])
    return int(idx)


def union_ns(intervals):
    total = 0
    end = None
    for start, stop in sorted(intervals):
        if end is None or start > end:
            total += stop - start
            end = stop
        elif stop > end:
            total += stop - end
            end = stop
    return total


def collect_events(prof):
    """Split raw Kineto events into host ops, CUDA runtime calls, phase ranges and GPU work."""
    raw = prof.profiler.kineto_results.events()
    host, ranges, gpu = [], {}, []
    op_thread, op_name = {}, {}
    for event in raw:
        name = event.name()
        start, end = event.start_ns(), event.end_ns()
        device = event.device_type()
        if device == DeviceType.CPU:
            if name in PHASES:
                ranges[name] = (start, end, event.start_thread_id())
            elif not event.is_user_annotation():
                runtime = name.startswith("cu")
                host.append(
                    [name, start, end, event.start_thread_id(), runtime,
                     event.linked_correlation_id()]
                )
                if not runtime:
                    op_thread[event.correlation_id()] = event.start_thread_id()
                    op_name[event.correlation_id()] = name
        elif device == DeviceType.CUDA and not event.is_user_annotation():
            kind = (
                "memcpy" if name.startswith("Memcpy")
                else "memset" if name.startswith("Memset")
                else "kernel"
            )
            gpu.append((name, start, end, kind, event.linked_correlation_id()))
    # CUDA runtime calls can carry a different thread id than the op that issued them;
    # attribute each to its op's thread, as torch.autograd.profiler does.
    for item in host:
        if item[4]:
            item[3] = op_thread.get(item[5], item[3])
    return {
        "event_count": len(raw),
        "host": host,
        "ranges": ranges,
        "gpu": gpu,
        "op_name": op_name,
    }


def window_breakdown(data, phase):
    """Totals for one phase range plus per-name self-time tables (all times in ns)."""
    start, end, thread = data["ranges"][phase]
    out = {"wall_ns": end - start}

    gpu = [g for g in data["gpu"] if start <= g[1] < end]
    for kind in ("kernel", "memcpy", "memset"):
        items = [g for g in gpu if g[3] == kind]
        out[f"{kind}_count"] = len(items)
        out[f"{kind}_ns"] = sum(g[2] - g[1] for g in items)
    out["gpu_active_ns"] = union_ns((g[1], min(g[2], end)) for g in gpu)

    host = [h for h in data["host"] if h[3] == thread and h[1] >= start and h[2] <= end]
    out["other_thread_host_events"] = sum(
        1 for h in data["host"] if h[3] != thread and h[1] >= start and h[2] <= end
    )
    out["host_in_torch_ns"] = union_ns((h[1], h[2]) for h in host)
    out["torch_op_calls"] = sum(1 for h in host if not h[4])
    for label, names in (("launch", LAUNCH_CALLS), ("sync", SYNC_CALLS), ("copy", COPY_CALLS)):
        calls = [h for h in host if h[4] and h[0] in names]
        out[f"{label}_calls"] = len(calls)
        out[f"{label}_call_ns"] = sum(h[2] - h[1] for h in calls)

    # Self time: an event's duration minus the part of it covered by its direct children.
    host.sort(key=lambda h: (h[1], -h[2]))
    self_ns = [h[2] - h[1] for h in host]
    stack = []
    for i, (_, s, e, *_rest) in enumerate(host):
        while stack and host[stack[-1]][2] <= s:
            stack.pop()
        if stack:
            parent = stack[-1]
            self_ns[parent] -= min(e, host[parent][2]) - s
        stack.append(i)
    cpu_table = defaultdict(lambda: [0, 0])
    for h, value in zip(host, self_ns):
        cpu_table[h[0]][0] += value
        cpu_table[h[0]][1] += 1

    op_table = defaultdict(lambda: [0, 0])
    kernel_table = defaultdict(lambda: [0, 0])
    for name, s, e, _kind, corr in gpu:
        owner = data["op_name"].get(corr, "(unattributed)")
        op_table[owner][0] += e - s
        op_table[owner][1] += 1
        kernel_table[name][0] += e - s
        kernel_table[name][1] += 1
    tables = {"self_cpu": cpu_table, "self_cuda_by_op": op_table, "cuda_kernels": kernel_table}
    return out, tables


def window_report(totals, steps=None):
    """Milliseconds, fractions and (for decode) per-step values from summed ns totals."""
    wall = totals["wall_ns"]
    ms = lambda ns: ns / 1e6  # noqa: E731
    report = {
        "wall_ms": ms(wall),
        "kernel_count": totals["kernel_count"],
        "kernel_ms": ms(totals["kernel_ns"]),
        "memcpy_count": totals["memcpy_count"],
        "memcpy_ms": ms(totals["memcpy_ns"]),
        "memset_count": totals["memset_count"],
        "memset_ms": ms(totals["memset_ns"]),
        "gpu_active_ms": ms(totals["gpu_active_ns"]),
        "gpu_busy_fraction": totals["kernel_ns"] / wall if wall else None,
        "gpu_active_fraction": totals["gpu_active_ns"] / wall if wall else None,
        "host_in_torch_ms": ms(totals["host_in_torch_ns"]),
        "host_in_torch_fraction": totals["host_in_torch_ns"] / wall if wall else None,
        "host_sync_wait_ms": ms(totals["sync_call_ns"]),
        "host_sync_wait_fraction": totals["sync_call_ns"] / wall if wall else None,
        "host_outside_torch_ms": ms(wall - totals["host_in_torch_ns"]),
        "torch_op_calls": totals["torch_op_calls"],
        "launch_calls": totals["launch_calls"],
        "launch_call_ms": ms(totals["launch_call_ns"]),
        "sync_calls": totals["sync_calls"],
        "copy_calls": totals["copy_calls"],
        "copy_call_ms": ms(totals["copy_call_ns"]),
        "other_thread_host_events": totals["other_thread_host_events"],
    }
    if steps:
        report["steps"] = steps
        report["per_step"] = {
            "wall_ms": ms(wall) / steps,
            "kernel_ms": ms(totals["kernel_ns"]) / steps,
            "kernel_launches": totals["kernel_count"] / steps,
            "host_in_torch_ms": ms(totals["host_in_torch_ns"]) / steps,
            "host_active_in_torch_ms": ms(totals["host_in_torch_ns"] - totals["sync_call_ns"])
            / steps,
            "host_sync_wait_ms": ms(totals["sync_call_ns"]) / steps,
            "host_outside_torch_ms": ms(wall - totals["host_in_torch_ns"]) / steps,
            "torch_op_calls": totals["torch_op_calls"] / steps,
            "sync_calls": totals["sync_calls"] / steps,
        }
    return report


def top_table(table, steps, total_ns):
    rows = sorted(table.items(), key=lambda item: item[1][0], reverse=True)[:TOP_N]
    return [
        {
            "name": name,
            "ms": value / 1e6,
            "calls": calls,
            "ms_per_step": value / 1e6 / steps if steps else None,
            "share": value / total_ns if total_ns else None,
        }
        for name, (value, calls) in rows
    ]


def top_tables(tables, steps, totals):
    host_total = sum(v[0] for v in tables["self_cpu"].values())
    return {
        "top_self_cpu": top_table(tables["self_cpu"], steps, host_total),
        "top_self_cuda_by_op": top_table(tables["self_cuda_by_op"], steps, totals["kernel_ns"]
                                         + totals["memcpy_ns"] + totals["memset_ns"]),
        "top_cuda_kernels": top_table(tables["cuda_kernels"], steps, totals["kernel_ns"]
                                      + totals["memcpy_ns"] + totals["memset_ns"]),
    }


def add_totals(into, totals):
    for key, value in totals.items():
        into[key] = into.get(key, 0) + value


def add_tables(into, tables):
    for kind, table in tables.items():
        target = into.setdefault(kind, defaultdict(lambda: [0, 0]))
        for name, (value, calls) in table.items():
            target[name][0] += value
            target[name][1] += calls


def profiled_generation(model, args, model_inputs):
    """The profiled run: returns (profiler, output ids, idx, generation stats, wall seconds)."""
    phases = PhaseRanges()
    first = "draft_init" if args.method == "vispec" else "decode"
    marker = (
        decode_starts_after_tree_init(phases)
        if args.method == "vispec"
        else contextlib.nullcontext()
    )
    stats = {}
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False
    ) as prof:
        torch.manual_seed(0)
        phases.switch("prefill")
        started = time.perf_counter()
        with marker:
            output_ids, _, idx, _, _ = generate(
                model,
                args,
                model_inputs,
                on_first_token=lambda token: phases.switch(first),
                generation_stats=stats,
            )
        phases.switch(None)
        wall = time.perf_counter() - started
    return prof, output_ids, idx, stats, wall


def timed_generation(model, args, model_inputs, tokenizer, kv_bpt):
    """One unprofiled run measured as evaluate.py measures each answer."""
    torch.manual_seed(0)
    measurement = GenerationMeasurement(
        model_inputs, phase_memory=True, kv_bytes_per_token=kv_bpt
    )
    stats = {}
    output_ids, new_token, idx, acceptance, decode_time = generate(
        model, args, model_inputs, on_first_token=measurement.mark, generation_stats=stats
    )
    wall = measurement.elapsed()
    output_ids = output_ids[0][model_inputs["input_ids"].shape[1]:]
    record = measurement.finish(output_ids, tokenizer, new_token, stats)
    return record, measurement.seconds, wall, decode_time, idx, stats, acceptance


def init_cupti(device):
    """Absorb CUPTI's one-time start-up cost before the first profiled sample."""
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]):
        torch.ones(8, device=device).add_(1)
        torch.cuda.synchronize()


@torch.inference_mode()
def run(args):
    output_dir = Path(args.output_dir)
    summary_path = output_dir / SUMMARY_NAME
    if summary_path.exists():
        raise FileExistsError(f"{summary_path} already exists; use a fresh --output-dir")
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_samples(args.benchmark, Path(args.data_dir), args.num_samples)
    processor, model, tokenizer = load_model(args)
    write_run_runtime(summary_path, model, args)
    kv_bpt = kv_bytes_per_token(model.base_model.config, model.base_model.dtype)
    device = str(model.base_model.device)
    print(
        f"profile method={args.method} benchmark={args.benchmark} "
        f"prompt_style={args.prompt_style} samples={len(samples)} device={device}"
    )
    inputs = [
        build_inputs(processor, args.benchmark, sample, args.prompt_style, device=device)
        for sample in samples
    ]

    # Phase 1: unprofiled timing, before any profiler starts in this process. Once CUPTI is
    # initialized it keeps adding host time to every kernel launch after the profiler stops
    # (about 10 ms per decode step in job 393027), so timed runs must come first.
    timed = []
    for model_inputs in inputs:
        warm_up(model, args, model_inputs)
        timed.append(timed_generation(model, args, model_inputs, tokenizer, kv_bpt))
    print(f"Unprofiled timing done for {len(timed)} samples", flush=True)

    # Phase 2: profiling.
    init_cupti(device)
    pooled = {phase: {} for phase in PHASES}
    pooled_tables = {}
    pooled_steps = 0
    pooled_unprofiled = defaultdict(float)
    records = []
    for index, (sample, model_inputs) in enumerate(zip(samples, inputs), start=1):
        input_len = model_inputs["input_ids"].shape[1]
        warm_up(model, args, model_inputs)

        prof, prof_ids, prof_idx, prof_stats, prof_wall = profiled_generation(
            model, args, model_inputs
        )
        prof_ids = prof_ids[0][input_len:].tolist()
        analysis_started = time.perf_counter()
        if index == 1:
            plain = output_dir / "trace_sample1.json"
            prof.export_chrome_trace(str(plain))
            with open(plain, "rb") as fin, gzip.open(output_dir / TRACE_NAME, "wb") as fout:
                shutil.copyfileobj(fin, fout)
            plain.unlink()
        data = collect_events(prof)
        del prof
        steps = decode_steps(args.method, prof_idx, prof_stats)
        windows = {}
        decode_totals = decode_tables = None
        for phase in PHASES:
            if phase not in data["ranges"]:
                continue
            totals, tables = window_breakdown(data, phase)
            windows[phase] = window_report(totals, steps if phase == "decode" else None)
            add_totals(pooled[phase], totals)
            if phase == "decode":
                decode_totals, decode_tables = totals, tables
                add_tables(pooled_tables, tables)
        if decode_totals is None or decode_totals["kernel_count"] == 0:
            raise RuntimeError(
                "The decode range recorded no CUDA kernels; is CUPTI available to torch.profiler?"
            )
        analysis_seconds = time.perf_counter() - analysis_started
        event_count = data["event_count"]
        del data

        # Same generation again, unprofiled but after CUPTI has started: identical GPU work
        # with extra host time per launch, which shows how step time responds to host cost.
        post = timed_generation(model, args, model_inputs, tokenizer, kv_bpt)
        post_steps = decode_steps(args.method, post[4], post[5])
        post_profiler = {
            "decode_ms": post[3] * 1e3,
            "decode_ms_per_step": post[3] * 1e3 / post_steps if post_steps else None,
            "steps": post_steps,
            "output_matches_unprofiled": post[0]["output_token_ids"]
            == timed[index - 1][0]["output_token_ids"],
        }

        record, ttft, wall, decode_time, idx, stats, acceptance = timed[index - 1]
        tokens = record["output_tokens"]
        timed_steps = decode_steps(args.method, idx, stats)
        unprofiled = {
            "wall_ms": wall * 1e3,
            "ttft_ms": ttft * 1e3,
            "decode_ms": decode_time * 1e3,
            "between_ttft_and_decode_ms": (wall - ttft - decode_time) * 1e3,
            "tpot_ms": (wall - ttft) * 1e3 / (tokens - 1) if tokens > 1 else None,
            "decode_ms_per_step": decode_time * 1e3 / timed_steps if timed_steps else None,
            "steps": timed_steps,
        }
        decode = windows["decode"]
        decode["estimated_unprofiled_gpu_busy_fraction"] = (
            decode["kernel_ms"] / unprofiled["decode_ms"] if decode_time > 0 else None
        )
        sample_record = {
            "index": index,
            "question_id": sample.question_id,
            "input_tokens": input_len,
            "output_tokens": tokens,
            "stop_reason": record["stop_reason"],
            "iterations": record["iterations"],
            "decode_steps": steps,
            "accepted_draft_tokens_mean": sum(acceptance) / len(acceptance)
            if acceptance
            else None,
            "profiled_output_matches_unprofiled": prof_ids == record["output_token_ids"],
            "unprofiled": unprofiled,
            "post_profiler_unprofiled": post_profiler,
            "profiled": {
                "wall_ms": prof_wall * 1e3,
                "event_count": event_count,
                "analysis_seconds": analysis_seconds,
                "windows": windows,
            },
            "decode_tables": top_tables(decode_tables, steps, decode_totals),
        }
        records.append(sample_record)
        pooled_steps += steps
        pooled_unprofiled["decode_ms"] += unprofiled["decode_ms"]
        pooled_unprofiled["steps"] += timed_steps
        pooled_unprofiled["post_decode_ms"] += post_profiler["decode_ms"]
        pooled_unprofiled["post_steps"] += post_steps
        pooled_unprofiled["after_ttft_ms"] += unprofiled["wall_ms"] - unprofiled["ttft_ms"]
        pooled_unprofiled["tokens_after_first"] += tokens - 1
        pooled_unprofiled["wall_ms"] += unprofiled["wall_ms"]
        per_step = decode["per_step"]
        print(
            f"sample {index} qid={sample.question_id} tokens={tokens} steps={steps} "
            f"match={sample_record['profiled_output_matches_unprofiled']} "
            f"gpu_busy={decode['gpu_busy_fraction']:.3f} "
            f"est_unprofiled_busy={decode['estimated_unprofiled_gpu_busy_fraction']:.3f} "
            f"launches/step={per_step['kernel_launches']:.1f} "
            f"kernel_ms/step={per_step['kernel_ms']:.2f} "
            f"host_in_torch_ms/step={per_step['host_in_torch_ms']:.2f} "
            f"sync_wait_ms/step={per_step['host_sync_wait_ms']:.2f} "
            f"profiled_wall_ms/step={per_step['wall_ms']:.2f} "
            f"unprofiled_decode_ms/step={unprofiled['decode_ms_per_step']:.2f} "
            f"post_profiler_decode_ms/step={post_profiler['decode_ms_per_step']:.2f} "
            f"tpot_ms={unprofiled['tpot_ms']:.2f} analysis_s={analysis_seconds:.1f}",
            flush=True,
        )

    aggregate_windows = {
        phase: window_report(totals, pooled_steps if phase == "decode" else None)
        for phase, totals in pooled.items()
        if totals
    }
    aggregate_windows["decode"]["estimated_unprofiled_gpu_busy_fraction"] = (
        aggregate_windows["decode"]["kernel_ms"] / pooled_unprofiled["decode_ms"]
    )
    summary = {
        "schema_version": 1,
        "method": args.method,
        "benchmark": args.benchmark,
        "prompt_style": args.prompt_style,
        "generation_args": vars(args),
        "protocol": {
            "warmups_per_sample": WARMUPS,
            "timed_run": "phase 1: per sample, 3 warmups then 1 unprofiled run timed as "
            "evaluate.py, before any profiler starts in the process",
            "profiled_run": "phase 2: after CUPTI start-up, per sample, 3 warmups then a "
            "fourth run under torch.profiler",
            "post_profiler_run": "phase 2: a fifth, unprofiled run of the same sample after "
            "the profiled run, with CUPTI still initialized in the process",
            "profiler": {
                "activities": ["CPU", "CUDA"],
                "record_shapes": False,
                "profile_memory": False,
                "with_stack": False,
            },
            "phases": list(PHASES),
            "step_unit": "verification iteration" if args.method == "vispec"
            else "decode forward pass (one token)",
        },
        "trace": TRACE_NAME,
        "runtime": summary_path.with_suffix(".runtime.json").name,
        "samples": records,
        "aggregate": {
            "samples": len(records),
            "all_profiled_outputs_match": all(
                r["profiled_output_matches_unprofiled"] for r in records
            ),
            "all_post_profiler_outputs_match": all(
                r["post_profiler_unprofiled"]["output_matches_unprofiled"] for r in records
            ),
            "profiled_windows": aggregate_windows,
            "unprofiled": {
                "wall_ms_mean": pooled_unprofiled["wall_ms"] / len(records),
                "decode_ms_per_step": pooled_unprofiled["decode_ms"]
                / pooled_unprofiled["steps"],
                "post_profiler_decode_ms_per_step": pooled_unprofiled["post_decode_ms"]
                / pooled_unprofiled["post_steps"],
                "ms_per_token_after_first": pooled_unprofiled["after_ttft_ms"]
                / pooled_unprofiled["tokens_after_first"]
                if pooled_unprofiled["tokens_after_first"]
                else None,
            },
            "decode_tables": top_tables(pooled_tables, pooled_steps, pooled["decode"]),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    run(parse_args())
