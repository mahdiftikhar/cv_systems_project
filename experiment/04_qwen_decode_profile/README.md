# 04 — Qwen2.5-VL decode profile

## Purpose and hypothesis

This experiment profiles where decode time goes for Qwen2.5-VL-3B-Instruct under standard
autoregressive (AR) decoding and under ViSpec speculative decoding. It is the first Week 2 task.
Its findings decide whether the later visual-sparsity work should target latency or memory.

The Week 1 runs ([02](../02_qwen_autoregressive/README.md) and [03](../03_qwen_vispec/README.md))
measured about 37.5 to 38.4 ms per output token for AR decoding and about 50 ms per ViSpec
verification iteration. Reading the bf16 language-model weights once per token at the A6000's
peak bandwidth would take about 8 to 9 ms. The host is a Xeon Silver 4114 at 2.2 GHz with 4
allocated cores, and AR decoding synchronizes once per step (`input_id.item()` in
`vispec/model/autoregressive.py`). The hypothesis is therefore that Week 1 decoding is
host-bound: the GPU waits for the CPU to issue kernels, so the GPU is idle for much of each step.
If this holds, removing visual tokens from the KV cache would shorten GPU work that is not on
the critical path, and latency would barely move.

Status: **measured** (Slurm job 393060, exit code 0). The GPU executes kernels for at most about
39% of an unprofiled AR step and 50% of a ViSpec iteration, and unprofiled step time works out to
21 to 23 µs per kernel launched. From this we infer that decoding is host-bound for both methods,
and that the per-step synchronization is not where the time goes. The results and the inferences
are [below](#results).

## Changes from the baseline

This experiment answers no benchmark questions and computes no quality scores. It runs
`python -m vispec.evaluation.profile_generation` instead of the evaluator. The profiler reuses
the evaluator's model loading, input building and generation call (`load_model`, `build_inputs`
and `generate` from `vispec/evaluation/evaluate.py`), so both methods decode through exactly the
code paths that 02 and 03 measured.

To share that code, the argument definitions, model loading, runtime-file writing and warmup
loop were factored out of `evaluate.py` into importable functions. Its CLI and outputs are
unchanged: `--help` is byte-identical, and the same job re-runs the first 2 TextVQA questions
through `evaluate.py` and compares their token IDs with the Week 1 03 run (see the identity
check below).

## Models

| Role | Model |
| --- | --- |
| Target | `Qwen/Qwen2.5-VL-3B-Instruct`, local copy `models/Qwen2.5-VL-3B-Instruct` |
| Draft | `JLKang/ViSpec-Qwen2.5-VL-3B-Instruct`, local copy `models/ViSpec-Qwen2.5-VL-3B-Instruct`. It proposes tokens for ViSpec and is loaded but unused for AR |

## Data, splits and subsets

| Benchmark | Input | Profiled samples |
| --- | --- | --- |
| TextVQA | `bench_data/textvqa/TextVQA_0.5.1_val.json` + `train_images/` | The first 3 questions of the 100-question seed-42 subset used by 02 and 03 |
| COCO Captions | `bench_data/coco/karpathy_test.parquet` + `val2014/` | The first 3 images of the 100-image seed-42 subset used by 02 and 03 |

Each benchmark uses the `long` prompt style, because its answers run to hundreds of tokens and
decoding dominates the wall time. Three samples per benchmark are enough to characterize a
per-step cost that repeats hundreds of times per answer. They are not a benchmark measurement.

## Settings

The settings are the Week 1 settings: greedy decoding (temperature 0.0), bf16, batch size 1, at
most 1,024 new tokens, slim prefill on, a 16,384-token KV capacity, and for ViSpec depth 3, top-k
8, total-token 30 and `num_q` 2. Image limits are `min_pixels = 256*28*28` and
`max_pixels = 1280*28*28`. The runner fixes all of these. It reads `PROMPT_STYLE` (required),
`BENCH_DIR`, `BASE_MODEL_PATH`, `SPEC_DIR`, `PYTHON_BIN` and `IDENTITY_REFERENCE`.

Each profiler invocation (one method, one benchmark) loads the model once and runs two phases:

1. **Unprofiled timing.** Before any profiler starts in the process, each sample runs three
   untimed warmups, seeded as `evaluate.py` seeds them. It then runs one generation that is timed
   exactly as `evaluate.py` times an answer.
2. **Profiling.** After a one-off CUPTI start-up, each sample again runs three untimed warmups. A
   fourth, untimed run follows under `torch.profiler` with CPU and CUDA activities and
   `record_shapes=False` (no memory profiling and no stacks). A fifth run is unprofiled but
   happens after the profiler has run (the post-profiler run).

The timed runs must come first. Once CUPTI has been initialized in a process, it keeps adding
host time to every kernel launch after the profiler stops. The first launch of this experiment
(job 393027) timed each sample after its profiled run and measured 59.12 ms per ViSpec iteration
for TextVQA question 35785. `evaluate.py` measured 48.97 ms for the same question in the same job,
and 48.93 ms in Week 1. The post-profiler run keeps that effect as a measurement: it repeats the
same GPU work with more host time per launch.

The profiled run is split into `torch.profiler.record_function` ranges, with a device
synchronization at every boundary, so each GPU activity belongs to the range in which it starts:

| Range | AR | ViSpec |
| --- | --- | --- |
| `prefill` | Start of generation to the first token (TTFT) | Start of generation to the first target token (TTFT) |
| `draft_init` | none | First target token to the end of `initialize_tree`: the draft prefill and the first draft tree |
| `decode` | The decode loop, one forward pass per token | The verification loop, exactly the window that `decode_time` measures |

The ViSpec `decode` range opens when `initialize_tree` returns. The profiler marks that point by
wrapping `spec_model_ours.initialize_tree` for the duration of the profiled call only, so the
model code is not edited. A step is one decode forward pass for AR and one verification
iteration for ViSpec.

## Cluster prerequisites

1. Follow the [Ampere setup guide](../AMPERE_SETUP.md). The batch job checks for the pinned
   48 GB `NVIDIA RTX A6000` and exits with status 3 on any other GPU.
2. Prepare assets once with `bash prepare.sh --suite qwen`. The runner repeats the offline check
   (`--check-only --suite qwen`) after the device capture.
3. The identity check reads the Week 1 03 answers for the chosen style from
   `experiment/03_qwen_vispec/runs/`, which Git ignores. Without that folder the check is recorded
   as `skipped_missing_reference`; set `IDENTITY_REFERENCE` to another 03 `test.jsonl` to use it.

## Run command

From the repository root on a login or VS Code shell:

```bash
PROMPT_STYLE=long CLUSTER_PARTITION=preempt SBATCH_EXTRA="--cpus-per-task=4 --mem=22G --time=01:00:00" \
  bash experiment/submit_ampere.sh experiment/04_qwen_decode_profile
```

These are the Week 1 resources (4 CPUs, 22 GB), so the allocation matches 02 and 03, although
the node and card can differ. Inside an
existing one-GPU allocation, run `PROMPT_STYLE=long bash experiment/04_qwen_decode_profile/run.sh`.
That skips the GPU-name check.

## Run artifacts

Each launch creates `runs/<UTC>_<commit12>_<style>_XXXXXX/`. It holds the full commit,
working-tree status and tracked diff, the runner snapshot, a copy of
`profile_generation.py` (the launch diff omits untracked files), packages, the device report
(`device.txt`), the scheduler job ID and timestamps in `metadata.txt`, combined stdout/stderr
(`output.log`) and `exit-code.txt`. There is no report step. The profiles are written to:

```text
profile/<method>_<benchmark>/profile_summary.json          per-sample and pooled metrics
profile/<method>_<benchmark>/profile_summary.runtime.json  GPU UUID, allocation, settings
profile/<method>_<benchmark>/trace_sample1.json.gz         Chrome trace of sample 1's profiled run
identity_check/results/textvqa_test/vispec_qwen/test.jsonl evaluate.py answers for 2 questions
identity_check/identity_check.json                         token-ID comparison with Week 1
```

Open a trace in `chrome://tracing` or [Perfetto](https://ui.perfetto.dev). Generated runs stay
out of Git and are preserved on project storage.

## Metric definitions

All profiled metrics come from the raw Kineto events of the profiled run, restricted to one
range. Host events are those on the thread that opened the range; CUDA runtime calls are
attributed to the thread of the op that issued them.

| Metric | Definition |
| --- | --- |
| Kernel time | Sum of CUDA kernel durations that start in the range. Memcpy and memset are counted separately |
| GPU-busy fraction | Kernel time divided by the range's wall time. Work runs on one stream, so this is the share of time the GPU executes kernels |
| Estimated unprofiled GPU-busy fraction | Profiled decode kernel time divided by the unprofiled run's decode time for the same sample and output |
| Kernel launches per step | Kernels in `decode` divided by decode steps |
| Host time in torch ops | Union of the intervals the host spends inside torch ops or CUDA runtime calls |
| Sync wait | Time inside `cudaStreamSynchronize`, `cudaDeviceSynchronize` and `cudaEventSynchronize`, where the host waits for the GPU |
| Host outside torch | Range wall time minus host time in torch ops: Python and framework glue between ops |
| Top ops | The 15 names with the most self-CPU time (ops and runtime calls), the 15 ops with the most CUDA time from kernels they launched, and the 15 most expensive kernels |
| Unprofiled ms per token | `(latency - TTFT) / (output tokens - 1)` in the phase-1 timed run, the Week 1 TPOT definition |
| Unprofiled decode ms per step | `decode_time` divided by decode steps in the phase-1 timed run |
| Post-profiler decode ms per step | The same quantity in the post-profiler run, with CUPTI initialized |

Self time is an event's duration minus the time covered by its direct children, the same rule
as `torch.profiler`'s self-CPU column. Pooled values sum the numerators and the denominators over
the 3 samples before dividing. The profiled and post-profiler outputs of every sample are checked
against its phase-1 output (`profiled_output_matches_unprofiled` and
`post_profiler_unprofiled.output_matches_unprofiled`). A GPU smoke check on one sample per method
confirmed that the per-range GPU totals equal `torch.profiler`'s own `device_time_total` for the
same range to within 0.01%, and that the wall times are identical.

The profiler records host-side events, so it lengthens host time and leaves kernel durations
almost unchanged. Profiled wall times and the GPU-busy fraction are therefore biased toward a
host-bound reading. Read the ratios and counts first, and use the unprofiled wall times as the
absolute reference. The estimated unprofiled GPU-busy fraction avoids most of that bias, because
its numerator is GPU-timed and its denominator is unprofiled.

## Results

All figures come from one run (Slurm job 393060), `long` style, 3 samples per benchmark, greedy
decoding, bf16, batch size 1, at most 1,024 new tokens, depth 3, top-k 8, total-token 30,
`num_q` 2, slim prefill on and a 16,384-token KV capacity. Values are pooled over the 3 samples
(sums before division). Per-sample values and the top 15 tables are in each
`profile/<method>_<benchmark>/profile_summary.json`. Every profiled and post-profiler output
matched its phase-1 output, and all 12 phase-1 outputs have the same token counts as the Week 1
answers to the same questions.

### Hardware

| Run | Job / partition | Node and host CPU | GPU | UUID | Driver | Allocation | Device report |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Run of record | 393060 / `preempt`, 00:21:28 | `c1-8`, Intel Xeon Silver 4116 @ 2.10 GHz (48 logical CPUs on the host) | NVIDIA RTX A6000 | `GPU-8d51152c-22f1-372d-aef5-063afee0d488` | 615.71.09 | 1 GPU, 4 CPUs (affinity 4, 14, 28, 38), 22,528 MB, 1 h limit | [device.txt](runs/20261019T230855Z_65078a0dbe9b_long_J4OYTT/device.txt) |
| First launch (timing superseded) | 393027 / `preempt`, 00:16:14 | `c2-2`, Intel Xeon Silver 4114 @ 2.20 GHz (40 logical CPUs on the host) | NVIDIA RTX A6000 | `GPU-a2e40ccd-5141-9e51-2b3f-0e9d71fabf73` | 615.71.09 | 1 GPU, 4 CPUs (affinity 9, 12, 29, 32), 22,528 MB, 1 h limit | [device.txt](runs/20261019T223554Z_65078a0dbe9b_long_0i4Hy6/device.txt) |

The run of record used a different node, host CPU and card from Week 1, which ran on `c2-2`
(`GPU-a2e40ccd-…`). Its unprofiled decode times still match Week 1 for the same questions within
1% in three rows and 6% in the fourth (table below). The links point at generated run folders,
which Git ignores; they are preserved on project storage. Device reports are launch snapshots.

### Where decode time goes

A step is one decode forward pass for AR and one verification iteration for ViSpec. Profiled
columns come from the profiled run, and the estimated unprofiled GPU-busy fraction divides that
run's kernel time by the phase-1 decode time.

| Method | Benchmark | Steps (3 samples) | GPU-busy, profiled | GPU-busy, estimated unprofiled | Kernel launches per step | Kernel launches per output token | CUDA kernel time per step (ms) | Host time in torch ops per step (ms, profiled) | Host sync wait per step (ms, profiled) | Host outside torch per step (ms, profiled) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AR | TextVQA | 669 | 0.202 | 0.386 | 1,789 | 1,789 | 14.58 | 50.69 | 0.72 | 21.48 |
| ViSpec | TextVQA | 220 | 0.296 | 0.501 | 2,270 | 719 | 24.95 | 59.92 | 0.94 | 24.46 |
| AR | COCO Captions | 295 | 0.206 | 0.364 | 1,789 | 1,789 | 13.63 | 46.94 | 0.75 | 19.35 |
| ViSpec | COCO Captions | 92 | 0.254 | 0.431 | 2,275 | 669 | 22.50 | 62.49 | 0.91 | 25.98 |

ViSpec commits 3.16 (TextVQA) and 3.40 (COCO) output tokens per iteration in these samples, which
gives the launches per output token. Memcpy and memset add less than 0.001 to the busy fractions.
The host issues about 9,300 torch op calls per AR step and 11,850 per ViSpec iteration, nested
calls included. It synchronizes about 3 times per AR step and 32 times per ViSpec iteration.

### Unprofiled wall times from the same job

| Method | Benchmark | ms per output token | Decode ms per step | Week 1 decode ms per step, same 3 questions | Post-profiler decode ms per step | Profiled decode ms per step |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| AR | TextVQA | 37.82 | 37.82 | 37.69 | 49.76 | 72.17 |
| ViSpec | TextVQA | 15.83 | 49.84 | 49.38 | 59.59 | 84.38 |
| AR | COCO Captions | 37.45 | 37.45 | 37.21 | 45.85 | 66.29 |
| ViSpec | COCO Captions | 15.47 | 52.24 | 55.60 | 63.12 | 88.47 |

The ms per output token column uses the Week 1 TPOT definition; the Week 1 figures come from the
saved 02 and 03 `long` answers. The profiler lengthens a decode step 1.7 to 1.9 times.

### Top ops in the decode range

Self-CPU time (profiled, so inflated by the profiler) and CUDA time by launching op, in ms per
step. The top 15 of each, and the top 15 kernels by name, are in the summaries.

| Rank | AR TextVQA, self-CPU | AR COCO, self-CPU | AR TextVQA, CUDA | AR COCO, CUDA |
| ---: | --- | --- | --- | --- |
| 1 | `cudaLaunchKernel` 12.54 | `cudaLaunchKernel` 11.53 | `aten::mm` 8.88 | `aten::mm` 9.00 |
| 2 | `aten::mm` 3.78 | `aten::mm` 3.55 | `aten::copy_` 1.45 | `aten::addmm` 1.16 |
| 3 | `aten::mul` 3.40 | `aten::mul` 3.21 | `aten::addmm` 1.17 | `aten::copy_` 0.99 |
| 4 | `aten::addmm` 2.93 | `aten::addmm` 2.73 | `aten::_flash_attention_forward` 1.01 | `aten::mul` 0.63 |
| 5 | `aten::copy_` 2.36 | `aten::copy_` 2.23 | `aten::mul` 0.64 | `aten::cat` 0.44 |

| Rank | ViSpec TextVQA, self-CPU | ViSpec COCO, self-CPU | ViSpec TextVQA, CUDA | ViSpec COCO, CUDA |
| ---: | --- | --- | --- | --- |
| 1 | `cudaLaunchKernel` 14.35 | `cudaLaunchKernel` 15.10 | `aten::mm` 14.62 | `aten::mm` 14.65 |
| 2 | `aten::mm` 4.36 | `aten::mm` 4.60 | `aten::_efficient_attention_forward` 2.97 | `aten::addmm` 1.61 |
| 3 | `aten::mul` 3.58 | `aten::mul` 3.76 | `aten::copy_` 1.84 | `aten::copy_` 1.32 |
| 4 | `aten::addmm` 3.43 | `aten::addmm` 3.57 | `aten::addmm` 1.62 | `aten::_efficient_attention_forward` 1.11 |
| 5 | `aten::copy_` 3.01 | `aten::copy_` 3.16 | `aten::mul` 0.93 | `aten::mul` 0.91 |

`cudaLaunchKernel` takes about a quarter of the host's in-op time under the profiler. CUPTI
instruments that call, so this share overstates its unprofiled cost.

### Reading the profile

The measurements above are direct. The explanations in this section are inferences from them.

**Decode is host-bound.** In an unprofiled AR step the GPU runs kernels for at most about 36 to
39% of the time, and in a ViSpec iteration for 43 to 50%. The host waits on the GPU at
synchronization points for only 0.7 to 0.9 ms per step, about 1% of the profiled decode time, so
the GPU finishes each step's work soon after the host issues the last kernel. Dividing unprofiled
step time by kernels per step gives 20.9 to 23.0 µs of host time per kernel, nearly the same for
both methods and both benchmarks. The AR step issues exactly 1,789 kernels for TextVQA prompts
(1,084 to 1,346 tokens) and COCO prompts (398 tokens), so the host cost per step does not depend on
context length at these sizes. The GPU side is also above the weight-read floor. An AR step's
13.6 to 14.6 ms of kernels is 1.7 to 1.8 times the roughly 8 ms needed to read the weights once,
and the batch-1 GEMMs (`aten::mm`) account for about 9 ms of it. A faster GPU would not remove the
host's cost of issuing the step, which is close to the full 38 ms.

**The per-step `.item()` synchronization is not the bottleneck.** It costs well under 1 ms of
waiting per step. The time goes into issuing about 9,300 op calls and 1,789 kernels per step from
Python on a 2.1 to 2.2 GHz Skylake-SP core.

**ViSpec's advantage follows launches per token.** AR takes 2.39 (TextVQA) and 2.42 (COCO) times
as long per output token as ViSpec. ViSpec needs 2.49 and 2.67 times fewer kernel launches per
token, but only 1.85 and 2.06 times less kernel time per token. Its gain comes from issuing fewer
kernels per committed token rather than from less GPU work.

**Extra host cost lengthens the step.** The post-profiler run repeats the same GPU work and
gives the same outputs, but with CUPTI initialized in the process. It lengthens the decode step by
8.4 (COCO) and 11.9 ms (TextVQA) for AR, and by 10.9 and 9.8 ms for ViSpec, which is 4.3 to 6.7 µs
per kernel launched. A GPU-bound loop would have absorbed that extra host time in idle GPU time.

**What this means for visual sparsity.** Attention over the KV cache is a small part of the
decode-range GPU work. It is 1.01 ms per AR step for TextVQA and 0.43 ms for COCO (7% and 3% of
kernel time), and 2.97 and 1.11 ms per ViSpec iteration. That GPU work is off the critical path,
and evicting visual KV entries does not reduce the number of kernels. Target-side eviction is
therefore unlikely to shorten decode latency measurably in this eager harness. Its measurable
effect should be on memory. Latency would move with fewer kernel launches or cheaper launches,
for example CUDA graphs or fused kernels. Neither was tested here.

### Prefill, for context

The prefill range looks different. For TextVQA, kernels occupy 77% (AR) and 79% (ViSpec) of it,
and the host spends 68 to 72% of it waiting at synchronizations. About 8,600 synchronizations
happen per TextVQA prefill. For COCO's smaller prompts, the busy fraction is 0.58 and 0.57. This
suggests that the GPU, not the host, limits TextVQA prefill. Only the decode range has op tables,
so this profile does not attribute prefill time to ops. Target-side eviction happens after prefill
and cannot change TTFT.

### Identity check and the first launch

The `evaluate.py` refactor kept its outputs. In both launches, `evaluate.py --method vispec
--benchmark textvqa --prompt-style long --num-samples 2` reproduced the Week 1 03 token IDs exactly
for questions 36137 (116 tokens) and 35785 (340 tokens) (`identity_check/identity_check.json`),
and `--help` is byte-identical to the pre-refactor output.

The first launch (job 393027,
[run folder](runs/20261019T223554Z_65078a0dbe9b_long_0i4Hy6/)) timed each sample right after its
profiled run. It ran on the Week 1 node and cores, and it measured 46.70 ms per AR step and 60.52 ms
per ViSpec iteration for TextVQA. Its own `evaluate.py` run measured 48.97 ms for question 35785, and
Week 1 measured 48.93 ms. That led to the two-phase protocol. Its profiled metrics agree with the
run of record on a different card and host CPU: 1,789 and 2,271 launches per step, 14.64 and 25.24 ms
of kernels per step for TextVQA, and GPU-busy fractions of 0.21 to 0.30. Its timing columns are
superseded.

## Limitations

- Three samples per benchmark, one run, in the `long` style only. The profile characterizes
  per-step costs; it is not a benchmark result and has no repeat variability estimate.
- Profiler overhead inflates host times and profiled wall times (see the metric definitions).
  The post-profiler run also shows that a process stays slower after CUPTI starts, so profiling
  and timing must not share a process in that order.
- The run of record used host CPU `c1-8` (Xeon Silver 4116), not the Week 1 host `c2-2` (Xeon
  Silver 4114). In a host-bound loop, step time depends on the host CPU.
- The estimated unprofiled GPU-busy fraction assumes that CUPTI does not change kernel
  durations.
- The ranges synchronize the device at each boundary, as the Week 1 timers already do at the
  same points, so the profiled run adds no synchronization inside the decode loop.
- The profile records time, not cause. Statements about why decoding is host-bound or GPU-bound
  are inferences from these measurements.
- Short-style decoding, Re:Verse page windows and long contexts were not profiled.
