# Experiments

We compare standard autoregressive generation (**00**) with ViSpec speculative
decoding (**01**) using **LLaVA-1.6 Vicuna 7B**. The benchmark suite is
**TextVQA → GQA → MM-Vet**. Runs execute on the UCF cluster.

Standard autoregressive generation produces one token at a time. ViSpec uses a
smaller draft model to propose several tokens, then asks the original model to
verify them together. We want faster completed answers without changing their
content or reducing correctness.

**TextVQA** tests reading text inside images, **GQA** tests visual reasoning, and
**MM-Vet** tests broader open-ended visual understanding. We use 100, 100 and 218
questions, respectively. These results describe those subsets.

## Latest LLaVA results (00 and 01)

[Plots and how to read them](plots/README.md) · [Full metric tables by benchmark](COMPARISON.md).

**ViSpec took about 62% less generation time overall.** This is a promising
timing result, but we have not validated that it preserves the baseline output:
51 of 418 responses differ, and 14 responses per method hit the output limit.

Both methods completed all questions. Times below are mean generation latency
in **seconds per question**. Lower is better.

| Benchmark | Questions per method | 00: Standard | 01: ViSpec | Observed speedup |
| --------- | -------------------: | -----------: | ---------: | ---------------: |
| TextVQA   |                  100 |      9.675 s |    3.365 s |            2.88× |
| GQA       |                  100 |      1.380 s |    0.640 s |            2.16× |
| MM-Vet    |                  218 |      4.109 s |    1.651 s |            2.49× |

### Overall LLaVA results

| Method       | Mean generation time | Mean time to first token |
| ------------ | -------------------: | -----------------------: |
| 00: Standard |              4.788 s |                  0.278 s |
| 01: ViSpec   |              1.819 s |                  0.276 s |

The overall ratio is **2.63×**. Each of the 418 questions contributes equally to
these means. Generation time includes prefill and decoding. It excludes model
loading, prompt preparation, converting token IDs to text, and scoring.

### What the headline metrics mean

**Generation time** measures how long the model takes to finish an answer.
**Time to first token (TTFT)** measures how soon it can begin answering. We
measure both because earlier completion does not necessarily mean an earlier
start. The observed speedup divides standard mean generation time by ViSpec's:
a 2.63× ratio means ViSpec took about 38% as long in this pair.

**Output agreement** checks whether both methods say the same thing.
**Quality** checks their answers against references. We need both measures
because identical answers can still be wrong. A **capped** answer was stopped by
the output limit and may be incomplete. A **pending** result is unavailable,
not zero.

The detailed tables also report the **mean** (average), **median** (middle value)
and **p95** (a threshold covering about 95% of measurements). The
[metric guide](METRICS.md#what-each-metric-means) explains these and the remaining
measurements, including throughput, memory and draft acceptance.

### What still needs work

- **Output agreement:** exact token sequences match on 76/100 TextVQA,
  99/100 GQA, and 192/218 MM-Vet questions: **367/418 overall (87.8%)**.
- **Long answers:** 11 TextVQA and 3 MM-Vet responses per method reached the
  1,024-token cap. Agreement on a capped response only checks its saved prefix.
- **Answer quality:** both methods scored 0% on TextVQA consensus and GQA exact
  match, with 100/100 questions scored for each benchmark. Verbose answers can
  fail strict scoring even when they contain the right answer. MM-Vet judging
  is pending (0/218 scored). Quality scores are not combined across benchmarks.
- **Comparison conditions:** the jobs overlapped on the same node but used
  different physical GPUs and CPU bindings. We have one run per method, so
  repeat variability is unknown.

Next: diagnose the token differences, then repeat with the same GPU and CPU
binding. Resolve answer formatting before claiming preserved answer quality.
The [detailed review](01_baseline_draft/RESULTS_20261017.md) contains every
first token mismatch, memory measurements, artifact checks, and limitations.

## Experiments and run evidence

| Experiment                                              | Purpose                           | Latest status                                    |
| ------------------------------------------------------- | --------------------------------- | ------------------------------------------------ |
| [00: Standard baseline](00_baseline_original/README.md) | Measure autoregressive generation | 418/418 completed                                |
| [01: ViSpec baseline](01_baseline_draft/README.md)      | Measure speculative generation    | 418/418 completed. Output equivalence unresolved |
| [02: Qwen autoregressive](02_qwen_autoregressive/README.md) | Qwen2.5-VL-3B reference decoding | Week 1 measured, `long` and `short`, 400/400 completed |
| [03: Qwen ViSpec](03_qwen_vispec/README.md)             | Qwen2.5-VL-3B speculative decoding | Week 1 measured, 400/400 completed. Output equivalence unresolved (all divergences are bf16 near-ties) |
| [04: Qwen decode profile](04_qwen_decode_profile/README.md) | Where AR and ViSpec decode time goes (`torch.profiler`) | Measured (job 393060), `long`, 3 samples per benchmark. Decode is inferred to be host-bound |
| [05: Qwen draft pruning](05_qwen_draft_pruning/README.md) | ViSpec with a pruned draft image-adaptor input, vs the same arm at 100% | Week 2 measured (jobs 393336, 393337), retention 1.0 and 0.5, 100/100 per benchmark. Identity gate passed. Draft-only diagnosis failed for `long` 0.5:attention (1 unexplained case). 0.25, 0.75 and combined arms pending |
| [06: Qwen KV eviction](06_qwen_kv_eviction/README.md) | Target visual KV eviction after prefill, vs the same method at 100% | Week 2 measured (jobs 393336, 393337), retention 1.0 and 0.5, 100/100 per benchmark. Identity gate and bookkeeping check passed. 0.25, 0.75 and combined arms pending |
| [07: Qwen decode drift](07_qwen_decode_drift/README.md) | Fork cached AR decode vs stock HF cached and full-recompute decoding (Week 2 case 35537:162) | Pending |
| [08: Qwen combined sparsity](08_qwen_combined_sparsity/README.md) | ViSpec with the same retention and policy on draft pruning and target KV eviction (diagonal arms only), vs the same-job 03 reference and the same-policy 1.0 arm | Pending, no run yet |

Experiments 02 and 03 are a separate, later study from the LLaVA pair above. They use
Qwen2.5-VL-3B-Instruct on TextVQA and COCO Captions (100 samples each) in two prompt styles,
`long` and `short`, on the Ampere cluster's pinned A6000 nodes. Their numbers are not
comparable with the LLaVA numbers, so they are listed in their own table below.

### Qwen Week 1 results (02 and 03)

One run per method per style on one `NVIDIA RTX A6000` (UUID `GPU-a2e40ccd-5141-9e51-2b3f-0e9d71fabf73`,
driver 615.71.09, node `c2-2`, 1 GPU, 4 CPUs, 22,528 MB, `preempt` partition): Slurm job 392660 for
`long` and 392661 for `short`, commit `a06dd479abba`. Device reports:
[long](03_qwen_vispec/runs/20261019T054357Z_a06dd479abba_long_kDp191/device.txt) and
[short](03_qwen_vispec/runs/20261019T063023Z_a06dd479abba_short_ogIM31/device.txt). Times are mean
generation latency in seconds per question (lower is better), and the ratio is 02 divided by 03. Each
row has 100 samples per method. These are observed ratios, not validated speedups, because the outputs
differ between methods. The Qwen run links point to run folders that Git ignores; they are preserved
on project storage and resolve only in a checkout that has them.

| Style | Benchmark | 02 latency (s) | 03 latency (s) | Observed ratio | Token agreement | 02 / 03 quality |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| long (speed measurement) | TextVQA | 11.538 | 5.366 | 2.150 | 22/100 | 0.0 / 0.0 % VQA (secondary) |
| long (speed measurement) | COCO Captions | 4.318 | 2.030 | 2.127 | 28/100 | 1.87e-05 / 1.90e-05 CIDEr-D x100 (secondary) |
| short (quality measurement) | TextVQA | 1.108 | 1.159 | 0.956 | 96/100 | 71.0 / 70.0 % VQA |
| short (quality measurement) | COCO Captions | 0.754 | 0.516 | 1.461 | 88/100 | 121.40 / 119.68 CIDEr-D x100 |

Short-style speed is secondary because prefill dominates its wall time. Long-style quality is
secondary because strict scoring rejects paragraph answers. The Week 1 gate outcome: every greedy
divergence is a bf16 near-tie (150 of 150 in `long`, 16 of 16 in `short`, none unexplained), so the
low long-style token agreement is consistent with bf16 tie-breaking and with no decoding bug. The most
likely mechanism, which was not measured, is the different kernel shapes of batched tree verification
compared with one-token decoding. Validated equivalent-output speedup is still unestablished, and repeat-run variability
is **pending**. Full tables, memory, acceptance and the diagnosis are in the
[03 README](03_qwen_vispec/README.md#results) and [02 README](02_qwen_autoregressive/README.md#results).

### Qwen decode profile (04)

Experiment 04 profiled 3 `long`-style samples per benchmark with `torch.profiler` on one
`NVIDIA RTX A6000` (UUID `GPU-8d51152c-22f1-372d-aef5-063afee0d488`, driver 615.71.09, node `c1-8`,
Xeon Silver 4116, 1 GPU, 4 CPUs, 22,528 MB, `preempt` partition): Slurm job 393060,
[device report](04_qwen_decode_profile/runs/20261019T230855Z_65078a0dbe9b_long_J4OYTT/device.txt).
A step is one decode forward pass for AR and one verification iteration for ViSpec. The GPU-busy
fraction is CUDA kernel time divided by decode time. The profiled value is biased low by profiler
overhead, so the table also gives an estimate that divides the same kernel time by the unprofiled
decode time.

| Method | Benchmark | GPU-busy, profiled / estimated unprofiled | Kernel launches per step | Launches per output token | Kernel ms per step | Unprofiled ms per output token | Top self-CPU op (profiled ms per step) |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| AR | TextVQA | 0.202 / 0.385 | 1,789 | 1,789 | 14.58 | 37.82 | `cudaLaunchKernel` 12.54 |
| ViSpec | TextVQA | 0.296 / 0.501 | 2,270 | 719 | 24.95 | 15.83 | `cudaLaunchKernel` 14.35 |
| AR | COCO Captions | 0.206 / 0.364 | 1,789 | 1,789 | 13.63 | 37.45 | `cudaLaunchKernel` 11.53 |
| ViSpec | COCO Captions | 0.254 / 0.431 | 2,275 | 669 | 22.50 | 15.47 | `cudaLaunchKernel` 15.10 |

The measurements are these. By the unprofiled estimate, the GPU is idle for about half or more
of each step (61 to 64% for AR, 50 to 57% for ViSpec). The host waits at synchronizations for under
1 ms per step, and unprofiled step time works out to 21 to 23 µs per kernel launched. Attention
over the KV cache is 3 to 12% of decode kernel time (the attention kernel alone, closer to 10% for AR TextVQA with
the context-dependent copies).

From these we infer that decode is probably host-bound for both methods, and that ViSpec's gain follows
from its 2.5 to 2.7 times fewer launches per output token. Visual KV eviction would shorten GPU
work that is off the critical path, so it is unlikely to reduce decode latency in this harness.
Its measurable effect should be on memory. The
[04 README](04_qwen_decode_profile/README.md#results) has the full tables, the top ops and the
limits.

### Qwen Week 2 results (05 and 06)

Experiments 05 (draft-side pruning) and 06 (target-side KV eviction) ran at retention 1.0 and 0.5 with the
`spatial` and `attention` policies, one run per arm and style, 100 samples per benchmark, in one allocation
per style with the 02 and 03 references (commit `66f492a5dc65`, clean tree, all runs exit 0). Both
styles used `NVIDIA RTX A6000` cards, but on different nodes, so their latencies are not comparable.

| Style | Job | Node and host CPU | GPU UUID | Allocation | Device reports |
| --- | --- | --- | --- | --- | --- |
| long | 393336 (`preempt`, 03:29:51) | `c1-8`, Xeon Silver 4116 | `GPU-8d51152c-22f1-372d-aef5-063afee0d488` | 1 GPU, 4 CPUs, 22,528 MB | [03 reference](03_qwen_vispec/runs/20261020T090922Z_66f492a5dc65_long_wDElc3/device.txt); every arm folder has its own `device.txt` (see the [05](05_qwen_draft_pruning/README.md#hardware) and [06](06_qwen_kv_eviction/README.md#hardware) READMEs) |
| short | 393337 (`preempt`, 00:51:00) | `c2-2`, Xeon Silver 4114 | `GPU-a2e40ccd-5141-9e51-2b3f-0e9d71fabf73` | 1 GPU, 4 CPUs, 22,528 MB | [03 reference](03_qwen_vispec/runs/20261020T073727Z_66f492a5dc65_short_AFFHib/device.txt); every arm folder has its own `device.txt` |

The links point at run folders that Git ignores; they are preserved on project storage and resolve only in a
checkout that has them. Ratios are reference over candidate against the same-policy 1.0 arm (above 1 means the
candidate was faster). The live KV ratio is candidate over reference. Quality is TextVQA VQA accuracy in percent
or COCO CIDEr-D x 100 (primary in `short`, secondary in `long`, where Week 1 scored about 0), as 1.0 arm to 0.5 arm.
Draft-only arms leave the target unchanged, so in exact arithmetic they would reproduce the reference. Exact
agreement reflects bf16 near-ties as well as method effects, and with n = 100 the counts are noisy.

| Style | Benchmark | Arm (0.5 vs same-policy 1.0) | Exact agreement | Accepted draft tokens per iteration | Latency ratio | TTFT ratio | Live KV ratio | Quality |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| long | TextVQA | 05 ViSpec 0.5:spatial | 56/100 | 2.141 to 2.145 | 1.007 | 1.001 | 1.001 | 0.0 to 0.0 |
| long | TextVQA | 05 ViSpec 0.5:attention | 48/100 | 2.141 to 2.129 | 1.017 | 1.000 | 0.995 | 0.0 to 0.0 |
| long | TextVQA | 06 AR 0.5:spatial | 0/100 | n/a (AR) | 0.890 | 1.001 | 0.662 | 0.0 to 0.0 |
| long | TextVQA | 06 ViSpec 0.5:spatial | 0/100 | 2.141 to 2.265 | 0.905 | 1.003 | 0.671 | 0.0 to 0.0 |
| long | TextVQA | 06 ViSpec 0.5:attention | 6/100 | 2.141 to 2.255 | 0.935 | 1.003 | 0.661 | 0.0 to 0.0 |
| long | COCO Captions | 05 ViSpec 0.5:spatial | 77/100 | 2.284 to 2.344 | 0.989 | 1.008 | 1.011 | 1.90e-05 to 1.90e-05 |
| long | COCO Captions | 05 ViSpec 0.5:attention | 72/100 | 2.284 to 2.294 | 1.004 | 1.001 | 1.001 | 1.90e-05 to 1.90e-05 |
| long | COCO Captions | 06 AR 0.5:spatial | 1/100 | n/a (AR) | 0.996 | 1.002 | 0.661 | 1.87e-05 to 8.84e-07 |
| long | COCO Captions | 06 ViSpec 0.5:spatial | 2/100 | 2.284 to 2.291 | 0.993 | 0.997 | 0.659 | 1.90e-05 to 1.95e-05 |
| long | COCO Captions | 06 ViSpec 0.5:attention | 1/100 | 2.284 to 2.352 | 1.061 | 1.002 | 0.650 | 1.90e-05 to 3.49e-06 |
| short | TextVQA | 05 ViSpec 0.5:spatial | 100/100 | 0.690 to 0.679 | 1.016 | 1.005 | 1.000 | 70.0 to 70.0 |
| short | TextVQA | 05 ViSpec 0.5:attention | 100/100 | 0.690 to 0.685 | 1.002 | 1.003 | 1.000 | 70.0 to 70.0 |
| short | TextVQA | 06 AR 0.5:spatial | 82/100 | n/a (AR) | 0.907 | 0.999 | 0.534 | 71.0 to 68.4 |
| short | TextVQA | 06 ViSpec 0.5:spatial | 83/100 | 0.690 to 0.983 | 0.965 | 1.003 | 0.535 | 70.0 to 68.4 |
| short | TextVQA | 06 ViSpec 0.5:attention | 96/100 | 0.690 to 0.696 | 0.998 | 0.994 | 0.532 | 70.0 to 70.0 |
| short | COCO Captions | 05 ViSpec 0.5:spatial | 100/100 | 1.746 to 1.736 | 1.008 | 1.010 | 1.000 | 119.68 to 119.68 |
| short | COCO Captions | 05 ViSpec 0.5:attention | 100/100 | 1.746 to 1.731 | 1.007 | 1.014 | 1.000 | 119.68 to 119.68 |
| short | COCO Captions | 06 AR 0.5:spatial | 39/100 | n/a (AR) | 0.966 | 0.987 | 0.580 | 121.40 to 116.49 |
| short | COCO Captions | 06 ViSpec 0.5:spatial | 38/100 | 1.746 to 1.774 | 1.098 | 1.041 | 0.581 | 119.68 to 117.07 |
| short | COCO Captions | 06 ViSpec 0.5:attention | 43/100 | 1.746 to 1.736 | 1.000 | 0.996 | 0.582 | 119.68 to 110.84 |

What the measurements show:

- **Identity gate passed.** Every 1.0 arm matches its null reference on 100/100 for both benchmarks and both
  styles. The Week 2 compact references equal the Week 1 fixed-layout 02 and 03 runs token for token on 100/100.
- **Bookkeeping check at 0.5 passed.** AR and ViSpec eviction arms agree on 19/100 and 18/100 (`long`) and 97/100 and
  87/100 (`short`) on TextVQA and COCO, against 22, 28, 96 and 88 for the unpruned pair. Long COCO is exactly at
  the 10.0-point limit (18 against 28); the rule was "more than 10".
- **Draft pruning (05)** cost acceptance at most slightly and did not move latency, as predicted. Exact agreement
  is 100/100 in `short` and 48 to 77 of 100 in `long`. The draft-only diagnosis **failed** for `long` 0.5:attention
  (79 near-ties and 1 unexplained case); `long` 0.5:spatial passed (67 of 67). The controller's evidence points to
  the shared cached decode and not to pruning, which is an inference, and the underlying drift is open.
- **KV eviction (06)** cut live KV to 0.65 to 0.67 (`long`) and 0.53 to 0.58 (`short`) of the 1.0 arm and moved the
  decode peak by about 0.2% or less. Latency changed little per token, but long TextVQA end-to-end latency rose as
  outputs got longer. Long-style outputs diverge heavily from the 1.0 arm. In `short`, the spatial arms lost 1.6 to 2.6 VQA points on TextVQA and 2.61 to 4.91 CIDEr-D on COCO. The
  ViSpec attention arm lost 8.85 CIDEr-D on COCO and nothing on TextVQA.
- **Pending (Week 3):** retention 0.25 and 0.75, the combined draft-plus-KV arms, and the fp32 confirmation of
  the draft-arm divergences.

Full tables, memory, hardware and the predictions scored against the measurements are in the
[05 README](05_qwen_draft_pruning/README.md#results) and [06 README](06_qwen_kv_eviction/README.md#results).

### LLaVA run conditions (00 and 01)

The latest LLaVA pair (00 and 01) used temperature **0.0 (greedy)**, FP16, and a maximum of
**1,024 output token IDs, including EOS**. Both launch trees were clean at
commit `771911bd8665846e9ab8cb698cb22c2929d296b5`.

Each job received **one H100 PCIe with 79.2 GiB visible VRAM, 8 logical CPUs,
and 64 GiB RAM** on `evc46`. These are job resources, not host totals.

| Run             | Physical GPU UUID                      | Hardware evidence                                                                          |
| --------------- | -------------------------------------- | ------------------------------------------------------------------------------------------ |
| 00 · job 950793 | `c8aca2f5-7741-b9ef-76aa-48901fe44401` | [Device report](00_baseline_original/runs/20261017T211853Z_771911bd8665_9RRsjY/device.txt) |
| 01 · job 950803 | `cd5e5946-fc28-f3cf-978d-635f8e613aa4` | [Device report](01_baseline_draft/runs/20261017T211906Z_771911bd8665_ei1koG/device.txt)    |

## Run an experiment

Follow the [Newton setup guide](NEWTON_SETUP.md), then submit from the repository
root:

```bash
# 00: Standard baseline
bash experiment/submit.sh experiment/00_baseline_original

# 01: ViSpec baseline
bash experiment/submit.sh experiment/01_baseline_draft
```

Both runners enforce greedy decoding and the 1,024-token cap. The `TEMPERATURE`
environment variable is ignored. Each experiment has its own `run.sbatch` and
reuses the prepared environment. Both check CUDA before loading models.

Scheduler logs go to `$PROJECT_ROOT/logs/`. Answers, reports, and launch evidence
go to a unique `<experiment>/runs/<run>/` directory. The current snapshots for
jobs **950793** and **950803** are versioned so teammates can inspect all answers,
raw token IDs, reports, logs and launch evidence. Other generated runs remain
ignored by default. Preserve these snapshots as evidence and use new run folders
for reruns.
See the [metrics guide](METRICS.md) for scoring and comparison commands.

The Qwen experiments are submitted with the Ampere helper, one prompt style per job with 02
and 03 in the same allocation. This is the Week 1 command for the `long` style; use
`PROMPT_STYLE=short` for the other job:

```bash
PROMPT_STYLE=long CLUSTER_PARTITION=preempt SBATCH_EXTRA="--cpus-per-task=4 --mem=22G --time=03:00:00" \
  bash experiment/submit_ampere.sh experiment/02_qwen_autoregressive experiment/03_qwen_vispec
```

Other resources or a separate job per experiment give a different allocation, which
`compare_metrics` refuses to pair (see the 02 and 03 READMEs).

See the [Ampere setup guide](AMPERE_SETUP.md) for the GPU pin and partitions.

## Adding experiments

Use `NN_descriptive_name/` with a runnable shell script and a brief README covering
the hypothesis, changes, models, data/subsets, settings, cluster prerequisites,
exact command, metrics, results, and limitations. Update this register when
verified measurements arrive. Mark unmeasured results **pending**.

Each launch must preserve the full commit, working-tree status/diff, commands,
environment, scheduler job ID, timestamps, combined stdout/stderr, raw outputs,
and exit status. Prefer committed code and preserve reruns separately.

Capture hardware before evaluation with [capture_device.sh](capture_device.sh).
Report allocated resources separately from host totals, link device evidence,
and keep different hardware/settings in separate result rows. Device snapshots
do not measure runtime peak memory, energy, or utilization. Report benchmark
counts before aggregating. Never substitute paper results for measured results.
