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

## Latest results

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

### Overall results

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

The latest pair used temperature **0.0 (greedy)**, FP16, and a maximum of
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
