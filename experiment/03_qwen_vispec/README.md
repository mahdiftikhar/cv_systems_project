# 03 — Qwen2.5-VL ViSpec speculative decoding

## Purpose and hypothesis

This experiment measures ViSpec speculative decoding on Qwen2.5-VL-3B-Instruct for the same TextVQA and COCO Captions subsets as the [autoregressive arm (02)](../02_qwen_autoregressive/README.md). The hypothesis is that the draft model lowers generation time without changing the greedy output, which the saved token IDs let us check against 02.

This is a different measurement from the earlier LLaVA pair (00 and 01). Those runs used
LLaVA-1.6 Vicuna 7B on TextVQA, GQA and MM-Vet. The Qwen experiments (02 and later) use
Qwen2.5-VL-3B-Instruct on TextVQA and COCO Captions, run on the Ampere cluster, and their
numbers must not be mixed with the LLaVA rows. See the
[design spec](../../docs/superpowers/specs/2026-10-18-qwen-dynamic-visual-sparsity-design.md)
for the goals.

Status: **measured for Week 1**. One run per prompt style (jobs 392660 and 392661) completed with exit code 0. Greedy outputs often differ from 02. Exact token agreement with 02 is 22/100 (TextVQA) and 28/100 (COCO Captions) in the long style, and 96/100 and 88/100 in the short style. Every difference is a bf16 near-tie (see the diagnosis below). Validated equivalent-output speedup is not established.

## Changes from the baseline

Compared with experiment 00, this experiment changes the model family, the benchmarks and the
evaluator. It runs the unified evaluator `python -m vispec.evaluation.evaluate` instead of the
per-benchmark `gen_*_answer_*` scripts. The runner requires `PROMPT_STYLE=long` or `short` and
exits with status 2 otherwise. Each style is a separate run group, because the inherited
verbose prompts suit speed measurement but defeat strict answer scoring.

Two engineering settings apply to every Qwen arm. Prefill slimming (on by default, with
`--no-slim-prefill` as the off switch) avoids computing logits and keeping every hidden-state
layer for the whole prompt. The KV cache is preallocated at 16,384 tokens instead of the
model's 128,000-token default.

## Models

| Role | Model |
| --- | --- |
| Target | `Qwen/Qwen2.5-VL-3B-Instruct`, local copy `models/Qwen2.5-VL-3B-Instruct` |
| Draft | `JLKang/ViSpec-Qwen2.5-VL-3B-Instruct`, local copy `models/ViSpec-Qwen2.5-VL-3B-Instruct`, used to propose tokens |

## Data, splits and subsets

| Benchmark | Input | Subset |
| --- | --- | --- |
| TextVQA | `bench_data/textvqa/TextVQA_0.5.1_val.json` + `train_images/` | 100 questions, seed-42 shuffle (same permutation as experiment 00) |
| COCO Captions | `bench_data/coco/karpathy_test.parquet` + `val2014/` | 100 Karpathy-test images, seed-42 shuffle |

Image size limits are `min_pixels = 256*28*28` and `max_pixels = 1280*28*28` for both
benchmarks and both styles. These are the measured subsets, not full-benchmark scores.

The COCO subset comes from the Karpathy-test parquet that preparation pins
(`yerevann/coco-karpathy`). Spec hazard 7 asked whether it matches the subset that the inherited
`HuggingFaceM4/COCO` loader in `gen_*_answer_coco_caption.py` would select. That was never
checked. No recorded result depends on it, because no run used the inherited loader.

| Style | TextVQA instruction | COCO instruction |
| --- | --- | --- |
| `long` | OCR extraction with a detailed explanation | Detailed description of the image |
| `short` | Answer using a single word or phrase | One-sentence caption |

## Settings

Decoding is greedy (temperature 0.0, fixed; `TEMPERATURE` is ignored), bf16, batch size 1, at
most 1,024 new tokens (counting all returned token IDs, including EOS), and 100 samples per
benchmark. The draft tree uses depth 3, top-k 8, total-token 30 and `num_q` 2. Slim prefill is on and the KV capacity is 16,384 tokens.

| Variable | Default | Purpose |
| --- | --- | --- |
| `PROMPT_STYLE` | none (required) | `long` or `short` |
| `BENCH_DIR` | `<repo>/bench_data` | Benchmark data root |
| `BASE_MODEL_PATH` | `<repo>/models/Qwen2.5-VL-3B-Instruct` | Target checkpoint |
| `SPEC_DIR` | `<repo>/models/ViSpec-Qwen2.5-VL-3B-Instruct` | Draft checkpoint |
| `PYTHON_BIN` | `/home/aeternum/anaconda3/envs/vipsec/bin/python` | Python executable |
| `DEPTH` | `3` | Draft tree depth |
| `TOP_K` | `8` | Draft candidates per step |
| `TOTAL_TOKEN` | `30` | Tokens in the draft tree |
| `NUM_Q` | `2` | Learned image queries seen by the draft |

## Cluster prerequisites

1. Follow the [Ampere setup guide](../AMPERE_SETUP.md). Jobs run on the pinned 48 GB
   `NVIDIA RTX A6000` nodes and fail with status 3 on any other GPU model. That GPU-name check
   lives only in `run_sequence.sbatch`, so a direct `run.sh` or `gpu_check.sh` launch skips it.
   Results of record must be submitted through `submit_ampere.sh`.
2. Prepare assets once with `bash prepare.sh --suite qwen` inside a Slurm allocation. The runner
   repeats the offline check (`--check-only --suite qwen`) after the device capture and stops
   before any model loads if an asset is missing.
3. Commit code before launching, so the run records a clean commit.

Week 1 runs use `CLUSTER_PARTITION=preempt`, which reaches the same pinned A6000 nodes. The
`gpu` partition queue was 4 to 7 days. The cluster default is `JobRequeue=0`, so
`run_sequence.sbatch` requests `--requeue`; without it a preempted job is cancelled. A requeued
job reruns `run_sequence.sbatch` from the start, so 02 and 03 each get a fresh run folder and
no answer path is reused. The partial folders from the preempted attempt stay on disk as
evidence. Their reports may be missing, because the preempt partition has `GraceTime=0` and
Slurm sends SIGKILL 30 seconds (`KillWait`) after SIGTERM.

Run folders from the same attempt share `slurm_job_id` and `slurm_restart_count` in
`metadata.txt`, and the job log `logs/ampere-sequence-<job>.log` keeps every attempt because it
is opened in append mode. The Week 1 jobs were submitted before `--requeue` was added. Neither
was preempted: each job log has one start line.

## Run command

Week 1 submitted one job per prompt style, and each job ran 02 and then 03 in one allocation.
To reproduce it, run these commands from the repository root on a login or VS Code shell:

```bash
PROMPT_STYLE=long CLUSTER_PARTITION=preempt SBATCH_EXTRA="--cpus-per-task=4 --mem=22G --time=03:00:00" \
  bash experiment/submit_ampere.sh experiment/02_qwen_autoregressive experiment/03_qwen_vispec
PROMPT_STYLE=short CLUSTER_PARTITION=preempt SBATCH_EXTRA="--cpus-per-task=4 --mem=22G --time=03:00:00" \
  bash experiment/submit_ampere.sh experiment/02_qwen_autoregressive experiment/03_qwen_vispec
```

The pair must run in the same job. Running 03 alone, or with the default resources of
`run_sequence.sbatch` (8 CPUs, 64G, 8 hours), gives a different allocation and possibly a
different card. `compare_metrics` compares the GPU UUID, the allocated CPU count and memory, and
the CPU affinity, and it refuses to pair runs whose values differ.

Runs submitted through the helper record the exact `sbatch` command (`submit_command=`) and
`SBATCH_EXTRA` (`sbatch_extra=`, empty when none was given) in `metadata.txt`. Both read `unset`
for a launch that did not use the helper.

Inside an existing one-GPU allocation, run `PROMPT_STYLE=long bash experiment/03_qwen_vispec/run.sh`.
Such a run skips the GPU-name check (see the prerequisites).
The runner prints each exact evaluator command before running it, so `output.log` records
every generation argument. TextVQA runs first, then COCO Captions, and the run stops on the
first failure.

## Run artifacts

Each launch creates `runs/<UTC>_<commit12>_<style>_XXXXXX/`. It holds the full commit, working-tree
status and tracked diff, the runner snapshot, packages, scheduler job ID, timestamps,
combined stdout/stderr (`output.log`), `metadata.txt`, `exit-code.txt`, the device report
(`device.txt`, from [capture_device.sh](../capture_device.sh)), and the automatic report under
`reports/`. Answers are written to:

```text
results/textvqa_test/vispec_qwen/test.jsonl
results/coco_caption_test/vispec_qwen/test.jsonl
```

Generated runs stay out of Git and are preserved on project storage. Untracked source files
are not captured in the launch diff. Exit code 0 means completion.

## Metric definitions

Per-benchmark results and sample counts are reported before any aggregation. Latency
is mean wall time per question in seconds, covering prefill and decoding but not model loading,
prompt preparation or scoring. Time to first token (TTFT) is the wait before the first answer
token is ready. Token counts are the returned token IDs saved with each answer, including EOS.
See the [metrics guide](../METRICS.md) for the shared definitions.

Quality scores are TextVQA consensus accuracy (the downloaded `m4c_evaluator.py`) and COCO
CIDEr-D. CIDEr-D inverse document frequency is computed over the evaluated 100-image subset, so
the values are subset measurements and are not comparable to full-corpus CIDEr.

Memory is reported in two phases. The prefill-phase peak covers processing of the image and
question; the decode-phase peak covers token generation. For ViSpec the phase boundary is the first target token, so the draft's prefill counts toward the decode-phase peak. Prefill slimming is
output-preserving (slim and no-slim tokens were identical in a 5-sample smoke run) and lowers
the decode-phase peak (maximum 8.49 versus 8.85 GiB in that smoke run). It does not lower the
prefill-phase peak, which is set before the language-model head, most likely by the vision
encoder. `kv_live_bytes` is a whole-model figure repeated in each device entry, so do not
sum it across GPUs. The KV cache capacity is 16,384 tokens, not the model's 128k default.
These smoke figures are not results of this experiment.

`kv_tokens` is the target KV cache length when generation ends. In this arm it equals input
tokens plus output tokens. In the autoregressive arm (02) it is one less, because the last
generated token is never fed back through the model, so live-KV figures differ by one token's
worth between the arms for the same output. Both relations hold for every Week 1 record.
`kv_live_bytes` is `kv_tokens` times the bytes one token occupies across all layers. It is
arithmetic, not an allocator measurement.

The recorded `decode_time` starts after `initialize_tree` returns
(`vispec/model/spec_model_ours.py`, `specgenerate`), while TTFT ends at the first target token
inside `initialize_tree`. The draft prefill and the first draft tree therefore fall in neither
TTFT nor `decode_time`. In Week 1 that window, latency minus TTFT minus `decode_time`, averaged
11.6 to 12.5 ms per question in each benchmark and style, against 0.4 ms in 02 (from the saved
per-question records in `results/*/test.jsonl`). Compare the arms with latency and TTFT, not
`decode_time`.

## Results

All figures come from one run per method per prompt style, 100 samples per benchmark, greedy decoding, bf16, batch size 1, at most 1,024 new tokens, depth 3, top-k 8, total-token 30, `num_q` 2, slim prefill on, 16,384-token KV capacity. Per-benchmark results are listed before anything else, and no cross-benchmark aggregate is computed. The 02 numbers for the same runs are in the [02 README](../02_qwen_autoregressive/README.md#results).

### Hardware

Each 02 and 03 pair ran inside one Slurm job on the same physical card, so the pair shares hardware. The diagnosis is a separate check on different cards and is not a timed result.

| Style | Job / partition | GPU | UUID | Driver | Allocation | Device report |
| --- | --- | --- | --- | --- | --- | --- |
| long (02 and 03 runs) | 392660 / `preempt`, node `c2-2`, 00:42:57 | NVIDIA RTX A6000 (47.5 GiB visible) | `GPU-a2e40ccd-5141-9e51-2b3f-0e9d71fabf73` | 615.71.09 | 1 GPU, 4 CPUs (affinity 9, 12, 29, 32), 22,528 MB, 3 h limit | [03 device.txt](runs/20261019T054357Z_a06dd479abba_long_kDp191/device.txt) |
| short (02 and 03 runs) | 392661 / `preempt`, node `c2-2`, 00:08:38 | NVIDIA RTX A6000 (47.5 GiB visible) | `GPU-a2e40ccd-5141-9e51-2b3f-0e9d71fabf73` | 615.71.09 | 1 GPU, 4 CPUs (affinity 9, 12, 29, 32), 22,528 MB, 3 h limit | [03 device.txt](runs/20261019T063023Z_a06dd479abba_short_ogIM31/device.txt) |
| long diagnosis | 392676 / `preempt`, node `c1-8` | NVIDIA RTX A6000 | `GPU-eb032c66-7f3f-d1b6-96cf-c9db5600d457` | 615.71.09 | one GPU | [diagnosis device report](runs/20261019T054357Z_a06dd479abba_long_kDp191/diagnosis/20261019T060057Z_device.txt) |
| short diagnosis | 392714 / `preempt`, node `c1-8` | NVIDIA RTX A6000 | `8d51152c-22f1-372d-aef5-063afee0d488` (from `diagnosis.json`) | pending (no device report saved) | one GPU | [diagnosis.json](runs/20261019T063023Z_a06dd479abba_short_ogIM31/diagnosis/20261019T085127Z/diagnosis.json) |

The links point at generated run folders, which are ignored by Git and preserved on project storage, so they resolve only in a checkout that has those folders. Device reports are launch snapshots and do not measure runtime peaks.

### Reading these results

The `long` style is the speed and acceptance measurement. Its strict quality scores are secondary and near zero, because CIDEr-D's length penalty and TextVQA's exact consensus reject paragraph answers for both methods. The `short` style is the quality measurement. Its speed is secondary, because answers average 5 tokens (TextVQA) and 15 tokens (COCO) and prefill dominates the wall time.

Units: latency is wall time per question in seconds for prefill plus decoding, excluding model loading, prompt preparation and scoring. TTFT is the wait before the first answer token. TPOT is `(latency - TTFT) / (output tokens - 1)` in milliseconds, averaged per question. Throughput is the sum of output tokens divided by the sum of latencies. Output tokens include EOS. Accepted draft tokens per iteration is the mean over all verification iterations of the draft tokens the target accepted, so output tokens per iteration is that value plus one. The latency ratio divides the 02 mean latency by the 03 mean latency, so a value above 1 means 03 was faster. The ratio is an observed ratio only. The comparison marks `controlled_latency_speedup` as unavailable because greedy output equivalence is not validated. Memory is `torch.cuda.max_memory_allocated` per question in GiB (mean and max over 100 questions) and live KV is the whole-model figure in MiB. For ViSpec the phase boundary is the first target token, so the draft prefill and the first draft tree count toward the decode-phase peak. The 03 decode-peak mean is 0.06 to 0.07 GiB above 02's in both benchmarks and both styles (tables here and in the 02 README), although TextVQA prompts average 1,040 to 1,063 tokens and COCO prompts 405 to 406 (mean saved `input_ids_tokens` per style). A gap that does not grow with prompt length is likely the draft tree or top-k buffers rather than the draft prefill. This was not measured. Phase peaks and live KV come from the saved per-question measurements in `results/*/test.jsonl`.

### Style `long` (speed measurement), 03 ViSpec

| Benchmark | Samples | Mean / median / p95 latency (s) | Mean TTFT (s) | Mean TPOT (ms) | Throughput (tokens/s) | Mean output tokens | Truncated |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| TextVQA | 100 | 5.366 / 4.236 / 12.557 | 0.952 | 17.1 | 51.75 | 277.70 | 3/100 (0.03) |
| COCO Captions | 100 | 2.030 / 1.921 / 3.319 | 0.229 | 16.6 | 54.12 | 109.89 | 0/100 (0.00) |

| Benchmark | Prefill peak mean / max (GiB) | Decode peak mean / max (GiB) | Live KV mean / max (MiB) | Accepted draft tokens per iteration (iterations) | Quality | Scored |
| --- | --- | --- | --- | --- | ---: | ---: |
| TextVQA | 10.69 / 12.08 | 8.48 / 8.49 | 47.1 / 81.9 | 2.141 (8841) | 0.0% VQA accuracy | 100/100 |
| COCO Captions | 8.71 / 9.08 | 8.45 / 8.45 | 18.1 / 24.1 | 2.284 (3346) | 1.90e-05 CIDEr-D x100 | 100/100 |

### Style `short` (quality measurement), 03 ViSpec

| Benchmark | Samples | Mean / median / p95 latency (s) | Mean TTFT (s) | Mean TPOT (ms) | Throughput (tokens/s) | Mean output tokens | Truncated |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| TextVQA | 100 | 1.159 / 1.119 / 1.588 | 0.976 | 69.3 | 4.52 | 5.24 | 0/100 (0.00) |
| COCO Captions | 100 | 0.516 / 0.509 / 0.660 | 0.226 | 20.9 | 29.17 | 15.05 | 0/100 (0.00) |

| Benchmark | Prefill peak mean / max (GiB) | Decode peak mean / max (GiB) | Live KV mean / max (MiB) | Accepted draft tokens per iteration (iterations) | Quality | Scored |
| --- | --- | --- | --- | --- | ---: | ---: |
| TextVQA | 10.69 / 12.08 | 8.47 / 8.49 | 36.7 / 46.7 | 0.690 (310) | 70.0% VQA accuracy | 100/100 |
| COCO Captions | 8.71 / 9.08 | 8.44 / 8.45 | 14.8 / 21.1 | 1.746 (548) | 119.68 CIDEr-D x100 | 100/100 |

### Comparison with 02

Comparison reports: [long](runs/20261019T054357Z_a06dd479abba_long_kDp191/comparisons/20261019T055847Z/comparison.md) and [short](runs/20261019T063023Z_a06dd479abba_short_ogIM31/comparisons/20261019T063643Z/comparison.md). Same hardware, settings and commit (`a06dd479abba`) for both methods in each style.

**Long style (speed measurement)**

| Benchmark | Samples | Latency 02 (s) | Latency 03 (s) | Latency ratio 02/03 | TTFT ratio 02/03 | Exact token agreement | Decoded-text agreement |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| TextVQA | 100 | 11.538 | 5.366 | 2.150 | 0.997 | 22/100 (0.22) | 0.22 |
| COCO Captions | 100 | 4.318 | 2.030 | 2.127 | 0.965 | 28/100 (0.28) | 0.28 |

Quality in the long style is secondary.

| Benchmark | Scored | 02 quality | 03 quality | Difference (03 minus 02) |
| --- | ---: | ---: | ---: | ---: |
| TextVQA VQA accuracy (%) | 100/100 | 0.0 | 0.0 | +0.0 pp |
| COCO CIDEr-D x100 | 100/100 | 1.87e-05 | 1.90e-05 | +3.17e-07 |

**Short style (quality measurement)**

| Benchmark | Samples | Latency 02 (s) | Latency 03 (s) | Latency ratio 02/03 | TTFT ratio 02/03 | Exact token agreement | Decoded-text agreement |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| TextVQA | 100 | 1.108 | 1.159 | 0.956 | 0.979 | 96/100 (0.96) | 0.96 |
| COCO Captions | 100 | 0.754 | 0.516 | 1.461 | 0.998 | 88/100 (0.88) | 0.88 |

| Benchmark | Scored | 02 quality | 03 quality | Difference (03 minus 02) |
| --- | ---: | ---: | ---: | ---: |
| TextVQA VQA accuracy (%) | 100/100 | 71.0 | 70.0 | -1.0 pp |
| COCO CIDEr-D x100 | 100/100 | 121.40 | 119.68 | -1.72 |

Short-style speed is secondary. Mean TTFT is 0.956 s (02) and 0.976 s (03) for TextVQA out of 1.108 s and 1.159 s mean latency, so prefill is most of the wall time and the 03 TextVQA ratio of 0.956 shows no gain with 0.690 accepted draft tokens per iteration. COCO Captions, with longer answers, shows a 1.461 ratio.

### Agreement and divergence diagnosis

Greedy decoding at temperature 0 did not give identical sequences. Token agreement equals decoded-text agreement in every row, so every differing sequence also decodes to different text. In the long style, 78 of 100 TextVQA and 72 of 100 COCO sequences diverge, with the first mismatch between output index 1 and 983. In the short style, 4 TextVQA and 12 COCO sequences diverge, with the first mismatch at index 1 to 45.

[`diagnose_divergence`](../../vispec/evaluation/diagnose_divergence.py) rebuilds each prompt (checked against the saved input fingerprints, 0 mismatches in both styles), appends the shared output prefix and runs one full forward pass of the stock model in bf16 and then fp32 (TF32 off) to read the logits at the first differing position. A case is `near_tie` if the bf16 top-2 margin is at most 0.5 logits, otherwise `unexplained`. Both diagnoses ran on `c1-8` cards that differ from the timed runs' card, so they are checks and not timed results.

**Long style**

| Benchmark | Mismatched turns | near_tie | unexplained | fp32 agrees with 02 | fp32 agrees with 03 | fp32 agrees with neither | bf16 margin min / median / max | fp32 margin min / median / max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| TextVQA | 78 | 78 | 0 | 50 | 28 | 0 | 0.0000 / 0.0000 / 0.3750 | 0.0049 / 0.0717 / 0.3692 |
| COCO Captions | 72 | 72 | 0 | 35 | 33 | 4 | 0.0000 / 0.1250 / 0.2500 | 0.0047 / 0.0683 / 0.2631 |
| Both | 150 | 150 | 0 | 85 | 61 | 4 | 0.0000 / 0.1250 / 0.3750 | 0.0047 / 0.0683 / 0.3692 |

**Short style**

| Benchmark | Mismatched turns | near_tie | unexplained | fp32 agrees with 02 | fp32 agrees with 03 | fp32 agrees with neither | bf16 margin min / median / max | fp32 margin min / median / max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| TextVQA | 4 | 4 | 0 | 3 | 1 | 0 | 0.0000 / 0.0000 / 0.1250 | 0.0593 / 0.1456 / 0.2375 |
| COCO Captions | 12 | 12 | 0 | 3 | 9 | 0 | 0.0000 / 0.1250 / 0.1250 | 0.0035 / 0.0616 / 0.2493 |
| Both | 16 | 16 | 0 | 6 | 10 | 0 | 0.0000 / 0.0625 / 0.1250 | 0.0035 / 0.0670 / 0.2493 |

Every divergence in both styles is a bf16 near-tie: 150 of 150 in the long style and 16 of 16 in the short style, with none unexplained. The bf16 margin is at most 0.375 logits, and many are exact ties (margin 0.0000), where `argmax` takes the first index and the winner depends on rounding. The fp32 pass often prefers the other method's token (61 of 150 long, 10 of 16 short for 03's token), and in 4 long COCO cases it prefers neither. The fp32 margins are also small (at most 0.369).

### Week 1 gate outcome

The gate passes. The low long-style token agreement (0.22 on TextVQA and 0.28 on COCO) is consistent with bf16 tie-breaking, and it is consistent with no decoding bug. The most likely mechanism, which was not measured, is the different kernel shapes of batched tree verification compared with one-token decoding. Once two sequences diverge at a near-tie, the rest of the text differs, so a single tie can change a whole long paragraph.

A validated equivalent-output speedup remains **unestablished**, because the outputs differ. The 2.15x (TextVQA) and 2.13x (COCO) long-style latency ratios are observed ratios between two output sets of similar length (mean output tokens 283.31 vs 277.70 and 109.96 vs 109.89), not controlled speedups.

### Open items

- Repeat-run variability: **pending**. There is one run per method per style.
- Energy and non-PyTorch GPU memory: **pending** (not measured).
- Controlled latency speedup: **pending**, until a configuration with equivalent outputs exists.

## Limitations

- Scores cover 100-sample subsets, and each configuration has one run, so repeat variability
  is unknown.
- Model and dataset revisions are pinned by preparation; the recorded values are in
  `.preparation/assets_qwen.json`, which each run copies as `prepared-assets.json`.
- `device.txt` is a launch snapshot. It does not measure peak memory, energy or utilization
  during evaluation, and energy and non-PyTorch GPU memory are not measured.
- The draft model has a fixed 8,192-position RoPE table, so prompts must stay below that
  length. These benchmarks are far below it.
- CIDEr-D is a subset measurement (see above), and the long prompt style is expected to
  score poorly on strict TextVQA matching.
- Decoding is likely host-bound. The paired AR arm takes about 38 ms per token (mean TPOT in
  the 02 README), while reading the weights once per token would take about 8 to 9 ms: the
  6.17 GB of bf16 language-model weights at the A6000's 768 GB/s peak bandwidth is 8.0 ms. The
  host is a Xeon Silver 4114 with 4 allocated cores. This is inferred from these numbers and was
  not profiled. A profiling pass is planned before Week 3.
- The COCO subset was not checked against the inherited `HuggingFaceM4/COCO` loader's subset
  (see the data section).
- ViSpec output equivalence with 02 is checked from saved token IDs and is not assumed. `compare_metrics` requires `method=vispec_speculative` here and `method=autoregressive` in 02, with matching `prompt_style`, `slim_prefill`, `kv_capacity`, `num_samples` and `max_new_tokens`.