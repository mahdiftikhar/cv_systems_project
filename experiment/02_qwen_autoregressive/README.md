# 02 — Qwen2.5-VL autoregressive baseline

## Purpose and hypothesis

This experiment measures how fast Qwen2.5-VL-3B-Instruct answers TextVQA questions and writes COCO captions with standard autoregressive decoding. It is the reference for the [ViSpec arm (03)](../03_qwen_vispec/README.md) and for later visual-sparsity arms. We expect it to be slower than 03 while producing the reference outputs that 03 must match under greedy decoding.

This is a different measurement from the earlier LLaVA pair (00 and 01). Those runs used
LLaVA-1.6 Vicuna 7B on TextVQA, GQA and MM-Vet. The Qwen experiments (02 and later) use
Qwen2.5-VL-3B-Instruct on TextVQA and COCO Captions, run on the Ampere cluster, and their
numbers must not be mixed with the LLaVA rows. See the
[design spec](../../docs/superpowers/specs/2026-10-18-qwen-dynamic-visual-sparsity-design.md)
for the goals.

Status: **measured for Week 1**. One run per prompt style (jobs 392660 and 392661) completed with exit code 0. Results are below, and the comparison with 03 is in the [03 README](../03_qwen_vispec/README.md#results).

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
| Draft | `JLKang/ViSpec-Qwen2.5-VL-3B-Instruct`, local copy `models/ViSpec-Qwen2.5-VL-3B-Instruct`. It is loaded by the wrapper but not used for decoding |

## Data, splits and subsets

| Benchmark | Input | Subset |
| --- | --- | --- |
| TextVQA | `bench_data/textvqa/TextVQA_0.5.1_val.json` + `train_images/` | 100 questions, seed-42 shuffle (same permutation as experiment 00) |
| COCO Captions | `bench_data/coco/karpathy_test.parquet` + `val2014/` | 100 Karpathy-test images, seed-42 shuffle |

Image size limits are `min_pixels = 256*28*28` and `max_pixels = 1280*28*28` for both
benchmarks and both styles. These are the measured subsets, not full-benchmark scores.

| Style | TextVQA instruction | COCO instruction |
| --- | --- | --- |
| `long` | OCR extraction with a detailed explanation | Detailed description of the image |
| `short` | Answer using a single word or phrase | One-sentence caption |

## Settings

Decoding is greedy (temperature 0.0, fixed; `TEMPERATURE` is ignored), bf16, batch size 1, at
most 1,024 new tokens (counting all returned token IDs, including EOS), and 100 samples per
benchmark. Slim prefill is on and the KV capacity is 16,384 tokens.

| Variable | Default | Purpose |
| --- | --- | --- |
| `PROMPT_STYLE` | none (required) | `long` or `short` |
| `BENCH_DIR` | `<repo>/bench_data` | Benchmark data root |
| `BASE_MODEL_PATH` | `<repo>/models/Qwen2.5-VL-3B-Instruct` | Target checkpoint |
| `SPEC_DIR` | `<repo>/models/ViSpec-Qwen2.5-VL-3B-Instruct` | Draft checkpoint (loaded, unused for decoding) |
| `PYTHON_BIN` | `/home/aeternum/anaconda3/envs/vipsec/bin/python` | Python executable |

## Cluster prerequisites

1. Follow the [Ampere setup guide](../AMPERE_SETUP.md). Jobs run on the pinned 48 GB
   `NVIDIA RTX A6000` nodes and fail with status 3 on any other GPU model.
2. Prepare assets once with `bash prepare.sh --suite qwen` inside a Slurm allocation. The runner
   repeats the offline check (`--check-only --suite qwen`) after the device capture and stops
   before any model loads if an asset is missing.
3. Commit code before launching, so the run records a clean commit.

Week 1 runs use `CLUSTER_PARTITION=preempt`, which reaches the same pinned A6000 nodes. The
`gpu` partition queue was 4 to 7 days. A preempted job is requeued and reruns the runner from
the start. Each invocation creates a fresh run folder and never reuses answer paths, so the
partial folder from the preempted attempt stays on disk as evidence.

## Run command

From the repository root on a login or VS Code shell, submit one style per job:

```bash
PROMPT_STYLE=long  CLUSTER_PARTITION=preempt bash experiment/submit_ampere.sh experiment/02_qwen_autoregressive
PROMPT_STYLE=short CLUSTER_PARTITION=preempt bash experiment/submit_ampere.sh experiment/02_qwen_autoregressive
```

Inside an existing one-GPU allocation, run `PROMPT_STYLE=long bash experiment/02_qwen_autoregressive/run.sh`.
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
results/textvqa_test/autoregressive_qwen/test.jsonl
results/coco_caption_test/autoregressive_qwen/test.jsonl
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
question; the decode-phase peak covers token generation. In this arm the phase boundary is the first generated token. Prefill slimming is
output-preserving (slim and no-slim tokens were identical in a 5-sample smoke run) and lowers
the decode-phase peak (maximum 8.49 versus 8.85 GiB in that smoke run). It does not lower the
prefill-phase peak, which is set before the language-model head, most likely by the vision
encoder. `kv_live_bytes` is a whole-model figure repeated in each device entry, so do not
sum it across GPUs. The KV cache capacity is 16,384 tokens, not the model's 128k default.
These smoke figures are not results of this experiment.

## Results

All figures come from one run per method per prompt style (Slurm jobs 392660 for `long` and 392661 for `short`), 100 samples per benchmark, greedy decoding, bf16, batch size 1, at most 1,024 new tokens, slim prefill on, 16,384-token KV capacity. Per-benchmark results are listed before anything else, and no cross-benchmark aggregate is computed. Quality scores are subset measurements.

### Hardware

| Style | Job / partition | GPU | UUID | Driver | Allocation | Device report |
| --- | --- | --- | --- | --- | --- | --- |
| long | 392660 / `preempt`, node `c2-2`, 00:42:57 for the 02 and 03 pair | NVIDIA RTX A6000 (47.5 GiB visible) | `GPU-a2e40ccd-5141-9e51-2b3f-0e9d71fabf73` | 615.71.09 | 1 GPU, 4 CPUs (affinity 9, 12, 29, 32), 22,528 MB, 3 h limit | [02 device.txt](runs/20261019T051448Z_a06dd479abba_long_tWbqsW/device.txt) |
| short | 392661 / `preempt`, node `c2-2`, 00:08:38 for the 02 and 03 pair | NVIDIA RTX A6000 (47.5 GiB visible) | `GPU-a2e40ccd-5141-9e51-2b3f-0e9d71fabf73` | 615.71.09 | 1 GPU, 4 CPUs (affinity 9, 12, 29, 32), 22,528 MB, 3 h limit | [02 device.txt](runs/20261019T062551Z_a06dd479abba_short_Jjml6V/device.txt) |

Both jobs ran on the same physical card and host (Intel Xeon Silver 4114, 40 logical CPUs on the host, 4 allocated). The paired 03 run in each job used the same card and allocation. The links point at generated run folders, which are ignored by Git and preserved on project storage, so they resolve only in a checkout that has those folders. Device reports are launch snapshots and do not measure runtime peaks.

### Reading these results

The `long` style is the speed measurement, because answers run to hundreds of tokens and decoding dominates. Its quality scores are secondary: TextVQA consensus accuracy requires an exact short answer and CIDEr-D's length penalty punishes paragraphs, so both are near zero for any method. The `short` style is the quality measurement. Its speed is secondary, because answers are only a few tokens and prefill dominates the wall time, especially for TextVQA. The head-to-head comparison, token agreement and divergence diagnosis are in the [03 README](../03_qwen_vispec/README.md#results).

Units: latency is wall time per question in seconds for prefill plus decoding, excluding model loading, prompt preparation and scoring. TTFT is the wait before the first answer token. TPOT is `(latency - TTFT) / (output tokens - 1)` in milliseconds, averaged per question for answers with more than one token. Throughput is the sum of output tokens divided by the sum of latencies. Output tokens include EOS. Truncated counts answers that hit the 1,024-token cap. Memory is `torch.cuda.max_memory_allocated` per question in GiB (mean and max over the 100 questions), and live KV is the whole-model figure in MiB. Phase peaks and live KV are read from the saved per-question measurements in `results/*/test.jsonl`, not from `summary.json`. Quality is VQA consensus accuracy in percent for TextVQA and CIDEr-D multiplied by 100 for COCO (IDF over the 100-image subset).

### Style `long` (speed measurement)

| Benchmark | Samples | Mean / median / p95 latency (s) | Mean TTFT (s) | Mean TPOT (ms) | Throughput (tokens/s) | Mean output tokens | Truncated |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| TextVQA | 100 | 11.538 / 8.461 / 31.777 | 0.949 | 37.5 | 24.55 | 283.31 | 4/100 (0.04) |
| COCO Captions | 100 | 4.318 / 3.989 / 7.766 | 0.221 | 37.6 | 25.47 | 109.96 | 0/100 (0.00) |

| Benchmark | Prefill peak mean / max (GiB) | Decode peak mean / max (GiB) | Live KV mean / max (MiB) | Quality | Scored |
| --- | --- | --- | --- | ---: | ---: |
| TextVQA | 10.69 / 12.08 | 8.41 / 8.42 | 47.3 / 81.9 | 0.0% VQA accuracy | 100/100 |
| COCO Captions | 8.71 / 9.08 | 8.38 / 8.38 | 18.1 / 24.1 | 1.87e-05 CIDEr-D x100 | 100/100 |

### Style `short` (quality measurement)

| Benchmark | Samples | Mean / median / p95 latency (s) | Mean TTFT (s) | Mean TPOT (ms) | Throughput (tokens/s) | Mean output tokens | Truncated |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| TextVQA | 100 | 1.108 / 1.037 / 1.552 | 0.956 | 38.2 | 4.54 | 5.03 | 0/100 (0.00) |
| COCO Captions | 100 | 0.754 / 0.732 / 0.978 | 0.226 | 38.4 | 19.59 | 14.76 | 0/100 (0.00) |

| Benchmark | Prefill peak mean / max (GiB) | Decode peak mean / max (GiB) | Live KV mean / max (MiB) | Quality | Scored |
| --- | --- | --- | --- | ---: | ---: |
| TextVQA | 10.69 / 12.08 | 8.41 / 8.41 | 36.7 / 46.6 | 71.0% VQA accuracy | 100/100 |
| COCO Captions | 8.71 / 9.08 | 8.38 / 8.38 | 14.8 / 21.0 | 121.40 CIDEr-D x100 | 100/100 |

### Open items

- Repeat-run variability: **pending**. There is one run per method per style.
- Energy and non-PyTorch GPU memory: **pending** (not measured).
- Validated equivalent-output speedup: **pending**. See the 03 README for the agreement and diagnosis results.

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
- This arm uses no draft tree, so draft settings do not apply. `compare_metrics` requires `method=autoregressive` here and a matching `prompt_style` in the paired 03 run.