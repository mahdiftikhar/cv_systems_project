# 06 — Qwen2.5-VL target-side visual KV eviction

## Purpose and hypothesis

This experiment measures Qwen2.5-VL-3B-Instruct when the target's KV cache keeps only a fraction of
each image's tokens after the prefill. Each arm runs under autoregressive or ViSpec decoding and is
compared with the same method at retention 1.0, run in the same job.

Predictions, not measurements. The [04 profile](../04_qwen_decode_profile/README.md) found decode
host-bound: the GPU is idle for at least half of each step and attention over the KV cache is 3 to
12% of kernel time. Eviction should therefore lower memory (live KV and decode peak) but barely move
latency. It can change the output, because later tokens attend to fewer image positions, so output
agreement and quality against the reference are measured rather than assumed.

Status: **pending**. No run has been made.

## Changes from the reference

The reference is 02 (autoregressive) or 03 (ViSpec), run in the same job with `KV_LAYOUT=compact`.
Each arm adds eviction after the prefill:

- `--kv-retention` keeps a fraction of each image's prompt tokens (0.25, 0.5, 0.75 or 1.0).
- `--kv-policy` picks which ones: `spatial` (uniform raster stride) or `attention` (top prefill
  attention from the prompt tokens after the last image token).
- Eviction happens once, removes only image-token positions and keeps at least 1 token per image.
- `--kv-layout compact` is forced, because eviction needs the per-sample cache.

## Models

| Role | Model |
| --- | --- |
| Target | `Qwen/Qwen2.5-VL-3B-Instruct`, local copy `models/Qwen2.5-VL-3B-Instruct` |
| Draft | `JLKang/ViSpec-Qwen2.5-VL-3B-Instruct`, local copy `models/ViSpec-Qwen2.5-VL-3B-Instruct` (used by ViSpec arms) |

## Data, splits and subsets

The same 100 TextVQA questions (seed-42 shuffle) and 100 COCO Karpathy-test images as 02 and 03,
in the `long` and `short` styles. See the [03 README](../03_qwen_vispec/README.md#data-splits-and-subsets).

## Settings

Greedy decoding (temperature 0.0, fixed), bf16, batch size 1, at most 1,024 new tokens, 100 samples
per benchmark, depth 3, top-k 8, total-token 30, `num_q` 2, slim prefill on, KV capacity 16,384 as an
upper bound for the compact layout.

| Variable | Default | Purpose |
| --- | --- | --- |
| `PROMPT_STYLE` | none (required) | `long` or `short` |
| `KV_ARMS` | `0.5:spatial` | Space-separated `<retention>:<policy>[:<method>]`. Method is `vispec` (default) or `autoregressive` |
| `BENCH_DIR`, `BASE_MODEL_PATH`, `SPEC_DIR`, `PYTHON_BIN` | as in 03 | Data and model locations |
| `DEPTH`, `TOP_K`, `TOTAL_TOKEN`, `NUM_Q` | 3, 8, 30, 2 | Draft tree (ViSpec arms) |

The runner validates every arm before it creates a run folder; an invalid list exits with status 2.
Each arm creates `runs/<UTC>_<commit12>_<style>_r<retention>_<policy>[_<method>]_XXXXXX/`, where the
method suffix appears only when the spec gives one. `metadata.txt` records `method`
(`vispec_speculative` or `autoregressive`), `kv_layout`, the four sparsity settings and `arm=<spec>`.
Shared code is in [`../qwen_arm_runner.sh`](../qwen_arm_runner.sh).

A failed arm is recorded and the remaining arms still run, but a failed benchmark stops that arm's remaining benchmarks; the runner exits 1 if any arm failed, and `run_sequence.sbatch` still continues to the next experiment.

## How to run

References first, then arms, in one job so the allocation matches:

```bash
PROMPT_STYLE=long KV_LAYOUT=compact KV_ARMS="0.5:spatial 0.5:attention 0.5:spatial:autoregressive" \
  CLUSTER_PARTITION=preempt SBATCH_EXTRA="--cpus-per-task=4 --mem=22G --time=<per task>" \
  bash experiment/submit_ampere.sh experiment/02_qwen_autoregressive experiment/03_qwen_vispec \
  experiment/06_qwen_kv_eviction
```

Compare an arm with the same method's reference:

```bash
python -m vispec.evaluation.compare_metrics --mode sparsity --original <reference report>... --draft <06 report>... --output-dir <new_dir>
```

## Results

**Pending.** Live KV, decode peak memory, latency, output agreement and quality are not measured yet.

## Limits

- Predictions above are untested.
- `diagnose_divergence` refuses eviction runs (`kv_retention` below 1.0), because its stock full-KV
  model cannot reproduce them. Divergences from the reference cannot be classified as near-ties.
- `kv_live_bytes` is a whole-model figure. One run per arm: repeat variability is unknown.
  Subsets are 100 samples, not full benchmarks.
