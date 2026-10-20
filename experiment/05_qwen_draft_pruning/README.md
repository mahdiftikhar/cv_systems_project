# 05 — Qwen2.5-VL draft-side visual pruning

## Purpose and hypothesis

This experiment measures ViSpec speculative decoding on Qwen2.5-VL-3B-Instruct when the draft's image
adaptor sees only a fraction of each image's tokens. The target model is untouched, so the target still
verifies every draft and the greedy output should stay the same as the same-job reference at retention
1.0, up to the bf16 near-tie divergences already seen between 02 and 03.

Predictions, not measurements. Draft pruning should cost acceptance length at most slightly, because the
adaptor compresses the image tokens to `num_q` = 2 queries either way. It should barely move latency,
because the [04 profile](../04_qwen_decode_profile/README.md) found decode host-bound, and the
adaptor runs once per sample.

Status: **pending**. No run has been made.

## Changes from the reference

The reference is 03 run in the same job with `KV_LAYOUT=compact` (retention 1.0, no pruning). Each arm
changes only the draft side:

- `--draft-retention` keeps a fraction of each image's tokens (0.25, 0.5, 0.75 or 1.0).
- `--draft-policy` picks which ones: `spatial` (uniform raster stride) or `attention` (top attention
  from the adaptor's own learned queries).

Every arm also runs with `--kv-layout compact`, which sizes the target KV cache to each sample instead
of preallocating 16,384 slots. This makes live KV comparable with the 06 arms. The reference uses the
same layout, so layout is not a confound within a comparison.

## Models

| Role | Model |
| --- | --- |
| Target | `Qwen/Qwen2.5-VL-3B-Instruct`, local copy `models/Qwen2.5-VL-3B-Instruct` |
| Draft | `JLKang/ViSpec-Qwen2.5-VL-3B-Instruct`, local copy `models/ViSpec-Qwen2.5-VL-3B-Instruct` |

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
| `DRAFT_ARMS` | `0.5:spatial` | Space-separated `<retention>:<policy>`. Retention is 0.25, 0.5, 0.75 or 1.0. Policy is `spatial` or `attention`. No method suffix: every arm is ViSpec |
| `BENCH_DIR`, `BASE_MODEL_PATH`, `SPEC_DIR`, `PYTHON_BIN` | as in 03 | Data and model locations |
| `DEPTH`, `TOP_K`, `TOTAL_TOKEN`, `NUM_Q` | 3, 8, 30, 2 | Draft tree |

The runner validates every arm before it creates a run folder; an invalid list exits with status 2.
Each arm creates `runs/<UTC>_<commit12>_<style>_r<retention>_<policy>_XXXXXX/` with the 03 evidence
set plus `kv_layout`, the four sparsity settings and `arm=<spec>` in `metadata.txt`. Shared code is in
[`../qwen_arm_runner.sh`](../qwen_arm_runner.sh).

A failed arm is recorded and the remaining arms still run, but a failed benchmark stops that arm's remaining benchmarks; the runner exits 1 if any arm failed, and `run_sequence.sbatch` still continues to the next experiment.

## How to run

Run the references first, then the arms, in one job so the allocation matches:

```bash
PROMPT_STYLE=long KV_LAYOUT=compact DRAFT_ARMS="0.5:spatial 0.5:attention" CLUSTER_PARTITION=preempt \
  SBATCH_EXTRA="--cpus-per-task=4 --mem=22G --time=<per task>" \
  bash experiment/submit_ampere.sh experiment/03_qwen_vispec experiment/05_qwen_draft_pruning
```

`KV_LAYOUT=compact` applies to 03 (the reference) and is ignored by 05, which always uses compact.
Compare an arm with its reference:

```bash
python -m vispec.evaluation.compare_metrics --mode sparsity --original <03 report>... --draft <05 report>... --output-dir <new_dir>
```

## Results

**Pending.** Acceptance length, latency, memory and token agreement with the reference are not measured yet.

## Limits

- Predictions above are untested.
- Pruning changes only the draft input. It cannot shrink target KV memory; see
  [06](../06_qwen_kv_eviction/README.md) for that.
- One run per arm: repeat variability is unknown. Subsets are 100 samples, not full benchmarks.
- A near-tie divergence from the reference is not evidence of a bug. Use
  `vispec.evaluation.diagnose_divergence` to classify differences; draft-only arms are diagnosable.
