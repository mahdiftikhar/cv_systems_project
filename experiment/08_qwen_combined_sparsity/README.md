# 08 — Qwen2.5-VL combined draft and KV sparsity

## Purpose and hypothesis

This experiment measures ViSpec speculative decoding on Qwen2.5-VL-3B-Instruct when both sides see
fewer image tokens at once: the target's KV cache keeps a fraction of each image's tokens after the
prefill (as in [06](../06_qwen_kv_eviction/README.md)), and the draft's image adaptor sees the same
fraction (as in [05](../05_qwen_draft_pruning/README.md)). Each arm applies the same retention and the
same policy to both sides. The question is whether the two effects add, cancel or interact.

**Diagonal only.** Only the diagonal of the retention-by-retention grid is run: an arm
`R:P` sets `--kv-retention R --kv-policy P --draft-retention R --draft-policy P`. Off-diagonal pairs
(for example target 0.5 with draft 1.0, or different policies on the two sides) are not run. The
single-side effects are already given by 05 and 06, and the diagonal keeps the arm count at
retentions x policies per style. The cost is that an interaction between unequal budgets cannot be
separated here.

Predictions, recorded before the runs and framed from the Week 2 measurements. In Week 2, eviction
dominated the effects. At 0.5 it lowered live KV to 0.65 to 0.67 of the 1.0 arm in `long` and 0.53 to
0.58 in `short`, and it changed the output strongly: 0 to 6 of 100 exact agreement on `long` TextVQA
and 38 to 43 of 100 on `short` COCO. Draft pruning barely changed acceptance length (2.141 to 2.145 on
`long` TextVQA 0.5:spatial) or latency (`long` latency ratios against the same-policy 1.0 arm 0.989 to 1.017), and its agreement with the reference
was high on `short` (100/100) and mixed on `long` (48 to 77 of 100). We therefore predict:

- Memory and quality of a combined arm track the matching 06 ViSpec arm. Live KV falls by the same
  amount, and agreement and quality against the ViSpec reference sit close to the 06 arm's.
- Draft pruning adds little on top of eviction: acceptance length and latency stay within the spread
  that separates the 06 arm from its 1.0 arm, and any gap between a combined arm and the 06 arm of
  the same budget and policy is small next to the gap between that 06 arm and the reference.
- Latency barely moves, for the reason given in 05 and 06 (the [04 profile](../04_qwen_decode_profile/README.md)
  inferred that decode is host-bound). Longer outputs under eviction can still move end-to-end latency, so
  TPOT and TTFT are read alongside it.

These are predictions, not results. The Results section is `pending`.

Status: **pending**. No 08 run has been made.

## Changes from the reference

The reference is 03 (ViSpec), run in the same job with `KV_LAYOUT=compact` and no sparsity flags. Each
arm adds, for retention `R` and policy `P`:

- `--kv-retention R --kv-policy P`: target-side eviction of image tokens once after the prefill.
- `--draft-retention R --draft-policy P`: draft-side pruning of the image adaptor input.
- `--kv-layout compact`, forced, because eviction needs the per-sample cache.

Combined arms are ViSpec only (the draft does not exist under autoregressive decoding), so the
arm list takes no method suffix and `metadata.txt` records `method=vispec_speculative`.

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
upper bound for the compact layout. Budgets are 0.25, 0.5, 0.75 and 1.0. Policies are `spatial` and
`attention`.

| Variable | Default | Purpose |
| --- | --- | --- |
| `PROMPT_STYLE` | none (required) | `long` or `short` |
| `COMBINED_ARMS` | `0.5:spatial` | Space-separated `<retention>:<policy>`. A method suffix is rejected |
| `BENCH_DIR`, `BASE_MODEL_PATH`, `SPEC_DIR`, `PYTHON_BIN` | as in 03 | Data and model locations |
| `DEPTH`, `TOP_K`, `TOTAL_TOKEN`, `NUM_Q` | 3, 8, 30, 2 | Draft tree |

The runner validates every arm before it creates a run folder; an invalid list exits with status 2.
Each arm creates `runs/<UTC>_<commit12>_<style>_r<retention>_<policy>_XXXXXX/`. `metadata.txt` records
`method=vispec_speculative`, `kv_layout`, `sweep=combined`, the four sparsity settings and `arm=<spec>`.
Shared code is in [`../qwen_arm_runner.sh`](../qwen_arm_runner.sh).

A failed arm is recorded and the remaining arms still run, but a failed benchmark stops that arm's remaining benchmarks; the runner exits 1 if any arm failed, and `run_sequence.sbatch` still continues to the next experiment.

## Cluster prerequisites

1. Follow the [Ampere setup guide](../AMPERE_SETUP.md). Jobs run on the pinned 48 GB
   `NVIDIA RTX A6000` nodes and fail with status 3 on any other GPU model. That check lives only in
   `run_sequence.sbatch`, so results of record must be submitted through `submit_ampere.sh`.
2. Prepare assets once with `bash prepare.sh --suite qwen` inside a Slurm allocation. The runner
   repeats the offline check after the device capture and stops before any model loads if an asset is
   missing.
3. Commit code before launching, so the run records a clean commit.
4. Submit the 02 and 03 references with `KV_LAYOUT=compact` in the same job, so the allocation
   matches. `CLUSTER_PARTITION=preempt` reaches the same pinned nodes; a requeued job reruns the
   runner into fresh run folders.

## How to run

References first, then arms, in one job:

```bash
PROMPT_STYLE=long KV_LAYOUT=compact COMBINED_ARMS="0.5:spatial 0.5:attention" \
  CLUSTER_PARTITION=preempt SBATCH_EXTRA="--cpus-per-task=4 --mem=22G --time=<per task>" \
  bash experiment/submit_ampere.sh experiment/02_qwen_autoregressive experiment/03_qwen_vispec \
  experiment/08_qwen_combined_sparsity
```

Compare an arm with the ViSpec reference and with the matching 05 and 06 arms of the same job:

```bash
python -m vispec.evaluation.compare_metrics --mode sparsity --original <reference report>... --draft <08 report>... --output-dir <new_dir>
```

Latency and TTFT are compared with the same-policy 1.0 arm of the same job. Speedup against
autoregressive decoding uses the same job's 02 run. Ratios never span two jobs.
`diagnose_divergence` refuses these runs, because they evict target KV.

## Metric definitions

As in the [03 README](../03_qwen_vispec/README.md#metric-definitions) and the [metrics guide](../METRICS.md):
latency and TTFT in seconds per question, TPOT in milliseconds, accepted draft tokens per iteration,
exact token-ID agreement over 100 samples, and task quality (strict TextVQA score and CIDEr-D, the
latter computed over the 100-image subset). Memory comes from `kv_live_bytes` (arithmetic, repeated per
device entry) and the prefill and decode peaks.

## Results

**Pending.** No run exists yet. When the runs finish, this section follows the layout of the
[06 README](../06_qwen_kv_eviction/README.md#results): hardware and run folders, an identity gate at
retention 1.0, per-style tables, memory, and a table scoring each prediction above against the
measurements.

## Limits

- Diagonal arms only: the same retention and policy on both sides. Unequal budgets are not tested.
- One run per arm and style, once measured. Subsets are 100 samples, so agreement counts and quality
  differences are noisy, and no confidence intervals are computed.
- A combined arm changes the target's attention, so divergence from the reference cannot be classified
  as bf16 near-ties by `diagnose_divergence`, and cannot be split between the draft and target effects
  without the matching 05 and 06 arms.
- `long` and `short` ran on different nodes in Week 2, so their latencies are never pooled.
