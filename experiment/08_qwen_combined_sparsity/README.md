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

The [Week 3 results](#week-3-results-retention-sweep) score each prediction against the measurements.

Status: **measured for Week 3** at commit `c3a36ac25f95` in jobs 398885 (`long`) and 398923 (`short`), diagonal arms with both policies, 100/100 samples per benchmark, all runs exit 0 (the `long` style ran 1.0, 0.5 and 0.25, the `short` style also 0.75). In those jobs ViSpec ran 2.158 (TextVQA) and 2.315 (COCO) times faster than autoregressive decoding in the `long` style, at 2.141 and 2.284 accepted draft tokens per iteration, and 0.981 and 1.485 times in the `short` style (job 398923). A combined arm tracks the matching [06](../06_qwen_kv_eviction/README.md) eviction-only arm closely: live KV (arithmetic) within 0.005 of it, exact agreement within 1 of 100 samples, and `short` quality within 0.58 CIDEr-D points. Draft pruning therefore adds almost nothing to eviction, which is the ceiling argument again: the KV cache is 0.5% of the weight read per step at 1,000 tokens and 3.4% at 7,000. The 1.0 identity gate passed on every arm. Pending: GPU job 402165. It runs the fp32 offset oracle for the combined 0.25 spatial arm.

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

## Week 3 results (retention sweep)

### Summary

Speculative decoding comes first. In these jobs ViSpec completed Qwen2.5-VL-3B answers 2.2 to 2.3 times faster than autoregressive (AR) decoding in the `long` style, and about as fast as AR or 1.5 times faster in the `short` style. The table gives the same-job references, which every arm below is compared with.

| Style | Job | Benchmark | AR latency (s) | ViSpec latency (s) | ViSpec speedup vs AR | ViSpec accepted draft tokens per iteration | Exact agreement, ViSpec vs AR | Quality, AR / ViSpec |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| long | 398885 | TextVQA | 12.185 | 5.647 | 2.158 | 2.141 | 22/100 | near zero (not meaningful) |
| long | 398885 | COCO Captions | 4.584 | 1.980 | 2.315 | 2.284 | 28/100 | near zero (not meaningful) |
| short | 398923 | TextVQA | 1.100 | 1.122 | 0.981 | 0.690 | 96/100 | 71.0 / 70.0 |
| short | 398923 | COCO Captions | 0.784 | 0.528 | 1.485 | 1.746 | 88/100 | 121.40 / 119.68 |

In `long`, the speedup comes with 2.141 (TextVQA) and 2.284 (COCO) accepted draft tokens per iteration. In `short` the answers are 5 and 15 tokens long and prefill dominates: TTFT is 0.940 s of the 1.122 s ViSpec latency on TextVQA, so ViSpec shows no speedup there (0.981). ViSpec and AR outputs differ in token IDs on `long` (22/100 and 28/100 agree exactly) and nearly agree on `short` (96/100 and 88/100). In bf16 the batched tree verification and one-token decoding use different kernel shapes, so near-tied logits can flip; the Week 1 diagnosis found every divergence to be a near-tie ([03 README](../03_qwen_vispec/README.md#results)).

Sparsity comes second. Pruning the draft input on top of target eviction, at the same budget and policy, adds almost nothing to eviction alone:

- **Memory.** The live-KV ratio of a combined arm differs from the matching [06](../06_qwen_kv_eviction/README.md#week-3-results-retention-sweep) arm by at most 0.005 (`long`) and 0.0002 (`short`). It is 0.465 to 0.673 on `long` and 0.298 to 0.791 on `short`. The decode peak moved by 0.3% or less (ratio 0.997 to 1.000).
- **Output and quality.** Exact agreement with the ViSpec reference differs from the 06 arm by at most 1 of 100 samples. `short` quality is the same as the 06 arm on TextVQA and within 0.58 CIDEr-D points on COCO. The spatial policy lost quality at 0.25 (TextVQA 70.0 to 60.8, COCO 119.68 to 104.98), and the attention policy kept it (70.7 and 122.86), within the noise of 100 samples.
- **Acceptance and latency.** Acceptance rose under eviction as in 06 and differs from the 06 arm by at most 0.045 (`long`) and 0.024 (`short`). TTFT is unchanged (ratios 0.987 to 1.011). End-to-end latency ratios run from 0.902 to 1.130 and follow the output length (see [Ratios that run against the trend](#ratios-that-run-against-the-trend)).

### Why the latency effect is small

This is the explanation for small effects, not a new method. On this model the KV cache costs 36,864 bytes per token (36 layers x 2 x 2 KV heads x 128 x 2 bytes), while every decode step reads about 6.99 GiB of weights. The cache is therefore 0.5% of the weight read at 1,000 tokens and 3.4% at 7,000 tokens, and the two are equal only at about 204,000 tokens. Decode is also host-bound: the [04 profile](../04_qwen_decode_profile/README.md) found the GPU busy for 36 to 50% of each step. Eviction runs once after prefill, so it cannot lower TTFT. The adaptor compresses the image to `num_q` = 2 queries at every budget, so pruning its input leaves little for the draft to lose or gain. Together these fit the near-identical combined and 06 results. The host-bound reading is an inference from the Week 2 profile and is not tested by these runs.

### Setup

Four Slurm jobs, all at commit `c3a36ac25f95` with a clean tree, greedy decoding (temperature 0.0), bf16, batch size 1, at most 1,024 new tokens, depth 3, top-k 8, total-token 30, `num_q` 2, slim prefill on and the compact KV layout. Every run exited 0 and scored 100/100 samples on both benchmarks. This README uses job 398885 (`long`) and job 398923 (`short`). The other `long` jobs belong to [05](../05_qwen_draft_pruning/README.md#week-3-results-retention-sweep) (398884) and [06](../06_qwen_kv_eviction/README.md#week-3-results-retention-sweep) (398883).

**The budgets differ by style.** The `long` style ran retention 1.0, 0.5 and 0.25. The 0.75 budget was dropped from `long` to save GPU time. The `short` style ran all four (1.0, 0.75, 0.5 and 0.25). Both policies (`spatial`, `attention`) ran at every budget, with the same budget and policy on both sides (the diagonal).

Each arm is compared with the same-policy 1.0 arm from the same job (the primary reading, because the 1.0 arm runs the same selection code) and with the unpruned 03 ViSpec reference of that job (the second reading). The job's 1.0 arms against that reference give the overhead and noise floor. The matching 06 arms ran in another job for `long` and in the same job for `short`, so the `long` comparison with 06 compares ratios, never absolute times.

### Week 3 hardware

Each job ran on one physical card, so an arm and its references share hardware. The `long` job here (398885) and the `short` job ran on the same card at different times. **Absolute times are compared only within a job, never across jobs.** All ratios below stay inside one job.

| Style | Job / partition (complete attempt) | Node and host CPU | GPU | UUID | Driver | Allocation | Device report (one per job; every run folder has its own, listed below) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| long | 398885 / `preempt`, restart 1, 2026-11-03, 22:01 to 2026-11-04, 00:15 UTC, 4 h limit | `c1-8`, Intel Xeon Silver 4116 @ 2.10 GHz | NVIDIA RTX A6000 (49,140 MiB total) | `GPU-eb032c66-7f3f-d1b6-96cf-c9db5600d457` | 615.71.09 | 1 GPU, 4 CPUs (affinity 4, 18, 28, 42), 22,528 MB | [08 1.0:spatial device.txt](runs/20261103T224725Z_c3a36ac25f95_long_r1.0_spatial_zLGPCZ/device.txt) |
| short | 398923 / `preempt`, restart 0, 2026-11-01, 12:19 to 14:06 UTC, 4 h limit | `c1-8`, Intel Xeon Silver 4116 @ 2.10 GHz | NVIDIA RTX A6000 (49,140 MiB total) | `GPU-eb032c66-7f3f-d1b6-96cf-c9db5600d457` | 615.71.09 | 1 GPU, 4 CPUs (affinity 2, 18, 26, 42), 22,528 MB | [08 1.0:spatial device.txt](runs/20261101T133421Z_c3a36ac25f95_short_r1.0_spatial_wkj6oq/device.txt) |

The links point at generated run folders, which Git ignores and which are preserved on project storage, so they resolve only in a checkout that has them. Device reports are launch snapshots and do not measure runtime peaks. The job command is recorded in each run's `metadata.txt`. A requeued job reruns the runner into new folders, and only the complete attempt of each job is used; interrupted attempts are excluded.

#### Week 3 run folders and device reports

| Style | Job | Run | Arm | Run folder (preserved on project storage) | Device report |
| --- | --- | --- | --- | --- | --- |
| long | 398885 | 02 | reference | [`20261103T220127Z_c3a36ac25f95_long_3e78UW`](../02_qwen_autoregressive/runs/20261103T220127Z_c3a36ac25f95_long_3e78UW/) | [device.txt](../02_qwen_autoregressive/runs/20261103T220127Z_c3a36ac25f95_long_3e78UW/device.txt) |
| long | 398885 | 03 | reference | [`20261103T223313Z_c3a36ac25f95_long_K6xO4a`](../03_qwen_vispec/runs/20261103T223313Z_c3a36ac25f95_long_K6xO4a/) | [device.txt](../03_qwen_vispec/runs/20261103T223313Z_c3a36ac25f95_long_K6xO4a/device.txt) |
| long | 398885 | 08 | 1.0:attention | [`20261103T230148Z_c3a36ac25f95_long_r1.0_attention_UX80y3`](runs/20261103T230148Z_c3a36ac25f95_long_r1.0_attention_UX80y3/) | [device.txt](runs/20261103T230148Z_c3a36ac25f95_long_r1.0_attention_UX80y3/device.txt) |
| long | 398885 | 08 | 1.0:spatial | [`20261103T224725Z_c3a36ac25f95_long_r1.0_spatial_zLGPCZ`](runs/20261103T224725Z_c3a36ac25f95_long_r1.0_spatial_zLGPCZ/) | [device.txt](runs/20261103T224725Z_c3a36ac25f95_long_r1.0_spatial_zLGPCZ/device.txt) |
| long | 398885 | 08 | 0.5:attention | [`20261103T233125Z_c3a36ac25f95_long_r0.5_attention_HnYw9U`](runs/20261103T233125Z_c3a36ac25f95_long_r0.5_attention_HnYw9U/) | [device.txt](runs/20261103T233125Z_c3a36ac25f95_long_r0.5_attention_HnYw9U/device.txt) |
| long | 398885 | 08 | 0.5:spatial | [`20261103T231602Z_c3a36ac25f95_long_r0.5_spatial_YDWAF1`](runs/20261103T231602Z_c3a36ac25f95_long_r0.5_spatial_YDWAF1/) | [device.txt](runs/20261103T231602Z_c3a36ac25f95_long_r0.5_spatial_YDWAF1/device.txt) |
| long | 398885 | 08 | 0.25:attention | [`20261104T000122Z_c3a36ac25f95_long_r0.25_attention_7EjIiP`](runs/20261104T000122Z_c3a36ac25f95_long_r0.25_attention_7EjIiP/) | [device.txt](runs/20261104T000122Z_c3a36ac25f95_long_r0.25_attention_7EjIiP/device.txt) |
| long | 398885 | 08 | 0.25:spatial | [`20261103T234624Z_c3a36ac25f95_long_r0.25_spatial_EYraap`](runs/20261103T234624Z_c3a36ac25f95_long_r0.25_spatial_EYraap/) | [device.txt](runs/20261103T234624Z_c3a36ac25f95_long_r0.25_spatial_EYraap/device.txt) |
| short | 398923 | 02 | reference | [`20261101T121923Z_c3a36ac25f95_short_J5AdE5`](../02_qwen_autoregressive/runs/20261101T121923Z_c3a36ac25f95_short_J5AdE5/) | [device.txt](../02_qwen_autoregressive/runs/20261101T121923Z_c3a36ac25f95_short_J5AdE5/device.txt) |
| short | 398923 | 03 | reference | [`20261101T122530Z_c3a36ac25f95_short_ePf9XE`](../03_qwen_vispec/runs/20261101T122530Z_c3a36ac25f95_short_ePf9XE/) | [device.txt](../03_qwen_vispec/runs/20261101T122530Z_c3a36ac25f95_short_ePf9XE/device.txt) |
| short | 398923 | 08 | 1.0:attention | [`20261101T133823Z_c3a36ac25f95_short_r1.0_attention_s1WTiN`](runs/20261101T133823Z_c3a36ac25f95_short_r1.0_attention_s1WTiN/) | [device.txt](runs/20261101T133823Z_c3a36ac25f95_short_r1.0_attention_s1WTiN/device.txt) |
| short | 398923 | 08 | 1.0:spatial | [`20261101T133421Z_c3a36ac25f95_short_r1.0_spatial_wkj6oq`](runs/20261101T133421Z_c3a36ac25f95_short_r1.0_spatial_wkj6oq/) | [device.txt](runs/20261101T133421Z_c3a36ac25f95_short_r1.0_spatial_wkj6oq/device.txt) |
| short | 398923 | 08 | 0.75:attention | [`20261101T134628Z_c3a36ac25f95_short_r0.75_attention_rhcIG2`](runs/20261101T134628Z_c3a36ac25f95_short_r0.75_attention_rhcIG2/) | [device.txt](runs/20261101T134628Z_c3a36ac25f95_short_r0.75_attention_rhcIG2/device.txt) |
| short | 398923 | 08 | 0.75:spatial | [`20261101T134226Z_c3a36ac25f95_short_r0.75_spatial_DP8uZj`](runs/20261101T134226Z_c3a36ac25f95_short_r0.75_spatial_DP8uZj/) | [device.txt](runs/20261101T134226Z_c3a36ac25f95_short_r0.75_spatial_DP8uZj/device.txt) |
| short | 398923 | 08 | 0.5:attention | [`20261101T135438Z_c3a36ac25f95_short_r0.5_attention_okSOsU`](runs/20261101T135438Z_c3a36ac25f95_short_r0.5_attention_okSOsU/) | [device.txt](runs/20261101T135438Z_c3a36ac25f95_short_r0.5_attention_okSOsU/device.txt) |
| short | 398923 | 08 | 0.5:spatial | [`20261101T135032Z_c3a36ac25f95_short_r0.5_spatial_bJEWpe`](runs/20261101T135032Z_c3a36ac25f95_short_r0.5_spatial_bJEWpe/) | [device.txt](runs/20261101T135032Z_c3a36ac25f95_short_r0.5_spatial_bJEWpe/device.txt) |
| short | 398923 | 08 | 0.25:attention | [`20261101T140246Z_c3a36ac25f95_short_r0.25_attention_WBOSjc`](runs/20261101T140246Z_c3a36ac25f95_short_r0.25_attention_WBOSjc/) | [device.txt](runs/20261101T140246Z_c3a36ac25f95_short_r0.25_attention_WBOSjc/device.txt) |
| short | 398923 | 08 | 0.25:spatial | [`20261101T135839Z_c3a36ac25f95_short_r0.25_spatial_PWxgvF`](runs/20261101T135839Z_c3a36ac25f95_short_r0.25_spatial_PWxgvF/) | [device.txt](runs/20261101T135839Z_c3a36ac25f95_short_r0.25_spatial_PWxgvF/device.txt) |

### Reading the Week 3 tables

Units follow the [03 README](../03_qwen_vispec/README.md#reading-these-results) and the [metrics guide](../METRICS.md). Latency and TTFT are seconds per question, TPOT is milliseconds, and mean output tokens count EOS (in parentheses, the answers that hit the 1,024-token cap). Decode ms per iteration is the mean decode time over the run's iterations (mean decode time per question times 100, divided by the iteration count), computed from each run's `summary.json`. Quality is TextVQA consensus accuracy in percent or COCO CIDEr-D x 100 (inverse document frequency from the 100-image subset, so a subset measurement).

- **Direction of ratios.** `Speedup vs AR` and both latency ratios are reference over arm, so a value above 1 means the arm was faster. Memory ratios are arm over reference, so a value above 1 means more memory.
- **Two latency readings, shown side by side.** The column against the same-policy 1.0 arm and the column against the unpruned ViSpec reference are both given and neither is preferred after the fact. A 1.0 arm has no same-policy reference, so its row shows only the reference reading, which measures overhead plus run-to-run noise.
- **Agreement and bf16.** Exact agreement is token-ID equality through the first EOS over 100 samples. Greedy decoding in bf16 flips near-tied logits when kernel shapes differ, so agreement reflects near-ties as well as the method effect, and it is a count out of 100 without a confidence interval. Agreement with AR is the secondary column. `diagnose_divergence` refuses combined arms because they evict target KV, so divergences cannot be classified as near-ties here.
- **Live KV is arithmetic.** `kv_tokens` times 36,864 bytes, not an allocator measurement. It includes the output tokens, so a ratio is larger than the retention budget, and an arm with longer or shorter outputs shifts its ratio. Compare with the output-token column.
- **Steady-state decode memory.** The post-compaction peak is the figure for steady-state decode memory, and its ratio is given beside the decode-peak ratio.
- **Long style.** Long answers are paragraphs, so strict VQA accuracy and CIDEr-D are near zero and not meaningful there. The `long` tables give agreement with the reference and output length instead of quality. Quality numbers come from the `short` style.

### Week 3 identity gate at retention 1.0

The gate passed: all 12 1.0 arms of the Week 3 jobs of 05, 06 and 08 (both policies, both styles) reproduce their same-job ViSpec reference token for token, 100/100 on TextVQA and 100/100 on COCO. The 08 arms are the 100/100 entries on the 1.0 rows below. The comparison files are in each arm's `comparisons/20261105T192451Z_identity_vs_03-ref/`.

### Style `long`, Week 3 (speed and agreement measurement)

| Benchmark | Arm | Latency (s) | TTFT (s) | Mean output tokens (capped) | TPOT (ms) | Speedup vs AR | Latency ratio vs same-policy 1.0 | Latency ratio vs unpruned ViSpec ref |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TextVQA | 02 reference (AR) | 12.185 | 0.932 | 283.31 (4) | 39.9 | 1.000 | n/a | n/a |
| TextVQA | 03 reference (ViSpec) | 5.647 | 0.936 | 277.70 (3) | 18.2 | 2.158 | n/a | n/a |
| TextVQA | 08 ViSpec 1.0:attention | 5.610 | 0.942 | 277.70 (3) | 18.1 | 2.172 | 1.000 | 1.006 |
| TextVQA | 08 ViSpec 1.0:spatial | 5.699 | 0.937 | 277.70 (3) | 18.3 | 2.138 | 1.000 | 0.991 |
| TextVQA | 08 ViSpec 0.5:attention | 6.120 | 0.938 | 315.94 (7) | 17.8 | 1.991 | 0.917 | 0.923 |
| TextVQA | 08 ViSpec 0.5:spatial | 6.315 | 0.940 | 328.27 (7) | 17.4 | 1.929 | 0.902 | 0.894 |
| TextVQA | 08 ViSpec 0.25:attention | 5.693 | 0.941 | 296.97 (5) | 17.8 | 2.140 | 0.986 | 0.992 |
| TextVQA | 08 ViSpec 0.25:spatial | 6.037 | 0.936 | 312.16 (4) | 17.0 | 2.018 | 0.944 | 0.935 |
| COCO Captions | 02 reference (AR) | 4.584 | 0.222 | 109.96 (0) | 40.0 | 1.000 | n/a | n/a |
| COCO Captions | 03 reference (ViSpec) | 1.980 | 0.221 | 109.89 (0) | 16.2 | 2.315 | n/a | n/a |
| COCO Captions | 08 ViSpec 1.0:attention | 2.059 | 0.232 | 109.89 (0) | 16.8 | 2.226 | 1.000 | 0.962 |
| COCO Captions | 08 ViSpec 1.0:spatial | 2.052 | 0.225 | 109.89 (0) | 16.9 | 2.234 | 1.000 | 0.965 |
| COCO Captions | 08 ViSpec 0.5:attention | 1.930 | 0.230 | 106.06 (0) | 16.3 | 2.375 | 1.067 | 1.026 |
| COCO Captions | 08 ViSpec 0.5:spatial | 2.050 | 0.223 | 110.29 (0) | 16.8 | 2.236 | 1.001 | 0.966 |
| COCO Captions | 08 ViSpec 0.25:attention | 1.822 | 0.231 | 98.60 (0) | 16.4 | 2.516 | 1.130 | 1.087 |
| COCO Captions | 08 ViSpec 0.25:spatial | 1.948 | 0.223 | 107.45 (0) | 16.3 | 2.354 | 1.054 | 1.017 |

*Exact agreement counts identical token IDs in bf16, where near-tied logits can flip with kernel shape, so it reflects near-ties as well as any method effect.*

| Benchmark | Arm | Accepted draft tokens per iteration (iterations) | Decode ms per iteration | Exact agreement with ViSpec reference | Exact agreement with AR (secondary) |
| --- | --- | --- | ---: | ---: | ---: |
| TextVQA | 02 reference (AR) | n/a (AR) | n/a | 22/100 | n/a |
| TextVQA | 03 reference (ViSpec) | 2.141 (8841) | 53.1 | n/a | 22/100 |
| TextVQA | 08 ViSpec 1.0:attention | 2.141 (8841) | 52.7 | 100/100 | 22/100 |
| TextVQA | 08 ViSpec 1.0:spatial | 2.141 (8841) | 53.7 | 100/100 | 22/100 |
| TextVQA | 08 ViSpec 0.5:attention | 2.265 (9677) | 53.4 | 7/100 | 5/100 |
| TextVQA | 08 ViSpec 0.5:spatial | 2.310 (9917) | 54.1 | 0/100 | 0/100 |
| TextVQA | 08 ViSpec 0.25:attention | 2.356 (8848) | 53.6 | 3/100 | 2/100 |
| TextVQA | 08 ViSpec 0.25:spatial | 2.272 (9540) | 53.3 | 0/100 | 0/100 |
| COCO Captions | 02 reference (AR) | n/a (AR) | n/a | 28/100 | n/a |
| COCO Captions | 03 reference (ViSpec) | 2.284 (3346) | 52.2 | n/a | 28/100 |
| COCO Captions | 08 ViSpec 1.0:attention | 2.284 (3346) | 54.2 | 100/100 | 28/100 |
| COCO Captions | 08 ViSpec 1.0:spatial | 2.284 (3346) | 54.2 | 100/100 | 28/100 |
| COCO Captions | 08 ViSpec 0.5:attention | 2.351 (3165) | 53.3 | 1/100 | 1/100 |
| COCO Captions | 08 ViSpec 0.5:spatial | 2.276 (3367) | 53.9 | 1/100 | 1/100 |
| COCO Captions | 08 ViSpec 0.25:attention | 2.362 (2933) | 53.8 | 2/100 | 1/100 |
| COCO Captions | 08 ViSpec 0.25:spatial | 2.387 (3172) | 54.0 | 0/100 | 0/100 |

*Live KV is arithmetic (`kv_tokens` times 36,864 bytes, output tokens included), so its ratio shifts with output length. The post-compaction peak is the steady-state decode figure.*

| Benchmark | Arm | Smallest kept count per image | Live KV mean (MiB, arithmetic) | Live KV ratio vs same-policy 1.0 | Decode peak mean / max (GiB) | Decode peak ratio vs same-policy 1.0 | Post-compaction peak ratio vs same-policy 1.0 |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| TextVQA | 02 reference (AR) | n/a | 47.3 | n/a | 7.945 / 7.970 | n/a | n/a |
| TextVQA | 03 reference (ViSpec) | n/a | 47.1 | n/a | 7.993 / 8.015 | n/a | n/a |
| TextVQA | 08 ViSpec 1.0:attention | 729 | 47.1 | 1.000 | 7.993 / 8.015 | 1.000 | 1.000 |
| TextVQA | 08 ViSpec 1.0:spatial | 729 | 47.1 | 1.000 | 7.993 / 8.015 | 1.000 | 1.000 |
| TextVQA | 08 ViSpec 0.5:attention | 364 | 31.3 | 0.664 | 7.977 / 7.996 | 0.998 | 0.998 |
| TextVQA | 08 ViSpec 0.5:spatial | 364 | 31.7 | 0.673 | 7.977 / 7.997 | 0.998 | 0.998 |
| TextVQA | 08 ViSpec 0.25:attention | 182 | 22.0 | 0.467 | 7.968 / 7.981 | 0.997 | 0.997 |
| TextVQA | 08 ViSpec 0.25:spatial | 182 | 22.5 | 0.478 | 7.968 / 7.985 | 0.997 | 0.997 |
| COCO Captions | 02 reference (AR) | n/a | 18.1 | n/a | 7.878 / 7.893 | n/a | n/a |
| COCO Captions | 03 reference (ViSpec) | n/a | 18.1 | n/a | 7.934 / 7.946 | n/a | n/a |
| COCO Captions | 08 ViSpec 1.0:attention | 264 | 18.1 | 1.000 | 7.934 / 7.946 | 1.000 | 1.000 |
| COCO Captions | 08 ViSpec 1.0:spatial | 264 | 18.1 | 1.000 | 7.934 / 7.946 | 1.000 | 1.000 |
| COCO Captions | 08 ViSpec 0.5:attention | 132 | 11.8 | 0.651 | 7.929 / 7.936 | 0.999 | 0.999 |
| COCO Captions | 08 ViSpec 0.5:spatial | 132 | 11.9 | 0.659 | 7.928 / 7.938 | 0.999 | 0.999 |
| COCO Captions | 08 ViSpec 0.25:attention | 66 | 8.4 | 0.465 | 7.925 / 7.933 | 0.999 | 0.999 |
| COCO Captions | 08 ViSpec 0.25:spatial | 66 | 8.7 | 0.482 | 7.925 / 7.934 | 0.999 | 0.999 |

Retention 0.75 was not run in `long`. The smallest image keeps 182 of 729 tokens at 0.25 and 364 at 0.5 on TextVQA, and 66 of 264 and 132 on COCO, on both sides.

### Style `short`, Week 3 (quality measurement)

| Benchmark | Arm | Latency (s) | TTFT (s) | Mean output tokens (capped) | TPOT (ms) | Speedup vs AR | Latency ratio vs same-policy 1.0 | Latency ratio vs unpruned ViSpec ref |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TextVQA | 02 reference (AR) | 1.100 | 0.934 | 5.03 (0) | 41.6 | 1.000 | n/a | n/a |
| TextVQA | 03 reference (ViSpec) | 1.122 | 0.940 | 5.24 (0) | 69.2 | 0.981 | n/a | n/a |
| TextVQA | 08 ViSpec 1.0:attention | 1.125 | 0.943 | 5.24 (0) | 69.5 | 0.977 | 1.000 | 0.997 |
| TextVQA | 08 ViSpec 1.0:spatial | 1.122 | 0.939 | 5.24 (0) | 69.5 | 0.980 | 1.000 | 1.000 |
| TextVQA | 08 ViSpec 0.75:attention | 1.125 | 0.943 | 5.31 (0) | 70.1 | 0.978 | 1.001 | 0.997 |
| TextVQA | 08 ViSpec 0.75:spatial | 1.130 | 0.939 | 5.69 (0) | 70.0 | 0.973 | 0.993 | 0.993 |
| TextVQA | 08 ViSpec 0.5:attention | 1.127 | 0.941 | 5.31 (0) | 70.7 | 0.976 | 0.998 | 0.995 |
| TextVQA | 08 ViSpec 0.5:spatial | 1.174 | 0.940 | 7.88 (0) | 70.3 | 0.937 | 0.956 | 0.955 |
| TextVQA | 08 ViSpec 0.25:attention | 1.127 | 0.940 | 5.43 (0) | 70.3 | 0.976 | 0.998 | 0.995 |
| TextVQA | 08 ViSpec 0.25:spatial | 1.123 | 0.938 | 5.41 (0) | 69.1 | 0.979 | 0.999 | 0.999 |
| COCO Captions | 02 reference (AR) | 0.784 | 0.222 | 14.76 (0) | 40.9 | 1.000 | n/a | n/a |
| COCO Captions | 03 reference (ViSpec) | 0.528 | 0.222 | 15.05 (0) | 22.2 | 1.485 | n/a | n/a |
| COCO Captions | 08 ViSpec 1.0:attention | 0.540 | 0.231 | 15.05 (0) | 22.4 | 1.454 | 1.000 | 0.979 |
| COCO Captions | 08 ViSpec 1.0:spatial | 0.529 | 0.222 | 15.05 (0) | 22.3 | 1.482 | 1.000 | 0.999 |
| COCO Captions | 08 ViSpec 0.75:attention | 0.532 | 0.230 | 14.83 (0) | 22.1 | 1.475 | 1.015 | 0.994 |
| COCO Captions | 08 ViSpec 0.75:spatial | 0.535 | 0.222 | 15.16 (0) | 22.4 | 1.467 | 0.990 | 0.988 |
| COCO Captions | 08 ViSpec 0.5:attention | 0.543 | 0.231 | 15.15 (0) | 22.4 | 1.445 | 0.994 | 0.973 |
| COCO Captions | 08 ViSpec 0.5:spatial | 0.520 | 0.224 | 14.48 (0) | 22.3 | 1.508 | 1.017 | 1.016 |
| COCO Captions | 08 ViSpec 0.25:attention | 0.525 | 0.231 | 14.51 (0) | 22.0 | 1.493 | 1.027 | 1.006 |
| COCO Captions | 08 ViSpec 0.25:spatial | 0.536 | 0.223 | 15.27 (0) | 22.2 | 1.465 | 0.988 | 0.987 |

*Exact agreement counts identical token IDs in bf16, where near-tied logits can flip with kernel shape, so it reflects near-ties as well as any method effect.*

| Benchmark | Arm | Accepted draft tokens per iteration (iterations) | Decode ms per iteration | Exact agreement with ViSpec reference | Exact agreement with AR (secondary) | Quality | Change vs ViSpec reference |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| TextVQA | 02 reference (AR) | n/a (AR) | n/a | 96/100 | n/a | 71.0 | n/a |
| TextVQA | 03 reference (ViSpec) | 0.690 (310) | 54.6 | n/a | 96/100 | 70.0 | n/a |
| TextVQA | 08 ViSpec 1.0:attention | 0.690 (310) | 54.7 | 100/100 | 96/100 | 70.0 | +0.0 |
| TextVQA | 08 ViSpec 1.0:spatial | 0.690 (310) | 54.8 | 100/100 | 96/100 | 70.0 | +0.0 |
| TextVQA | 08 ViSpec 0.75:attention | 0.718 (309) | 54.7 | 95/100 | 95/100 | 71.0 | +1.0 |
| TextVQA | 08 ViSpec 0.75:spatial | 0.745 (326) | 54.7 | 89/100 | 87/100 | 69.6 | -0.4 |
| TextVQA | 08 ViSpec 0.5:attention | 0.696 (313) | 55.2 | 96/100 | 94/100 | 70.0 | +0.0 |
| TextVQA | 08 ViSpec 0.5:spatial | 0.960 (402) | 55.1 | 83/100 | 83/100 | 68.4 | -1.6 |
| TextVQA | 08 ViSpec 0.25:attention | 0.718 (316) | 55.1 | 93/100 | 92/100 | 70.7 | +0.7 |
| TextVQA | 08 ViSpec 0.25:spatial | 0.717 (315) | 54.7 | 69/100 | 70/100 | 60.8 | -9.2 |
| COCO Captions | 02 reference (AR) | n/a (AR) | n/a | 88/100 | n/a | 121.40 | n/a |
| COCO Captions | 03 reference (ViSpec) | 1.746 (548) | 53.7 | n/a | 88/100 | 119.68 | n/a |
| COCO Captions | 08 ViSpec 1.0:attention | 1.746 (548) | 54.1 | 100/100 | 88/100 | 119.68 | +0.00 |
| COCO Captions | 08 ViSpec 1.0:spatial | 1.746 (548) | 53.9 | 100/100 | 88/100 | 119.68 | +0.00 |
| COCO Captions | 08 ViSpec 0.75:attention | 1.757 (538) | 53.8 | 77/100 | 76/100 | 122.27 | +2.59 |
| COCO Captions | 08 ViSpec 0.75:spatial | 1.727 (556) | 54.1 | 49/100 | 49/100 | 120.75 | +1.07 |
| COCO Captions | 08 ViSpec 0.5:attention | 1.760 (549) | 54.5 | 44/100 | 43/100 | 110.26 | -9.42 |
| COCO Captions | 08 ViSpec 0.5:spatial | 1.763 (524) | 54.2 | 38/100 | 41/100 | 117.07 | -2.61 |
| COCO Captions | 08 ViSpec 0.25:attention | 1.796 (519) | 54.3 | 22/100 | 21/100 | 122.86 | +3.18 |
| COCO Captions | 08 ViSpec 0.25:spatial | 1.766 (552) | 54.4 | 10/100 | 9/100 | 104.98 | -14.70 |

*Live KV is arithmetic (`kv_tokens` times 36,864 bytes, output tokens included), so its ratio shifts with output length. The post-compaction peak is the steady-state decode figure.*

| Benchmark | Arm | Smallest kept count per image | Live KV mean (MiB, arithmetic) | Live KV ratio vs same-policy 1.0 | Decode peak mean / max (GiB) | Decode peak ratio vs same-policy 1.0 | Post-compaction peak ratio vs same-policy 1.0 |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| TextVQA | 02 reference (AR) | n/a | 36.7 | n/a | 7.944 / 7.968 | n/a | n/a |
| TextVQA | 03 reference (ViSpec) | n/a | 36.7 | n/a | 7.982 / 8.005 | n/a | n/a |
| TextVQA | 08 ViSpec 1.0:attention | 729 | 36.7 | 1.000 | 7.982 / 8.005 | 1.000 | 1.000 |
| TextVQA | 08 ViSpec 1.0:spatial | 729 | 36.7 | 1.000 | 7.982 / 8.005 | 1.000 | 1.000 |
| TextVQA | 08 ViSpec 0.75:attention | 547 | 28.1 | 0.766 | 7.973 / 7.992 | 0.999 | 0.999 |
| TextVQA | 08 ViSpec 0.75:spatial | 547 | 28.2 | 0.766 | 7.973 / 7.993 | 0.999 | 0.999 |
| TextVQA | 08 ViSpec 0.5:attention | 364 | 19.5 | 0.532 | 7.965 / 7.984 | 0.998 | 0.998 |
| TextVQA | 08 ViSpec 0.5:spatial | 364 | 19.6 | 0.534 | 7.965 / 7.983 | 0.998 | 0.998 |
| TextVQA | 08 ViSpec 0.25:attention | 182 | 10.9 | 0.298 | 7.956 / 7.974 | 0.997 | 0.997 |
| TextVQA | 08 ViSpec 0.25:spatial | 182 | 10.9 | 0.298 | 7.957 / 7.972 | 0.997 | 0.997 |
| COCO Captions | 02 reference (AR) | n/a | 14.8 | n/a | 7.878 / 7.893 | n/a | n/a |
| COCO Captions | 03 reference (ViSpec) | n/a | 14.8 | n/a | 7.930 / 7.948 | n/a | n/a |
| COCO Captions | 08 ViSpec 1.0:attention | 264 | 14.8 | 1.000 | 7.930 / 7.948 | 1.000 | 1.000 |
| COCO Captions | 08 ViSpec 1.0:spatial | 264 | 14.8 | 1.000 | 7.930 / 7.948 | 1.000 | 1.000 |
| COCO Captions | 08 ViSpec 0.75:attention | 198 | 11.7 | 0.791 | 7.927 / 7.944 | 1.000 | 1.000 |
| COCO Captions | 08 ViSpec 0.75:spatial | 198 | 11.7 | 0.791 | 7.927 / 7.942 | 1.000 | 1.000 |
| COCO Captions | 08 ViSpec 0.5:attention | 132 | 8.6 | 0.582 | 7.924 / 7.938 | 0.999 | 0.999 |
| COCO Captions | 08 ViSpec 0.5:spatial | 132 | 8.6 | 0.581 | 7.923 / 7.936 | 0.999 | 0.999 |
| COCO Captions | 08 ViSpec 0.25:attention | 66 | 5.5 | 0.371 | 7.921 / 7.933 | 0.999 | 0.999 |
| COCO Captions | 08 ViSpec 0.25:spatial | 66 | 5.5 | 0.373 | 7.921 / 7.933 | 0.999 | 0.999 |

The `short` arms of 08 repeat the 06 results at every budget (next section), so the quality findings are those of 06: at 0.25 the spatial policy lost 9.2 TextVQA points and 14.70 CIDEr-D points against the reference, and the attention policy lost none. COCO attention quality is not monotone in the budget (122.27, 110.26 and 122.86 at 0.75, 0.5 and 0.25), which is within CIDEr-D noise at 100 images.

### Acceptance and the 06 arms

Each combined arm is shown next to the 06 arm of the same budget and policy (`long`: job 398883, a different job than 08's 398885; `short`: the same job). The `long` columns compare ratios, never absolute times.

Style `long`:

*Exact agreement counts identical token IDs in bf16, where near-tied logits can flip with kernel shape, so it reflects near-ties as well as any method effect.*

| Benchmark | Arm | Exact agreement with ViSpec reference, 06 / 08 | Accepted draft tokens per iteration, 06 / 08 | Latency ratio vs same-policy 1.0, 06 / 08 | Live KV ratio vs same-policy 1.0, 06 / 08 |
| --- | --- | --- | --- | --- | --- |
| TextVQA | 0.5:attention | 6/100 / 7/100 | 2.255 / 2.265 | 0.932 / 0.917 | 0.661 / 0.664 |
| TextVQA | 0.5:spatial | 0/100 / 0/100 | 2.265 / 2.310 | 0.875 / 0.902 | 0.671 / 0.673 |
| TextVQA | 0.25:attention | 2/100 / 3/100 | 2.324 / 2.356 | 0.983 / 0.986 | 0.470 / 0.467 |
| TextVQA | 0.25:spatial | 0/100 / 0/100 | 2.286 / 2.272 | 0.897 / 0.944 | 0.484 / 0.478 |
| COCO Captions | 0.5:attention | 1/100 / 1/100 | 2.352 / 2.351 | 1.041 / 1.067 | 0.650 / 0.651 |
| COCO Captions | 0.5:spatial | 2/100 / 1/100 | 2.291 / 2.276 | 0.997 / 1.001 | 0.659 / 0.659 |
| COCO Captions | 0.25:attention | 1/100 / 2/100 | 2.360 / 2.362 | 1.106 / 1.130 | 0.468 / 0.465 |
| COCO Captions | 0.25:spatial | 0/100 / 0/100 | 2.379 / 2.387 | 1.078 / 1.054 | 0.482 / 0.482 |

Style `short`:

*Exact agreement counts identical token IDs in bf16, where near-tied logits can flip with kernel shape, so it reflects near-ties as well as any method effect.*

| Benchmark | Arm | Exact agreement with ViSpec reference, 06 / 08 | Accepted draft tokens per iteration, 06 / 08 | Latency ratio vs same-policy 1.0, 06 / 08 | Live KV ratio vs same-policy 1.0, 06 / 08 | Quality, 06 / 08 |
| --- | --- | --- | --- | --- | --- | --- |
| TextVQA | 0.75:attention | 95/100 / 95/100 | 0.718 / 0.718 | 1.002 / 1.001 | 0.766 / 0.766 | 71.0 / 71.0 |
| TextVQA | 0.75:spatial | 89/100 / 89/100 | 0.745 / 0.745 | 0.995 / 0.993 | 0.766 / 0.766 | 69.6 / 69.6 |
| TextVQA | 0.5:attention | 96/100 / 96/100 | 0.696 / 0.696 | 0.999 / 0.998 | 0.532 / 0.532 | 70.0 / 70.0 |
| TextVQA | 0.5:spatial | 83/100 / 83/100 | 0.983 / 0.960 | 0.955 / 0.956 | 0.535 / 0.534 | 68.4 / 68.4 |
| TextVQA | 0.25:attention | 93/100 / 93/100 | 0.718 / 0.718 | 0.998 / 0.998 | 0.298 / 0.298 | 70.7 / 70.7 |
| TextVQA | 0.25:spatial | 69/100 / 69/100 | 0.717 / 0.717 | 0.996 / 0.999 | 0.298 / 0.298 | 60.8 / 60.8 |
| COCO Captions | 0.75:attention | 77/100 / 77/100 | 1.746 / 1.757 | 1.011 / 1.015 | 0.791 / 0.791 | 121.84 / 122.27 |
| COCO Captions | 0.75:spatial | 49/100 / 49/100 | 1.727 / 1.727 | 0.989 / 0.990 | 0.791 / 0.791 | 120.75 / 120.75 |
| COCO Captions | 0.5:attention | 43/100 / 44/100 | 1.736 / 1.760 | 0.998 / 0.994 | 0.582 / 0.582 | 110.84 / 110.26 |
| COCO Captions | 0.5:spatial | 38/100 / 38/100 | 1.774 / 1.763 | 1.013 / 1.017 | 0.581 / 0.581 | 117.07 / 117.07 |
| COCO Captions | 0.25:attention | 22/100 / 22/100 | 1.790 / 1.796 | 1.034 / 1.027 | 0.371 / 0.371 | 122.86 / 122.86 |
| COCO Captions | 0.25:spatial | 10/100 / 10/100 | 1.756 / 1.766 | 0.985 / 0.988 | 0.373 / 0.373 | 104.98 / 104.98 |

Memory and quality track the 06 arm closely: the live-KV ratios differ by at most 0.005 on `long` and 0.0002 on `short`, agreement by at most 1 of 100, and `short` quality by 0.58 CIDEr-D points at most (COCO 0.5:attention; TextVQA is identical).

Acceptance rises under eviction in the same way as in 06. On `long` TextVQA it rises from 2.141 to 2.265 (attention) and 2.310 (spatial) at 0.5, and to 2.356 (attention) and 2.272 (spatial) at 0.25. On COCO it rises from 2.284 to 2.351 (attention) and 2.276 (spatial) at 0.5, and to 2.362 (attention) and 2.387 (spatial) at 0.25. As in 06, the TextVQA rise is partly an effect of capped answers: the eviction arms have 4 to 7 answers at the cap against 3 in the reference. Over the 92 to 94 TextVQA questions below the cap in the reference and the arm, computed from each run's `samples.jsonl`, the rise is +0.02 to +0.13 (arms 2.135, 2.140, 2.164 and 2.218 against references of 2.082 to 2.114), against +0.12 to +0.21 for the full samples. COCO has no capped answer in this job, so its rise (-0.008 to +0.103) is the full-sample one. Decode time per iteration stays between 52.2 and 54.2 ms in every ViSpec run of the job, so we infer that part of the small per-token gain comes from the accepted tokens and not from cheaper steps. That is the same inference as in 06, with the same limits: different text per arm, one run, and the cause not tested.

### Ratios that run against the trend

- **Long TextVQA, all four arms slower (0.902 to 0.986).** The arms' answers are longer: 328.27 and 315.94 tokens at 0.5 (spatial, attention) and 312.16 and 296.97 at 0.25, against 277.70, and 4 to 7 answers reach the cap against 3. TPOT is the same or lower (17.0 to 17.8 ms against 18.1 and 18.3 for the 1.0 arms). It is a longer-answer effect and not a per-token slowdown.
- **Long COCO, faster with shorter answers (1.067 and 1.130 for attention at 0.5 and 0.25, and 1.054 for 0.25:spatial).** The answers are shorter (106.06, 98.60 and 107.45 tokens against 109.89) and acceptance is higher. The unpruned-reference readings are 1.026, 1.087 and 1.017, about 4% lower for the first two, because both 1.0 arms ran slower than the job's reference (0.965 and 0.962). The two readings differ by that gap, and one run per arm cannot separate it from run-to-run spread.
- **Long COCO 0.5:spatial (1.001) and its acceptance of 2.276.** The arm's answers are as long as the reference's (110.29 against 109.89 tokens), and acceptance is 0.008 below it. This arm shows no eviction speedup and no acceptance rise, as in 06 (0.997, 2.291), so it is the exception in the COCO pattern.
- **Short TextVQA 0.5:spatial (0.956).** The mean answer grew from 5.24 to 7.88 tokens (no answer capped), so the slowdown and the high acceptance (0.960) both describe longer text. TTFT is 0.940 s.
- **Short COCO 0.75:attention (1.015) and 0.25:attention (1.027).** The mean answers are shorter than the 1.0 arm's 15.05 tokens (14.83 and 14.51).

### Predictions against the Week 3 measurements

The predictions are those recorded before the runs (see [Purpose and hypothesis](#purpose-and-hypothesis)).

| Prediction (written before the runs) | Outcome | Evidence |
| --- | --- | --- |
| Memory and quality of a combined arm track the matching 06 arm | **Held** | Live-KV ratios differ from the 06 arm by 0.005 (`long`) and 0.0002 (`short`) at most. Agreement differs by at most 1 of 100. `short` quality differs by 0 (TextVQA) and 0.58 CIDEr-D points at most (COCO). |
| Draft pruning adds little on top of eviction: acceptance and latency stay within the spread that separates the 06 arm from its 1.0 arm | **Held** | Acceptance differs from the 06 arm by 0.045 (`long`) and 0.024 (`short`) at most, against rises of up to +0.21 from the 1.0 arm. Latency ratios differ by 0.047 (`long`, TextVQA 0.25:spatial, 0.944 against 0.897) and 0.007 (`short`) at most, against a gap of up to 0.125 between a 06 arm and its 1.0 arm. The `long` comparison crosses two jobs, it is the loosest result, and with one run per arm it cannot be separated from run-to-run spread. |
| Latency barely moves, for the reason given in 05 and 06; TPOT and TTFT are read alongside | **Held for TTFT and per-iteration time, partly held end to end** | TTFT ratios 0.987 to 1.011. Decode time per iteration 52.2 to 54.2 ms in every ViSpec run of the long job. End-to-end ratios 0.902 to 1.130 follow output length and acceptance. |

### Pending: fp32 offset oracle

Pending: GPU job 402165. It runs `offset_check` in fp32 on 5 samples at combined 0.25 spatial, with the draft retention and policy set as well. The result will be filled in here.

## Limits

- Diagonal arms only: the same retention and policy on both sides. Unequal budgets are not tested.
- One run per arm and style. Subsets are 100 samples, so agreement counts and quality
  differences are noisy, and no confidence intervals are computed.
- A combined arm changes the target's attention, so divergence from the reference cannot be classified
  as bf16 near-ties by `diagnose_divergence`, and cannot be split between the draft and target effects
  without the matching 05 and 06 arms.
- The `long` style did not run the 0.75 budget (dropped to save GPU time). The `long` 08 job and the `short` job ran at different times, and the three `long` jobs ran on two cards, so no absolute time is compared across jobs.
- The 08 and 06 `long` arms come from different jobs, so their comparison uses ratios only. The TextVQA acceptance rise is partly an effect of more answers at the token cap (see the Week 3 section).
- The fp32 offset oracle for the combined arm: Pending: GPU job 402165.
