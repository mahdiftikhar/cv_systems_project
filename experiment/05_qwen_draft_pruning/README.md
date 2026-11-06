# 05 — Qwen2.5-VL draft-side visual pruning

## Purpose and hypothesis

This experiment measures ViSpec speculative decoding on Qwen2.5-VL-3B-Instruct when the draft's image
adaptor sees only a fraction of each image's tokens. The target model is untouched, so the target still
verifies every draft and the greedy output should stay the same as the same-job reference at retention
1.0, up to the bf16 near-tie divergences already seen between 02 and 03.

Predictions, recorded before the runs. Draft pruning should cost acceptance length at most slightly, because the
adaptor compresses the image tokens to `num_q` = 2 queries either way. It should barely move latency,
because the [04 profile](../04_qwen_decode_profile/README.md) inferred that decode is host-bound, and the
adaptor runs once per sample. The [Results](#predictions-against-the-measurements) section scores each prediction.

Status: **measured for Weeks 2 and 3.** Week 3 swept the retention budgets at commit `c3a36ac25f95` in jobs 398884 (`long`) and 398923 (`short`), both policies, 100/100 samples per benchmark, all runs exit 0 (the `long` style ran 1.0, 0.5 and 0.25, the `short` style also 0.75). In those jobs ViSpec ran 2.190 (TextVQA) and 2.304 (COCO) times faster than autoregressive decoding in the `long` style, at 2.141 and 2.284 accepted draft tokens per iteration, and 0.981 and 1.485 times in the `short` style (job 398923). Draft pruning left both where the unpruned reference had them: accepted tokens per iteration stayed within -0.015 to +0.010 of the 1.0 arm everywhere except `long` COCO spatial, whose rise comes from one answer that ran to the token cap, and latency ratios against the same-policy 1.0 arm lie between 0.976 and 1.037. The 1.0 identity gate passed on every arm in both weeks. In the `short` style every arm agrees with the reference on 99 or 100 of 100 samples, and in the `long` style 40 to 77 of 100 agree. The Week 2 unexplained case (`long` 0.5:attention, TextVQA question 35537 at output index 162) was traced by [experiment 07](../07_qwen_decode_drift/README.md): at that token the fork's bf16 logits tie exactly, and it was judged bf16 implementation noise, a known limitation. The Week 3 diagnosis (job 402165 diagnosed the 10 draft-only Week 3 arms with the stock full-KV model (fp32 on the first 50 cases of each arm). Nine pass, with every case a bf16 near-tie (at most 0.375 logits). `long` 0.5:attention fails, as in Week 2, on the one known case: TextVQA question 35537 at output index 162, where the reference has "." (token 13) and the pruned arm "**:" (token 95518). The stock bf16 margin is 0.625 logits and the fp32 margin 0.724 logits, and both pick "**:". [Experiment 07](../07_qwen_decode_drift/README.md) traced that case to an exact tie in the fork's own cached bf16 logits, and it was judged bf16 implementation noise, a known limitation. The threshold stays at 0.5 logits) is in [the Week 3 diagnosis](#draft-only-divergence-diagnosis-week-3).

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

Run the references first, then the arms, in one job so the allocation matches:

```bash
PROMPT_STYLE=long KV_LAYOUT=compact DRAFT_ARMS="0.5:spatial 0.5:attention" CLUSTER_PARTITION=preempt \
  SBATCH_EXTRA="--cpus-per-task=4 --mem=22G --time=<per task>" \
  bash experiment/submit_ampere.sh experiment/03_qwen_vispec experiment/05_qwen_draft_pruning
```

`KV_LAYOUT=compact` applies to 03 (the reference) and is ignored by 05, which always uses compact.

The Week 2 jobs ran 02, 03, 06 and 05 in one allocation per style, with `--cpus-per-task=4 --mem=22G` and a
6 h (`long`) or 2 h (`short`) limit. The arm lists are recorded as `arm_list` in each run's `metadata.txt`:
`1.0:spatial 1.0:attention 0.5:spatial 0.5:attention` for 05.
The Week 3 jobs ran 02, 03 and the arm experiments with `--cpus-per-task=4 --mem=22G --time=04:00:00`. The `long` style ran one job per experiment (05 in job 398884), and the `short` style ran 02, 03, 06, 05 and 08 together in job 398923. The `long` arm list is `1.0:spatial 1.0:attention 0.5:spatial 0.5:attention 0.25:spatial 0.25:attention`, and the `short` list adds `0.75:spatial 0.75:attention` after the 1.0 arms (`arm_list` in each `metadata.txt`). The 0.75 budget was dropped from `long` to save GPU time.

Compare an arm with its reference:

```bash
python -m vispec.evaluation.compare_metrics --mode sparsity --original <03 report>... --draft <05 report>... --output-dir <new_dir>
```

## Results

All figures come from one run per arm per prompt style: job 393336 for `long` and job 393337 for `short`, each ran 02, 03, 06 and 05 in one allocation at commit `66f492a5dc65` with a clean tree. Every arm used greedy decoding (temperature 0.0), bf16, batch size 1, at most 1,024 new tokens, depth 3, top-k 8, total-token 30, `num_q` 2, slim prefill on and the compact KV layout. Every run exited 0 and scored 100/100 samples on both benchmarks. Per-benchmark results are listed first, and no cross-benchmark aggregate is computed. The reference for every arm is the 03 run from the same job.

This section is the Week 2 record (retention 1.0 and 0.5, jobs 393336 and 393337). The Week 3 sweep, with retention 0.25 and 0.75 and different jobs, follows in [Week 3 results](#week-3-results-retention-sweep).

### Hardware

Each style ran inside one Slurm job on one physical card, so an arm and its references share hardware. The two styles ran on different nodes, host CPUs and cards, so their rows are never pooled or compared in latency. Week 1 `long` ran on `c2-2`, so the Week 2 `long` timings also differ in host from Week 1 and are comparable only within this job. The diagnosis (job 393579) ran on `c2-2` on the card of the `short` job, which is not the card of the timed `long` job, so it is a check and not a timed result.

| Style | Job / partition | Node and host CPU | GPU | UUID | Driver | Allocation | Device report (one per run, all listed below) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| long | 393336 / `preempt`, 03:29:51, 6 h limit | `c1-8`, Intel Xeon Silver 4116 @ 2.10 GHz | NVIDIA RTX A6000 (47.5 GiB visible) | `GPU-8d51152c-22f1-372d-aef5-063afee0d488` | 615.71.09 | 1 GPU, 4 CPUs (affinity 4, 14, 28, 38), 22,528 MB | [05 1.0:spatial device.txt](runs/20261020T111712Z_66f492a5dc65_long_r1.0_spatial_UQQwCS/device.txt) |
| short | 393337 / `preempt`, 00:51:00, 2 h limit | `c2-2`, Intel Xeon Silver 4114 @ 2.20 GHz | NVIDIA RTX A6000 (47.5 GiB visible) | `GPU-a2e40ccd-5141-9e51-2b3f-0e9d71fabf73` | 615.71.09 | 1 GPU, 4 CPUs (affinity 9, 15, 29, 35), 22,528 MB | [05 1.0:spatial device.txt](runs/20261020T080655Z_66f492a5dc65_short_r1.0_spatial_37v3lR/device.txt) |

The links point at generated run folders, which Git ignores and which are preserved on project storage, so they resolve only in a checkout that has them. Device reports are launch snapshots and do not measure runtime peaks. The job command is recorded in each run's `metadata.txt` (`submit_command`, `arm_list`). The job ran the experiments in the order 02, 03, 06, 05.

#### All run folders and device reports

| Style | Run | Arm | Run folder (preserved on project storage) | Device report |
| --- | --- | --- | --- | --- |
| long | 03 | reference | [`20261020T090922Z_66f492a5dc65_long_wDElc3`](../03_qwen_vispec/runs/20261020T090922Z_66f492a5dc65_long_wDElc3/) | [device.txt](../03_qwen_vispec/runs/20261020T090922Z_66f492a5dc65_long_wDElc3/device.txt) |
| long | 05 | 1.0:spatial | [`20261020T111712Z_66f492a5dc65_long_r1.0_spatial_UQQwCS`](runs/20261020T111712Z_66f492a5dc65_long_r1.0_spatial_UQQwCS/) | [device.txt](runs/20261020T111712Z_66f492a5dc65_long_r1.0_spatial_UQQwCS/device.txt) |
| long | 05 | 1.0:attention | [`20261020T113046Z_66f492a5dc65_long_r1.0_attention_cbhyi1`](runs/20261020T113046Z_66f492a5dc65_long_r1.0_attention_cbhyi1/) | [device.txt](runs/20261020T113046Z_66f492a5dc65_long_r1.0_attention_cbhyi1/device.txt) |
| long | 05 | 0.5:spatial | [`20261020T114412Z_66f492a5dc65_long_r0.5_spatial_ici1mV`](runs/20261020T114412Z_66f492a5dc65_long_r0.5_spatial_ici1mV/) | [device.txt](runs/20261020T114412Z_66f492a5dc65_long_r0.5_spatial_ici1mV/device.txt) |
| long | 05 | 0.5:attention | [`20261020T115745Z_66f492a5dc65_long_r0.5_attention_gdjuqN`](runs/20261020T115745Z_66f492a5dc65_long_r0.5_attention_gdjuqN/) | [device.txt](runs/20261020T115745Z_66f492a5dc65_long_r0.5_attention_gdjuqN/device.txt) |
| short | 03 | reference | [`20261020T073727Z_66f492a5dc65_short_AFFHib`](../03_qwen_vispec/runs/20261020T073727Z_66f492a5dc65_short_AFFHib/) | [device.txt](../03_qwen_vispec/runs/20261020T073727Z_66f492a5dc65_short_AFFHib/device.txt) |
| short | 05 | 1.0:spatial | [`20261020T080655Z_66f492a5dc65_short_r1.0_spatial_37v3lR`](runs/20261020T080655Z_66f492a5dc65_short_r1.0_spatial_37v3lR/) | [device.txt](runs/20261020T080655Z_66f492a5dc65_short_r1.0_spatial_37v3lR/device.txt) |
| short | 05 | 1.0:attention | [`20261020T081100Z_66f492a5dc65_short_r1.0_attention_jNHMtk`](runs/20261020T081100Z_66f492a5dc65_short_r1.0_attention_jNHMtk/) | [device.txt](runs/20261020T081100Z_66f492a5dc65_short_r1.0_attention_jNHMtk/device.txt) |
| short | 05 | 0.5:spatial | [`20261020T081501Z_66f492a5dc65_short_r0.5_spatial_PwVHz0`](runs/20261020T081501Z_66f492a5dc65_short_r0.5_spatial_PwVHz0/) | [device.txt](runs/20261020T081501Z_66f492a5dc65_short_r0.5_spatial_PwVHz0/device.txt) |
| short | 05 | 0.5:attention | [`20261020T081904Z_66f492a5dc65_short_r0.5_attention_q93FHe`](runs/20261020T081904Z_66f492a5dc65_short_r0.5_attention_q93FHe/) | [device.txt](runs/20261020T081904Z_66f492a5dc65_short_r0.5_attention_q93FHe/device.txt) |

### Reading these results

Units follow the [03 README](../03_qwen_vispec/README.md#reading-these-results) and the [metrics guide](../METRICS.md). Latency is wall time per question in seconds and TTFT is the wait before the first target token. TPOT is `(latency - TTFT) / (output tokens - 1)` in milliseconds, averaged per question. Output tokens include EOS. Accepted draft tokens per iteration is the mean over verification iterations, and the number in parentheses is the iteration count. Quality is TextVQA consensus accuracy in percent, or COCO CIDEr-D x 100 with inverse document frequency from the 100-image subset, so it is a subset measurement. Live KV is the whole-model figure in MiB (`kv_tokens` times the bytes per token, which includes the output tokens), and peaks are `torch.cuda.max_memory_allocated` per question in GiB (mean and max over 100 questions). The memory columns come from the `memory` block of each `summary.json`.

Ratios are reference over candidate, so a value above 1 means the candidate was faster. Each 0.5 arm is compared with the same-policy 1.0 arm (primary), because the 1.0 arm runs the same selection code (mask, score capture) as the 0.5 arm. The null reference, the 03 run without any sparsity code, is the secondary column. It also anchors the identity gate. A 1.0 arm has no same-policy reference, so its row shows ratios against the null reference only. Differences between a 1.0 arm and the null reference measure overhead plus run-to-run noise together. They range from 0.974 to 1.009 in latency and from 0.994 to 1.006 in TTFT here, and with one run per arm they cannot be separated.

The `long` style is the speed and acceptance measurement, and its strict quality scores are near zero for the reasons given in 03. The `short` style is the quality measurement. Agreement is exact token-ID agreement over 100 samples. With n = 100, agreement counts and quality differences are noisy, and no confidence intervals are computed.

### Identity gate at retention 1.0

The gate passed. Every 1.0 arm reproduces its null reference token for token on 100 of 100 samples, on both benchmarks and in both styles.

| Arm (vs its null reference) | long TextVQA | long COCO | short TextVQA | short COCO |
| --- | ---: | ---: | ---: | ---: |
| 05 ViSpec 1.0:spatial vs 03 | 100/100 | 100/100 | 100/100 | 100/100 |
| 05 ViSpec 1.0:attention vs 03 | 100/100 | 100/100 | 100/100 | 100/100 |

A separate check shows that the Week 2 compact references (02 and 03, `long`) equal the Week 1 fixed-layout runs token for token on 100/100 for TextVQA and for COCO Captions. The same holds for the `short` style. The compact layout therefore changed no output. The Week 1 runs are jobs 392660 and 392661 (see the [03 README](../03_qwen_vispec/README.md#results)).

### Style `long` (speed and acceptance measurement)

Draft-only arms leave the target unchanged, so in exact arithmetic they would reproduce the reference. The exact-agreement column therefore reflects bf16 near-ties as well as any method effect.

| Benchmark | Arm | Latency (s) | TTFT (s) | Mean output tokens | Truncated at cap | TPOT (ms) | Latency ratio vs same-policy 1.0 | TTFT ratio vs same-policy 1.0 | Latency ratio vs null ref | TTFT ratio vs null ref |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TextVQA | 03 reference (ViSpec) | 5.395 | 0.940 | 277.70 | 3 | 17.2 | n/a | n/a | n/a | n/a |
| TextVQA | 05 ViSpec 1.0:spatial | 5.406 | 0.938 | 277.70 | 3 | 17.2 | n/a | n/a | 0.998 | 1.002 |
| TextVQA | 05 ViSpec 1.0:attention | 5.349 | 0.937 | 277.70 | 3 | 17.0 | n/a | n/a | 1.009 | 1.003 |
| TextVQA | 05 ViSpec 0.5:spatial | 5.370 | 0.937 | 279.18 | 2 | 17.0 | 1.007 | 1.001 | 1.005 | 1.003 |
| TextVQA | 05 ViSpec 0.5:attention | 5.258 | 0.937 | 270.76 | 4 | 17.1 | 1.017 | 1.000 | 1.026 | 1.003 |
| COCO Captions | 03 reference (ViSpec) | 1.897 | 0.221 | 109.89 | 0 | 15.5 | n/a | n/a | n/a | n/a |
| COCO Captions | 05 ViSpec 1.0:spatial | 1.914 | 0.222 | 109.89 | 0 | 15.6 | n/a | n/a | 0.991 | 0.997 |
| COCO Captions | 05 ViSpec 1.0:attention | 1.903 | 0.221 | 109.89 | 0 | 15.5 | n/a | n/a | 0.997 | 1.000 |
| COCO Captions | 05 ViSpec 0.5:spatial | 1.935 | 0.220 | 115.43 | 1 | 15.3 | 0.989 | 1.008 | 0.981 | 1.004 |
| COCO Captions | 05 ViSpec 0.5:attention | 1.896 | 0.221 | 110.31 | 0 | 15.4 | 1.004 | 1.001 | 1.001 | 1.001 |

| Benchmark | Arm | Accepted draft tokens per iteration (iterations) | Exact agreement with reference | First mismatch, min / median (count below 5) | Quality | Change vs reference |
| --- | --- | --- | ---: | --- | ---: | ---: |
| TextVQA | 03 reference (ViSpec) | 2.141 (8841) | n/a | n/a | 0.0 | n/a |
| TextVQA | 05 ViSpec 1.0:spatial | 2.141 (8841) | 100/100 | none | 0.0 | +0.0 |
| TextVQA | 05 ViSpec 1.0:attention | 2.141 (8841) | 100/100 | none | 0.0 | +0.0 |
| TextVQA | 05 ViSpec 0.5:spatial | 2.145 (8878) | 56/100 | 32 / 168 (0) | 0.0 | +0.0 |
| TextVQA | 05 ViSpec 0.5:attention | 2.129 (8652) | 48/100 | 19 / 161 (0) | 0.0 | +0.0 |
| COCO Captions | 03 reference (ViSpec) | 2.284 (3346) | n/a | n/a | 1.90e-05 | n/a |
| COCO Captions | 05 ViSpec 1.0:spatial | 2.284 (3346) | 100/100 | none | 1.90e-05 | +0.00e+00 |
| COCO Captions | 05 ViSpec 1.0:attention | 2.284 (3346) | 100/100 | none | 1.90e-05 | +0.00e+00 |
| COCO Captions | 05 ViSpec 0.5:spatial | 2.344 (3452) | 77/100 | 21 / 61 (0) | 1.90e-05 | -4.42e-12 |
| COCO Captions | 05 ViSpec 0.5:attention | 2.294 (3349) | 72/100 | 16 / 76 (0) | 1.90e-05 | -2.89e-10 |

| Benchmark | Arm | Live KV mean / max (MiB) | Decode peak mean / max (GiB) | Prefill peak mean / max (GiB) | Evicted tokens mean / max | Prompt tokens mean |
| --- | --- | --- | --- | --- | --- | --- |
| TextVQA | 03 reference (ViSpec) | 47.1 / 81.9 | 7.993 / 8.015 | 10.16 / 11.57 | n/a | 1062.9 |
| TextVQA | 05 ViSpec 1.0:spatial | 47.1 / 81.9 | 7.993 / 8.015 | 10.16 / 11.57 | n/a | 1062.9 |
| TextVQA | 05 ViSpec 1.0:attention | 47.1 / 81.9 | 7.993 / 8.015 | 10.16 / 11.57 | n/a | 1062.9 |
| TextVQA | 05 ViSpec 0.5:spatial | 47.2 / 81.9 | 7.993 / 8.014 | 10.16 / 11.57 | n/a | 1062.9 |
| TextVQA | 05 ViSpec 0.5:attention | 46.9 / 81.9 | 7.993 / 8.015 | 10.16 / 11.57 | n/a | 1062.9 |
| COCO Captions | 03 reference (ViSpec) | 18.1 / 24.1 | 7.934 / 7.946 | 8.17 / 8.54 | n/a | 405.5 |
| COCO Captions | 05 ViSpec 1.0:spatial | 18.1 / 24.1 | 7.934 / 7.946 | 8.17 / 8.54 | n/a | 405.5 |
| COCO Captions | 05 ViSpec 1.0:attention | 18.1 / 24.1 | 7.934 / 7.946 | 8.17 / 8.54 | n/a | 405.5 |
| COCO Captions | 05 ViSpec 0.5:spatial | 18.3 / 47.2 | 7.934 / 7.947 | 8.17 / 8.54 | n/a | 405.5 |
| COCO Captions | 05 ViSpec 0.5:attention | 18.1 / 24.1 | 7.935 / 7.949 | 8.17 / 8.54 | n/a | 405.5 |


COCO Captions 0.5:spatial shows a live-KV maximum of 47.2 MiB against 24.1 MiB for the other arms. One sample (question 456521) ran to the 1,024-token cap, against 337 tokens and EOS in the reference, and `kv_tokens` includes the output tokens. The memory is therefore an output-length effect, not a pruning effect.

### Style `short` (quality measurement)

| Benchmark | Arm | Latency (s) | TTFT (s) | Mean output tokens | Truncated at cap | TPOT (ms) | Latency ratio vs same-policy 1.0 | TTFT ratio vs same-policy 1.0 | Latency ratio vs null ref | TTFT ratio vs null ref |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TextVQA | 03 reference (ViSpec) | 1.140 | 0.966 | 5.24 | 0 | 66.0 | n/a | n/a | n/a | n/a |
| TextVQA | 05 ViSpec 1.0:spatial | 1.153 | 0.965 | 5.24 | 0 | 71.6 | n/a | n/a | 0.988 | 1.001 |
| TextVQA | 05 ViSpec 1.0:attention | 1.135 | 0.960 | 5.24 | 0 | 66.5 | n/a | n/a | 1.004 | 1.006 |
| TextVQA | 05 ViSpec 0.5:spatial | 1.135 | 0.960 | 5.24 | 0 | 66.3 | 1.016 | 1.005 | 1.004 | 1.006 |
| TextVQA | 05 ViSpec 0.5:attention | 1.133 | 0.957 | 5.24 | 0 | 66.8 | 1.002 | 1.003 | 1.006 | 1.009 |
| COCO Captions | 03 reference (ViSpec) | 0.518 | 0.225 | 15.05 | 0 | 21.1 | n/a | n/a | n/a | n/a |
| COCO Captions | 05 ViSpec 1.0:spatial | 0.532 | 0.227 | 15.05 | 0 | 22.1 | n/a | n/a | 0.974 | 0.994 |
| COCO Captions | 05 ViSpec 1.0:attention | 0.522 | 0.225 | 15.05 | 0 | 21.5 | n/a | n/a | 0.991 | 0.999 |
| COCO Captions | 05 ViSpec 0.5:spatial | 0.527 | 0.224 | 15.05 | 0 | 21.9 | 1.008 | 1.010 | 0.982 | 1.004 |
| COCO Captions | 05 ViSpec 0.5:attention | 0.519 | 0.222 | 15.05 | 0 | 21.4 | 1.007 | 1.014 | 0.998 | 1.013 |

| Benchmark | Arm | Accepted draft tokens per iteration (iterations) | Exact agreement with reference | First mismatch, min / median (count below 5) | Quality | Change vs reference |
| --- | --- | --- | ---: | --- | ---: | ---: |
| TextVQA | 03 reference (ViSpec) | 0.690 (310) | n/a | n/a | 70.0 | n/a |
| TextVQA | 05 ViSpec 1.0:spatial | 0.690 (310) | 100/100 | none | 70.0 | +0.0 |
| TextVQA | 05 ViSpec 1.0:attention | 0.690 (310) | 100/100 | none | 70.0 | +0.0 |
| TextVQA | 05 ViSpec 0.5:spatial | 0.679 (312) | 100/100 | none | 70.0 | +0.0 |
| TextVQA | 05 ViSpec 0.5:attention | 0.685 (311) | 100/100 | none | 70.0 | +0.0 |
| COCO Captions | 03 reference (ViSpec) | 1.746 (548) | n/a | n/a | 119.68 | n/a |
| COCO Captions | 05 ViSpec 1.0:spatial | 1.746 (548) | 100/100 | none | 119.68 | +0.00e+00 |
| COCO Captions | 05 ViSpec 1.0:attention | 1.746 (548) | 100/100 | none | 119.68 | +0.00e+00 |
| COCO Captions | 05 ViSpec 0.5:spatial | 1.736 (550) | 100/100 | none | 119.68 | +0.00e+00 |
| COCO Captions | 05 ViSpec 0.5:attention | 1.731 (551) | 100/100 | none | 119.68 | +0.00e+00 |

| Benchmark | Arm | Live KV mean / max (MiB) | Decode peak mean / max (GiB) | Prefill peak mean / max (GiB) | Evicted tokens mean / max | Prompt tokens mean |
| --- | --- | --- | --- | --- | --- | --- |
| TextVQA | 03 reference (ViSpec) | 36.7 / 46.7 | 7.982 / 8.005 | 10.16 / 11.57 | n/a | 1039.9 |
| TextVQA | 05 ViSpec 1.0:spatial | 36.7 / 46.7 | 7.982 / 8.005 | 10.16 / 11.57 | n/a | 1039.9 |
| TextVQA | 05 ViSpec 1.0:attention | 36.7 / 46.7 | 7.982 / 8.005 | 10.16 / 11.57 | n/a | 1039.9 |
| TextVQA | 05 ViSpec 0.5:spatial | 36.7 / 46.7 | 7.982 / 8.007 | 10.16 / 11.57 | n/a | 1039.9 |
| TextVQA | 05 ViSpec 0.5:attention | 36.7 / 46.7 | 7.981 / 8.005 | 10.16 / 11.57 | n/a | 1039.9 |
| COCO Captions | 03 reference (ViSpec) | 14.8 / 21.1 | 7.930 / 7.948 | 8.17 / 8.54 | n/a | 406.5 |
| COCO Captions | 05 ViSpec 1.0:spatial | 14.8 / 21.1 | 7.930 / 7.948 | 8.17 / 8.54 | n/a | 406.5 |
| COCO Captions | 05 ViSpec 1.0:attention | 14.8 / 21.1 | 7.930 / 7.948 | 8.17 / 8.54 | n/a | 406.5 |
| COCO Captions | 05 ViSpec 0.5:spatial | 14.8 / 21.1 | 7.930 / 7.947 | 8.17 / 8.54 | n/a | 406.5 |
| COCO Captions | 05 ViSpec 0.5:attention | 14.8 / 21.1 | 7.930 / 7.947 | 8.17 / 8.54 | n/a | 406.5 |


### Predictions against the measurements

| Prediction (written before the runs) | Outcome | Evidence |
| --- | --- | --- |
| Draft pruning costs acceptance at most slightly | **Held** | Accepted draft tokens per iteration at 0.5 differ from the same-policy 1.0 arm by -0.015 to +0.060 across the 8 style, benchmark and policy cells. The largest rise is long COCO spatial (2.284 to 2.344). Where outputs differ, the two means average over different text. |
| Draft pruning barely moves latency | **Held** | Latency ratios against the same-policy 1.0 arm lie between 0.989 and 1.017, and TTFT ratios between 1.000 and 1.014. These are inside the 0.974 to 1.009 spread that the 1.0 arms already show against the null reference. |
| The target still verifies every draft, so greedy output matches the 1.0 reference up to bf16 near-ties | **Held for `short`, partly held for `long`** | Both `short` arms agree on 100/100 in both benchmarks. `long` agrees on 56/100 (TextVQA) and 77/100 (COCO) for spatial and 48/100 and 72/100 for attention. The first mismatch is at index 16 or later and never below 5. The `long` divergences are classified below. |

The latency outcome is stated from the table above. The reason it did not move is an inference, not a measurement: the [04 profile](../04_qwen_decode_profile/README.md) inferred that decode is host-bound and the adaptor runs once per sample.

### Draft-only divergence diagnosis

`diagnose_divergence` classifies each first mismatch between a 0.5 arm and the 03 reference with the stock full-KV model, teacher-forced on the shared prefix. A case is `near_tie` if the bf16 top-2 margin is at most 0.5 logits. The gate passes only when no case is `unexplained` or `both_wrong`. Prompt fingerprints matched in every case (0 mismatches). The fp32 pass ran on the first 50 cases of each arm only (`fp32_limit` 50).

| Style | Arm | Cases | near_tie | unexplained | both_wrong | bf16 margin median / max (logits) | Draft-only gate |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| long | 0.5:spatial | 67 | 67 | 0 | 0 | 0.000 / 0.250 | passes |
| long | 0.5:attention | 80 | 79 | 1 | 0 | 0.125 / 0.625 | **fails** |
| short | 0.5:spatial | 0 | 0 | 0 | 0 | n/a | passes (no divergences) |
| short | 0.5:attention | 0 | 0 | 0 | 0 | n/a | passes (no divergences) |

Diagnosis files:

- long 0.5:spatial: [`diagnosis.json`](runs/20261020T114412Z_66f492a5dc65_long_r0.5_spatial_ici1mV/comparisons/20261020T121115Z_nullref_vs_03-ref/diagnosis/diagnosis.json)
- long 0.5:attention: [`diagnosis.json`](runs/20261020T115745Z_66f492a5dc65_long_r0.5_attention_gdjuqN/comparisons/20261020T121115Z_nullref_vs_03-ref/diagnosis/diagnosis.json)
- short 0.5:spatial: [`diagnosis.json`](runs/20261020T081501Z_66f492a5dc65_short_r0.5_spatial_PwVHz0/comparisons/20261020T082423Z_nullref_vs_03-ref/diagnosis/diagnosis.json)
- short 0.5:attention: [`diagnosis.json`](runs/20261020T081904Z_66f492a5dc65_short_r0.5_attention_q93FHe/comparisons/20261020T082423Z_nullref_vs_03-ref/diagnosis/diagnosis.json)

**The `long` 0.5:attention gate failed**: 79 near-ties and 1 unexplained case of 80. The threshold of 0.5 logits was not changed. `long` 0.5:spatial passed with 67 of 67 near-ties, and both `short` arms had 100/100 agreement and so no cases.

The unexplained case is TextVQA question 35537, first mismatch at output index 162. Examining it gave this evidence:

- The reference (unpruned ViSpec, run 03) emitted "." (token 13) at index 162. The Week 1 02 and 03 runs and the Week 2 02 and 03 references emit the same token there.
- The stock model, teacher-forced on the shared 162-token prefix, prefers "**:" (token 95518) over "." by 0.625 logits in bf16 and 0.724 in fp32.
- The pruned candidate emitted "**:", the stock argmax.

From this we infer that the deviating token belongs to the shared incremental KV-cached decode, which standard autoregressive decoding and ViSpec both use, and that pruning did not cause it. It is therefore not a Week 2 regression. This is an inference. Experiment 07 later traced the case ([07 README](../07_qwen_decode_drift/README.md)): at index 162 the fork's bf16 logits for tokens 13 and 95518 tie exactly, and the case was judged bf16 implementation noise, a known limitation. The mechanism behind the gap to stock cached decoding is not isolated. The draft-only gate is still recorded as failed for this arm as it was run.

### Notes on the attention policy

The attention policy ranks the adaptor's image tokens by attention from its own learned queries and keeps the top fraction. In a smoke run on 5 `long` TextVQA samples, the attention-selected and spatial-selected halves overlapped at a Jaccard of 0.31 to 0.35, which is about the 1/3 expected for two unrelated half-subsets. So the attention policy is not a disguised stride pattern. The selected indices are not saved with the Week 2 runs, so the overlap was not re-measured there. Attention scoring also runs at retention 1.0, so the 1.0 attention arm carries that overhead, which is part of why the primary latency reference is the same-policy 1.0 arm.

## Week 3 results (retention sweep)

### Summary

Speculative decoding comes first. In these jobs ViSpec completed Qwen2.5-VL-3B answers 2.2 to 2.3 times faster than autoregressive (AR) decoding in the `long` style, and about as fast as AR or 1.5 times faster in the `short` style. The table gives the same-job references, which every arm below is compared with.

| Style | Job | Benchmark | AR latency (s) | ViSpec latency (s) | ViSpec speedup vs AR | ViSpec accepted draft tokens per iteration | Exact agreement, ViSpec vs AR | Quality, AR / ViSpec |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| long | 398884 | TextVQA | 12.485 | 5.700 | 2.190 | 2.141 | 22/100 | near zero (not meaningful) |
| long | 398884 | COCO Captions | 4.697 | 2.039 | 2.304 | 2.284 | 28/100 | near zero (not meaningful) |
| short | 398923 | TextVQA | 1.100 | 1.122 | 0.981 | 0.690 | 96/100 | 71.0 / 70.0 |
| short | 398923 | COCO Captions | 0.784 | 0.528 | 1.485 | 1.746 | 88/100 | 121.40 / 119.68 |

In `long`, the speedup comes with 2.141 (TextVQA) and 2.284 (COCO) accepted draft tokens per iteration. In `short` the answers are 5 and 15 tokens long and prefill dominates: TTFT is 0.940 s of the 1.122 s ViSpec latency on TextVQA, so ViSpec shows no speedup there (0.981). ViSpec and AR outputs differ in token IDs on `long` (22/100 and 28/100 agree exactly) and nearly agree on `short` (96/100 and 88/100). In bf16 the batched tree verification and one-token decoding use different kernel shapes, so near-tied logits can flip; the Week 1 diagnosis found every divergence to be a near-tie ([03 README](../03_qwen_vispec/README.md#results)).

Sparsity comes second, and draft pruning leaves speculative decoding where it was. Pruning the draft's image input at retention 0.5 and 0.25 (and 0.75 in `short`), against the same ViSpec method at retention 1.0:

- **Acceptance** stayed close to the reference. Accepted draft tokens per iteration were 2.129 to 2.145 on `long` TextVQA (reference 2.141), 2.289 to 2.360 on `long` COCO (2.284), 0.679 to 0.690 on `short` TextVQA (0.690) and 1.731 to 1.751 on `short` COCO (1.746). The two largest COCO rises come from one answer that ran to the token cap (see [below](#acceptance-after-removing-one-answer)).
- **Latency** did not move beyond the spread of a single run. Ratios against the same-policy 1.0 arm are 0.976 to 1.037 on `long` and 0.991 to 1.002 on `short`. The 1.0 arms themselves differ from the unpruned reference by 0.992 to 1.012 on `long` and 0.996 to 1.004 on `short`. TTFT is unchanged.
- **Output** is the same as the reference on 99 or 100 of 100 samples in `short` and on 40 to 77 of 100 in `long`, where the diagnosis classifies all but one case as a bf16 near-tie (see [the diagnosis](#draft-only-divergence-diagnosis-week-3)). `short` quality is unchanged: TextVQA 70.0 in every arm, COCO CIDEr-D 119.68 to 119.93.
- **Memory** is unchanged, because the target and its KV cache are untouched. The live-KV ratio is 0.995 to 1.012, which comes from output length alone.

### Why the effect is small

This is the explanation for small effects, not a new method. The adaptor compresses each image's tokens to `num_q` = 2 queries whatever the input size, so the draft sees two image positions at 1.0 and at 0.25. It runs once per sample, and the [04 profile](../04_qwen_decode_profile/README.md) found decode host-bound, with the GPU busy for 36 to 50% of each step. The target still verifies every draft and holds the full cache, which costs 36,864 bytes per token (36 layers x 2 x 2 KV heads x 128 x 2 bytes) against about 5.75 GiB of language-model weights read in every step (measured from the checkpoint files; the 1.25 GiB vision encoder is not read in decode steps): the cache is 0.6% of the per-step weight read at 1,000 tokens and 4.2% at 7,000 tokens, and the two are equal only near 167,000 tokens. Pruning the draft input cannot shrink the target cache at all; [06](../06_qwen_kv_eviction/README.md#why-the-latency-effect-is-small) covers that side. The host-bound reading is an inference from the Week 2 profile and is not tested by these runs.

### Setup

Four Slurm jobs, all at commit `c3a36ac25f95` with a clean tree, greedy decoding (temperature 0.0), bf16, batch size 1, at most 1,024 new tokens, depth 3, top-k 8, total-token 30, `num_q` 2, slim prefill on and the compact KV layout. Every run exited 0 and scored 100/100 samples on both benchmarks. This README uses job 398884 (`long`) and job 398923 (`short`). The other `long` jobs belong to [06](../06_qwen_kv_eviction/README.md#week-3-results-retention-sweep) (398883) and [08](../08_qwen_combined_sparsity/README.md#week-3-results-retention-sweep) (398885).

**The budgets differ by style.** The `long` style ran retention 1.0, 0.5 and 0.25. The 0.75 budget was dropped from `long` to save GPU time. The `short` style ran all four (1.0, 0.75, 0.5 and 0.25). Both policies (`spatial`, `attention`) ran at every budget.

Each arm is compared with the same-policy 1.0 arm from the same job (the primary reading, because the 1.0 arm runs the same selection code) and with the unpruned 03 ViSpec reference of that job (the second reading). The job's 1.0 arms against that reference give the overhead and noise floor.

### Week 3 hardware

Each job ran on one physical card, so an arm and its references share hardware. The `long` job here (398884) and the `short` job ran on the same card at different times. **Absolute times are compared only within a job, never across jobs.** All ratios below stay inside one job.

| Style | Job / partition (complete attempt) | Node and host CPU | GPU | UUID | Driver | Allocation | Device report (one per job; every run folder has its own, listed below) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| long | 398884 / `preempt`, restart 2, 2026-11-04, 00:15 to 02:30 UTC, 4 h limit | `c1-8`, Intel Xeon Silver 4116 @ 2.10 GHz | NVIDIA RTX A6000 (49,140 MiB total) | `GPU-eb032c66-7f3f-d1b6-96cf-c9db5600d457` | 615.71.09 | 1 GPU, 4 CPUs (affinity 4, 18, 28, 42), 22,528 MB | [05 1.0:spatial device.txt](runs/20261104T010325Z_c3a36ac25f95_long_r1.0_spatial_nyBi6I/device.txt) |
| short | 398923 / `preempt`, restart 0, 2026-11-01, 12:19 to 14:06 UTC, 4 h limit | `c1-8`, Intel Xeon Silver 4116 @ 2.10 GHz | NVIDIA RTX A6000 (49,140 MiB total) | `GPU-eb032c66-7f3f-d1b6-96cf-c9db5600d457` | 615.71.09 | 1 GPU, 4 CPUs (affinity 2, 18, 26, 42), 22,528 MB | [05 1.0:spatial device.txt](runs/20261101T130208Z_c3a36ac25f95_short_r1.0_spatial_lbrC6E/device.txt) |

The links point at generated run folders, which Git ignores and which are preserved on project storage, so they resolve only in a checkout that has them. Device reports are launch snapshots and do not measure runtime peaks. The job command is recorded in each run's `metadata.txt`. A requeued job reruns the runner into new folders, and only the complete attempt of each job is used; interrupted attempts are excluded.

#### Week 3 run folders and device reports

| Style | Job | Run | Arm | Run folder (preserved on project storage) | Device report |
| --- | --- | --- | --- | --- | --- |
| long | 398884 | 02 | reference | [`20261104T001540Z_c3a36ac25f95_long_9RqlFI`](../02_qwen_autoregressive/runs/20261104T001540Z_c3a36ac25f95_long_9RqlFI/) | [device.txt](../02_qwen_autoregressive/runs/20261104T001540Z_c3a36ac25f95_long_9RqlFI/device.txt) |
| long | 398884 | 03 | reference | [`20261104T004818Z_c3a36ac25f95_long_umvxvi`](../03_qwen_vispec/runs/20261104T004818Z_c3a36ac25f95_long_umvxvi/) | [device.txt](../03_qwen_vispec/runs/20261104T004818Z_c3a36ac25f95_long_umvxvi/device.txt) |
| long | 398884 | 05 | 1.0:attention | [`20261104T011816Z_c3a36ac25f95_long_r1.0_attention_77b4EN`](runs/20261104T011816Z_c3a36ac25f95_long_r1.0_attention_77b4EN/) | [device.txt](runs/20261104T011816Z_c3a36ac25f95_long_r1.0_attention_77b4EN/device.txt) |
| long | 398884 | 05 | 1.0:spatial | [`20261104T010325Z_c3a36ac25f95_long_r1.0_spatial_nyBi6I`](runs/20261104T010325Z_c3a36ac25f95_long_r1.0_spatial_nyBi6I/) | [device.txt](runs/20261104T010325Z_c3a36ac25f95_long_r1.0_spatial_nyBi6I/device.txt) |
| long | 398884 | 05 | 0.5:attention | [`20261104T014656Z_c3a36ac25f95_long_r0.5_attention_ufNrw8`](runs/20261104T014656Z_c3a36ac25f95_long_r0.5_attention_ufNrw8/) | [device.txt](runs/20261104T014656Z_c3a36ac25f95_long_r0.5_attention_ufNrw8/device.txt) |
| long | 398884 | 05 | 0.5:spatial | [`20261104T013237Z_c3a36ac25f95_long_r0.5_spatial_TWABnt`](runs/20261104T013237Z_c3a36ac25f95_long_r0.5_spatial_TWABnt/) | [device.txt](runs/20261104T013237Z_c3a36ac25f95_long_r0.5_spatial_TWABnt/device.txt) |
| long | 398884 | 05 | 0.25:attention | [`20261104T021535Z_c3a36ac25f95_long_r0.25_attention_OBIe3y`](runs/20261104T021535Z_c3a36ac25f95_long_r0.25_attention_OBIe3y/) | [device.txt](runs/20261104T021535Z_c3a36ac25f95_long_r0.25_attention_OBIe3y/device.txt) |
| long | 398884 | 05 | 0.25:spatial | [`20261104T020058Z_c3a36ac25f95_long_r0.25_spatial_BZAvRH`](runs/20261104T020058Z_c3a36ac25f95_long_r0.25_spatial_BZAvRH/) | [device.txt](runs/20261104T020058Z_c3a36ac25f95_long_r0.25_spatial_BZAvRH/device.txt) |
| short | 398923 | 02 | reference | [`20261101T121923Z_c3a36ac25f95_short_J5AdE5`](../02_qwen_autoregressive/runs/20261101T121923Z_c3a36ac25f95_short_J5AdE5/) | [device.txt](../02_qwen_autoregressive/runs/20261101T121923Z_c3a36ac25f95_short_J5AdE5/device.txt) |
| short | 398923 | 03 | reference | [`20261101T122530Z_c3a36ac25f95_short_ePf9XE`](../03_qwen_vispec/runs/20261101T122530Z_c3a36ac25f95_short_ePf9XE/) | [device.txt](../03_qwen_vispec/runs/20261101T122530Z_c3a36ac25f95_short_ePf9XE/device.txt) |
| short | 398923 | 05 | 1.0:attention | [`20261101T130611Z_c3a36ac25f95_short_r1.0_attention_vyq7vi`](runs/20261101T130611Z_c3a36ac25f95_short_r1.0_attention_vyq7vi/) | [device.txt](runs/20261101T130611Z_c3a36ac25f95_short_r1.0_attention_vyq7vi/device.txt) |
| short | 398923 | 05 | 1.0:spatial | [`20261101T130208Z_c3a36ac25f95_short_r1.0_spatial_lbrC6E`](runs/20261101T130208Z_c3a36ac25f95_short_r1.0_spatial_lbrC6E/) | [device.txt](runs/20261101T130208Z_c3a36ac25f95_short_r1.0_spatial_lbrC6E/device.txt) |
| short | 398923 | 05 | 0.75:attention | [`20261101T131413Z_c3a36ac25f95_short_r0.75_attention_zKV3y1`](runs/20261101T131413Z_c3a36ac25f95_short_r0.75_attention_zKV3y1/) | [device.txt](runs/20261101T131413Z_c3a36ac25f95_short_r0.75_attention_zKV3y1/device.txt) |
| short | 398923 | 05 | 0.75:spatial | [`20261101T131010Z_c3a36ac25f95_short_r0.75_spatial_yNBPQI`](runs/20261101T131010Z_c3a36ac25f95_short_r0.75_spatial_yNBPQI/) | [device.txt](runs/20261101T131010Z_c3a36ac25f95_short_r0.75_spatial_yNBPQI/device.txt) |
| short | 398923 | 05 | 0.5:attention | [`20261101T132217Z_c3a36ac25f95_short_r0.5_attention_siMBDH`](runs/20261101T132217Z_c3a36ac25f95_short_r0.5_attention_siMBDH/) | [device.txt](runs/20261101T132217Z_c3a36ac25f95_short_r0.5_attention_siMBDH/device.txt) |
| short | 398923 | 05 | 0.5:spatial | [`20261101T131814Z_c3a36ac25f95_short_r0.5_spatial_mDMaO8`](runs/20261101T131814Z_c3a36ac25f95_short_r0.5_spatial_mDMaO8/) | [device.txt](runs/20261101T131814Z_c3a36ac25f95_short_r0.5_spatial_mDMaO8/device.txt) |
| short | 398923 | 05 | 0.25:attention | [`20261101T133019Z_c3a36ac25f95_short_r0.25_attention_jNhSuR`](runs/20261101T133019Z_c3a36ac25f95_short_r0.25_attention_jNhSuR/) | [device.txt](runs/20261101T133019Z_c3a36ac25f95_short_r0.25_attention_jNhSuR/device.txt) |
| short | 398923 | 05 | 0.25:spatial | [`20261101T132617Z_c3a36ac25f95_short_r0.25_spatial_4Trrz4`](runs/20261101T132617Z_c3a36ac25f95_short_r0.25_spatial_4Trrz4/) | [device.txt](runs/20261101T132617Z_c3a36ac25f95_short_r0.25_spatial_4Trrz4/device.txt) |

### Reading the Week 3 tables

Units follow the [03 README](../03_qwen_vispec/README.md#reading-these-results) and the [metrics guide](../METRICS.md). Latency and TTFT are seconds per question, TPOT is milliseconds, and mean output tokens count EOS (in parentheses, the answers that hit the 1,024-token cap). Decode ms per iteration is the mean decode time over the run's iterations (mean decode time per question times 100, divided by the iteration count), computed from each run's `summary.json`. Quality is TextVQA consensus accuracy in percent or COCO CIDEr-D x 100 (inverse document frequency from the 100-image subset, so a subset measurement).

- **Direction of ratios.** `Speedup vs AR` and both latency ratios are reference over arm, so a value above 1 means the arm was faster. Memory ratios are arm over reference, so a value above 1 means more memory.
- **Two latency readings, shown side by side.** The column against the same-policy 1.0 arm and the column against the unpruned ViSpec reference are both given and neither is preferred after the fact. A 1.0 arm has no same-policy reference, so its row shows only the reference reading, which measures overhead plus run-to-run noise.
- **Agreement and bf16.** Exact agreement is token-ID equality through the first EOS over 100 samples. Draft-only arms leave the target unchanged, so in exact arithmetic they would reproduce the reference; in bf16 near-tied logits flip when kernel shapes differ, so agreement reflects near-ties as well as any method effect. Agreement with AR is the secondary column.
- **Live KV is arithmetic.** `kv_tokens` times 36,864 bytes, not an allocator measurement. It includes the output tokens, so a ratio shifts with output length. Compare with the output-token column.
- **Steady-state decode memory.** The post-compaction peak is the figure for steady-state decode memory, and its ratio is given beside the decode-peak ratio.
- **Long style.** Long answers are paragraphs, so strict VQA accuracy and CIDEr-D are near zero and not meaningful there. The `long` tables give agreement with the reference and output length instead of quality. Quality numbers come from the `short` style.

### Week 3 identity gate at retention 1.0

The gate passed: all 12 1.0 arms of the Week 3 jobs of 05, 06 and 08 (both policies, both styles) reproduce their same-job ViSpec reference token for token, 100/100 on TextVQA and 100/100 on COCO. The 05 arms are the 100/100 entries on the 1.0 rows below. The comparison files are in each arm's `comparisons/20261105T192451Z_identity_vs_03-ref/`.

### Style `long`, Week 3 (speed and agreement measurement)

| Benchmark | Arm | Latency (s) | TTFT (s) | Mean output tokens (capped) | TPOT (ms) | Speedup vs AR | Latency ratio vs same-policy 1.0 | Latency ratio vs unpruned ViSpec ref |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TextVQA | 02 reference (AR) | 12.485 | 0.937 | 283.31 (4) | 40.9 | 1.000 | n/a | n/a |
| TextVQA | 03 reference (ViSpec) | 5.700 | 0.941 | 277.70 (3) | 18.3 | 2.190 | n/a | n/a |
| TextVQA | 05 ViSpec 1.0:attention | 5.705 | 0.942 | 277.70 (3) | 18.4 | 2.189 | 1.000 | 0.999 |
| TextVQA | 05 ViSpec 1.0:spatial | 5.747 | 0.937 | 277.70 (3) | 18.5 | 2.172 | 1.000 | 0.992 |
| TextVQA | 05 ViSpec 0.5:attention | 5.503 | 0.940 | 270.76 (4) | 18.0 | 2.269 | 1.037 | 1.036 |
| TextVQA | 05 ViSpec 0.5:spatial | 5.694 | 0.940 | 279.18 (2) | 18.2 | 2.193 | 1.009 | 1.001 |
| TextVQA | 05 ViSpec 0.25:attention | 5.724 | 0.939 | 280.76 (3) | 18.3 | 2.181 | 0.997 | 0.996 |
| TextVQA | 05 ViSpec 0.25:spatial | 5.796 | 0.937 | 282.04 (3) | 18.4 | 2.154 | 0.992 | 0.983 |
| COCO Captions | 02 reference (AR) | 4.697 | 0.228 | 109.96 (0) | 41.1 | 1.000 | n/a | n/a |
| COCO Captions | 03 reference (ViSpec) | 2.039 | 0.222 | 109.89 (0) | 16.8 | 2.304 | n/a | n/a |
| COCO Captions | 05 ViSpec 1.0:attention | 2.037 | 0.222 | 109.89 (0) | 16.7 | 2.306 | 1.000 | 1.001 |
| COCO Captions | 05 ViSpec 1.0:spatial | 2.015 | 0.221 | 109.89 (0) | 16.5 | 2.331 | 1.000 | 1.012 |
| COCO Captions | 05 ViSpec 0.5:attention | 2.023 | 0.222 | 110.31 (0) | 16.6 | 2.322 | 1.007 | 1.008 |
| COCO Captions | 05 ViSpec 0.5:spatial | 1.998 | 0.221 | 115.43 (1) | 15.9 | 2.351 | 1.008 | 1.020 |
| COCO Captions | 05 ViSpec 0.25:attention | 2.037 | 0.223 | 110.22 (0) | 16.6 | 2.306 | 1.000 | 1.001 |
| COCO Captions | 05 ViSpec 0.25:spatial | 2.064 | 0.221 | 115.87 (1) | 16.4 | 2.275 | 0.976 | 0.988 |

*Exact agreement counts identical token IDs in bf16, where near-tied logits can flip with kernel shape, so it reflects near-ties as well as any method effect.*

| Benchmark | Arm | Accepted draft tokens per iteration (iterations) | Decode ms per iteration | Exact agreement with ViSpec reference | Exact agreement with AR (secondary) |
| --- | --- | --- | ---: | ---: | ---: |
| TextVQA | 02 reference (AR) | n/a (AR) | n/a | 22/100 | n/a |
| TextVQA | 03 reference (ViSpec) | 2.141 (8841) | 53.7 | n/a | 22/100 |
| TextVQA | 05 ViSpec 1.0:attention | 2.141 (8841) | 53.7 | 100/100 | 22/100 |
| TextVQA | 05 ViSpec 1.0:spatial | 2.141 (8841) | 54.3 | 100/100 | 22/100 |
| TextVQA | 05 ViSpec 0.5:attention | 2.129 (8652) | 52.6 | 48/100 | 22/100 |
| TextVQA | 05 ViSpec 0.5:spatial | 2.145 (8878) | 53.4 | 56/100 | 25/100 |
| TextVQA | 05 ViSpec 0.25:attention | 2.142 (8937) | 53.4 | 40/100 | 19/100 |
| TextVQA | 05 ViSpec 0.25:spatial | 2.141 (8980) | 54.0 | 53/100 | 23/100 |
| COCO Captions | 02 reference (AR) | n/a (AR) | n/a | 28/100 | n/a |
| COCO Captions | 03 reference (ViSpec) | 2.284 (3346) | 53.9 | n/a | 28/100 |
| COCO Captions | 05 ViSpec 1.0:attention | 2.284 (3346) | 53.9 | 100/100 | 28/100 |
| COCO Captions | 05 ViSpec 1.0:spatial | 2.284 (3346) | 53.3 | 100/100 | 28/100 |
| COCO Captions | 05 ViSpec 0.5:attention | 2.294 (3349) | 53.4 | 72/100 | 31/100 |
| COCO Captions | 05 ViSpec 0.5:spatial | 2.344 (3452) | 51.2 | 77/100 | 29/100 |
| COCO Captions | 05 ViSpec 0.25:attention | 2.289 (3351) | 53.8 | 58/100 | 29/100 |
| COCO Captions | 05 ViSpec 0.25:spatial | 2.360 (3448) | 53.1 | 75/100 | 24/100 |

*Live KV is arithmetic (`kv_tokens` times 36,864 bytes, output tokens included), so its ratio shifts with output length. The post-compaction peak is the steady-state decode figure.*

| Benchmark | Arm | Smallest kept count per image | Live KV mean (MiB, arithmetic) | Live KV ratio vs same-policy 1.0 | Decode peak mean / max (GiB) | Decode peak ratio vs same-policy 1.0 | Post-compaction peak ratio vs same-policy 1.0 |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| TextVQA | 02 reference (AR) | n/a | 47.3 | n/a | 7.945 / 7.970 | n/a | n/a |
| TextVQA | 03 reference (ViSpec) | n/a | 47.1 | n/a | 7.993 / 8.015 | n/a | n/a |
| TextVQA | 05 ViSpec 1.0:attention | 729 | 47.1 | 1.000 | 7.993 / 8.015 | 1.000 | 1.000 |
| TextVQA | 05 ViSpec 1.0:spatial | 729 | 47.1 | 1.000 | 7.993 / 8.015 | 1.000 | 1.000 |
| TextVQA | 05 ViSpec 0.5:attention | 364 | 46.9 | 0.995 | 7.993 / 8.015 | 1.000 | 1.000 |
| TextVQA | 05 ViSpec 0.5:spatial | 364 | 47.2 | 1.001 | 7.993 / 8.014 | 1.000 | 1.000 |
| TextVQA | 05 ViSpec 0.25:attention | 182 | 47.2 | 1.002 | 7.993 / 8.016 | 1.000 | 1.000 |
| TextVQA | 05 ViSpec 0.25:spatial | 182 | 47.3 | 1.003 | 7.993 / 8.013 | 1.000 | 1.000 |
| COCO Captions | 02 reference (AR) | n/a | 18.1 | n/a | 7.878 / 7.893 | n/a | n/a |
| COCO Captions | 03 reference (ViSpec) | n/a | 18.1 | n/a | 7.934 / 7.946 | n/a | n/a |
| COCO Captions | 05 ViSpec 1.0:attention | 264 | 18.1 | 1.000 | 7.934 / 7.946 | 1.000 | 1.000 |
| COCO Captions | 05 ViSpec 1.0:spatial | 264 | 18.1 | 1.000 | 7.934 / 7.946 | 1.000 | 1.000 |
| COCO Captions | 05 ViSpec 0.5:attention | 132 | 18.1 | 1.001 | 7.935 / 7.949 | 1.000 | 1.000 |
| COCO Captions | 05 ViSpec 0.5:spatial | 132 | 18.3 | 1.011 | 7.934 / 7.947 | 1.000 | 1.000 |
| COCO Captions | 05 ViSpec 0.25:attention | 66 | 18.1 | 1.001 | 7.935 / 7.949 | 1.000 | 1.000 |
| COCO Captions | 05 ViSpec 0.25:spatial | 66 | 18.3 | 1.012 | 7.934 / 7.947 | 1.000 | 1.000 |

Retention 0.75 was not run in `long`. The smallest image keeps 182 of 729 tokens at 0.25 and 364 at 0.5 on TextVQA, and 66 of 264 and 132 on COCO, in the draft's adaptor input.

### Style `short`, Week 3 (quality measurement)

| Benchmark | Arm | Latency (s) | TTFT (s) | Mean output tokens (capped) | TPOT (ms) | Speedup vs AR | Latency ratio vs same-policy 1.0 | Latency ratio vs unpruned ViSpec ref |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TextVQA | 02 reference (AR) | 1.100 | 0.934 | 5.03 (0) | 41.6 | 1.000 | n/a | n/a |
| TextVQA | 03 reference (ViSpec) | 1.122 | 0.940 | 5.24 (0) | 69.2 | 0.981 | n/a | n/a |
| TextVQA | 05 ViSpec 1.0:attention | 1.119 | 0.937 | 5.24 (0) | 68.9 | 0.983 | 1.000 | 1.002 |
| TextVQA | 05 ViSpec 1.0:spatial | 1.122 | 0.940 | 5.24 (0) | 69.3 | 0.980 | 1.000 | 0.999 |
| TextVQA | 05 ViSpec 0.75:attention | 1.120 | 0.938 | 5.24 (0) | 68.6 | 0.982 | 0.999 | 1.002 |
| TextVQA | 05 ViSpec 0.75:spatial | 1.121 | 0.940 | 5.24 (0) | 69.1 | 0.981 | 1.001 | 1.000 |
| TextVQA | 05 ViSpec 0.5:attention | 1.122 | 0.943 | 5.24 (0) | 67.6 | 0.980 | 0.997 | 0.999 |
| TextVQA | 05 ViSpec 0.5:spatial | 1.122 | 0.938 | 5.24 (0) | 69.2 | 0.981 | 1.001 | 1.000 |
| TextVQA | 05 ViSpec 0.25:attention | 1.118 | 0.936 | 5.24 (0) | 69.0 | 0.984 | 1.001 | 1.003 |
| TextVQA | 05 ViSpec 0.25:spatial | 1.120 | 0.939 | 5.24 (0) | 68.7 | 0.982 | 1.002 | 1.001 |
| COCO Captions | 02 reference (AR) | 0.784 | 0.222 | 14.76 (0) | 40.9 | 1.000 | n/a | n/a |
| COCO Captions | 03 reference (ViSpec) | 0.528 | 0.222 | 15.05 (0) | 22.2 | 1.485 | n/a | n/a |
| COCO Captions | 05 ViSpec 1.0:attention | 0.530 | 0.222 | 15.05 (0) | 22.3 | 1.479 | 1.000 | 0.996 |
| COCO Captions | 05 ViSpec 1.0:spatial | 0.526 | 0.222 | 15.05 (0) | 22.0 | 1.491 | 1.000 | 1.004 |
| COCO Captions | 05 ViSpec 0.75:attention | 0.535 | 0.222 | 15.05 (0) | 22.6 | 1.467 | 0.991 | 0.988 |
| COCO Captions | 05 ViSpec 0.75:spatial | 0.529 | 0.222 | 15.06 (0) | 22.1 | 1.484 | 0.995 | 1.000 |
| COCO Captions | 05 ViSpec 0.5:attention | 0.531 | 0.222 | 15.05 (0) | 22.3 | 1.478 | 0.999 | 0.996 |
| COCO Captions | 05 ViSpec 0.5:spatial | 0.530 | 0.224 | 15.05 (0) | 22.1 | 1.481 | 0.993 | 0.998 |
| COCO Captions | 05 ViSpec 0.25:attention | 0.530 | 0.222 | 15.06 (0) | 22.3 | 1.480 | 1.001 | 0.997 |
| COCO Captions | 05 ViSpec 0.25:spatial | 0.530 | 0.222 | 15.05 (0) | 22.3 | 1.479 | 0.992 | 0.996 |

*Exact agreement counts identical token IDs in bf16, where near-tied logits can flip with kernel shape, so it reflects near-ties as well as any method effect.*

| Benchmark | Arm | Accepted draft tokens per iteration (iterations) | Decode ms per iteration | Exact agreement with ViSpec reference | Exact agreement with AR (secondary) | Quality | Change vs ViSpec reference |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| TextVQA | 02 reference (AR) | n/a (AR) | n/a | 96/100 | n/a | 71.0 | n/a |
| TextVQA | 03 reference (ViSpec) | 0.690 (310) | 54.6 | n/a | 96/100 | 70.0 | n/a |
| TextVQA | 05 ViSpec 1.0:attention | 0.690 (310) | 54.5 | 100/100 | 96/100 | 70.0 | +0.0 |
| TextVQA | 05 ViSpec 1.0:spatial | 0.690 (310) | 54.6 | 100/100 | 96/100 | 70.0 | +0.0 |
| TextVQA | 05 ViSpec 0.75:attention | 0.685 (311) | 54.2 | 100/100 | 96/100 | 70.0 | +0.0 |
| TextVQA | 05 ViSpec 0.75:spatial | 0.690 (310) | 54.4 | 100/100 | 96/100 | 70.0 | +0.0 |
| TextVQA | 05 ViSpec 0.5:attention | 0.685 (311) | 53.5 | 100/100 | 96/100 | 70.0 | +0.0 |
| TextVQA | 05 ViSpec 0.5:spatial | 0.679 (312) | 54.6 | 100/100 | 96/100 | 70.0 | +0.0 |
| TextVQA | 05 ViSpec 0.25:attention | 0.679 (312) | 54.2 | 100/100 | 96/100 | 70.0 | +0.0 |
| TextVQA | 05 ViSpec 0.25:spatial | 0.690 (310) | 54.3 | 100/100 | 96/100 | 70.0 | +0.0 |
| COCO Captions | 02 reference (AR) | n/a (AR) | n/a | 88/100 | n/a | 121.40 | n/a |
| COCO Captions | 03 reference (ViSpec) | 1.746 (548) | 53.7 | n/a | 88/100 | 119.68 | n/a |
| COCO Captions | 05 ViSpec 1.0:attention | 1.746 (548) | 54.0 | 100/100 | 88/100 | 119.68 | +0.00 |
| COCO Captions | 05 ViSpec 1.0:spatial | 1.746 (548) | 53.4 | 100/100 | 88/100 | 119.68 | +0.00 |
| COCO Captions | 05 ViSpec 0.75:attention | 1.741 (549) | 54.6 | 100/100 | 88/100 | 119.68 | +0.00 |
| COCO Captions | 05 ViSpec 0.75:spatial | 1.738 (550) | 53.5 | 99/100 | 87/100 | 119.93 | +0.24 |
| COCO Captions | 05 ViSpec 0.5:attention | 1.731 (551) | 53.8 | 100/100 | 88/100 | 119.68 | +0.00 |
| COCO Captions | 05 ViSpec 0.5:spatial | 1.736 (550) | 53.3 | 100/100 | 88/100 | 119.68 | +0.00 |
| COCO Captions | 05 ViSpec 0.25:attention | 1.733 (551) | 53.6 | 99/100 | 87/100 | 119.72 | +0.04 |
| COCO Captions | 05 ViSpec 0.25:spatial | 1.751 (547) | 54.1 | 100/100 | 88/100 | 119.68 | +0.00 |

*Live KV is arithmetic (`kv_tokens` times 36,864 bytes, output tokens included), so its ratio shifts with output length. The post-compaction peak is the steady-state decode figure.*

| Benchmark | Arm | Smallest kept count per image | Live KV mean (MiB, arithmetic) | Live KV ratio vs same-policy 1.0 | Decode peak mean / max (GiB) | Decode peak ratio vs same-policy 1.0 | Post-compaction peak ratio vs same-policy 1.0 |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| TextVQA | 02 reference (AR) | n/a | 36.7 | n/a | 7.944 / 7.968 | n/a | n/a |
| TextVQA | 03 reference (ViSpec) | n/a | 36.7 | n/a | 7.982 / 8.005 | n/a | n/a |
| TextVQA | 05 ViSpec 1.0:attention | 729 | 36.7 | 1.000 | 7.982 / 8.005 | 1.000 | 1.000 |
| TextVQA | 05 ViSpec 1.0:spatial | 729 | 36.7 | 1.000 | 7.982 / 8.005 | 1.000 | 1.000 |
| TextVQA | 05 ViSpec 0.75:attention | 547 | 36.7 | 1.000 | 7.982 / 8.004 | 1.000 | 1.000 |
| TextVQA | 05 ViSpec 0.75:spatial | 547 | 36.7 | 1.000 | 7.982 / 8.007 | 1.000 | 1.000 |
| TextVQA | 05 ViSpec 0.5:attention | 364 | 36.7 | 1.000 | 7.981 / 8.005 | 1.000 | 1.000 |
| TextVQA | 05 ViSpec 0.5:spatial | 364 | 36.7 | 1.000 | 7.982 / 8.007 | 1.000 | 1.000 |
| TextVQA | 05 ViSpec 0.25:attention | 182 | 36.7 | 1.000 | 7.981 / 8.003 | 1.000 | 1.000 |
| TextVQA | 05 ViSpec 0.25:spatial | 182 | 36.7 | 1.000 | 7.982 / 8.007 | 1.000 | 1.000 |
| COCO Captions | 02 reference (AR) | n/a | 14.8 | n/a | 7.878 / 7.893 | n/a | n/a |
| COCO Captions | 03 reference (ViSpec) | n/a | 14.8 | n/a | 7.930 / 7.948 | n/a | n/a |
| COCO Captions | 05 ViSpec 1.0:attention | 264 | 14.8 | 1.000 | 7.930 / 7.948 | 1.000 | 1.000 |
| COCO Captions | 05 ViSpec 1.0:spatial | 264 | 14.8 | 1.000 | 7.930 / 7.948 | 1.000 | 1.000 |
| COCO Captions | 05 ViSpec 0.75:attention | 198 | 14.8 | 1.000 | 7.930 / 7.947 | 1.000 | 1.000 |
| COCO Captions | 05 ViSpec 0.75:spatial | 198 | 14.8 | 1.000 | 7.930 / 7.947 | 1.000 | 1.000 |
| COCO Captions | 05 ViSpec 0.5:attention | 132 | 14.8 | 1.000 | 7.930 / 7.947 | 1.000 | 1.000 |
| COCO Captions | 05 ViSpec 0.5:spatial | 132 | 14.8 | 1.000 | 7.930 / 7.947 | 1.000 | 1.000 |
| COCO Captions | 05 ViSpec 0.25:attention | 66 | 14.8 | 1.000 | 7.930 / 7.947 | 1.000 | 1.000 |
| COCO Captions | 05 ViSpec 0.25:spatial | 66 | 14.8 | 1.000 | 7.930 / 7.947 | 1.000 | 1.000 |

All six pruned `short` arms keep TextVQA accuracy at 70.0. On COCO, four of six give exactly the reference CIDEr-D (119.68). The other two, 0.75:spatial and 0.25:attention, differ from the reference on 1 of 100 samples and score 119.93 and 119.72.

### Acceptance after removing one answer

On `long` COCO the two spatial arms at 0.5 and 0.25 show higher acceptance than the reference (2.344 and 2.360 against 2.284). One answer causes it. COCO question 456521 ends at 337 tokens in the reference but runs to the 1,024-token cap in both arms (mean output 115.43 and 115.87 tokens against 109.89, and 3,452 and 3,448 iterations against 3,346), with 2.908 accepted tokens per iteration. Computed from each run's `samples.jsonl`, the means over the other 99 samples are 2.297 (0.5:spatial) and 2.315 (0.25:spatial) against 2.319 for the reference. So pruning did not raise acceptance there.

### Ratios that run against the trend

Draft pruning should leave latency at the 1.0 level. Three `long` ratios run against that (the 1.0 arms differ from the reference by 0.992 to 1.012):

- **Long TextVQA 0.5:attention, 1.037 (unpruned reference 1.036).** The arm's answers are shorter (270.76 tokens against 277.70) and it needs fewer iterations (8,652 against 8,841). Decode time per iteration is 52.6 ms against 53.7 ms, inside the 52.6 to 54.3 ms spread of this job's TextVQA ViSpec runs. The ratio therefore describes shorter text and not a faster step.
- **Long COCO 0.25:spatial, 0.976 (unpruned reference 0.988).** One answer ran to the cap (see above), so the arm's mean output is 5.4% longer (115.87 tokens), which accounts for the slower mean latency.
- **Long COCO 0.5:spatial, 1.008 (unpruned reference 1.020, outside the 1.0 arms' spread), also with the capped answer.** The arm is faster despite longer output. Decode time per iteration is 51.2 ms against 53.3 ms for the 1.0 spatial arm, 4% lower, and the cause is not identified. It is the fastest COCO arm of the job, and with one run per arm it cannot be separated from run-to-run spread.

On `short` every ratio is within 0.988 to 1.003, in line with the 1.0 arms (0.996 to 1.004).

### Predictions against the Week 3 measurements

The predictions are those recorded before Week 2 (see [Purpose and hypothesis](#purpose-and-hypothesis)).

| Prediction (written before the runs) | Outcome | Evidence |
| --- | --- | --- |
| Draft pruning costs acceptance at most slightly | **Held** | Accepted tokens per iteration differ from the 1.0 arm by -0.012 to +0.004 on TextVQA (`long`) and -0.011 to +0.000 (`short`). On COCO the changes are +0.005 to +0.076 on `long` (the two large rises come from one capped answer) and -0.015 to +0.005 on `short`. |
| Draft pruning barely moves latency | **Held** | Latency ratios against the same-policy 1.0 arm: 0.976 to 1.037 on `long` and 0.991 to 1.002 on `short`. TTFT ratios (1.0 arm over arm) 0.989 to 1.003. The two outliers on `long` follow output length (see above). |
| The target still verifies every draft, so greedy output matches the 1.0 reference up to bf16 near-ties | **Held for `short`, held for `long` apart from one known case** | `short` agrees on 99 or 100 of 100 in every arm. `long` agrees on 40 to 56 (TextVQA) and 58 to 77 (COCO), lower at 0.25 than at 0.5. The diagnosis (below) classifies every `long` divergence except one as a bf16 near-tie, and the two `short` divergences as near-ties. |

The latency outcome is stated from the tables. The reason it did not move is an inference, not a measurement.

### Draft-only divergence diagnosis, Week 3

GPU job 402165 ran `diagnose_divergence` on the 10 draft-only Week 3 arms (`short` 0.75, 0.5 and 0.25, and `long` 0.5 and 0.25, each with both policies), comparing each arm with its same-job 03 reference, with the stock full-KV model teacher-forced on the shared prefix. A case is `near_tie` if the bf16 top-2 margin is at most 0.5 logits, and the gate passes only when no case is `unexplained` or `both_wrong`. Prompt fingerprints matched in every case (0 mismatches). The fp32 pass ran on the first 50 cases of each arm (`fp32_limit` 50), as in Week 2, so the cases after the first 50 are classified by bf16 margin only: 52 cases of `long` 0.25:attention, 22 of 0.25:spatial, 30 of 0.5:attention and 17 of 0.5:spatial. The Week 2 note about extending fp32 beyond 50 cases was not carried out. This job ran on node `c4-3` and card `GPU-786d8051-7e60-0cbf-22b0-defdaae84a18` at commit `c3a36ac25f95`, which is not the card of any timed job, so it is a check and not a timed result. The agreement columns count which token the stock model's argmax picks; near-ties flip in bf16 as well as in fp32.

| Style | Arm | Cases (TextVQA / COCO) | near_tie | unexplained | both_wrong | bf16 margin median / max (logits) | bf16 argmax agrees with AR / ViSpec / neither | fp32 argmax agrees with AR / ViSpec / neither (not run) | Draft-only gate |
| --- | --- | --- | ---: | ---: | ---: | --- | --- | --- | --- |
| long | 0.25:attention | 102 (60 / 42) | 102 | 0 | 0 | 0.125 / 0.375 | 52 / 45 / 5 | 22 / 26 / 2 (52) | passes |
| long | 0.25:spatial | 72 (47 / 25) | 72 | 0 | 0 | 0.094 / 0.375 | 37 / 31 / 4 | 34 / 16 / 0 (22) | passes |
| long | 0.5:attention | 80 (52 / 28) | 79 | 1 | 0 | 0.125 / 0.625 | 34 / 44 / 2 | 22 / 28 / 0 (30) | **fails** |
| long | 0.5:spatial | 67 (44 / 23) | 67 | 0 | 0 | 0.000 / 0.250 | 40 / 25 / 2 | 29 / 19 / 2 (17) | passes |
| short | 0.25:attention | 1 (0 / 1) | 1 | 0 | 0 | 0.125 / 0.125 | 0 / 1 / 0 | 0 / 1 / 0 (0) | passes |
| short | 0.25:spatial | 0 (0 / 0) | 0 | 0 | 0 | n/a | 0 / 0 / 0 | 0 / 0 / 0 (0) | passes (no cases) |
| short | 0.5:attention | 0 (0 / 0) | 0 | 0 | 0 | n/a | 0 / 0 / 0 | 0 / 0 / 0 (0) | passes (no cases) |
| short | 0.5:spatial | 0 (0 / 0) | 0 | 0 | 0 | n/a | 0 / 0 / 0 | 0 / 0 / 0 (0) | passes (no cases) |
| short | 0.75:attention | 0 (0 / 0) | 0 | 0 | 0 | n/a | 0 / 0 / 0 | 0 / 0 / 0 (0) | passes (no cases) |
| short | 0.75:spatial | 1 (0 / 1) | 1 | 0 | 0 | 0.125 / 0.125 | 1 / 0 / 0 | 0 / 1 / 0 (0) | passes |

**Nine of the 10 arms pass.** The 4 `short` arms with no divergence and the `short` 0.25:attention and 0.75:spatial arms (1 case each, both near-ties) pass, and so do `long` 0.25:attention (102 near-ties), 0.25:spatial (72) and 0.5:spatial (67). No case in any arm is `both_wrong`. The long arms' bf16 margins reach at most 0.375 logits, except for the one case below.

**The `long` 0.5:attention gate fails on the known case, and nothing else.** It has 79 near-ties and 1 unexplained case of 80, the same counts as Week 2. The unexplained case is TextVQA question 35537 at output index 162: the reference emits "." (token 13), the pruned arm emits "**:" (token 95518), and the stock model prefers "**:" by 0.625 logits in bf16 (24.625 against 24.0) and 0.724 in fp32. This is the case that [experiment 07](../07_qwen_decode_drift/README.md) traced to an exact tie in the fork's own cached bf16 logits, and it was judged bf16 implementation noise, a known limitation, so it is not a pruning effect. The threshold of 0.5 logits was not changed and the gate is recorded as failed for this arm.

Diagnosis files are under each arm's `comparisons/20261105T192451Z_nullref_vs_03-ref/diagnosis/` (`diagnosis.json` and `diagnosis.md`); the list of the 10 is in the Week 3 sweep record.

## Limits

- One run per arm and style: repeat variability is unknown. Subsets are 100 samples, not full benchmarks, so agreement counts and quality differences are noisy.
- The `long` and `short` styles ran on different nodes, host CPUs and cards. Do not compare their latencies.
- The draft-only diagnosis gate failed for `long` 0.5:attention, with one unexplained case. Our reading (a shared cached-decode effect, not pruning) is an inference from one case. Experiment 07 judged the case bf16 implementation noise, a known limitation ([07 README](../07_qwen_decode_drift/README.md)), and its mechanism is not isolated.
- Exact agreement reflects bf16 near-ties and method effects together. Draft-only arms would reproduce the reference in exact arithmetic.
- Pruning changes only the draft input. It cannot shrink target KV memory; see [06](../06_qwen_kv_eviction/README.md) for that.
- The attention-policy overlap comes from a 5-sample smoke run and was not re-measured in these runs.
- Week 3 has one run per arm and style as well. The `long` style did not run the 0.75 budget (dropped to save GPU time), and its three jobs ran on two cards at different times, so no absolute time is compared across jobs.
- The Week 3 diagnosis covers fp32 on the first 50 cases of each arm only. Cases beyond that (52, 22, 30 and 17 for `long` 0.25:attention, 0.25:spatial, 0.5:attention and 0.5:spatial) are classified by bf16 margin only, and the Week 2 extension was not carried out.
- The `long` and `short` Week 3 agreement counts include bf16 near-tie flips, which exact agreement cannot separate from pruning effects.
