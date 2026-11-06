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

Status: **measured for Week 2** at retention 1.0 and 0.5 (jobs 393336 and 393337, both exit 0, 100/100
samples per benchmark). The 1.0 identity gate passed on every arm. Acceptance and latency predictions held. At
0.5 in the `short` style every arm agrees with the reference on 100/100 samples, and in the `long` style
48 to 77 of 100 agree. The draft-only divergence diagnosis failed for `long` 0.5:attention (79 near-ties and 1
unexplained case), which the controller's evidence traces to the shared cached decode and not to pruning, an
inference left open in [Results](#draft-only-divergence-diagnosis). Retention 0.25 and 0.75 and the combined arms are
Week 3, pending.

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

The Week 2 jobs ran 02, 03, 06 and 05 in one allocation per style, with `--cpus-per-task=4 --mem=22G` and a
6 h (`long`) or 2 h (`short`) limit. The arm lists are recorded as `arm_list` in each run's `metadata.txt`:
`1.0:spatial 1.0:attention 0.5:spatial 0.5:attention` for 05.
Compare an arm with its reference:

```bash
python -m vispec.evaluation.compare_metrics --mode sparsity --original <03 report>... --draft <05 report>... --output-dir <new_dir>
```

## Results

All figures come from one run per arm per prompt style: job 393336 for `long` and job 393337 for `short`, each ran 02, 03, 06 and 05 in one allocation at commit `66f492a5dc65` with a clean tree. Every arm used greedy decoding (temperature 0.0), bf16, batch size 1, at most 1,024 new tokens, depth 3, top-k 8, total-token 30, `num_q` 2, slim prefill on and the compact KV layout. Every run exited 0 and scored 100/100 samples on both benchmarks. Per-benchmark results are listed first, and no cross-benchmark aggregate is computed. The reference for every arm is the 03 run from the same job.

Retention 0.25, retention 0.75 and the combined draft-plus-KV arms are **Week 3, pending**. Only 1.0 and 0.5 were run.

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

A controller check adds that the Week 2 compact references (02 and 03, `long`) equal the Week 1 fixed-layout runs token for token on 100/100 for TextVQA and for COCO Captions. The same holds for the `short` style. The compact layout therefore changed no output. The Week 1 runs are jobs 392660 and 392661 (see the [03 README](../03_qwen_vispec/README.md#results)).

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

The unexplained case is TextVQA question 35537, first mismatch at output index 162. The controller examined it and recorded this evidence:

- The reference (unpruned ViSpec, run 03) emitted "." (token 13) at index 162. The Week 1 02 and 03 runs and the Week 2 02 and 03 references emit the same token there.
- The stock model, teacher-forced on the shared 162-token prefix, prefers "**:" (token 95518) over "." by 0.625 logits in bf16 and 0.724 in fp32.
- The pruned candidate emitted "**:", the stock argmax.

From this we infer that the deviating token belongs to the shared incremental KV-cached decode, which standard autoregressive decoding and ViSpec both use, and that pruning did not cause it. It is therefore not a Week 2 regression. This is an inference. The cause of the drift between cached decode and full recompute (0.72 fp32 logits at a 162-token prefix) is unexplained and open. The draft-only gate is still recorded as failed for this arm.

### Notes on the attention policy

The attention policy ranks the adaptor's image tokens by attention from its own learned queries and keeps the top fraction. In the Task 6 smoke run on 5 `long` TextVQA samples, the attention-selected and spatial-selected halves overlapped at a Jaccard of 0.31 to 0.35, which is about the 1/3 expected for two unrelated half-subsets. So the attention policy is not a disguised stride pattern. The selected indices are not saved with the Week 2 runs, so the overlap was not re-measured there. Attention scoring also runs at retention 1.0, so the 1.0 attention arm carries that overhead, which is part of why the primary latency reference is the same-policy 1.0 arm.

### Week 3 (pending)

- Retention 0.25 and 0.75.
- Combined draft-plus-KV arms.
- fp32 confirmation of the draft-arm divergences, beyond the first 50 cases per arm, and a repeat of the failed `long` 0.5:attention gate.

## Limits

- One run per arm and style: repeat variability is unknown. Subsets are 100 samples, not full benchmarks, so agreement counts and quality differences are noisy.
- The `long` and `short` styles ran on different nodes, host CPUs and cards. Do not compare their latencies.
- The draft-only diagnosis gate failed for `long` 0.5:attention, with one unexplained case. The controller's reading (a shared cached-decode effect, not pruning) is an inference from one case, and the underlying drift is open.
- Exact agreement reflects bf16 near-ties and method effects together. Draft-only arms would reproduce the reference in exact arithmetic.
- Pruning changes only the draft input. It cannot shrink target KV memory; see [06](../06_qwen_kv_eviction/README.md) for that.
- The attention-policy overlap comes from a 5-sample smoke run and was not re-measured in these runs.
- Only retention 1.0 and 0.5 were run. Retention 0.25 and 0.75 and combined arms are Week 3 and pending.
