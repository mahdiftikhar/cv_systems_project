# 06 — Qwen2.5-VL target-side visual KV eviction

## Purpose and hypothesis

This experiment measures Qwen2.5-VL-3B-Instruct when the target's KV cache keeps only a fraction of
each image's tokens after the prefill. Each arm runs under autoregressive or ViSpec decoding and is
compared with the same method at retention 1.0, run in the same job.

Predictions, recorded before the runs. The [04 profile](../04_qwen_decode_profile/README.md) inferred that decode is
host-bound: the GPU is idle for about half or more of each step and attention over the KV cache is 3 to
12% of kernel time (the attention kernel alone, closer to 10% for AR TextVQA once the context-dependent copies are added). Eviction should therefore lower memory (live KV and decode peak) but barely move
latency. It can change the output, because later tokens attend to fewer image positions, so output
agreement and quality against the reference are measured rather than assumed. The
[Results](#predictions-against-the-measurements) section scores each prediction.

Status: **measured for Week 2** at retention 1.0 and 0.5 (jobs 393336 and 393337, both exit 0, 100/100
samples per benchmark). The 1.0 identity gate passed on every arm, and the AR-versus-ViSpec bookkeeping check at 0.5
passed (long COCO exactly at the 10.0-point limit). Live KV fell to 0.65 to 0.67 of the 1.0 arm in `long` and 0.53 to
0.58 in `short`, and the decode peak moved by about 0.2% or less. Long-style outputs diverge heavily from the 1.0
arm, and short-style quality dropped for the spatial arms. Retention 0.25 and 0.75 and the combined arms are Week 3, pending.

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

The Week 2 jobs ran 02, 03, 06 and 05 in one allocation per style, with `--cpus-per-task=4 --mem=22G` and a
6 h (`long`) or 2 h (`short`) limit. The arm list is recorded as `arm_list` in each run's `metadata.txt`:
`1.0:spatial 1.0:attention 0.5:spatial 0.5:attention 1.0:spatial:autoregressive 0.5:spatial:autoregressive`.

Compare an arm with the same method's reference:

```bash
python -m vispec.evaluation.compare_metrics --mode sparsity --original <reference report>... --draft <06 report>... --output-dir <new_dir>
```

## Results

All figures come from one run per arm per prompt style: job 393336 for `long` and job 393337 for `short`, each ran 02, 03, 06 and 05 in one allocation at commit `66f492a5dc65` with a clean tree. Every arm used greedy decoding (temperature 0.0), bf16, batch size 1, at most 1,024 new tokens, depth 3, top-k 8, total-token 30, `num_q` 2, slim prefill on and the compact KV layout. Every run exited 0 and scored 100/100 samples on both benchmarks. Per-benchmark results are listed first, and no cross-benchmark aggregate is computed. Each arm's reference is the same decoding method at retention 1.0 from the same job.

Retention 0.25, retention 0.75 and the combined draft-plus-KV arms are **Week 3, pending**. Only 1.0 and 0.5 were run.

### Hardware

Each style ran inside one Slurm job on one physical card, so an arm and its references share hardware. The two styles ran on different nodes, host CPUs and cards, so their rows are never pooled or compared in latency. Week 1 `long` ran on `c2-2`, so the Week 2 `long` timings also differ in host from Week 1 and are comparable only within this job.

| Style | Job / partition | Node and host CPU | GPU | UUID | Driver | Allocation | Device report (one per run, all listed below) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| long | 393336 / `preempt`, 03:29:51, 6 h limit | `c1-8`, Intel Xeon Silver 4116 @ 2.10 GHz | NVIDIA RTX A6000 (47.5 GiB visible) | `GPU-8d51152c-22f1-372d-aef5-063afee0d488` | 615.71.09 | 1 GPU, 4 CPUs (affinity 4, 14, 28, 38), 22,528 MB | [06 1.0:spatial:autoregressive device.txt](runs/20261020T101850Z_66f492a5dc65_long_r1.0_spatial_autoregressive_DjLwjt/device.txt) |
| short | 393337 / `preempt`, 00:51:00, 2 h limit | `c2-2`, Intel Xeon Silver 4114 @ 2.20 GHz | NVIDIA RTX A6000 (47.5 GiB visible) | `GPU-a2e40ccd-5141-9e51-2b3f-0e9d71fabf73` | 615.71.09 | 1 GPU, 4 CPUs (affinity 9, 15, 29, 35), 22,528 MB | [06 1.0:spatial:autoregressive device.txt](runs/20261020T075750Z_66f492a5dc65_short_r1.0_spatial_autoregressive_oGVPxT/device.txt) |

The links point at generated run folders, which Git ignores and which are preserved on project storage, so they resolve only in a checkout that has them. Device reports are launch snapshots and do not measure runtime peaks. The job command is recorded in each run's `metadata.txt` (`submit_command`, `arm_list`). The job ran the experiments in the order 02, 03, 06, 05.

#### All run folders and device reports

| Style | Run | Arm | Run folder (preserved on project storage) | Device report |
| --- | --- | --- | --- | --- |
| long | 02 | reference | [`20261020T084113Z_66f492a5dc65_long_2P5sjB`](../02_qwen_autoregressive/runs/20261020T084113Z_66f492a5dc65_long_2P5sjB/) | [device.txt](../02_qwen_autoregressive/runs/20261020T084113Z_66f492a5dc65_long_2P5sjB/device.txt) |
| long | 06 | 1.0:spatial:autoregressive | [`20261020T101850Z_66f492a5dc65_long_r1.0_spatial_autoregressive_DjLwjt`](runs/20261020T101850Z_66f492a5dc65_long_r1.0_spatial_autoregressive_DjLwjt/) | [device.txt](runs/20261020T101850Z_66f492a5dc65_long_r1.0_spatial_autoregressive_DjLwjt/device.txt) |
| long | 06 | 0.5:spatial:autoregressive | [`20261020T104647Z_66f492a5dc65_long_r0.5_spatial_autoregressive_OIyV5v`](runs/20261020T104647Z_66f492a5dc65_long_r0.5_spatial_autoregressive_OIyV5v/) | [device.txt](runs/20261020T104647Z_66f492a5dc65_long_r0.5_spatial_autoregressive_OIyV5v/device.txt) |
| long | 03 | reference | [`20261020T090922Z_66f492a5dc65_long_wDElc3`](../03_qwen_vispec/runs/20261020T090922Z_66f492a5dc65_long_wDElc3/) | [device.txt](../03_qwen_vispec/runs/20261020T090922Z_66f492a5dc65_long_wDElc3/device.txt) |
| long | 06 | 1.0:spatial | [`20261020T092258Z_66f492a5dc65_long_r1.0_spatial_LDfpdZ`](runs/20261020T092258Z_66f492a5dc65_long_r1.0_spatial_LDfpdZ/) | [device.txt](runs/20261020T092258Z_66f492a5dc65_long_r1.0_spatial_LDfpdZ/device.txt) |
| long | 06 | 1.0:attention | [`20261020T093629Z_66f492a5dc65_long_r1.0_attention_yrAMKW`](runs/20261020T093629Z_66f492a5dc65_long_r1.0_attention_yrAMKW/) | [device.txt](runs/20261020T093629Z_66f492a5dc65_long_r1.0_attention_yrAMKW/device.txt) |
| long | 06 | 0.5:spatial | [`20261020T095000Z_66f492a5dc65_long_r0.5_spatial_jQnrVo`](runs/20261020T095000Z_66f492a5dc65_long_r0.5_spatial_jQnrVo/) | [device.txt](runs/20261020T095000Z_66f492a5dc65_long_r0.5_spatial_jQnrVo/device.txt) |
| long | 06 | 0.5:attention | [`20261020T100447Z_66f492a5dc65_long_r0.5_attention_Mi2col`](runs/20261020T100447Z_66f492a5dc65_long_r0.5_attention_Mi2col/) | [device.txt](runs/20261020T100447Z_66f492a5dc65_long_r0.5_attention_Mi2col/device.txt) |
| short | 02 | reference | [`20261020T073211Z_66f492a5dc65_short_dv0FuB`](../02_qwen_autoregressive/runs/20261020T073211Z_66f492a5dc65_short_dv0FuB/) | [device.txt](../02_qwen_autoregressive/runs/20261020T073211Z_66f492a5dc65_short_dv0FuB/device.txt) |
| short | 06 | 1.0:spatial:autoregressive | [`20261020T075750Z_66f492a5dc65_short_r1.0_spatial_autoregressive_oGVPxT`](runs/20261020T075750Z_66f492a5dc65_short_r1.0_spatial_autoregressive_oGVPxT/) | [device.txt](runs/20261020T075750Z_66f492a5dc65_short_r1.0_spatial_autoregressive_oGVPxT/device.txt) |
| short | 06 | 0.5:spatial:autoregressive | [`20261020T080215Z_66f492a5dc65_short_r0.5_spatial_autoregressive_M42YCK`](runs/20261020T080215Z_66f492a5dc65_short_r0.5_spatial_autoregressive_M42YCK/) | [device.txt](runs/20261020T080215Z_66f492a5dc65_short_r0.5_spatial_autoregressive_M42YCK/device.txt) |
| short | 03 | reference | [`20261020T073727Z_66f492a5dc65_short_AFFHib`](../03_qwen_vispec/runs/20261020T073727Z_66f492a5dc65_short_AFFHib/) | [device.txt](../03_qwen_vispec/runs/20261020T073727Z_66f492a5dc65_short_AFFHib/device.txt) |
| short | 06 | 1.0:spatial | [`20261020T074127Z_66f492a5dc65_short_r1.0_spatial_LDXAj8`](runs/20261020T074127Z_66f492a5dc65_short_r1.0_spatial_LDXAj8/) | [device.txt](runs/20261020T074127Z_66f492a5dc65_short_r1.0_spatial_LDXAj8/device.txt) |
| short | 06 | 1.0:attention | [`20261020T074535Z_66f492a5dc65_short_r1.0_attention_fpjQ1y`](runs/20261020T074535Z_66f492a5dc65_short_r1.0_attention_fpjQ1y/) | [device.txt](runs/20261020T074535Z_66f492a5dc65_short_r1.0_attention_fpjQ1y/device.txt) |
| short | 06 | 0.5:spatial | [`20261020T074943Z_66f492a5dc65_short_r0.5_spatial_wiF6ah`](runs/20261020T074943Z_66f492a5dc65_short_r0.5_spatial_wiF6ah/) | [device.txt](runs/20261020T074943Z_66f492a5dc65_short_r0.5_spatial_wiF6ah/device.txt) |
| short | 06 | 0.5:attention | [`20261020T075347Z_66f492a5dc65_short_r0.5_attention_94d5ZE`](runs/20261020T075347Z_66f492a5dc65_short_r0.5_attention_94d5ZE/) | [device.txt](runs/20261020T075347Z_66f492a5dc65_short_r0.5_attention_94d5ZE/device.txt) |

### Reading these results

Units follow the [03 README](../03_qwen_vispec/README.md#reading-these-results) and the [metrics guide](../METRICS.md). Latency is wall time per question in seconds and TTFT is the wait before the first target token. TPOT is `(latency - TTFT) / (output tokens - 1)` in milliseconds, averaged per question. Output tokens include EOS. Accepted draft tokens per iteration is the mean over verification iterations, and the number in parentheses is the iteration count. Quality is TextVQA consensus accuracy in percent, or COCO CIDEr-D x 100 with inverse document frequency from the 100-image subset, so it is a subset measurement. The memory columns come from the `memory` block of each `summary.json` ([metrics guide](../METRICS.md#phase-memory-and-live-kv-memory-block)). Live KV is the whole-model figure in MiB, and `kv_tokens` includes the output tokens. Peaks are `torch.cuda.max_memory_allocated` per question in GiB (mean and max over 100 questions). Evicted tokens (`kv_evicted_tokens`) and prompt tokens (`input_ids_tokens`) are per-question counts from `results/*/test.jsonl`.

Ratios are reference over candidate, so a value above 1 means the candidate was faster. Each 0.5 arm is compared with the same-policy 1.0 arm (primary), because the 1.0 arm pays the same image-mask and score-capture overhead as the 0.5 arm. The null reference, the 02 or 03 run without any sparsity code, is the secondary column. It also anchors the identity gate. A 1.0 arm has no same-policy reference, so its row shows ratios against the null reference only. Those 1.0 rows measure overhead plus run-to-run noise together: latency ratios range from 0.930 to 1.017 and TTFT ratios from 0.972 to 1.010, and the largest departure is short COCO ViSpec 1.0:spatial (0.930). With one run per arm they cannot be separated.

The `long` style is the speed measurement. Its strict quality scores are near zero (Week 1 scored about 0), so quality is secondary there. The `short` style is the quality measurement, and quality is primary there. Agreement is exact token-ID agreement over 100 samples. With n = 100, agreement counts and quality differences are noisy, and no confidence intervals are computed. One TextVQA question is worth 1 point of the 100-question accuracy.

### Identity gate at retention 1.0

The gate passed. Every 1.0 arm reproduces its null reference token for token on 100 of 100 samples, on both benchmarks and in both styles, for both methods and both policies.

| Arm (vs its null reference) | long TextVQA | long COCO | short TextVQA | short COCO |
| --- | ---: | ---: | ---: | ---: |
| 06 AR 1.0:spatial vs 02 | 100/100 | 100/100 | 100/100 | 100/100 |
| 06 ViSpec 1.0:spatial vs 03 | 100/100 | 100/100 | 100/100 | 100/100 |
| 06 ViSpec 1.0:attention vs 03 | 100/100 | 100/100 | 100/100 | 100/100 |

A controller check adds that the Week 2 compact references (02 and 03, `long`) equal the Week 1 fixed-layout runs token for token on 100/100 for TextVQA and for COCO Captions. The same holds for the `short` style. The compact layout therefore changed no output. The Week 1 runs are jobs 392660 and 392661 (see the [02](../02_qwen_autoregressive/README.md#results) and [03](../03_qwen_vispec/README.md#results) READMEs).

### Bookkeeping check at 0.5

The check compares the AR and ViSpec eviction arms at 0.5:spatial with the unpruned 02 and 03 compact pair, to see whether eviction adds AR-versus-ViSpec disagreement beyond the bf16 near-tie disagreement that the unpruned pair already shows. The check passed.

| Style | Pair | Benchmark | Exact agreement | First mismatch min / median / max | First mismatch below 5 | First mismatch at 0 |
| --- | --- | --- | ---: | --- | ---: | ---: |
| long | AR 0.5:spatial vs ViSpec 0.5:spatial (eviction) | TextVQA | 19/100 | 1 / 51 / 399 | 5 | 0 |
| long | AR 0.5:spatial vs ViSpec 0.5:spatial (eviction) | COCO Captions | 18/100 | 2 / 41 / 132 | 4 | 0 |
| long | 02 vs 03 compact (no eviction) | TextVQA | 22/100 | 1 / 58.5 / 983 | 7 | 0 |
| long | 02 vs 03 compact (no eviction) | COCO Captions | 28/100 | 2 / 27.5 / 119 | 9 | 0 |
| short | AR 0.5:spatial vs ViSpec 0.5:spatial (eviction) | TextVQA | 97/100 | 3 / 18 / 70 | 1 | 0 |
| short | AR 0.5:spatial vs ViSpec 0.5:spatial (eviction) | COCO Captions | 87/100 | 2 / 7 / 12 | 4 | 0 |
| short | 02 vs 03 compact (no eviction) | TextVQA | 96/100 | 1 / 1.5 / 45 | 3 | 0 |
| short | 02 vs 03 compact (no eviction) | COCO Captions | 88/100 | 1 / 6 / 16 | 5 | 0 |

The other three cells are within 3 points of the unpruned pair (long TextVQA 19 against 22, short TextVQA 97 against 96, short COCO 87 against 88). Long COCO sits exactly at the limit: 18/100 against 28/100 is a gap of 10.0 points, and the rule flagged a gap of more than 10 points. The limit was reached and not exceeded, so the check passes. A first mismatch at index 0 never occurred. With n = 100, these agreement counts are noisy, and a 10-point gap on 100 samples is a small number of questions.

### Style `long` (speed measurement)

Eviction changes what the target attends to, so a 0.5 arm is expected to diverge from its own 1.0 arm. It does, heavily: exact agreement is 0 to 6 of 100 on TextVQA and 1 to 2 of 100 on COCO. Many first mismatches come early (index below 5 in 31 of 100 TextVQA questions for AR spatial, 25 for ViSpec spatial and 8 for ViSpec attention). The agreement columns reflect bf16 near-ties as well as the method effect, and `diagnose_divergence` cannot classify these arms (see the limits). Quality is secondary in this style.

| Benchmark | Arm | Latency (s) | TTFT (s) | Mean output tokens | Truncated at cap | TPOT (ms) | Latency ratio vs same-policy 1.0 | TTFT ratio vs same-policy 1.0 | Latency ratio vs null ref | TTFT ratio vs null ref |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TextVQA | 02 reference (AR) | 11.481 | 0.935 | 283.31 | 4 | 37.4 | n/a | n/a | n/a | n/a |
| TextVQA | 06 AR 1.0:spatial | 11.453 | 0.934 | 283.31 | 4 | 37.3 | n/a | n/a | 1.002 | 1.001 |
| TextVQA | 06 AR 0.5:spatial | 12.874 | 0.933 | 317.24 | 7 | 37.8 | 0.890 | 1.001 | 0.892 | 1.002 |
| TextVQA | 03 reference (ViSpec) | 5.395 | 0.940 | 277.70 | 3 | 17.2 | n/a | n/a | n/a | n/a |
| TextVQA | 06 ViSpec 1.0:spatial | 5.369 | 0.941 | 277.70 | 3 | 17.1 | n/a | n/a | 1.005 | 0.999 |
| TextVQA | 06 ViSpec 1.0:attention | 5.367 | 0.946 | 277.70 | 3 | 17.1 | n/a | n/a | 1.005 | 0.994 |
| TextVQA | 06 ViSpec 0.5:spatial | 5.932 | 0.938 | 325.93 | 6 | 16.1 | 0.905 | 1.003 | 0.910 | 1.002 |
| TextVQA | 06 ViSpec 0.5:attention | 5.738 | 0.943 | 312.26 | 6 | 16.6 | 0.935 | 1.003 | 0.940 | 0.997 |
| COCO Captions | 02 reference (AR) | 4.269 | 0.220 | 109.96 | 0 | 37.2 | n/a | n/a | n/a | n/a |
| COCO Captions | 06 AR 1.0:spatial | 4.339 | 0.221 | 109.96 | 0 | 37.8 | n/a | n/a | 0.984 | 0.995 |
| COCO Captions | 06 AR 0.5:spatial | 4.356 | 0.221 | 111.71 | 0 | 37.3 | 0.996 | 1.002 | 0.980 | 0.996 |
| COCO Captions | 03 reference (ViSpec) | 1.897 | 0.221 | 109.89 | 0 | 15.5 | n/a | n/a | n/a | n/a |
| COCO Captions | 06 ViSpec 1.0:spatial | 1.896 | 0.221 | 109.89 | 0 | 15.4 | n/a | n/a | 1.000 | 1.000 |
| COCO Captions | 06 ViSpec 1.0:attention | 1.910 | 0.227 | 109.89 | 0 | 15.5 | n/a | n/a | 0.994 | 0.975 |
| COCO Captions | 06 ViSpec 0.5:spatial | 1.909 | 0.222 | 110.19 | 0 | 15.5 | 0.993 | 0.997 | 0.994 | 0.997 |
| COCO Captions | 06 ViSpec 0.5:attention | 1.800 | 0.226 | 105.61 | 0 | 15.2 | 1.061 | 1.002 | 1.054 | 0.977 |

| Benchmark | Arm | Accepted draft tokens per iteration (iterations) | Exact agreement with reference | First mismatch, min / median (count below 5) | Quality | Change vs reference |
| --- | --- | --- | ---: | --- | ---: | ---: |
| TextVQA | 02 reference (AR) | n/a (AR) | n/a | n/a | 0.0 | n/a |
| TextVQA | 06 AR 1.0:spatial | n/a (AR) | 100/100 | none | 0.0 | +0.0 |
| TextVQA | 06 AR 0.5:spatial | n/a (AR) | 0/100 | 1 / 10 (31) | 0.0 | +0.0 |
| TextVQA | 03 reference (ViSpec) | 2.141 (8841) | n/a | n/a | 0.0 | n/a |
| TextVQA | 06 ViSpec 1.0:spatial | 2.141 (8841) | 100/100 | none | 0.0 | +0.0 |
| TextVQA | 06 ViSpec 1.0:attention | 2.141 (8841) | 100/100 | none | 0.0 | +0.0 |
| TextVQA | 06 ViSpec 0.5:spatial | 2.265 (9983) | 0/100 | 1 / 11 (25) | 0.0 | +0.0 |
| TextVQA | 06 ViSpec 0.5:attention | 2.255 (9594) | 6/100 | 1 / 29 (8) | 0.0 | +0.0 |
| COCO Captions | 02 reference (AR) | n/a (AR) | n/a | n/a | 1.87e-05 | n/a |
| COCO Captions | 06 AR 1.0:spatial | n/a (AR) | 100/100 | none | 1.87e-05 | +0.00e+00 |
| COCO Captions | 06 AR 0.5:spatial | n/a (AR) | 1/100 | 2 / 15 (19) | 8.84e-07 | -1.78e-05 |
| COCO Captions | 03 reference (ViSpec) | 2.284 (3346) | n/a | n/a | 1.90e-05 | n/a |
| COCO Captions | 06 ViSpec 1.0:spatial | 2.284 (3346) | 100/100 | none | 1.90e-05 | +0.00e+00 |
| COCO Captions | 06 ViSpec 1.0:attention | 2.284 (3346) | 100/100 | none | 1.90e-05 | +0.00e+00 |
| COCO Captions | 06 ViSpec 0.5:spatial | 2.291 (3348) | 2/100 | 2 / 15 (21) | 1.95e-05 | +5.61e-07 |
| COCO Captions | 06 ViSpec 0.5:attention | 2.352 (3151) | 1/100 | 2 / 19 (11) | 3.49e-06 | -1.55e-05 |

| Benchmark | Arm | Live KV mean / max (MiB) | Decode peak mean / max (GiB) | Prefill peak mean / max (GiB) | Evicted tokens mean / max | Prompt tokens mean |
| --- | --- | --- | --- | --- | --- | --- |
| TextVQA | 02 reference (AR) | 47.3 / 81.9 | 7.945 / 7.970 | 10.16 / 11.57 | n/a | 1062.9 |
| TextVQA | 06 AR 1.0:spatial | 47.3 / 81.9 | 7.946 / 7.970 | 10.16 / 11.57 | 0.0 / 0 | 1062.9 |
| TextVQA | 06 AR 0.5:spatial | 31.3 / 57.9 | 7.929 / 7.948 | 10.16 / 11.57 | 489.2 / 630 | 1062.9 |
| TextVQA | 03 reference (ViSpec) | 47.1 / 81.9 | 7.993 / 8.015 | 10.16 / 11.57 | n/a | 1062.9 |
| TextVQA | 06 ViSpec 1.0:spatial | 47.1 / 81.9 | 7.993 / 8.015 | 10.16 / 11.57 | 0.0 / 0 | 1062.9 |
| TextVQA | 06 ViSpec 1.0:attention | 47.1 / 81.9 | 7.993 / 8.015 | 10.16 / 11.57 | 0.0 / 0 | 1062.9 |
| TextVQA | 06 ViSpec 0.5:spatial | 31.6 / 57.9 | 7.977 / 7.996 | 10.16 / 11.57 | 489.2 / 630 | 1062.9 |
| TextVQA | 06 ViSpec 0.5:attention | 31.1 / 60.4 | 7.976 / 7.993 | 10.16 / 11.57 | 489.2 / 630 | 1062.9 |
| COCO Captions | 02 reference (AR) | 18.1 / 24.1 | 7.878 / 7.893 | 8.16 / 8.54 | n/a | 405.5 |
| COCO Captions | 06 AR 1.0:spatial | 18.1 / 24.1 | 7.878 / 7.893 | 8.16 / 8.54 | 0.0 / 0 | 405.5 |
| COCO Captions | 06 AR 0.5:spatial | 12.0 / 16.6 | 7.872 / 7.884 | 8.16 / 8.54 | 176.2 / 265 | 405.5 |
| COCO Captions | 03 reference (ViSpec) | 18.1 / 24.1 | 7.934 / 7.946 | 8.17 / 8.54 | n/a | 405.5 |
| COCO Captions | 06 ViSpec 1.0:spatial | 18.1 / 24.1 | 7.934 / 7.946 | 8.17 / 8.54 | 0.0 / 0 | 405.5 |
| COCO Captions | 06 ViSpec 1.0:attention | 18.1 / 24.1 | 7.934 / 7.946 | 8.17 / 8.54 | 0.0 / 0 | 405.5 |
| COCO Captions | 06 ViSpec 0.5:spatial | 11.9 / 16.6 | 7.928 / 7.935 | 8.17 / 8.54 | 176.2 / 265 | 405.5 |
| COCO Captions | 06 ViSpec 0.5:attention | 11.8 / 18.9 | 7.929 / 7.938 | 8.17 / 8.54 | 176.2 / 265 | 405.5 |


### Style `short` (quality measurement)

| Benchmark | Arm | Latency (s) | TTFT (s) | Mean output tokens | Truncated at cap | TPOT (ms) | Latency ratio vs same-policy 1.0 | TTFT ratio vs same-policy 1.0 | Latency ratio vs null ref | TTFT ratio vs null ref |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TextVQA | 02 reference (AR) | 1.132 | 0.966 | 5.03 | 0 | 41.8 | n/a | n/a | n/a | n/a |
| TextVQA | 06 AR 1.0:spatial | 1.118 | 0.957 | 5.03 | 0 | 40.7 | n/a | n/a | 1.013 | 1.010 |
| TextVQA | 06 AR 0.5:spatial | 1.232 | 0.958 | 7.93 | 0 | 41.0 | 0.907 | 0.999 | 0.919 | 1.009 |
| TextVQA | 03 reference (ViSpec) | 1.140 | 0.966 | 5.24 | 0 | 66.0 | n/a | n/a | n/a | n/a |
| TextVQA | 06 ViSpec 1.0:spatial | 1.137 | 0.959 | 5.24 | 0 | 67.4 | n/a | n/a | 1.002 | 1.007 |
| TextVQA | 06 ViSpec 1.0:attention | 1.143 | 0.961 | 5.24 | 0 | 69.2 | n/a | n/a | 0.997 | 1.005 |
| TextVQA | 06 ViSpec 0.5:spatial | 1.178 | 0.956 | 8.03 | 0 | 66.3 | 0.965 | 1.003 | 0.967 | 1.010 |
| TextVQA | 06 ViSpec 0.5:attention | 1.145 | 0.967 | 5.31 | 0 | 67.7 | 0.998 | 0.994 | 0.995 | 0.999 |
| COCO Captions | 02 reference (AR) | 0.774 | 0.226 | 14.76 | 0 | 39.8 | n/a | n/a | n/a | n/a |
| COCO Captions | 06 AR 1.0:spatial | 0.761 | 0.225 | 14.76 | 0 | 39.0 | n/a | n/a | 1.017 | 1.004 |
| COCO Captions | 06 AR 0.5:spatial | 0.788 | 0.228 | 14.42 | 0 | 41.8 | 0.966 | 0.987 | 0.982 | 0.991 |
| COCO Captions | 03 reference (ViSpec) | 0.518 | 0.225 | 15.05 | 0 | 21.1 | n/a | n/a | n/a | n/a |
| COCO Captions | 06 ViSpec 1.0:spatial | 0.557 | 0.232 | 15.05 | 0 | 23.5 | n/a | n/a | 0.930 | 0.972 |
| COCO Captions | 06 ViSpec 1.0:attention | 0.529 | 0.230 | 15.05 | 0 | 21.6 | n/a | n/a | 0.979 | 0.979 |
| COCO Captions | 06 ViSpec 0.5:spatial | 0.507 | 0.223 | 14.48 | 0 | 21.4 | 1.098 | 1.041 | 1.021 | 1.011 |
| COCO Captions | 06 ViSpec 0.5:attention | 0.529 | 0.231 | 15.13 | 0 | 21.4 | 1.000 | 0.996 | 0.979 | 0.975 |

| Benchmark | Arm | Accepted draft tokens per iteration (iterations) | Exact agreement with reference | First mismatch, min / median (count below 5) | Quality | Change vs reference |
| --- | --- | --- | ---: | --- | ---: | ---: |
| TextVQA | 02 reference (AR) | n/a (AR) | n/a | n/a | 71.0 | n/a |
| TextVQA | 06 AR 1.0:spatial | n/a (AR) | 100/100 | none | 71.0 | +0.0 |
| TextVQA | 06 AR 0.5:spatial | n/a (AR) | 82/100 | 1 / 2 (16) | 68.4 | -2.6 |
| TextVQA | 03 reference (ViSpec) | 0.690 (310) | n/a | n/a | 70.0 | n/a |
| TextVQA | 06 ViSpec 1.0:spatial | 0.690 (310) | 100/100 | none | 70.0 | +0.0 |
| TextVQA | 06 ViSpec 1.0:attention | 0.690 (310) | 100/100 | none | 70.0 | +0.0 |
| TextVQA | 06 ViSpec 0.5:spatial | 0.983 (405) | 83/100 | 1 / 2 (15) | 68.4 | -1.6 |
| TextVQA | 06 ViSpec 0.5:attention | 0.696 (313) | 96/100 | 1 / 1 (3) | 70.0 | +0.0 |
| COCO Captions | 02 reference (AR) | n/a (AR) | n/a | n/a | 121.40 | n/a |
| COCO Captions | 06 AR 1.0:spatial | n/a (AR) | 100/100 | none | 121.40 | +0.00e+00 |
| COCO Captions | 06 AR 0.5:spatial | n/a (AR) | 39/100 | 1 / 4 (31) | 116.49 | -4.91 |
| COCO Captions | 03 reference (ViSpec) | 1.746 (548) | n/a | n/a | 119.68 | n/a |
| COCO Captions | 06 ViSpec 1.0:spatial | 1.746 (548) | 100/100 | none | 119.68 | +0.00e+00 |
| COCO Captions | 06 ViSpec 1.0:attention | 1.746 (548) | 100/100 | none | 119.68 | +0.00e+00 |
| COCO Captions | 06 ViSpec 0.5:spatial | 1.774 (522) | 38/100 | 1 / 5 (31) | 117.07 | -2.61 |
| COCO Captions | 06 ViSpec 0.5:attention | 1.736 (553) | 43/100 | 1 / 7 (16) | 110.84 | -8.85 |

| Benchmark | Arm | Live KV mean / max (MiB) | Decode peak mean / max (GiB) | Prefill peak mean / max (GiB) | Evicted tokens mean / max | Prompt tokens mean |
| --- | --- | --- | --- | --- | --- | --- |
| TextVQA | 02 reference (AR) | 36.7 / 46.6 | 7.944 / 7.968 | 10.16 / 11.57 | n/a | 1039.9 |
| TextVQA | 06 AR 1.0:spatial | 36.7 / 46.6 | 7.944 / 7.969 | 10.16 / 11.57 | 0.0 / 0 | 1039.9 |
| TextVQA | 06 AR 0.5:spatial | 19.6 / 30.6 | 7.927 / 7.947 | 10.16 / 11.57 | 489.2 / 630 | 1039.9 |
| TextVQA | 03 reference (ViSpec) | 36.7 / 46.7 | 7.982 / 8.005 | 10.16 / 11.57 | n/a | 1039.9 |
| TextVQA | 06 ViSpec 1.0:spatial | 36.7 / 46.7 | 7.982 / 8.005 | 10.16 / 11.57 | 0.0 / 0 | 1039.9 |
| TextVQA | 06 ViSpec 1.0:attention | 36.7 / 46.7 | 7.982 / 8.005 | 10.16 / 11.57 | 0.0 / 0 | 1039.9 |
| TextVQA | 06 ViSpec 0.5:spatial | 19.6 / 31.1 | 7.965 / 7.981 | 10.16 / 11.57 | 489.2 / 630 | 1039.9 |
| TextVQA | 06 ViSpec 0.5:attention | 19.5 / 24.5 | 7.965 / 7.984 | 10.16 / 11.57 | 489.2 / 630 | 1039.9 |
| COCO Captions | 02 reference (AR) | 14.8 / 21.0 | 7.878 / 7.893 | 8.16 / 8.54 | n/a | 406.5 |
| COCO Captions | 06 AR 1.0:spatial | 14.8 / 21.0 | 7.878 / 7.893 | 8.16 / 8.54 | 0.0 / 0 | 406.5 |
| COCO Captions | 06 AR 0.5:spatial | 8.6 / 11.7 | 7.872 / 7.884 | 8.16 / 8.54 | 176.2 / 265 | 406.5 |
| COCO Captions | 03 reference (ViSpec) | 14.8 / 21.1 | 7.930 / 7.948 | 8.17 / 8.54 | n/a | 406.5 |
| COCO Captions | 06 ViSpec 1.0:spatial | 14.8 / 21.1 | 7.930 / 7.948 | 8.17 / 8.54 | 0.0 / 0 | 406.5 |
| COCO Captions | 06 ViSpec 1.0:attention | 14.8 / 21.1 | 7.930 / 7.948 | 8.17 / 8.54 | 0.0 / 0 | 406.5 |
| COCO Captions | 06 ViSpec 0.5:spatial | 8.6 / 11.8 | 7.923 / 7.936 | 8.17 / 8.54 | 176.2 / 265 | 406.5 |
| COCO Captions | 06 ViSpec 0.5:attention | 8.6 / 11.7 | 7.924 / 7.936 | 8.17 / 8.54 | 176.2 / 265 | 406.5 |


Short-style quality changes at 0.5 against the same-policy 1.0 arm:

| Arm | TextVQA VQA accuracy (%) | COCO CIDEr-D x 100 |
| --- | --- | --- |
| AR 0.5:spatial | 71.0 to 68.4 (-2.6) | 121.40 to 116.49 (-4.91) |
| ViSpec 0.5:spatial | 70.0 to 68.4 (-1.6) | 119.68 to 117.07 (-2.61) |
| ViSpec 0.5:attention | 70.0 to 70.0 (+0.0) | 119.68 to 110.84 (-8.85) |

Both spatial arms lose accuracy on both benchmarks, and the attention arm loses nothing on TextVQA but the most on COCO. Neither policy is better on both benchmarks in these single runs. The spatial arms' TextVQA answers are longer on average (8.03 tokens against 5.24 for ViSpec, 7.93 against 5.03 for AR), while the attention arm's are about the same length (5.31). We did not examine why.

### Memory

The 0.5 arms hold 0.650 to 0.671 of the 1.0 arm's live KV in the `long` style and 0.532 to 0.582 in the `short` style (`kv_live` ratio in each `comparison.json`). The ratio is above 0.5 because eviction removes only image tokens and `kv_tokens` still counts the output tokens. For ViSpec 0.5:spatial on long TextVQA the mean `kv_tokens` of 899.61 equals the mean prompt of 1,062.9 tokens, less 489.2 evicted, plus 325.93 output tokens. The evicted count does not depend on method or policy: it is a mean of 489.2 (max 630) tokens per TextVQA question and 176.2 (max 265) per COCO question, against mean prompts of 1,062.9 and 405.5 tokens in `long`, and 1,039.9 and 406.5 in `short`.

The decode-phase peak barely moves: its ratio is 0.9979 to 0.9993 for every 0.5 arm. The drop is about 6 to 17 MiB per arm, which is the same size as the drop in live KV (6.1 to 17.2 MiB), against a decode peak near 7.9 GiB. Live KV is under 1% of that peak, so the peak is dominated by other allocations such as the weights. This is an inference from the sizes. The prefill peak is unchanged to the digit shown, as expected for eviction after the prefill.

For AR the decode-phase peak is not steady-state decoding. It is the transient inside the cache compaction, when the prefill-sized buffer and the final buffer both exist. An AR decode-peak drop therefore reflects the smaller final buffer only. For ViSpec the phase boundary is the first target token, so the draft prefill counts toward the decode-phase peak.

### Predictions against the measurements

| Prediction (written before the runs) | Outcome | Evidence |
| --- | --- | --- |
| Eviction lowers live KV | **Held** | Live KV ratio 0.650 to 0.671 (`long`) and 0.532 to 0.582 (`short`), with a mean of 489.2 (TextVQA) and 176.2 (COCO) tokens evicted per question. |
| Eviction lowers decode peak memory | **Partly held** | The peak falls for every arm, by about 0.2% or less (ratio 0.9979 to 0.9993). The fall equals the live-KV fall, which is under 1% of the peak. For AR the fall reflects the smaller final buffer (see above). |
| Eviction barely moves latency | **Held for TTFT and per-token time, partly held end to end** | TTFT ratios against the same-policy 1.0 arm are 0.987 to 1.041. End-to-end latency ratios are 0.890 to 1.098. In long TextVQA they are 0.890 to 0.935 and come with longer outputs: 317.24 against 283.31 tokens for AR spatial, 325.93 against 277.70 for ViSpec spatial and 312.26 for ViSpec attention, with more answers at the 1,024-token cap (7 against 4 for AR, 6 against 3 for ViSpec). TPOT changes little there (AR 37.3 to 37.8 ms, ViSpec 17.1 to 16.1 and 16.6 ms). The `short` COCO ratio of 1.098 for ViSpec spatial is measured against a 1.0 arm that was itself 7% slower than its null reference (0.930), so its null-reference ratio of 1.021 is the more reliable reading. |
| Eviction can change the output | **Held, strongly** | See the agreement columns: 0 to 6 of 100 (`long` TextVQA), 1 to 2 of 100 (`long` COCO), 82 to 96 of 100 (`short` TextVQA) and 38 to 43 of 100 (`short` COCO). |

The reason latency did not move is an inference, not a measurement. The [04 profile](../04_qwen_decode_profile/README.md) inferred that decode is host-bound, which would make the GPU work that eviction shortens lie off the critical path. These runs are consistent with that inference. They do not test it.

### Notes on the attention policy

The score window is the fixed 6-token chat-template tail after the image, because the question comes before the image in these prompts. `kv_score_window` is 6 in every record of the 0.5 attention arms (400 records), so the attention policy ranks image tokens by how much that template tail attends to them. We infer from this that the ranking cannot depend on the question. The attention-selected and spatial-selected halves overlapped at a Jaccard of 0.32 to 0.34 on 5 `long` TextVQA samples in the Task 5 smoke run, which is about the 1/3 that two unrelated half-subsets give. So the attention policy is not a disguised stride pattern. The selected indices are not saved with the Week 2 runs, so the overlap was not re-measured there. Attention scoring also runs at retention 1.0, so the 1.0 attention arm carries that overhead, which is why the same-policy 1.0 arm is the primary latency reference.

### Week 3 (pending)

- Retention 0.25 and 0.75.
- Combined draft-plus-KV arms.
- fp32 confirmation of the draft-arm divergences (see [05](../05_qwen_draft_pruning/README.md#draft-only-divergence-diagnosis)).

## Limits

- One run per arm and style: repeat variability is unknown. Subsets are 100 samples, not full benchmarks, so agreement counts and quality differences are noisy. The `long` COCO bookkeeping gap sits exactly at the 10.0-point limit.
- The `long` and `short` styles ran on different nodes, host CPUs and cards. Do not compare their latencies.
- `diagnose_divergence` refuses eviction runs (`kv_retention` below 1.0), because its stock full-KV model cannot reproduce them. Divergences from the reference cannot be classified as near-ties, and the large `long` disagreement is attributed to eviction by expectation, not by diagnosis.
- `kv_live_bytes` is a whole-model figure and is arithmetic (`kv_tokens` times the bytes per token), not an allocator measurement. The AR decode-phase peak is the compaction transient and not steady-state decoding.
- The 0.5 arms often produce longer outputs, so end-to-end latency ratios mix per-token speed with output length. Read TPOT and TTFT alongside them.
- The attention-policy overlap comes from a 5-sample smoke run and was not re-measured in these runs.
- Only retention 1.0 and 0.5 were run. Retention 0.25 and 0.75 and combined arms are Week 3 and pending.
