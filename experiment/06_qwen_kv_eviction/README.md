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

Status: **measured for Weeks 2 and 3.** Week 3 swept the retention budgets at commit `c3a36ac25f95` in jobs 398883 (`long`) and 398923 (`short`), both policies, ViSpec arms only, 100/100 samples per benchmark, all runs exit 0 (the `long` style ran 1.0, 0.5 and 0.25, the `short` style also 0.75). In those jobs ViSpec ran 2.200 (TextVQA) and 2.294 (COCO) times faster than autoregressive decoding in the `long` style, at 2.141 and 2.284 accepted draft tokens per iteration, and 0.981 and 1.485 times in the `short` style (job 398923). Eviction cut live KV (arithmetic) to 0.468 to 0.671 of the 1.0 arm in `long` and 0.298 to 0.791 in `short`, and moved the decode peak by about 0.3%. The cache is too small next to the language-model weights for that to change per-step time: 0.6% of the per-step weight read at 1,000 tokens and 4.2% at 7,000 tokens. TTFT did not change. Accepted tokens per iteration rose under eviction, which we infer accounts for part of the small per-token gain. Long-style outputs diverge from the 1.0 arm in nearly every sample, and short-style quality dropped for the spatial policy at 0.25 (TextVQA 70.0 to 60.8, COCO CIDEr-D 119.68 to 104.98) and not for the attention policy at 0.25 (it lost 8.85 CIDEr-D on COCO at 0.5). The 1.0 identity gate passed on every arm in both weeks. Week 2 also passed its AR-versus-ViSpec bookkeeping check at 0.5 (`long` COCO exactly at the 10.0-point limit), but Week 3 ran no autoregressive eviction arms, so that check was not repeated. In the fp32 offset oracle, AR equalled ViSpec token for token on 5 of 5 long TextVQA questions at kv 0.25 spatial and at kv 0.75 spatial (GPU job 402165), so the cache bookkeeping showed no error there.

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

The Week 3 jobs ran 02, 03 and the arm experiments with `--cpus-per-task=4 --mem=22G --time=04:00:00`. The `long` style ran one job per experiment (06 in job 398883), and the `short` style ran 02, 03, 06, 05 and 08 together in job 398923. The arms were ViSpec only (no `:autoregressive` suffix). The `long` arm list is `1.0:spatial 1.0:attention 0.5:spatial 0.5:attention 0.25:spatial 0.25:attention`, and the `short` list adds `0.75:spatial 0.75:attention` after the 1.0 arms (`arm_list` in each `metadata.txt`). The 0.75 budget was dropped from `long` to save GPU time.

Compare an arm with the same method's reference:

```bash
python -m vispec.evaluation.compare_metrics --mode sparsity --original <reference report>... --draft <06 report>... --output-dir <new_dir>
```

## Results

All figures come from one run per arm per prompt style: job 393336 for `long` and job 393337 for `short`, each ran 02, 03, 06 and 05 in one allocation at commit `66f492a5dc65` with a clean tree. Every arm used greedy decoding (temperature 0.0), bf16, batch size 1, at most 1,024 new tokens, depth 3, top-k 8, total-token 30, `num_q` 2, slim prefill on and the compact KV layout. Every run exited 0 and scored 100/100 samples on both benchmarks. Per-benchmark results are listed first, and no cross-benchmark aggregate is computed. Each arm's reference is the same decoding method at retention 1.0 from the same job.

This section is the Week 2 record (retention 1.0 and 0.5, jobs 393336 and 393337). The Week 3 sweep, with retention 0.25 and 0.75 and different jobs, follows in [Week 3 results](#week-3-results-retention-sweep).

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

A separate check shows that the Week 2 compact references (02 and 03, `long`) equal the Week 1 fixed-layout runs token for token on 100/100 for TextVQA and for COCO Captions. The same holds for the `short` style. The compact layout therefore changed no output. The Week 1 runs are jobs 392660 and 392661 (see the [02](../02_qwen_autoregressive/README.md#results) and [03](../03_qwen_vispec/README.md#results) READMEs).

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
| Eviction barely moves latency | **Held for TTFT and per-token time, partly held end to end** | TTFT ratios against the same-policy 1.0 arm are 0.987 to 1.041. End-to-end latency ratios are 0.890 to 1.098. In long TextVQA they are 0.890 to 0.935 and come with longer outputs: 317.24 against 283.31 tokens for AR spatial, 325.93 against 277.70 for ViSpec spatial and 312.26 for ViSpec attention, with more answers at the 1,024-token cap (7 against 4 for AR, 6 against 3 for ViSpec). TPOT changes little there (AR 37.3 to 37.8 ms, ViSpec 17.1 to 16.1 and 16.6 ms). The `short` COCO ratio of 1.098 for ViSpec spatial is measured against a 1.0 arm that was itself 7% slower than its null reference (0.930), so the two readings differ (1.098 against 1.021) and neither is preferred here. |
| Eviction can change the output | **Held, strongly** | See the agreement columns: 0 to 6 of 100 (`long` TextVQA), 1 to 2 of 100 (`long` COCO), 82 to 96 of 100 (`short` TextVQA) and 38 to 43 of 100 (`short` COCO). |

The reason latency did not move is an inference, not a measurement. The [04 profile](../04_qwen_decode_profile/README.md) inferred that decode is host-bound, which would make the GPU work that eviction shortens lie off the critical path. These runs are consistent with that inference. They do not test it.

### Notes on the attention policy

The score window is the fixed 6-token chat-template tail after the image, because the question comes before the image in these prompts. `kv_score_window` is 6 in every record of the 0.5 attention arms (400 records), so the attention policy ranks image tokens by how much that template tail attends to them. We infer from this that the ranking cannot depend on the question. The attention-selected and spatial-selected halves overlapped at a Jaccard of 0.32 to 0.34 in a smoke run on 5 `long` TextVQA samples, which is about the 1/3 that two unrelated half-subsets give. So the attention policy is not a disguised stride pattern. The selected indices are not saved with the Week 2 runs, so the overlap was not re-measured there. Attention scoring also runs at retention 1.0, so the 1.0 attention arm carries that overhead, which is why the same-policy 1.0 arm is the primary latency reference.

## Week 3 results (retention sweep)

### Summary

Speculative decoding comes first. In these jobs ViSpec completed Qwen2.5-VL-3B answers 2.2 to 2.3 times faster than autoregressive (AR) decoding in the `long` style, and about as fast as AR or 1.5 times faster in the `short` style. The table gives the same-job references, which every arm below is compared with.

| Style | Job | Benchmark | AR latency (s) | ViSpec latency (s) | ViSpec speedup vs AR | ViSpec accepted draft tokens per iteration | Exact agreement, ViSpec vs AR | Quality, AR / ViSpec |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| long | 398883 | TextVQA | 12.353 | 5.615 | 2.200 | 2.141 | 22/100 | near zero (not meaningful) |
| long | 398883 | COCO Captions | 4.544 | 1.981 | 2.294 | 2.284 | 28/100 | near zero (not meaningful) |
| short | 398923 | TextVQA | 1.100 | 1.122 | 0.981 | 0.690 | 96/100 | 71.0 / 70.0 |
| short | 398923 | COCO Captions | 0.784 | 0.528 | 1.485 | 1.746 | 88/100 | 121.40 / 119.68 |

In `long`, the speedup comes with 2.141 (TextVQA) and 2.284 (COCO) accepted draft tokens per iteration. In `short` the answers are 5 and 15 tokens long and prefill dominates: TTFT is 0.940 s of the 1.122 s ViSpec latency on TextVQA, so ViSpec shows no speedup there (0.981). ViSpec and AR outputs differ in token IDs on `long` (22/100 and 28/100 agree exactly) and nearly agree on `short` (96/100 and 88/100). In bf16 the batched tree verification and one-token decoding use different kernel shapes, so near-tied logits can flip; the Week 1 diagnosis found every divergence to be a near-tie ([03 README](../03_qwen_vispec/README.md#results)).

Sparsity comes second, and its effect on latency is small for a reason that can be computed. Eviction, measured here against the same ViSpec method at retention 1.0:

- **Memory.** Live KV (arithmetic, `kv_tokens` times 36,864 bytes) fell to 0.468 to 0.671 of the 1.0 arm on `long` and 0.298 to 0.791 on `short`. The decode peak moved by about 0.3% (ratio 0.997 to 1.000), and the post-compaction peak the same.
- **Latency.** TTFT did not change (ratios, 1.0 arm over arm, 0.986 to 1.009), because eviction runs after prefill. End-to-end latency ratios ran from 0.875 to 1.106, and each of those that leaves the 0.97 to 1.03 band comes with a change in output length (see [Ratios that run against the trend](#ratios-that-run-against-the-trend)).
- **Acceptance.** Accepted draft tokens per iteration rose under eviction, from 2.141 to 2.255 through 2.324 on `long` TextVQA and from 2.284 to as much as 2.379 on COCO. Part of the TextVQA rise comes from more capped answers (see [Acceptance rises under eviction](#acceptance-rises-under-eviction)).
- **Output and quality.** Long-style outputs diverge from the 1.0 arm in nearly every sample (0 to 6 of 100 agree on TextVQA, 0 to 2 on COCO). On `short` the spatial policy lost quality at the lowest budget (TextVQA 70.0 to 60.8, COCO CIDEr-D 119.68 to 104.98), and the attention policy kept it (70.7 and 122.86), within the noise of 100 samples.

### Why the latency effect is small

This is the explanation for small effects, not a new method. On this model the KV cache costs 36,864 bytes per token (36 layers x 2 x 2 KV heads x 128 x 2 bytes), while every decode step reads about 5.75 GiB of language-model weights (measured from the checkpoint files; the 1.25 GiB vision encoder is not read in decode steps). The cache is therefore 0.6% of the per-step weight read at 1,000 tokens and 4.2% at 7,000 tokens, and the two are equal only at about 167,000 tokens. Evicting a share of a 1,000-token cache cannot save more than a fraction of a percent of the bytes a step reads. Decode is also host-bound: the [04 profile](../04_qwen_decode_profile/README.md) found the GPU busy for 36 to 50% of each step (profiled 0.20 to 0.30, estimated unprofiled 0.36 to 0.50), so time saved on the GPU mostly falls off the critical path. Eviction runs once after prefill, so it cannot lower TTFT either. The measured memory and latency effects below are consistent with this arithmetic. The host-bound reading is an inference from the Week 2 profile and is not tested by these runs.

### Setup

Four Slurm jobs, all at commit `c3a36ac25f95` with a clean tree, greedy decoding (temperature 0.0), bf16, batch size 1, at most 1,024 new tokens, depth 3, top-k 8, total-token 30, `num_q` 2, slim prefill on and the compact KV layout. Every run exited 0 and scored 100/100 samples on both benchmarks. This README uses job 398883 (`long`) and job 398923 (`short`). The other `long` jobs belong to [05](../05_qwen_draft_pruning/README.md#week-3-results-retention-sweep) (398884) and [08](../08_qwen_combined_sparsity/README.md#week-3-results-retention-sweep) (398885).

**The budgets differ by style.** The `long` style ran retention 1.0, 0.5 and 0.25. The 0.75 budget was dropped from `long` to save GPU time. The `short` style ran all four (1.0, 0.75, 0.5 and 0.25). Both policies (`spatial`, `attention`) ran at every budget. Only ViSpec arms ran in Week 3: there are no autoregressive eviction arms, so the AR-versus-ViSpec bookkeeping check of Week 2 was not repeated.

Each arm is compared with the same method at retention 1.0 and the same policy from the same job (the primary reading, because the 1.0 arm runs the same mask and score-capture code). The unpruned 03 ViSpec reference of the same job is the second reading. The job's 1.0 arms against that reference give the overhead and noise floor.

### Week 3 hardware

Each job ran on one physical card, so an arm and its references share hardware. The two `long` jobs that carry 05 and 08 ran on a different card, and the `long` KV job (this one) ran on `GPU-8d51152c`, the card that carried the Week 2 `long` job. **Absolute times are compared only within a job, never across jobs.** All ratios below stay inside one job.

| Style | Job / partition (complete attempt) | Node and host CPU | GPU | UUID | Driver | Allocation | Device report (one per job; every run folder has its own, listed below) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| long | 398883 / `preempt`, restart 0, 2026-10-31, 16:56 to 19:08 UTC, 4 h limit | `c1-8`, Intel Xeon Silver 4116 @ 2.10 GHz | NVIDIA RTX A6000 (49,140 MiB total) | `GPU-8d51152c-22f1-372d-aef5-063afee0d488` | 615.71.09 | 1 GPU, 4 CPUs (affinity 0, 18, 24, 42), 22,528 MB | [06 1.0:spatial device.txt](runs/20261031T174037Z_c3a36ac25f95_long_r1.0_spatial_1d932G/device.txt) |
| short | 398923 / `preempt`, restart 0, 2026-11-01, 12:19 to 14:06 UTC, 4 h limit | `c1-8`, Intel Xeon Silver 4116 @ 2.10 GHz | NVIDIA RTX A6000 (49,140 MiB total) | `GPU-eb032c66-7f3f-d1b6-96cf-c9db5600d457` | 615.71.09 | 1 GPU, 4 CPUs (affinity 2, 18, 26, 42), 22,528 MB | [06 1.0:spatial device.txt](runs/20261101T122933Z_c3a36ac25f95_short_r1.0_spatial_Yppgjg/device.txt) |

The links point at generated run folders, which Git ignores and which are preserved on project storage, so they resolve only in a checkout that has them. Device reports are launch snapshots and do not measure runtime peaks. The job command is recorded in each run's `metadata.txt`. A requeued job reruns the runner into new folders, and only the complete attempt of each job is used; interrupted attempts are excluded.

#### Week 3 run folders and device reports

| Style | Job | Run | Arm | Run folder (preserved on project storage) | Device report |
| --- | --- | --- | --- | --- | --- |
| long | 398883 | 02 | reference | [`20261031T165619Z_c3a36ac25f95_long_sQw0be`](../02_qwen_autoregressive/runs/20261031T165619Z_c3a36ac25f95_long_sQw0be/) | [device.txt](../02_qwen_autoregressive/runs/20261031T165619Z_c3a36ac25f95_long_sQw0be/device.txt) |
| long | 398883 | 03 | reference | [`20261031T172630Z_c3a36ac25f95_long_Z1rgED`](../03_qwen_vispec/runs/20261031T172630Z_c3a36ac25f95_long_Z1rgED/) | [device.txt](../03_qwen_vispec/runs/20261031T172630Z_c3a36ac25f95_long_Z1rgED/device.txt) |
| long | 398883 | 06 | 1.0:attention | [`20261031T175442Z_c3a36ac25f95_long_r1.0_attention_7mIeai`](runs/20261031T175442Z_c3a36ac25f95_long_r1.0_attention_7mIeai/) | [device.txt](runs/20261031T175442Z_c3a36ac25f95_long_r1.0_attention_7mIeai/device.txt) |
| long | 398883 | 06 | 1.0:spatial | [`20261031T174037Z_c3a36ac25f95_long_r1.0_spatial_1d932G`](runs/20261031T174037Z_c3a36ac25f95_long_r1.0_spatial_1d932G/) | [device.txt](runs/20261031T174037Z_c3a36ac25f95_long_r1.0_spatial_1d932G/device.txt) |
| long | 398883 | 06 | 0.5:attention | [`20261031T182438Z_c3a36ac25f95_long_r0.5_attention_30dMpp`](runs/20261031T182438Z_c3a36ac25f95_long_r0.5_attention_30dMpp/) | [device.txt](runs/20261031T182438Z_c3a36ac25f95_long_r0.5_attention_30dMpp/device.txt) |
| long | 398883 | 06 | 0.5:spatial | [`20261031T180850Z_c3a36ac25f95_long_r0.5_spatial_sLxadm`](runs/20261031T180850Z_c3a36ac25f95_long_r0.5_spatial_sLxadm/) | [device.txt](runs/20261031T180850Z_c3a36ac25f95_long_r0.5_spatial_sLxadm/device.txt) |
| long | 398883 | 06 | 0.25:attention | [`20261031T185434Z_c3a36ac25f95_long_r0.25_attention_YRKcfi`](runs/20261031T185434Z_c3a36ac25f95_long_r0.25_attention_YRKcfi/) | [device.txt](runs/20261031T185434Z_c3a36ac25f95_long_r0.25_attention_YRKcfi/device.txt) |
| long | 398883 | 06 | 0.25:spatial | [`20261031T183927Z_c3a36ac25f95_long_r0.25_spatial_qsqZuG`](runs/20261031T183927Z_c3a36ac25f95_long_r0.25_spatial_qsqZuG/) | [device.txt](runs/20261031T183927Z_c3a36ac25f95_long_r0.25_spatial_qsqZuG/device.txt) |
| short | 398923 | 02 | reference | [`20261101T121923Z_c3a36ac25f95_short_J5AdE5`](../02_qwen_autoregressive/runs/20261101T121923Z_c3a36ac25f95_short_J5AdE5/) | [device.txt](../02_qwen_autoregressive/runs/20261101T121923Z_c3a36ac25f95_short_J5AdE5/device.txt) |
| short | 398923 | 03 | reference | [`20261101T122530Z_c3a36ac25f95_short_ePf9XE`](../03_qwen_vispec/runs/20261101T122530Z_c3a36ac25f95_short_ePf9XE/) | [device.txt](../03_qwen_vispec/runs/20261101T122530Z_c3a36ac25f95_short_ePf9XE/device.txt) |
| short | 398923 | 06 | 1.0:attention | [`20261101T123335Z_c3a36ac25f95_short_r1.0_attention_a3ZqIS`](runs/20261101T123335Z_c3a36ac25f95_short_r1.0_attention_a3ZqIS/) | [device.txt](runs/20261101T123335Z_c3a36ac25f95_short_r1.0_attention_a3ZqIS/device.txt) |
| short | 398923 | 06 | 1.0:spatial | [`20261101T122933Z_c3a36ac25f95_short_r1.0_spatial_Yppgjg`](runs/20261101T122933Z_c3a36ac25f95_short_r1.0_spatial_Yppgjg/) | [device.txt](runs/20261101T122933Z_c3a36ac25f95_short_r1.0_spatial_Yppgjg/device.txt) |
| short | 398923 | 06 | 0.75:attention | [`20261101T124144Z_c3a36ac25f95_short_r0.75_attention_BiUB4f`](runs/20261101T124144Z_c3a36ac25f95_short_r0.75_attention_BiUB4f/) | [device.txt](runs/20261101T124144Z_c3a36ac25f95_short_r0.75_attention_BiUB4f/device.txt) |
| short | 398923 | 06 | 0.75:spatial | [`20261101T123741Z_c3a36ac25f95_short_r0.75_spatial_6YsPuz`](runs/20261101T123741Z_c3a36ac25f95_short_r0.75_spatial_6YsPuz/) | [device.txt](runs/20261101T123741Z_c3a36ac25f95_short_r0.75_spatial_6YsPuz/device.txt) |
| short | 398923 | 06 | 0.5:attention | [`20261101T124957Z_c3a36ac25f95_short_r0.5_attention_42wdgL`](runs/20261101T124957Z_c3a36ac25f95_short_r0.5_attention_42wdgL/) | [device.txt](runs/20261101T124957Z_c3a36ac25f95_short_r0.5_attention_42wdgL/device.txt) |
| short | 398923 | 06 | 0.5:spatial | [`20261101T124550Z_c3a36ac25f95_short_r0.5_spatial_34ruij`](runs/20261101T124550Z_c3a36ac25f95_short_r0.5_spatial_34ruij/) | [device.txt](runs/20261101T124550Z_c3a36ac25f95_short_r0.5_spatial_34ruij/device.txt) |
| short | 398923 | 06 | 0.25:attention | [`20261101T125806Z_c3a36ac25f95_short_r0.25_attention_n1aKQq`](runs/20261101T125806Z_c3a36ac25f95_short_r0.25_attention_n1aKQq/) | [device.txt](runs/20261101T125806Z_c3a36ac25f95_short_r0.25_attention_n1aKQq/device.txt) |
| short | 398923 | 06 | 0.25:spatial | [`20261101T125402Z_c3a36ac25f95_short_r0.25_spatial_Uqezcc`](runs/20261101T125402Z_c3a36ac25f95_short_r0.25_spatial_Uqezcc/) | [device.txt](runs/20261101T125402Z_c3a36ac25f95_short_r0.25_spatial_Uqezcc/device.txt) |

### Reading the Week 3 tables

Units follow the [03 README](../03_qwen_vispec/README.md#reading-these-results) and the [metrics guide](../METRICS.md). Latency and TTFT are seconds per question, TPOT is milliseconds, and mean output tokens count EOS (in parentheses, the answers that hit the 1,024-token cap). Decode ms per iteration is the mean decode time over the run's iterations (mean decode time per question times 100, divided by the iteration count). It is computed from each run's `summary.json`. Quality is TextVQA consensus accuracy in percent or COCO CIDEr-D x 100 (inverse document frequency from the 100-image subset, so a subset measurement).

- **Direction of ratios.** `Speedup vs AR` and both latency ratios are reference over arm, so a value above 1 means the arm was faster. Memory ratios are arm over reference, so a value above 1 means more memory.
- **Two latency readings, shown side by side.** The column against the same-policy 1.0 arm and the column against the unpruned ViSpec reference are both given and neither is preferred after the fact. A 1.0 arm has no same-policy reference, so its row shows only the reference reading, which measures overhead plus run-to-run noise.
- **Agreement and bf16.** Exact agreement is token-ID equality through the first EOS over 100 samples. Greedy decoding in bf16 flips near-tied logits when kernel shapes differ, so agreement reflects near-ties as well as the method effect, and it is a count out of 100 without a confidence interval. Agreement with AR is the secondary column.
- **Live KV is arithmetic.** `kv_tokens` times 36,864 bytes, not an allocator measurement. It includes the output tokens, so a ratio is larger than the retention budget, and an arm with longer or shorter outputs shifts its ratio. Compare with the output-token column before reading a live-KV ratio.
- **Steady-state decode memory.** The decode peak includes transient allocations at the phase boundary. The post-compaction peak (taken after the cache compaction) is the figure for steady-state decode memory, and its ratio is given beside the decode-peak ratio.
- **Long style.** Long answers are paragraphs, so strict VQA accuracy and CIDEr-D are near zero and not meaningful there. The `long` tables give agreement with the reference and output length instead of quality. Quality numbers come from the `short` style.

### Week 3 identity gate at retention 1.0

The gate passed: all 12 1.0 arms of the Week 3 jobs of 05, 06 and 08 (both policies, both styles) reproduce their same-job ViSpec reference token for token, 100/100 on TextVQA and 100/100 on COCO. The 06 arms are the `Exact agreement with ViSpec reference` entries of 100/100 on the 1.0 rows below. The comparison files are in each arm's `comparisons/20261105T192451Z_identity_vs_03-ref/`.

### Style `long`, Week 3 (speed and agreement measurement)

| Benchmark | Arm | Latency (s) | TTFT (s) | Mean output tokens (capped) | TPOT (ms) | Speedup vs AR | Latency ratio vs same-policy 1.0 | Latency ratio vs unpruned ViSpec ref |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TextVQA | 02 reference (AR) | 12.353 | 0.940 | 283.31 (4) | 40.4 | 1.000 | n/a | n/a |
| TextVQA | 03 reference (ViSpec) | 5.615 | 0.939 | 277.70 (3) | 18.0 | 2.200 | n/a | n/a |
| TextVQA | 06 ViSpec 1.0:attention | 5.588 | 0.943 | 277.70 (3) | 17.9 | 2.211 | 1.000 | 1.005 |
| TextVQA | 06 ViSpec 1.0:spatial | 5.555 | 0.939 | 277.70 (3) | 17.8 | 2.224 | 1.000 | 1.011 |
| TextVQA | 06 ViSpec 0.5:attention | 5.999 | 0.943 | 312.26 (6) | 17.5 | 2.059 | 0.932 | 0.936 |
| TextVQA | 06 ViSpec 0.5:spatial | 6.351 | 0.944 | 325.93 (6) | 17.5 | 1.945 | 0.875 | 0.884 |
| TextVQA | 06 ViSpec 0.25:attention | 5.686 | 0.940 | 300.60 (5) | 17.4 | 2.173 | 0.983 | 0.987 |
| TextVQA | 06 ViSpec 0.25:spatial | 6.193 | 0.942 | 319.22 (5) | 17.2 | 1.995 | 0.897 | 0.907 |
| COCO Captions | 02 reference (AR) | 4.544 | 0.223 | 109.96 (0) | 39.7 | 1.000 | n/a | n/a |
| COCO Captions | 03 reference (ViSpec) | 1.981 | 0.221 | 109.89 (0) | 16.2 | 2.294 | n/a | n/a |
| COCO Captions | 06 ViSpec 1.0:attention | 2.038 | 0.233 | 109.89 (0) | 16.6 | 2.229 | 1.000 | 0.972 |
| COCO Captions | 06 ViSpec 1.0:spatial | 2.040 | 0.225 | 109.89 (0) | 16.8 | 2.227 | 1.000 | 0.971 |
| COCO Captions | 06 ViSpec 0.5:attention | 1.958 | 0.235 | 105.61 (0) | 16.7 | 2.320 | 1.041 | 1.011 |
| COCO Captions | 06 ViSpec 0.5:spatial | 2.046 | 0.224 | 110.19 (0) | 16.8 | 2.221 | 0.997 | 0.968 |
| COCO Captions | 06 ViSpec 0.25:attention | 1.842 | 0.233 | 100.15 (0) | 16.4 | 2.466 | 1.106 | 1.075 |
| COCO Captions | 06 ViSpec 0.25:spatial | 1.892 | 0.223 | 107.38 (0) | 15.8 | 2.402 | 1.078 | 1.047 |

*Exact agreement counts identical token IDs in bf16, where near-tied logits can flip with kernel shape, so it reflects near-ties as well as any method effect.*

| Benchmark | Arm | Accepted draft tokens per iteration (iterations) | Decode ms per iteration | Exact agreement with ViSpec reference | Exact agreement with AR (secondary) |
| --- | --- | --- | ---: | ---: | ---: |
| TextVQA | 02 reference (AR) | n/a (AR) | n/a | 22/100 | n/a |
| TextVQA | 03 reference (ViSpec) | 2.141 (8841) | 52.7 | n/a | 22/100 |
| TextVQA | 06 ViSpec 1.0:attention | 2.141 (8841) | 52.4 | 100/100 | 22/100 |
| TextVQA | 06 ViSpec 1.0:spatial | 2.141 (8841) | 52.1 | 100/100 | 22/100 |
| TextVQA | 06 ViSpec 0.5:attention | 2.255 (9594) | 52.6 | 6/100 | 5/100 |
| TextVQA | 06 ViSpec 0.5:spatial | 2.265 (9983) | 54.0 | 0/100 | 0/100 |
| TextVQA | 06 ViSpec 0.25:attention | 2.324 (9044) | 52.3 | 2/100 | 2/100 |
| TextVQA | 06 ViSpec 0.25:spatial | 2.286 (9715) | 53.9 | 0/100 | 0/100 |
| COCO Captions | 02 reference (AR) | n/a (AR) | n/a | 28/100 | n/a |
| COCO Captions | 03 reference (ViSpec) | 2.284 (3346) | 52.2 | n/a | 28/100 |
| COCO Captions | 06 ViSpec 1.0:attention | 2.284 (3346) | 53.6 | 100/100 | 28/100 |
| COCO Captions | 06 ViSpec 1.0:spatial | 2.284 (3346) | 53.9 | 100/100 | 28/100 |
| COCO Captions | 06 ViSpec 0.5:attention | 2.352 (3151) | 54.3 | 1/100 | 0/100 |
| COCO Captions | 06 ViSpec 0.5:spatial | 2.291 (3348) | 54.0 | 2/100 | 2/100 |
| COCO Captions | 06 ViSpec 0.25:attention | 2.360 (2981) | 53.6 | 1/100 | 1/100 |
| COCO Captions | 06 ViSpec 0.25:spatial | 2.379 (3178) | 52.2 | 0/100 | 0/100 |

*Live KV is arithmetic (`kv_tokens` times 36,864 bytes, output tokens included), so its ratio shifts with output length. The post-compaction peak is the steady-state decode figure.*

| Benchmark | Arm | Smallest kept count per image | Live KV mean (MiB, arithmetic) | Live KV ratio vs same-policy 1.0 | Decode peak mean / max (GiB) | Decode peak ratio vs same-policy 1.0 | Post-compaction peak ratio vs same-policy 1.0 |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| TextVQA | 02 reference (AR) | n/a | 47.3 | n/a | 7.945 / 7.970 | n/a | n/a |
| TextVQA | 03 reference (ViSpec) | n/a | 47.1 | n/a | 7.993 / 8.015 | n/a | n/a |
| TextVQA | 06 ViSpec 1.0:attention | 729 | 47.1 | 1.000 | 7.993 / 8.015 | 1.000 | 1.000 |
| TextVQA | 06 ViSpec 1.0:spatial | 729 | 47.1 | 1.000 | 7.993 / 8.015 | 1.000 | 1.000 |
| TextVQA | 06 ViSpec 0.5:attention | 364 | 31.1 | 0.661 | 7.976 / 7.993 | 0.998 | 0.998 |
| TextVQA | 06 ViSpec 0.5:spatial | 364 | 31.6 | 0.671 | 7.977 / 7.996 | 0.998 | 0.998 |
| TextVQA | 06 ViSpec 0.25:attention | 182 | 22.1 | 0.470 | 7.968 / 7.986 | 0.997 | 0.997 |
| TextVQA | 06 ViSpec 0.25:spatial | 182 | 22.8 | 0.484 | 7.968 / 7.983 | 0.997 | 0.997 |
| COCO Captions | 02 reference (AR) | n/a | 18.1 | n/a | 7.878 / 7.893 | n/a | n/a |
| COCO Captions | 03 reference (ViSpec) | n/a | 18.1 | n/a | 7.934 / 7.946 | n/a | n/a |
| COCO Captions | 06 ViSpec 1.0:attention | 264 | 18.1 | 1.000 | 7.934 / 7.946 | 1.000 | 1.000 |
| COCO Captions | 06 ViSpec 1.0:spatial | 264 | 18.1 | 1.000 | 7.934 / 7.946 | 1.000 | 1.000 |
| COCO Captions | 06 ViSpec 0.5:attention | 132 | 11.8 | 0.650 | 7.929 / 7.938 | 0.999 | 0.999 |
| COCO Captions | 06 ViSpec 0.5:spatial | 132 | 11.9 | 0.659 | 7.928 / 7.935 | 0.999 | 0.999 |
| COCO Captions | 06 ViSpec 0.25:attention | 66 | 8.5 | 0.468 | 7.925 / 7.936 | 0.999 | 0.999 |
| COCO Captions | 06 ViSpec 0.25:spatial | 66 | 8.7 | 0.482 | 7.925 / 7.934 | 0.999 | 0.999 |

Retention 0.75 was not run in `long`. Kept image tokens follow the budget: the smallest image keeps 182 of 729 tokens at 0.25 and 364 at 0.5 on TextVQA, and 66 of 264 and 132 on COCO.

### Style `short`, Week 3 (quality measurement)

| Benchmark | Arm | Latency (s) | TTFT (s) | Mean output tokens (capped) | TPOT (ms) | Speedup vs AR | Latency ratio vs same-policy 1.0 | Latency ratio vs unpruned ViSpec ref |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TextVQA | 02 reference (AR) | 1.100 | 0.934 | 5.03 (0) | 41.6 | 1.000 | n/a | n/a |
| TextVQA | 03 reference (ViSpec) | 1.122 | 0.940 | 5.24 (0) | 69.2 | 0.981 | n/a | n/a |
| TextVQA | 06 ViSpec 1.0:attention | 1.124 | 0.943 | 5.24 (0) | 69.3 | 0.978 | 1.000 | 0.998 |
| TextVQA | 06 ViSpec 1.0:spatial | 1.121 | 0.938 | 5.24 (0) | 69.5 | 0.981 | 1.000 | 1.001 |
| TextVQA | 06 ViSpec 0.75:attention | 1.123 | 0.941 | 5.31 (0) | 70.2 | 0.980 | 1.002 | 0.999 |
| TextVQA | 06 ViSpec 0.75:spatial | 1.127 | 0.938 | 5.69 (0) | 69.0 | 0.976 | 0.995 | 0.996 |
| TextVQA | 06 ViSpec 0.5:attention | 1.126 | 0.941 | 5.31 (0) | 70.1 | 0.977 | 0.999 | 0.996 |
| TextVQA | 06 ViSpec 0.5:spatial | 1.173 | 0.939 | 8.03 (0) | 70.4 | 0.937 | 0.955 | 0.956 |
| TextVQA | 06 ViSpec 0.25:attention | 1.126 | 0.941 | 5.43 (0) | 70.0 | 0.977 | 0.998 | 0.996 |
| TextVQA | 06 ViSpec 0.25:spatial | 1.125 | 0.939 | 5.41 (0) | 69.4 | 0.978 | 0.996 | 0.997 |
| COCO Captions | 02 reference (AR) | 0.784 | 0.222 | 14.76 (0) | 40.9 | 1.000 | n/a | n/a |
| COCO Captions | 03 reference (ViSpec) | 0.528 | 0.222 | 15.05 (0) | 22.2 | 1.485 | n/a | n/a |
| COCO Captions | 06 ViSpec 1.0:attention | 0.544 | 0.232 | 15.05 (0) | 22.6 | 1.441 | 1.000 | 0.971 |
| COCO Captions | 06 ViSpec 1.0:spatial | 0.527 | 0.222 | 15.05 (0) | 22.1 | 1.488 | 1.000 | 1.003 |
| COCO Captions | 06 ViSpec 0.75:attention | 0.538 | 0.231 | 14.83 (0) | 22.5 | 1.458 | 1.011 | 0.982 |
| COCO Captions | 06 ViSpec 0.75:spatial | 0.533 | 0.222 | 15.16 (0) | 22.3 | 1.472 | 0.989 | 0.992 |
| COCO Captions | 06 ViSpec 0.5:attention | 0.545 | 0.231 | 15.13 (0) | 22.6 | 1.439 | 0.998 | 0.969 |
| COCO Captions | 06 ViSpec 0.5:spatial | 0.520 | 0.225 | 14.48 (0) | 22.3 | 1.508 | 1.013 | 1.016 |
| COCO Captions | 06 ViSpec 0.25:attention | 0.526 | 0.231 | 14.51 (0) | 22.1 | 1.490 | 1.034 | 1.004 |
| COCO Captions | 06 ViSpec 0.25:spatial | 0.535 | 0.222 | 15.27 (0) | 22.2 | 1.466 | 0.985 | 0.988 |

*Exact agreement counts identical token IDs in bf16, where near-tied logits can flip with kernel shape, so it reflects near-ties as well as any method effect.*

| Benchmark | Arm | Accepted draft tokens per iteration (iterations) | Decode ms per iteration | Exact agreement with ViSpec reference | Exact agreement with AR (secondary) | Quality | Change vs ViSpec reference |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| TextVQA | 02 reference (AR) | n/a (AR) | n/a | 96/100 | n/a | 71.0 | n/a |
| TextVQA | 03 reference (ViSpec) | 0.690 (310) | 54.6 | n/a | 96/100 | 70.0 | n/a |
| TextVQA | 06 ViSpec 1.0:attention | 0.690 (310) | 54.6 | 100/100 | 96/100 | 70.0 | +0.0 |
| TextVQA | 06 ViSpec 1.0:spatial | 0.690 (310) | 54.9 | 100/100 | 96/100 | 70.0 | +0.0 |
| TextVQA | 06 ViSpec 0.75:attention | 0.718 (309) | 54.8 | 95/100 | 95/100 | 71.0 | +1.0 |
| TextVQA | 06 ViSpec 0.75:spatial | 0.745 (326) | 54.1 | 89/100 | 87/100 | 69.6 | -0.4 |
| TextVQA | 06 ViSpec 0.5:attention | 0.696 (313) | 55.0 | 96/100 | 94/100 | 70.0 | +0.0 |
| TextVQA | 06 ViSpec 0.5:spatial | 0.983 (405) | 54.6 | 83/100 | 83/100 | 68.4 | -1.6 |
| TextVQA | 06 ViSpec 0.25:attention | 0.718 (316) | 54.9 | 93/100 | 92/100 | 70.7 | +0.7 |
| TextVQA | 06 ViSpec 0.25:spatial | 0.717 (315) | 55.2 | 69/100 | 70/100 | 60.8 | -9.2 |
| COCO Captions | 02 reference (AR) | n/a (AR) | n/a | 88/100 | n/a | 121.40 | n/a |
| COCO Captions | 03 reference (ViSpec) | 1.746 (548) | 53.7 | n/a | 88/100 | 119.68 | n/a |
| COCO Captions | 06 ViSpec 1.0:attention | 1.746 (548) | 54.8 | 100/100 | 88/100 | 119.68 | +0.00 |
| COCO Captions | 06 ViSpec 1.0:spatial | 1.746 (548) | 53.6 | 100/100 | 88/100 | 119.68 | +0.00 |
| COCO Captions | 06 ViSpec 0.75:attention | 1.746 (540) | 54.6 | 77/100 | 76/100 | 121.84 | +2.16 |
| COCO Captions | 06 ViSpec 0.75:spatial | 1.727 (556) | 53.8 | 49/100 | 49/100 | 120.75 | +1.07 |
| COCO Captions | 06 ViSpec 0.5:attention | 1.736 (553) | 54.6 | 43/100 | 42/100 | 110.84 | -8.85 |
| COCO Captions | 06 ViSpec 0.5:spatial | 1.774 (522) | 54.3 | 38/100 | 41/100 | 117.07 | -2.61 |
| COCO Captions | 06 ViSpec 0.25:attention | 1.790 (520) | 54.5 | 22/100 | 21/100 | 122.86 | +3.18 |
| COCO Captions | 06 ViSpec 0.25:spatial | 1.756 (554) | 54.3 | 10/100 | 9/100 | 104.98 | -14.70 |

*Live KV is arithmetic (`kv_tokens` times 36,864 bytes, output tokens included), so its ratio shifts with output length. The post-compaction peak is the steady-state decode figure.*

| Benchmark | Arm | Smallest kept count per image | Live KV mean (MiB, arithmetic) | Live KV ratio vs same-policy 1.0 | Decode peak mean / max (GiB) | Decode peak ratio vs same-policy 1.0 | Post-compaction peak ratio vs same-policy 1.0 |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| TextVQA | 02 reference (AR) | n/a | 36.7 | n/a | 7.944 / 7.968 | n/a | n/a |
| TextVQA | 03 reference (ViSpec) | n/a | 36.7 | n/a | 7.982 / 8.005 | n/a | n/a |
| TextVQA | 06 ViSpec 1.0:attention | 729 | 36.7 | 1.000 | 7.982 / 8.005 | 1.000 | 1.000 |
| TextVQA | 06 ViSpec 1.0:spatial | 729 | 36.7 | 1.000 | 7.982 / 8.005 | 1.000 | 1.000 |
| TextVQA | 06 ViSpec 0.75:attention | 547 | 28.1 | 0.766 | 7.973 / 7.991 | 0.999 | 0.999 |
| TextVQA | 06 ViSpec 0.75:spatial | 547 | 28.2 | 0.766 | 7.973 / 7.992 | 0.999 | 0.999 |
| TextVQA | 06 ViSpec 0.5:attention | 364 | 19.5 | 0.532 | 7.965 / 7.984 | 0.998 | 0.998 |
| TextVQA | 06 ViSpec 0.5:spatial | 364 | 19.6 | 0.535 | 7.965 / 7.981 | 0.998 | 0.998 |
| TextVQA | 06 ViSpec 0.25:attention | 182 | 10.9 | 0.298 | 7.957 / 7.974 | 0.997 | 0.997 |
| TextVQA | 06 ViSpec 0.25:spatial | 182 | 10.9 | 0.298 | 7.957 / 7.972 | 0.997 | 0.997 |
| COCO Captions | 02 reference (AR) | n/a | 14.8 | n/a | 7.878 / 7.893 | n/a | n/a |
| COCO Captions | 03 reference (ViSpec) | n/a | 14.8 | n/a | 7.930 / 7.948 | n/a | n/a |
| COCO Captions | 06 ViSpec 1.0:attention | 264 | 14.8 | 1.000 | 7.930 / 7.948 | 1.000 | 1.000 |
| COCO Captions | 06 ViSpec 1.0:spatial | 264 | 14.8 | 1.000 | 7.930 / 7.948 | 1.000 | 1.000 |
| COCO Captions | 06 ViSpec 0.75:attention | 198 | 11.7 | 0.791 | 7.927 / 7.944 | 1.000 | 1.000 |
| COCO Captions | 06 ViSpec 0.75:spatial | 198 | 11.7 | 0.791 | 7.927 / 7.944 | 1.000 | 1.000 |
| COCO Captions | 06 ViSpec 0.5:attention | 132 | 8.6 | 0.582 | 7.924 / 7.936 | 0.999 | 0.999 |
| COCO Captions | 06 ViSpec 0.5:spatial | 132 | 8.6 | 0.581 | 7.923 / 7.936 | 0.999 | 0.999 |
| COCO Captions | 06 ViSpec 0.25:attention | 66 | 5.5 | 0.371 | 7.921 / 7.933 | 0.999 | 0.999 |
| COCO Captions | 06 ViSpec 0.25:spatial | 66 | 5.5 | 0.373 | 7.921 / 7.933 | 0.999 | 0.999 |

What the `short` tables show:

- **Quality at the lowest budget depends on the policy.** At 0.25, TextVQA accuracy is 60.8 for `spatial` and 70.7 for `attention` against 70.0 for the reference. COCO CIDEr-D is 104.98 and 122.86 against 119.68. Agreement shows the same order: 69/100 against 93/100 on TextVQA, and 10/100 against 22/100 on COCO.
- **Quality is not monotone in the budget.** The COCO `attention` arm scores 121.84, 110.84 and 122.86 at 0.75, 0.5 and 0.25. With 100 images, a spread of this size is within the noise of CIDEr-D, so no ordering of the budgets is claimed. A caption that differs from the reference can score as well as the reference, so agreement and CIDEr-D can move apart.
- **Memory follows the budget.** Live KV ratios are 0.766, 0.532 to 0.535 and 0.298 on TextVQA, and 0.791, 0.581 to 0.582 and 0.371 to 0.373 on COCO, at 0.75, 0.5 and 0.25.

### Memory

Live KV in the `long` job fell from 47.1 MiB to 22.1 MiB on TextVQA at 0.25:attention (a saving of 25.0 MiB, arithmetic) and from 18.1 to 8.5 MiB on COCO. The same saving appears in the decode peak, which falls by 16.5 to 25.3 MiB on TextVQA and 5.9 to 9.2 MiB on COCO. Against a decode peak of about 7.99 GiB that is about 0.3%, so the peak ratio is 0.997 to 0.998 (TextVQA) and 0.999 (COCO), and the post-compaction peak ratio is identical. The saving is the whole cache difference and nothing more, which fits the weight-dominated peak described above. Prefill peaks are not affected, because eviction follows prefill.

### Acceptance rises under eviction

Accepted draft tokens per iteration rose under eviction. In the `long` job on TextVQA the 2.141 of the reference and the 1.0 arms became 2.255 (attention) and 2.265 (spatial) at 0.5, and 2.324 (attention) and 2.286 (spatial) at 0.25. On COCO the 2.284 became 2.352 and 2.291 at 0.5, and 2.360 and 2.379 at 0.25. The combined [08 arms](../08_qwen_combined_sparsity/README.md#acceptance-and-the-06-arms) show the same rise.

We infer that part of the small per-token gain comes from this rise and not from cheaper steps. Decode time per iteration stays between 52.1 and 54.3 ms in every ViSpec run of this job, the references and 1.0 arms included, while output tokens per iteration rose from 3.141 to 3.255 through 3.324 on TextVQA. Mean TPOT fell from 17.8 and 17.9 ms (the 1.0 arms) to 17.2 through 17.5 ms in the eviction arms. That is a drop of 2 to 3%, and the extra 4 to 6% of tokens per iteration at an unchanged step time is the size of it.

This is an inference with limits.

- **Different text.** The accepted tokens are counted over different text in each arm (agreement is 0 to 6 of 100).
- **Capped answers inflate the TextVQA rise.** Eviction arms have more answers at the 1,024-token cap (5 or 6 against 3), and those long answers pull the arm means up. Computed from each run's `samples.jsonl`, over the 93 TextVQA questions that are below the cap in the reference and in the arm, the reference mean is 2.082 to 2.105 and the arms are 2.118 (0.5:spatial), 2.122 (0.5:attention), 2.135 (0.25:attention) and 2.192 (0.25:spatial). The rise is then +0.02 to +0.10, smaller than the full-sample +0.11 to +0.18. COCO has no capped answers in this job, so its rise (+0.007 to +0.095) is the full-sample one.
- **One run.** The values are means over iterations from one run per arm, and the references and 1.0 arms alone already span 52.1 to 53.9 ms per iteration.

The cause of the rise is not tested, and the evidence is suggestive rather than conclusive.

### Ratios that run against the trend

Most eviction latency ratios are near 1 or follow the output length. Those that do not:

- **Long TextVQA, all four arms slower (0.875 to 0.983).** Higher acceptance and equal or lower TPOT would suggest faster arms. The arms are slower because their answers are longer: 312.26 and 325.93 tokens at 0.5 and 300.60 and 319.22 at 0.25, against 277.70 for the 1.0 arms, and more answers reach the 1,024-token cap (5 or 6 against 3). The arm with the smallest length increase (0.25:attention, +8.2%) has the smallest slowdown (0.983). It is a longer-answer effect, not a per-token slowdown.
- **Long COCO, faster with shorter answers (1.041, 1.078, 1.106).** The arms with ratios above 1.04 (0.5:attention, 0.25:spatial, 0.25:attention) have shorter answers than the 1.0 arms (105.61, 107.38 and 100.15 tokens against 109.89) and a higher acceptance. Decode time per iteration did not drop for 0.5:attention and 0.25:attention (54.3 and 53.6 ms against 53.6). The same-policy and unpruned-reference readings differ by about 3% here (for example 1.106 against 1.075), because both COCO 1.0 arms ran slower than the job's reference (ratios 0.972 and 0.971, TTFT 0.233 and 0.225 s against 0.221 s). The size of that gap is the noise floor of a single run per arm.
- **Short TextVQA 0.5:spatial (0.955) and its acceptance of 0.983.** The mean answer grew from 5.24 to 8.03 tokens (no answer capped), so the slowdown and the high acceptance both describe different, longer text. TTFT is 0.939 s, as in the other arms.
- **Short COCO 0.25:attention (1.034) and 0.5:spatial (1.013).** Mean answers are shorter than the 1.0 arm's 15.05 tokens (14.51 and 14.48).

### Predictions against the Week 3 measurements

The predictions are those recorded before Week 2 (see [Purpose and hypothesis](#purpose-and-hypothesis)).

| Prediction (written before the runs) | Outcome | Evidence |
| --- | --- | --- |
| Eviction lowers live KV | **Held** | Live KV ratio 0.468 to 0.671 on `long` and 0.298 to 0.791 on `short`, with the smallest image keeping 25% to 75% of its image tokens. |
| Eviction lowers decode peak memory | **Partly held** | The peak falls in every arm, but by about 0.3% (ratio 0.997 to 1.000), the size of the live-KV drop and under 1% of the peak. |
| Eviction barely moves latency | **Held for TTFT and per-iteration time, partly held end to end** | TTFT ratios (1.0 arm over arm) 0.986 to 1.009. Decode time per iteration 52.1 to 54.3 ms in every ViSpec run of the long job. End-to-end ratios 0.875 to 1.106 follow output length and acceptance (see above). |
| Eviction can change the output | **Held, strongly** | `long` agreement 0 to 6 of 100 (TextVQA) and 0 to 2 (COCO). `short` agreement 69 to 96 of 100 (TextVQA) and 10 to 77 (COCO). |

### fp32 offset oracle (exactness check)

`diagnose_divergence` refuses eviction arms, so `offset_check` is the check for the large cache offsets. Greedy ViSpec commits only tokens that the target itself picks, so AR and ViSpec with the same eviction must return the same tokens. In fp32 (matmul and cuDNN TF32 both off, recorded in the file) near-tie flips go away, so a mismatch would point at the cache bookkeeping. GPU job 402165 ran it on the first 5 `long` TextVQA questions at each setting, on node `c4-3` and card `GPU-786d8051-7e60-0cbf-22b0-defdaae84a18` at commit `c3a36ac25f95`; it is a check and not a timed result.

| Setting | Questions | AR equals ViSpec | KV counts match | kv_offset (evicted tokens) min to max | First mismatches |
| --- | ---: | ---: | --- | --- | --- |
| kv 0.25 spatial | 5 | 5/5 | yes | 694 to 944 | none |
| kv 0.75 spatial | 5 | 5/5 | yes | 231 to 314 | none |

The combined 0.25 spatial setting is in [08](../08_qwen_combined_sparsity/README.md#fp32-offset-oracle-exactness-check). The files are `experiment/sweeps/week3_offset_oracle/kv0.25_spatial/` and `kv0.75_spatial/` (`offset_check.json` and `offset_check.md`). Five questions at two budgets is a small sample, and the check says that AR and ViSpec agree under the same eviction. It does not say that eviction preserves answers.

## Limits

- One run per arm and style: repeat variability is unknown. Subsets are 100 samples, not full benchmarks, so agreement counts and quality differences are noisy. The `long` COCO bookkeeping gap sits exactly at the 10.0-point limit.
- The `long` and `short` styles ran on different nodes, host CPUs and cards. Do not compare their latencies.
- `diagnose_divergence` refuses eviction runs (`kv_retention` below 1.0), because its stock full-KV model cannot reproduce them. Divergences from the reference cannot be classified as near-ties, and the large `long` disagreement is attributed to eviction by expectation, not by diagnosis.
- `kv_live_bytes` is a whole-model figure and is arithmetic (`kv_tokens` times the bytes per token), not an allocator measurement. The AR decode-phase peak is the compaction transient and not steady-state decoding.
- The 0.5 arms often produce longer outputs, so end-to-end latency ratios mix per-token speed with output length. Read TPOT and TTFT alongside them.
- The attention-policy overlap comes from a 5-sample smoke run and was not re-measured in these runs.
- Week 3 has one run per arm and style as well. The `long` style did not run the 0.75 budget (dropped to save GPU time), and its three jobs ran on two cards at different times, so no absolute time is compared across jobs. Week 3 ran ViSpec arms only, so the AR-versus-ViSpec bookkeeping check was not repeated.
- The TextVQA acceptance rise is partly an effect of more answers at the token cap (see the Week 3 section). The inference that the rise accounts for part of the per-token gain is not tested.
- The fp32 offset oracle covers 5 `long` TextVQA questions at kv 0.25 and 0.75 spatial only, and tests AR-versus-ViSpec equality under eviction, not answer quality.
