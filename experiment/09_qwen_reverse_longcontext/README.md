# 09 — Qwen2.5-VL long-context (Re:Verse+ manga windows)

## Purpose and hypothesis

Experiments 02 to 08 use prompts of about 400 to 1,100 tokens. This experiment repeats the Week 2 and
Week 3 arms on `reverse_summary`: the model reads a window of 5 to 6 consecutive manga pages as images
and summarizes the story in order. Prompts are 5,805 to 6,967 tokens (mean 6,496.6, 99% image tokens).
With the output, the cache reaches up to 7,991 positions, near the 8,192-position RoPE table of the draft.
It asks whether visual KV eviction and draft pruning behave differently when the image tokens are almost
the whole context.

Every arm of every earlier sweep runs here. The arms can run in one Slurm job or be split across shorter
jobs, and each job runs its own same-job references:

| Sweep | Arms (default) | What it applies |
| --- | --- | --- |
| `reference` | `ar`, `vispec` | No sparsity flags. The same-job references |
| `kv` | `1.0:spatial 1.0:attention 0.75:spatial 0.5:spatial 0.5:attention 0.25:spatial 0.25:attention 0.5:spatial:autoregressive` | Target KV eviction (as in [06](../06_qwen_kv_eviction/README.md)) |
| `draft` | `0.5:spatial 0.25:spatial` | Draft adaptor pruning (as in [05](../05_qwen_draft_pruning/README.md)) |
| `combined` | `0.5:spatial 0.25:spatial` | Same retention and policy on both sides (as in [08](../08_qwen_combined_sparsity/README.md)) |

That is 14 run folders per job. The sweeps run in the order above.

Predictions, written before any run. Week 2 measured TextVQA `long` at a mean prompt of 1,062.9 tokens,
with 36 KiB of KV per token (36 layers, 2 KV heads, head size 128, bf16), so about 49 MiB live at
retention 1.0. A 7,500-token window prompt is about 7 times that:

1. **Live KV is about 6 times the TextVQA size.** Cutoff: the mean live KV of the `1.0:spatial` ViSpec
   arm is at least 4 times the `long` TextVQA figure of the 06 run (at least about 200 MiB).
2. **Eviction saves hundreds of MiB.** Image tokens are nearly the whole prompt, so the live KV ratio
   follows the retention more closely than at TextVQA scale. Cutoffs, against the `1.0` arm of the same
   policy and method: live KV at 0.5 is at most 0.6 times the 1.0 arm and at least 100 MiB lower; live KV
   at 0.25 is at most 0.4 times the 1.0 arm and at least 150 MiB lower. "Hundreds of MiB" holds only if
   the 0.25 arm saves at least 150 MiB and the sum over the 0.5 and 0.25 arms of one policy exceeds
   250 MiB.
3. **Latency may move slightly more than at TextVQA scale**, because the attention share of a decode step
   grows with context. Cutoff: the latency ratio against the same-policy 1.0 arm, which was 0.989 to
   1.017 at `long` TextVQA 0.5 in Week 2, is at or below 0.97 for some eviction arm at 0.25 (at least 3%
   faster). Otherwise the prediction is not supported, and decode stays host-bound as in
   [04](../04_qwen_decode_profile/README.md).

Quality (ROUGE-1 F1 and BERTScore F1) and token agreement are reported without a prediction, because
Week 2 eviction changed outputs strongly and no earlier run covers this task. The identity gate carries
forward: at retention 1.0 each arm must be token-identical to its same-job reference.

These are predictions, not results. The [Results](#results) section scores them.

Status: **partial (Week 4).** Slurm job 398924 completed 5 of the 14 run folders: both references, kv
`1.0:spatial`, kv `1.0:attention` and kv `0.75:spatial`, each with 55 of 55 windows. The identity gate passed.
ViSpec was 1.74 times faster than AR per window, and prompt processing (TTFT) was 44% of ViSpec's time. The
remaining arms are queued as jobs 402641 to 402644. The Re:Verse+ fp32 offset check and the divergence
diagnosis wait for GPU time. The project-wide results page is [RESULTS.md](../RESULTS.md).

## Changes from the reference

- The benchmark is `reverse_summary`, selected by `QWEN_BENCHMARKS` (see Settings). The runner only
  passes the name through to `vispec.evaluation.evaluate`. The loader, scorer and prompt belong to the
  evaluator (`vispec/evaluation/benchmarks.py`).
- The prompt style is `long` only. `run.sh` exports `PROMPT_STYLE=long`, even if the job carries another
  value for other experiments, and prints a notice when it overrides one.
- The `reference` sweep is new in `experiment/qwen_arm_runner.sh`. A reference arm is `ar` or `vispec`,
  runs `--kv-layout compact` and passes no sparsity flag. Its `metadata.txt` records `sweep=reference`,
  `arm=ar` or `arm=vispec`, `method=autoregressive` or `method=vispec_speculative`, and every sparsity
  setting as `none`. These are the same-job references that `sweep_table` and `compare_metrics --mode
  sparsity` expect: `sweep_table` takes a run with all sparsity settings null as the reference of its
  method. The folder name is `runs/<UTC>_<commit12>_long_ref_<ar|vispec>_XXXXXX/`.
- `QWEN_BENCHMARKS` is new in the runner and in the 02 and 03 `run.sh`. Its default is `textvqa
  coco_caption`, so experiments 02 to 08 run as before. `metadata.txt` has a new `qwen_benchmarks=` line.
- 09 replaces the 02 and 03 references for Re:Verse: it runs its own `ar` and `vispec` references, so do
  not submit 02 and 03 in the same job for this purpose.

## Models

| Role | Model |
| --- | --- |
| Target | `Qwen/Qwen2.5-VL-3B-Instruct`, local copy `models/Qwen2.5-VL-3B-Instruct` |
| Draft | `JLKang/ViSpec-Qwen2.5-VL-3B-Instruct`, local copy `models/ViSpec-Qwen2.5-VL-3B-Instruct` |
| BERTScore encoder | `bert-base-uncased` at revision `86b5e0934494bd15c9632b12f734a8a67f723594`, under `models/` |

## Data, splits and subsets

Volume 1 of Re:Verse+, a local frozen copy reached through the git-ignored link `bench_data/reverse_plus`.
Each chapter of `n` pages is split into `ceil(n/6)` consecutive, non-overlapping windows whose sizes
differ by at most 1, larger windows first. The 11 chapters (308 pages) give 55 windows of 5 to 6 pages.
The reference of a window is its pages' narration texts joined in page order, with the `<D>` and `<T>`
markup removed. Every window uses the same prompt, and the loader raises if a prompt exceeds 8,192
positions. The loader and its pixel limit are in `vispec/evaluation/benchmarks.py`, not in this runner.

**Dataset terms.** The data is for non-commercial research only. It must not be redistributed. Credit
Yen Press, and cite it as a re-aligned derivative of Re:Verse (arXiv:2508.08508). No page image or text is
committed to this repository, and result files keep only model outputs, scores and token counts.

## Settings

Greedy decoding (temperature 0.0, fixed), bf16, batch size 1, at most 1,024 new tokens, depth 3, top-k 8,
total-token 30, `num_q` 2, slim prefill on, `--kv-layout compact` in every arm, KV capacity 16,384 as an
upper bound for the compact layout. Budgets are 0.25, 0.5, 0.75 and 1.0. Policies are `spatial` and
`attention`.

| Variable | Default | Purpose |
| --- | --- | --- |
| `PROMPT_STYLE` | forced to `long` | `submit_ampere.sh` still requires `long` or `short` to be set |
| `QWEN_BENCHMARKS` | `reverse_summary` in 09 (`textvqa coco_caption` elsewhere) | Space-separated benchmark names |
| `REF_ARMS` | `ar vispec` | Reference arms |
| `KV_ARMS` | `1.0:spatial 1.0:attention 0.75:spatial 0.5:spatial 0.5:attention 0.25:spatial 0.25:attention 0.5:spatial:autoregressive` | `<retention>:<policy>[:vispec\|:autoregressive]` |
| `DRAFT_ARMS` | `0.5:spatial 0.25:spatial` | `<retention>:<policy>`, no method suffix |
| `COMBINED_ARMS` | `0.5:spatial 0.25:spatial` | `<retention>:<policy>`, no method suffix |
| `BENCH_DIR`, `BASE_MODEL_PATH`, `SPEC_DIR`, `PYTHON_BIN` | as in 03 | Data and model locations |
| `DEPTH`, `TOP_K`, `TOTAL_TOKEN`, `NUM_Q` | 3, 8, 30, 2 | Draft tree |

`run.sh` validates every arm of all four lists before it creates the first run folder. An invalid or
empty list exits with status 2 and creates no folder. Setting `KV_ARMS`, `DRAFT_ARMS` or `COMBINED_ARMS`
to `none` skips that sweep. This splits the arms across shorter jobs, which survive preemption better.
The references always run, so each job carries its own and every ratio stays inside one job. Each arm creates its own folder; `metadata.txt`
records `sweep`, `arm`, `arm_list`, `benchmarks`, `qwen_benchmarks`, `kv_layout=compact` and the four
sparsity settings. The runner passes `--num-samples 100` to the evaluator as in the other experiments.
The window count is 55, so the evaluator must treat that value as an upper bound.

A failed arm is recorded and the remaining arms still run, but a failed benchmark stops that arm's remaining benchmarks; the runner exits 1 if any arm failed, and `run_sequence.sbatch` still continues to the next experiment.

## Cluster prerequisites

1. Follow the [Ampere setup guide](../AMPERE_SETUP.md). Jobs run on the pinned 48 GB
   `NVIDIA RTX A6000` nodes and fail with status 3 on any other GPU model. That check lives only in
   `run_sequence.sbatch`, so results of record must be submitted through `submit_ampere.sh`.
2. Prepare assets once with `bash prepare.sh --suite qwen` inside a Slurm allocation. The runner
   repeats the offline check after the device capture and stops before any model loads if an asset is
   missing. The Re:Verse+ link `bench_data/reverse_plus`, the BERTScore weights and the `rouge-score` and
   `bert-score` packages must exist for scoring.
3. Commit code before launching, so the run records a clean commit.
4. Size the time limit for 14 arms of 55 windows each. The default `run_sequence.sbatch` limit is 8
   hours, which is probably too short; pass `--time` in `SBATCH_EXTRA`. `CLUSTER_PARTITION=preempt`
   reaches the same pinned nodes. A requeued job reruns the runner into fresh run folders, so all arms of
   one job must share one Slurm job ID for `sweep_table`. A requeue keeps the job ID and reruns `run.sh` from the start, so folders of an interrupted attempt sit next to the new ones, and `sweep_table` stops with "Two completed folders for one arm" if both are completed. Pass `sweep_table` and `compare_metrics` only the final attempt's folders: those with the highest `slurm_restart_count` and `exit_code=0` in `metadata.txt` or `exit-code.txt`.

## How to run

```bash
PROMPT_STYLE=long CLUSTER_PARTITION=preempt \
  SBATCH_EXTRA="--cpus-per-task=4 --mem=22G --time=<per task>" \
  bash experiment/submit_ampere.sh experiment/09_qwen_reverse_longcontext
```

Split the arms into shorter jobs. Each job runs both references, then its own arms:

```bash
PROMPT_STYLE=long CLUSTER_PARTITION=preempt KV_ARMS="0.5:spatial 0.5:attention" DRAFT_ARMS=none \
  COMBINED_ARMS=none SBATCH_EXTRA="--cpus-per-task=4 --mem=22G --time=02:00:00" \
  bash experiment/submit_ampere.sh experiment/09_qwen_reverse_longcontext
```

Narrow the sweep, for example for a smoke run:

```bash
PROMPT_STYLE=long REF_ARMS="ar vispec" KV_ARMS="1.0:spatial 0.5:spatial" DRAFT_ARMS="0.5:spatial" \
  COMBINED_ARMS="0.5:spatial" bash experiment/09_qwen_reverse_longcontext/run.sh
```

Verification without a GPU: `bash -n experiment/09_qwen_reverse_longcontext/run.sh`, and
`CLUSTER_PARTITION=preempt SBATCH_EXTRA=--test-only bash experiment/submit_ampere.sh experiment/09_qwen_reverse_longcontext`
(with `PROMPT_STYLE=long`), which schedules nothing.

Report a job's folders together:

```bash
python -m vispec.evaluation.sweep_table --help
python -m vispec.evaluation.compare_metrics --mode sparsity --original <reference report>... --draft <arm report>... --output-dir <new_dir>
```

Latency and TTFT are compared with the same-policy 1.0 arm of the same job. Speedup against
autoregressive decoding uses the job's `ar` reference. Ratios never span two jobs.
`diagnose_divergence` runs only on draft-only arms, with the 0.5-logit threshold unchanged; it refuses
eviction arms. `offset_check` is the fp32 oracle for eviction.

## Metric definitions

As in the [03 README](../03_qwen_vispec/README.md#metric-definitions) and the [metrics guide](../METRICS.md):
latency and TTFT in seconds per window, TPOT in milliseconds, accepted draft tokens per iteration, exact
token-ID agreement, and memory from `kv_live_bytes` (arithmetic, whole model, repeated per device entry)
and the prefill and decode peaks. Task quality is ROUGE-1 F1 (`rouge-score`, stemming on) and BERTScore F1
(`bert-base-uncased`, no baseline rescaling), each averaged over windows.

## Results

### Week 4 results (partial)

Pending items are marked "Pending GPU run (job N)", or "(no job yet)" for checks that have not been
submitted. Ratio directions:

- `*_speedup_vs_*` is the reference latency divided by the arm latency, so above 1 means the arm was faster.
- `*_arm_over_*` is the arm divided by the reference, so below 1 means the arm used less memory.
- Every timing ratio from these runs is inside job 398924.

#### Hardware and run folders

| Item | Value |
| --- | --- |
| Job | Slurm 398924, `preempt`, attempt `slurm_restart_count=1` (2026-11-04), cancelled afterwards. Other attempts (restart counts 0, 2 and 3) left incomplete or duplicate folders, which are not used |
| Node and card | `c1-8`, Intel Xeon Silver 4116, one `NVIDIA RTX A6000` (UUID `GPU-eb032c66-7f3f-d1b6-96cf-c9db5600d457`) |
| Allocation | 1 GPU, 4 CPUs, 22,528 MB, 5 h limit |
| Code | Commit `f90aa6a8f19f`, clean tree (empty `git-status.txt` and `tracked-changes.patch`) |
| Run location | The job ran from a separate Git worktree of this repository, so the folders are under that worktree's `experiment/09_qwen_reverse_longcontext/runs/` |

| Arm | Run folder |
| --- | --- |
| `ar` reference | `20261104T023022Z_f90aa6a8f19f_long_ref_ar_kvIBpD` |
| `vispec` reference | `20261104T025726Z_f90aa6a8f19f_long_ref_vispec_l2fLoy` |
| kv `1.0:spatial` | `20261104T031344Z_f90aa6a8f19f_long_r1.0_spatial_M8DSqk` |
| kv `1.0:attention` | `20261104T032954Z_f90aa6a8f19f_long_r1.0_attention_nduu4D` |
| kv `0.75:spatial` | `20261104T034607Z_f90aa6a8f19f_long_r0.75_spatial_3B5yP0` |

The run folders are git-ignored and kept on project storage. Every folder has its own `device.txt`.

#### Completion checks

Every folder passes these checks:

- `exit_code=0` and `report_exit_code=0`;
- one GPU UUID across all five;
- 55 of 55 windows, with no missing or unexpected IDs;
- commit `f90aa6a8f19f` with a clean tree;
- the expected metadata: `sweep`, `arm`, `method`, the sparsity settings, `kv_layout=compact`,
  `qwen_benchmarks=reverse_summary`, temperature 0.0, 1,024 new tokens, and the draft tree 3/8/30/2 for ViSpec.

`--num-samples 100` gave all 55 windows, as the evaluator treats it as a maximum.

#### Identity gate

Each 1.0 arm is token-identical to the same-job ViSpec reference on every window. The check compares the
full saved output token IDs with the reference, and it was repeated with
`compare_metrics --mode sparsity`.

| Arm | Identical outputs | Identical prompt fingerprints |
| --- | ---: | ---: |
| kv `1.0:spatial` | 55/55 | 55/55 |
| kv `1.0:attention` | 55/55 | 55/55 |

The gate passes. The two 1.0 arms ran 0.9% and 0.5% faster than the ViSpec reference with identical
tokens. That gives a rough floor for timing differences between arms inside this job.

#### ViSpec against AR

| Measure | AR | ViSpec |
| --- | ---: | ---: |
| Windows | 55 | 55 |
| Mean latency per window (s) | 26.237 | 15.105 |
| Mean TTFT (s) | 6.654 | 6.665 |
| Mean decode time, latency minus TTFT (s) | 19.583 | 8.441 |
| TTFT share of latency (ratio of means) | 25.4% | 44.1% |
| Mean time per output token (ms) | 39.47 | 20.74 |
| Mean output tokens | 497.3 | 424.4 |
| Windows at the 1,024-token cap | 12 | 6 |
| Accepted draft tokens per iteration | n/a | 1.702 |
| Prefill peak, mean (GiB) | 11.43 | 11.43 |
| Decode peak, mean (GiB) | 8.517 | 9.088 |
| Live KV, mean (MiB) | 245.8 | 243.3 |
| ROUGE-1 F1 (x 100) | 22.92 | 22.58 |
| BERTScore F1 (x 100) | 46.58 | 46.63 |

The ratios below divide AR by ViSpec, so above 1 means ViSpec is faster.

- **End to end:** 1.737. This is `speedup_vs_ar` in the sweep table.
- **Decode only, like for like:** 1.895 over the 42 windows where neither method hit the 1,024-token cap.
  Leaving out only the 8 windows where one method hit the cap and the other did not gives 1.981 (47
  windows).
- **Decode only, all 55 windows:** 2.320. This figure is inflated by those 8 windows (see the length gap
  below). The per-window median is 2.02.
- **Per output token:** 1.903.
- **The cap:** with TTFT unchanged, infinitely fast decoding would give at most 26.237 / 6.665 = 3.94.

**Comparison with single images.** On long TextVQA and COCO answers, speedups are:

- 2.13 to 2.32 end to end (Weeks 1 and 3);
- 2.28 to 2.48 decode-only below the cap (all samples: 2.39 to 2.48, Week 3 jobs);
- 2.20 to 2.47 per output token (Week 3 jobs).

So on these multi-page prompts the decode phase gains about 1.9 to 2.0 times, below the single-image figures.
That fits the lower acceptance (1.70 against 2.14 and 2.28 accepted draft tokens per iteration). Prompt
processing then dilutes the gain further, to 1.74 end to end. These runs do not split TTFT between the
vision encoder and the language model's prompt pass.

**The length gap.** AR's summaries average 497.3 tokens and ViSpec's 424.4. The gap comes mostly from a few
windows:

- 49 of the 55 windows differ in length, and 6 are equal. AR is longer in 29 and ViSpec in 20.
- AR hits the 1,024-token cap in 12 windows and ViSpec in 6: 5 both, 7 AR only, 1 ViSpec only.
- The 8 one-sided windows carry 3,816 of the 4,010-token total gap (95%). In the other 47 windows the gap is
  194 tokens, with AR longer in 22, ViSpec in 19, and 6 equal.

The outputs first differ at output index 7 to 443 (median 58), never at the first token. That fits the known
bf16 divergence between the two methods, which matched in every fp32 check so far: 15 checks on long
TextVQA, all with eviction. The Re:Verse+ fp32 check is pending. Why AR reached the cap more often is not
measured.

**Exact agreement** between AR and ViSpec is 0 of 55, although the prompts are identical (55 of 55
fingerprints). The divergence diagnosis that would classify these as near-ties is Pending GPU run (no job
yet).

#### Per arm (sweep table)

| Arm | Latency (s) | TTFT (s) | TPOT (ms) | Output tokens | At cap | Accepted per iteration | Live KV (MiB) | Decode peak (GiB) | Post-compaction peak (GiB) | `kv_min_kept_per_image` | Exact agreement with ViSpec ref | ROUGE-1 | BERTScore |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ar` reference | 26.237 | 6.654 | 39.47 | 497.3 | 12 | n/a | 245.8 | 8.517 | 8.347 | n/a | 0/55 | 22.92 | 46.58 |
| `vispec` reference | 15.105 | 6.665 | 20.74 | 424.4 | 6 | 1.702 | 243.3 | 9.088 | 8.488 | n/a | 55/55 | 22.58 | 46.63 |
| kv `1.0:spatial` | 14.970 | 6.643 | 20.40 | 424.4 | 6 | 1.702 | 243.3 | 9.088 | 8.489 | 1,148 | 55/55 | 22.58 | 46.63 |
| kv `1.0:attention` | 15.030 | 6.674 | 20.48 | 424.4 | 6 | 1.702 | 243.3 | 9.088 | 8.489 | 1,148 | 55/55 | 22.58 | 46.63 |
| kv `0.75:spatial` | 15.190 | 6.656 | 20.36 | 442.7 | 8 | 1.760 | 187.4 | 9.088 | 8.421 | 861 | 0/55 | 22.32 | 46.26 |

The ratios for kv `0.75:spatial` against the same-policy 1.0 arm are these:

- **`latency_speedup_vs_samepolicy_1_0` = 0.985.** The arm was 1.5% slower. Its summaries were 4% longer,
  while time per output token was 0.2% lower.
- **`live_kv_arm_over_samepolicy_1_0` = 0.770,** which is 55.9 MiB less.
- **`post_compaction_peak_arm_over_samepolicy_1_0` = 0.992,** which is 0.067 GiB less.
- **`decode_peak_arm_over_samepolicy_1_0` = 1.000.** The decode-phase peak covers the draft prefill and the
  compaction transient as well as decoding (see the [metrics guide](../METRICS.md)), and it did not move.

**Other measures for kv `0.75:spatial`:**

- **Against the ViSpec reference:** `latency_speedup_vs_vispec_ref` = 0.994, and the ROUGE-1 delta is -0.26
  points. The BERTScore delta is -0.37 points.
- **Against AR:** `speedup_vs_ar` = 1.727.
- **Kept tokens:** each image kept at least 861 of its tokens. The smallest image has 1,148 tokens at
  this pixel limit.
- **Acceptance:** it rose from 1.702 to 1.760 accepted draft tokens per iteration, as eviction did in Week 3.

#### Predictions against the measurements

| Prediction (written before the runs) | Outcome | Evidence |
| --- | --- | --- |
| 1. Live KV is about 6 times the TextVQA size: the `1.0:spatial` ViSpec arm is at least 4 times the long TextVQA 06 figure and at least about 200 MiB | **Held** | 243.3 MiB, 5.2 times the 47.1 MiB of the long TextVQA `1.0:spatial` arm in job 398883 |
| 2. Eviction saves hundreds of MiB (cutoffs at 0.5 and 0.25) | **Pending GPU run (jobs 402641, 402642)** | At 0.75 the live KV ratio is 0.770, which saves 55.9 MiB. It follows the budget closely, because image tokens are 99% of the prompt |
| 3. Latency moves more than at TextVQA scale: some 0.25 eviction arm at least 3% faster than its same-policy 1.0 arm | **Pending GPU run (job 402642)** | At 0.75 the arm was 1.5% slower (`latency_speedup_vs_samepolicy_1_0` = 0.985), and TTFT did not change |

#### Still waiting for GPU runs

| Arms or check | Status |
| --- | --- |
| KV `0.5:spatial` and `0.5:attention` | Pending GPU run (job 402641) |
| KV `0.25:spatial` and `0.25:attention` | Pending GPU run (job 402642) |
| KV `0.5:spatial:autoregressive`, draft `0.5:spatial` and `0.25:spatial` | Pending GPU run (job 402643) |
| Combined `0.5:spatial` and `0.25:spatial` | Pending GPU run (job 402644) |
| fp32 offset check at kv 1.0, kv 0.25 spatial and combined 0.25 on 3 windows | Pending GPU run (no job yet) |
| Divergence diagnosis of AR against ViSpec and of the draft-only arms | Pending GPU run (no job yet) |

Each queued job runs both references first, so its ratios stay inside that job. Its timing will not be
compared with job 398924.

#### Analysis files

These folders are git-ignored and kept on project storage:

- **Sweep table:** `experiment/sweeps/20261109T181008Z_week4_reverse_partial_job398924/`, which has
  `sweep.md`, `sweep.json` and `sweep.csv`. It was built from exactly the five folders above.
- **Comparisons:** `experiment/sweeps/20261109T181008Z_week4_reverse_partial_comparisons_job398924/`, which
  has one `comparison.json` and `comparison.md` per comparison, and `commands.txt`. The comparisons are:
  - ViSpec against AR (`--mode method`);
  - each 1.0 arm against the ViSpec reference;
  - `0.75:spatial` against the ViSpec reference and against `1.0:spatial` (`--mode sparsity`).
- **Trade-off figure:** `reverse_summary_long` in `experiment/sweeps/20261109T181249Z_tradeoffs_weeks3_4/`
  (see [RESULTS.md](../RESULTS.md#5-trade-off-figures)).

#### How the inputs differ from 02 to 08

- **Pixel limit:** pages are capped at `max_pixels` = 1200 x 28 x 28, lowered once for the whole benchmark
  from the 1280 x 28 x 28 of TextVQA and COCO. The cap lets the largest window (6,967 prompt tokens) fit
  the draft's 8,192 positions with 1,024 new tokens.
- **Prompt layout:** the images come before the instruction, unlike the single-image prompts of 02 to 08.
- **Vision encoding:** multi-page features come from one vision-encoder call per page. One call over 6 pages
  needed a 45 GiB attention buffer and ran out of memory. Per-page calls are mathematically the same as one
  batched call but not bit-identical: an fp32 CPU check differed by about 1e-4 absolute. Single-image
  prompts still take the original single call, so 02 to 08 are unchanged.

## Limits

- Single run per arm, 55 windows from one volume. Differences in quality and agreement are noisy, and no
  confidence intervals are computed.
- Prompts of 5,805 to 6,967 tokens plus up to 1,024 new tokens sit close to the draft's 8,192-position RoPE
  table. A window that exceeds it is a loader error, not a truncation.
- Pages are images only, with no OCR. ROUGE-1 and BERTScore compare a model summary with Yen Press-style
  narration, so absolute scores are low, and only differences between arms are meaningful.
- The cache is the compact layout in every arm, so memory is compared between arms of this experiment
  and not with the `fixed` layout of 02 and 03.
- The pixel limit is uniform for the benchmark, 1200 x 28 x 28, so these results do not compare with 02 to
  08 image sizes.
- BERTScore truncates at 512 word pieces, so it compares only the opening of 50 of the 55 references and
  cannot show whether later pages are covered. ROUGE-1 is the primary score.
- Only 5 of the 14 folders exist so far, and only one eviction arm (0.75 spatial) is below retention 1.0.
  Predictions 2 and 3 cannot be scored until the queued jobs run.
