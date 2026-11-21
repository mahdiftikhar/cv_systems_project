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

Status: **complete (Week 4), apart from the fp32 check and the divergence diagnosis.** Five Slurm jobs
(398924 and 402641 to 402644) completed all 14 arms, each with 55 of 55 windows, and the identity gate passed.
ViSpec was 1.74 to 1.87 times faster than AR per window across the five jobs, and prompt processing (TTFT)
was 41% to 44% of ViSpec's time. Eviction saved 110 to 170 MiB of live KV at 0.5 and 0.25 but did not make
decode steps cheaper, and spatial eviction made summaries longer and more repetitive. The fp32 check and the
diagnosis are Pending GPU run (job 412902). The project-wide results page is [RESULTS.md](../RESULTS.md).

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

### Week 4 results

Ratio directions:

- `*_speedup_vs_*` is the reference latency divided by the arm latency, so above 1 means the arm was faster.
- `*_arm_over_*` is the arm divided by the reference, so below 1 means the arm used less memory.
- Every timing ratio is taken inside one Slurm job. Each job ran both references first for this reason.
- The same-policy 1.0 arms ran only in job 398924. In the later jobs, ratios are taken against the same-job
  ViSpec reference instead. The identity gate shows that the 1.0 arms and the reference produce the same
  tokens and the same live KV (243.3 MiB), so the reference stands in for them.

#### Hardware and run folders

| Job | Attempt used | Started (UTC) | Node | GPU | Arms |
| --- | --- | --- | --- | --- | --- |
| 398924 | `slurm_restart_count=1` | 2026-11-04 02:30 | `c1-8` | `GPU-eb032c66-7f3f-d1b6-96cf-c9db5600d457` | references, kv `1.0:spatial`, kv `1.0:attention`, kv `0.75:spatial` |
| 402641 | `slurm_restart_count=0` | 2026-11-10 22:44 | `c1-8` | `GPU-eb032c66-7f3f-d1b6-96cf-c9db5600d457` | references, kv `0.5:spatial`, kv `0.5:attention` |
| 402642 | `slurm_restart_count=2` | 2026-11-11 12:06 | `c2-1` | `GPU-aed0f0cc-9707-a7b7-c1b8-7ee9958993e3` | references, kv `0.25:spatial`, kv `0.25:attention` |
| 402643 | `slurm_restart_count=0` | 2026-11-11 08:53 | `c1-8` | `GPU-eb032c66-7f3f-d1b6-96cf-c9db5600d457` | references, kv `0.5:spatial:autoregressive`, draft `0.5:spatial`, draft `0.25:spatial` |
| 402644 | `slurm_restart_count=0` | 2026-11-11 08:34 | `c2-1` | `GPU-aed0f0cc-9707-a7b7-c1b8-7ee9958993e3` | references, combined `0.5:spatial`, combined `0.25:spatial` |

- **Allocation:** one `NVIDIA RTX A6000`, 4 CPUs and 22,528 MB on the `preempt` partition. Job 398924 had a
  5 h limit and the later jobs a 2 h limit, because the runner was changed to skip sweeps so that the arms
  could run as shorter jobs.
- **Code:** job 398924 ran commit `f90aa6a8f19f` and jobs 402641 to 402644 ran commit `27405e81224d`, each
  with a clean tree. The second commit only adds the sweep-skip option to the runner.
- **Requeues:** job 398924 was requeued by preemption and then cancelled, and only its restart 1 is used.
  Job 402642 was requeued twice. Its restart 0 left one incomplete folder. Restarts 1 and 2 both finished
  all four arms with token-identical outputs (55 of 55 per arm), and restart 2, the final attempt, is used.
- **Run location:** the jobs ran from a separate Git worktree of this repository, so the folders are under
  that worktree's `experiment/09_qwen_reverse_longcontext/runs/`. They are git-ignored, kept on project
  storage, and each has its own `device.txt`.

| Job | Arm | Run folder |
| --- | --- | --- |
| 398924 | `ar` reference | `20261104T023022Z_f90aa6a8f19f_long_ref_ar_kvIBpD` |
| 398924 | `vispec` reference | `20261104T025726Z_f90aa6a8f19f_long_ref_vispec_l2fLoy` |
| 398924 | kv `1.0:spatial` | `20261104T031344Z_f90aa6a8f19f_long_r1.0_spatial_M8DSqk` |
| 398924 | kv `1.0:attention` | `20261104T032954Z_f90aa6a8f19f_long_r1.0_attention_nduu4D` |
| 398924 | kv `0.75:spatial` | `20261104T034607Z_f90aa6a8f19f_long_r0.75_spatial_3B5yP0` |
| 402641 | `ar` reference | `20261110T224455Z_27405e81224d_long_ref_ar_ABGQza` |
| 402641 | `vispec` reference | `20261110T232211Z_27405e81224d_long_ref_vispec_t2QV3S` |
| 402641 | kv `0.5:spatial` | `20261110T234020Z_27405e81224d_long_r0.5_spatial_ZEAYAZ` |
| 402641 | kv `0.5:attention` | `20261110T235958Z_27405e81224d_long_r0.5_attention_XpIckf` |
| 402642 | `ar` reference | `20261111T120623Z_27405e81224d_long_ref_ar_v4MlAG` |
| 402642 | `vispec` reference | `20261111T123632Z_27405e81224d_long_ref_vispec_PDwSyv` |
| 402642 | kv `0.25:spatial` | `20261111T125329Z_27405e81224d_long_r0.25_spatial_VoFxho` |
| 402642 | kv `0.25:attention` | `20261111T131230Z_27405e81224d_long_r0.25_attention_D8qhGT` |
| 402643 | `ar` reference | `20261111T085312Z_27405e81224d_long_ref_ar_mXgNyQ` |
| 402643 | `vispec` reference | `20261111T092054Z_27405e81224d_long_ref_vispec_6QivNj` |
| 402643 | kv `0.5:spatial:autoregressive` | `20261111T093842Z_27405e81224d_long_r0.5_spatial_autoregressive_JvjSEH` |
| 402643 | draft `0.5:spatial` | `20261111T100529Z_27405e81224d_long_r0.5_spatial_TVZasl` |
| 402643 | draft `0.25:spatial` | `20261111T102214Z_27405e81224d_long_r0.25_spatial_PsXBY6` |
| 402644 | `ar` reference | `20261111T083411Z_27405e81224d_long_ref_ar_iEeg2o` |
| 402644 | `vispec` reference | `20261111T090505Z_27405e81224d_long_ref_vispec_AqsFAM` |
| 402644 | combined `0.5:spatial` | `20261111T092136Z_27405e81224d_long_r0.5_spatial_HBeEy5` |
| 402644 | combined `0.25:spatial` | `20261111T093923Z_27405e81224d_long_r0.25_spatial_ZqErUr` |

#### Completion checks

Every one of the 22 folders passes these checks:

- `exit_code=0` and `report_exit_code=0`;
- one GPU UUID inside each job;
- 55 of 55 windows, with no missing or unexpected IDs;
- the job's commit with a clean tree;
- the expected metadata: `sweep`, `arm`, `method`, the sparsity settings, `kv_layout=compact`,
  `qwen_benchmarks=reverse_summary`, temperature 0.0, 1,024 new tokens, and the draft tree 3/8/30/2 for ViSpec.

`--num-samples 100` gave all 55 windows, as the evaluator treats it as a maximum.

#### Identity gate and reproducibility

Each 1.0 arm in job 398924 is token-identical to the same-job ViSpec reference on every window (kv
`1.0:spatial` 55/55, kv `1.0:attention` 55/55, with 55/55 identical prompt fingerprints). The check compares
the full saved output token IDs, and it was repeated with `compare_metrics --mode sparsity`. The gate passes.

The references also reproduce across jobs. The AR reference of each later job is token-identical to the AR
reference of job 398924 on 55 of 55 windows, and the same holds for the ViSpec reference, with identical prompt
fingerprints. Every arm of every job also has 55 of 55 prompt fingerprints equal to its reference.

#### ViSpec against AR, in five jobs

The five jobs repeat the same AR and ViSpec references with identical outputs, so their timing differences
come from the machine alone. Ratios divide AR by ViSpec, so above 1 means ViSpec is faster.

| Job | Node | AR latency (s) | ViSpec latency (s) | TTFT AR / ViSpec (s) | ViSpec TTFT share | End to end | Decode only, below the cap in both (42) | Decode only, no one-sided cap (47) | Decode only, all 55 | AR / ViSpec time per output token (ms) |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 398924 | `c1-8` | 26.237 | 15.105 | 6.654 / 6.665 | 44.1% | 1.737 | 1.895 | 1.981 | 2.320 | 39.47 / 20.74 |
| 402641 | `c1-8` | 31.802 | 17.027 | 6.901 / 6.927 | 40.7% | 1.868 | 2.009 | 2.101 | 2.466 | 50.18 / 24.79 |
| 402642 | `c2-1` | 28.016 | 15.512 | 6.793 / 6.762 | 43.6% | 1.806 | 1.979 | 2.063 | 2.425 | 42.80 / 21.45 |
| 402643 | `c1-8` | 26.593 | 15.211 | 6.636 / 6.680 | 43.9% | 1.748 | 1.925 | 1.997 | 2.339 | 40.24 / 20.80 |
| 402644 | `c2-1` | 27.199 | 15.247 | 6.771 / 6.750 | 44.3% | 1.784 | 1.967 | 2.049 | 2.404 | 41.22 / 20.85 |

- **End to end, ViSpec is 1.74 to 1.87 times faster than AR** (mean 1.79 over the five jobs).
- **Decode only, like for like, it is 1.90 to 2.01 times faster** over the 42 windows where neither method hit
  the 1,024-token cap. Leaving out only the 8 windows where one method hit the cap gives 1.98 to 2.10.
- **The same outputs took different times in different jobs.** AR's mean latency ranged from 26.24 to 31.80 s
  (21%) and ViSpec's from 15.11 to 17.03 s (13%). Job 402641 was slow for both methods, on the same node
  and GPU as jobs 398924 and 402643. A slower host fits the host-bound decode of
  [04](../04_qwen_decode_profile/README.md), but host load was not recorded, so this is an inference. The
  ratio between the methods moved much less than either latency.
- **The cap from prompt processing:** with TTFT unchanged, infinitely fast decoding would give at most 3.94
  (job 398924) to 4.59 (job 402641) times over AR.

Every job gives the same output-level figures:

| Measure | AR | ViSpec |
| --- | ---: | ---: |
| Mean output tokens | 497.3 | 424.4 |
| Windows at the 1,024-token cap | 12 | 6 |
| Accepted draft tokens per iteration | n/a | 1.702 |
| Prefill peak, mean (GiB) | 11.43 | 11.43 |
| Decode peak, mean (GiB) | 8.517 | 9.088 |
| Live KV, mean (MiB) | 245.8 | 243.3 |
| ROUGE-1 F1 (x 100) | 22.92 | 22.58 |
| BERTScore F1 (x 100) | 46.58 | 46.63 |

**Comparison with single images.** On long TextVQA and COCO answers, speedups are 2.13 to 2.32 end to end
(Weeks 1 and 3), 2.28 to 2.48 decode-only below the cap, and 2.20 to 2.47 per output token (Week 3 jobs). So
on these multi-page prompts the decode phase gains about 1.9 to 2.0 times, below the single-image figures.
That fits the lower acceptance (1.70 against 2.14 and 2.28 accepted draft tokens per iteration). Prompt
processing then dilutes the gain to 1.74 to 1.87 end to end. These runs do not split TTFT between the
vision encoder and the language model's prompt pass.

**The length gap.** AR's summaries average 497.3 tokens and ViSpec's 424.4. The gap comes mostly from a few
windows:

- 49 of the 55 windows differ in length, and 6 are equal. AR is longer in 29 and ViSpec in 20.
- AR hits the 1,024-token cap in 12 windows and ViSpec in 6: 5 both, 7 AR only, 1 ViSpec only.
- The 8 one-sided windows carry 3,816 of the 4,010-token total gap (95%). In the other 47 windows the gap is
  194 tokens, with AR longer in 22, ViSpec in 19, and 6 equal.

The outputs first differ at output index 7 to 443 (median 58), never at the first token. Exact agreement
between AR and ViSpec is 0 of 55, although the prompts are identical (55 of 55 fingerprints). The fp32 check
and the divergence diagnosis below test whether these are bf16 near-ties.

#### Per arm (sweep table)

TPOT is the mean time per output token. "Per iteration" is the total decode time divided by the total number
of decode iterations (verification steps for ViSpec, tokens for AR). Agreement and the deltas are against the
same-job reference of the same method.

| Job | Arm | Latency (s) | TPOT (ms) | Per iteration (ms) | Output tokens | At cap | Accepted per iteration | Live KV (MiB) | Post-compaction peak (GiB) | Kept per image, min | Exact agreement | ROUGE-1 | BERTScore |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 398924 | `vispec` reference | 15.105 | 20.74 | 53.6 | 424.4 | 6 | 1.702 | 243.3 | 8.488 | n/a | 55/55 | 22.58 | 46.63 |
| 398924 | kv `1.0:spatial` | 14.970 | 20.40 | 52.9 | 424.4 | 6 | 1.702 | 243.3 | 8.489 | 1,148 | 55/55 | 22.58 | 46.63 |
| 398924 | kv `1.0:attention` | 15.030 | 20.48 | 53.1 | 424.4 | 6 | 1.702 | 243.3 | 8.489 | 1,148 | 55/55 | 22.58 | 46.63 |
| 398924 | kv `0.75:spatial` | 15.190 | 20.36 | 53.1 | 442.7 | 8 | 1.760 | 187.4 | 8.421 | 861 | 0/55 | 22.32 | 46.26 |
| 402641 | `vispec` reference | 17.027 | 24.79 | 64.2 | 424.4 | 6 | 1.702 | 243.3 | 8.488 | n/a | 55/55 | 22.58 | 46.63 |
| 402641 | kv `0.5:spatial` | 18.082 | 23.50 | 65.0 | 511.4 | 12 | 1.984 | 133.3 | 8.361 | 574 | 0/55 | 22.50 | 45.72 |
| 402641 | kv `0.5:attention` | 16.616 | 24.56 | 64.6 | 412.8 | 6 | 1.758 | 129.9 | 8.360 | 574 | 0/55 | 22.07 | 46.65 |
| 402642 | `vispec` reference | 15.512 | 21.45 | 55.6 | 424.4 | 6 | 1.702 | 243.3 | 8.488 | n/a | 55/55 | 22.58 | 46.63 |
| 402642 | kv `0.25:spatial` | 17.868 | 18.41 | 55.7 | 644.1 | 23 | 2.237 | 81.5 | 8.307 | 287 | 0/55 | 20.41 | 44.21 |
| 402642 | kv `0.25:attention` | 14.709 | 20.47 | 54.0 | 405.2 | 6 | 1.758 | 73.1 | 8.305 | 287 | 0/55 | 21.99 | 45.97 |
| 402643 | `ar` reference | 26.593 | 40.24 | 40.1 | 497.3 | 12 | n/a | 245.8 | 8.347 | n/a | 55/55 | 22.92 | 46.58 |
| 402643 | kv `0.5:spatial:autoregressive` | 25.314 | 40.03 | 39.9 | 467.8 | 10 | n/a | 131.8 | 8.212 | 574 | 0/55 | 21.52 | 45.58 |
| 402643 | `vispec` reference | 15.211 | 20.80 | 54.2 | 424.4 | 6 | 1.702 | 243.3 | 8.488 | n/a | 55/55 | 22.58 | 46.63 |
| 402643 | draft `0.5:spatial` | 15.057 | 20.61 | 53.5 | 421.6 | 5 | 1.692 | 243.2 | 8.489 | 574 (draft) | 16/55 | 22.64 | 46.55 |
| 402643 | draft `0.25:spatial` | 14.857 | 20.29 | 53.3 | 421.1 | 6 | 1.740 | 243.2 | 8.488 | 287 (draft) | 18/55 | 22.58 | 46.56 |
| 402644 | `vispec` reference | 15.247 | 20.85 | 54.0 | 424.4 | 6 | 1.702 | 243.3 | 8.488 | n/a | 55/55 | 22.58 | 46.63 |
| 402644 | combined `0.5:spatial` | 16.526 | 19.31 | 54.0 | 547.6 | 16 | 2.033 | 134.6 | 8.361 | 574 | 0/55 | 22.01 | 45.58 |
| 402644 | combined `0.25:spatial` | 17.675 | 18.87 | 56.1 | 618.8 | 21 | 2.199 | 80.6 | 8.306 | 287 | 0/55 | 20.69 | 44.40 |

The decode peak is 9.088 GiB in every ViSpec arm, because it also covers the draft prefill and the
compaction transient (see the [metrics guide](../METRICS.md)). It fell from 8.517 to 8.409 GiB (ratio 0.987)
in the AR eviction arm. Every arm keeps at least 1 image token per page, and text is never evicted.

Ratios against the same-job reference of the same method:

| Arm | Latency speedup | Live KV ratio | Live KV saved (MiB) | Post-compaction peak ratio | Output tokens ratio | Per-iteration time ratio | ROUGE-1 delta | BERTScore delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| kv `0.75:spatial` (398924) | 0.994 | 0.770 | 55.9 | 0.992 | 1.043 | 0.991 | -0.26 | -0.37 |
| kv `0.5:spatial` (402641) | 0.942 | 0.548 | 110.0 | 0.985 | 1.205 | 1.012 | -0.08 | -0.91 |
| kv `0.5:attention` (402641) | 1.025 | 0.534 | 113.4 | 0.985 | 0.973 | 1.006 | -0.51 | +0.02 |
| kv `0.25:spatial` (402642) | 0.868 | 0.335 | 161.8 | 0.979 | 1.518 | 1.002 | -2.17 | -2.42 |
| kv `0.25:attention` (402642) | 1.055 | 0.300 | 170.2 | 0.978 | 0.955 | 0.971 | -0.59 | -0.66 |
| kv `0.5:spatial` on AR (402643) | 1.051 | 0.536 | 114.0 | 0.984 | 0.941 | 0.995 | -1.39 | -1.00 |
| draft `0.5:spatial` (402643) | 1.010 | 1.000 | 0.1 | 1.000 | 0.993 | 0.987 | +0.06 | -0.08 |
| draft `0.25:spatial` (402643) | 1.024 | 1.000 | 0.1 | 1.000 | 0.992 | 0.983 | -0.00 | -0.07 |
| combined `0.5:spatial` (402644) | 0.923 | 0.553 | 108.7 | 0.985 | 1.290 | 1.000 | -0.58 | -1.05 |
| combined `0.25:spatial` (402644) | 0.863 | 0.331 | 162.7 | 0.979 | 1.458 | 1.039 | -1.89 | -2.23 |

#### What visual sparsity did on manga windows

- **Memory.** Live KV follows the budget closely, because image tokens are 99% of the prompt: about 0.55 of
  the reference at 0.5 and 0.30 to 0.34 at 0.25, which saves 109 to 114 MiB and 162 to 170 MiB. Live KV
  counts output tokens too, so the spatial arms, whose summaries are longer, keep slightly more. The
  post-compaction peak fell by 1.5% at 0.5 and by 2.1% to 2.2% at 0.25 (up to 0.183 GiB). The ViSpec decode
  peak did not move.
- **Time per decode step barely moved.** Per iteration, the eviction and combined arms were within -2.9% and
  +3.9% of their reference, and the AR eviction arm within -0.5%. The cache is about 4.1% of the bytes a
  decode step reads at this length (see the [cost model](../RESULTS.md#3-visual-sparsity-why-it-barely-helps-at-batch-1)),
  so even evicting it entirely could save only about that much. The largest per-step saving, 2.9% at kv
  `0.25:attention`, is inside that ceiling.
- **End-to-end latency followed output length.** The arms that ran slower than their reference produced
  longer summaries: kv `0.5:spatial` 20% longer and 6% slower, kv `0.25:spatial` 52% longer and 15% slower,
  and the combined arms 29% and 46% longer and 8% and 16% slower. The attention arms produced 3% to 5%
  shorter summaries and ran 2% to 5% faster.
- **Spatial eviction pushed summaries into repetition.** Repetition was measured from the saved token IDs
  with two measures. Distinct 4-grams is the share of distinct token 4-grams in one output. Tail repetition
  is the share of 4-grams in the last 200 output tokens that already occur earlier in the same output. Both
  are averaged over windows.

  | Arm | Distinct 4-grams | Tail repetition | Windows at the cap |
  | --- | ---: | ---: | ---: |
  | `vispec` reference | 0.860 | 0.163 | 6 |
  | kv `0.75:spatial` | 0.833 | 0.201 | 8 |
  | kv `0.5:spatial` | 0.774 | 0.279 | 12 |
  | kv `0.25:spatial` | 0.630 | 0.452 | 23 |
  | kv `0.5:attention` | 0.849 | 0.177 | 6 |
  | kv `0.25:attention` | 0.879 | 0.141 | 6 |
  | combined `0.5:spatial` | 0.732 | 0.334 | 16 |
  | combined `0.25:spatial` | 0.655 | 0.415 | 21 |
  | draft `0.5:spatial` | 0.854 | 0.182 | 5 |
  | draft `0.25:spatial` | 0.845 | 0.191 | 6 |
  | `ar` reference | 0.784 | 0.275 | 12 |
  | kv `0.5:spatial:autoregressive` | 0.799 | 0.245 | 10 |

  Outputs that reach the cap are loops in every arm (distinct 4-grams 0.28 to 0.37 among capped windows).
  Spatial eviction raised the number of such windows from 6 to 23 at 0.25. The attention policy did not.
- **Acceptance rose most where repetition rose.** It went from 1.702 to 1.984 and 2.237 under spatial
  eviction at 0.5 and 0.25, and to 2.033 and 2.199 in the combined arms, against 1.758 for both attention
  arms. Repeated text is easy for the draft to predict. This is an association across arms, not a tested
  cause.
- **Quality.** At 0.25 the attention policy lost 0.59 ROUGE-1 and 0.66 BERTScore points. The spatial policy
  lost 2.17 and 2.42 points with eviction alone, and 1.89 and 2.23 points combined with draft pruning. That is
  the same direction as the `short` TextVQA and COCO results at 0.25 in Week 3. With one run of 55 windows, differences below
  about half a point are not distinguishable from noise.
- **Draft pruning** changed acceptance by -0.010 and +0.038 and made latency 1.0% and 2.3% lower, as in Week 3, because
  the draft already sees each page as 2 query positions. Its outputs still differ from the reference in 39
  and 37 of 55 windows (at 0.5 and 0.25). The divergence diagnosis below tests whether these are bf16 near-ties.

#### Predictions against the measurements

| Prediction (written before the runs) | Outcome | Evidence |
| --- | --- | --- |
| 1. Live KV is about 6 times the TextVQA size: the `1.0:spatial` ViSpec arm is at least 4 times the long TextVQA 06 figure and at least about 200 MiB | **Held** | 243.3 MiB, 5.2 times the 47.1 MiB of the long TextVQA `1.0:spatial` arm in job 398883 |
| 2. Eviction saves hundreds of MiB. At 0.5, live KV at most 0.6 times the 1.0 arm and at least 100 MiB lower. At 0.25, at most 0.4 times and at least 150 MiB lower. The 0.5 and 0.25 savings of one policy sum to more than 250 MiB | **Held, every cutoff** | Spatial: 0.548 and 110.0 MiB at 0.5, 0.335 and 161.8 MiB at 0.25, sum 271.8 MiB. Attention: 0.534 and 113.4 MiB, 0.300 and 170.2 MiB, sum 283.6 MiB. Against the same-job ViSpec reference, which has the 1.0 arms' tokens and live KV |
| 3. Latency moves more than at TextVQA scale: some 0.25 eviction arm at least 3% faster than its same-policy 1.0 arm | **Held as stated, mostly for another reason** | kv `0.25:attention` took 0.948 of the reference's time (5.2% less). Its decode time fell 9.1%. About two thirds of that came from fewer verification steps (4.5% fewer output tokens and higher acceptance) and one third from 2.9% cheaper steps. kv `0.25:spatial` took 15% longer, with 52% longer outputs |

#### fp32 offset check and divergence diagnosis

| Check | Status |
| --- | --- |
| fp32 offset check: AR equals ViSpec in fp32 on 3 windows at kv `1.0:spatial`, kv `0.25:spatial` and combined `0.25:spatial` | Pending GPU run (job 412902) |
| Divergence diagnosis: AR against ViSpec, and each draft-only arm against the ViSpec reference (job 402643) | Pending GPU run (job 412902) |

**Dataset freeze.** The Re:Verse+ folder was re-frozen on 2026-11-15. Its `FREEZE.json` now
carries the date 2026-11-16 and digest `d495889d…`, in place of `7b71fc7a…` from 2026-10-16 that the runs
above used. This project never writes to that folder. The volume 1 page images and page texts that this
experiment reads carry file dates of 2026-07-13 and 2026-08-30, before the first freeze, so they are
unchanged. The asset record was re-prepared against the new freeze for the checks above, and the old
record is kept as `.preparation/assets_reverse.freeze-2026-10-16.json`. The checks also compare each window's
prompt fingerprint with the runs.

#### Analysis files

These folders are git-ignored and kept on project storage, under the main checkout's `experiment/sweeps/`:

- **Sweep table, all five jobs:** `20261121T212041Z_week4_reverse_full/`, with `sweep.md`, `sweep.json`,
  `sweep.csv`, and `repetition.py` with its output `repetition.tsv`. It was built from exactly the 22
  folders above.
- **Comparisons, job 398924:** `20261109T181008Z_week4_reverse_partial_comparisons_job398924/` (ViSpec
  against AR, the two identity comparisons, and kv 0.75 spatial against the reference and the 1.0 arm).
- **Comparisons, jobs 402641 to 402644:** `20261121T211835Z_week4_reverse_comparisons/`, with ViSpec against
  AR for each job and each draft-only arm against the ViSpec reference of job 402643, and `commands.txt`.
- **fp32 offset check:** `week4_offset_oracle/` in the worktree that ran the jobs.
- **Trade-off figure:** `reverse_summary_long` in `20261121T212041Z_tradeoffs_weeks3_4_final/` (see
  [RESULTS.md](../RESULTS.md#5-trade-off-figures)).

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
