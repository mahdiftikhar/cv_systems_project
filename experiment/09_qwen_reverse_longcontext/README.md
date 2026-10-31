# 09 — Qwen2.5-VL long-context (Re:Verse+ manga windows)

## Purpose and hypothesis

Experiments 02 to 08 use prompts of about 400 to 1,100 tokens. This experiment repeats the Week 2 and
Week 3 arms on `reverse_summary`: the model reads a window of 5 to 6 consecutive manga pages as images
and summarizes the story in order. Prompts are about 7 to 8 thousand tokens, near the 8,192-position
RoPE table of the draft. It asks whether visual KV eviction and draft pruning behave differently when the
image tokens are almost the whole context.

Every arm of every earlier sweep runs here, in one Slurm job, with its own same-job references:

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

These are predictions, not results. The Results section is `pending`.

Status: **pending**. No 09 run has been made.

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
positions. The loader and its pixel limit are in Task 2's evaluator code, not in this runner.

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
empty list exits with status 2 and creates no folder. Each arm creates its own folder; `metadata.txt`
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
   one job must share one Slurm job ID for `sweep_table`.

## How to run

```bash
PROMPT_STYLE=long CLUSTER_PARTITION=preempt \
  SBATCH_EXTRA="--cpus-per-task=4 --mem=22G --time=<per task>" \
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

**Pending.** No run exists yet. When the runs finish, this section reports hardware and run folders, the
identity gate at retention 1.0, per-arm tables, memory, and a table scoring each prediction above
against the measurements with its cutoff.

## Limits

- Single run per arm, 55 windows from one volume. Differences in quality and agreement are noisy, and no
  confidence intervals are computed.
- Prompts of about 7 to 8 thousand tokens sit close to the draft's 8,192-position RoPE table. A window
  that exceeds it is a loader error, not a truncation.
- Pages are images only, with no OCR. ROUGE-1 and BERTScore compare a model summary with Yen Press-style
  narration, so absolute scores are low, and only differences between arms are meaningful.
- The cache is the compact layout in every arm, so memory is compared between arms of this experiment
  and not with the `fixed` layout of 02 and 03.
- The pixel limit is uniform for the benchmark. If it was lowered to fit 6 pages, the value is recorded
  by the evaluator and these results do not compare with 02 to 08 image sizes.
- The runner is verified here only with a stub interpreter. The first real job also tests the evaluator's
  handling of `--num-samples 100` for 55 windows.
