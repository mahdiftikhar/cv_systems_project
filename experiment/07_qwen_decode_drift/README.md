# 07 — Qwen2.5-VL cached-decode drift probe

## Question

Week 2's draft-only diagnosis (experiment 05, `long` 0.5:attention) failed by one case. On TextVQA
question 35537 at output index 162, the repository's KV-cache fork emitted "." (token 13) in its
greedy decode. Every Week 1 and Week 2 02 and 03 run emits the same token there. The stock
`transformers.Qwen2_5_VLForConditionalGeneration`, teacher-forced with a full recompute on the
same 162-token prefix, prefers "**:" (token 95518), by 0.625 logits in bf16 and 0.724 in fp32.

This experiment asks whether that gap is generic bf16 drift between cached (incremental) decoding
and a full recompute, which any cached decoder would show, or a deviation specific to our fork.
The difference matters because every Week 3 arm decodes through the fork. A fork-specific
deviation would be inherited by all of them.

Status: **measured** (Slurm job 398844, exit code 0). **Both stop conditions hold**, so by the
plan's rule the 35537 case points to a deviation specific to the fork. The fork and stock HF
greedy decoding agree exactly on 19 of 100 questions. At 35537:162 the fork's logits for 13 and
95518 tie exactly, while stock incremental decoding prefers 95518 by 0.375 logits. The results and
the inferences drawn from them are [below](#results).

## Decision rule

The Week 3 plan (Task 1, Step 6) defines two stop conditions. If either holds, the result points to
a deviation specific to the fork, and the controller stops and reports to the user.

- **(a)** In Part B, stock incremental decoding (ii) prefers 95518 while the fork (i) prefers 13.
- **(b)** In Part A, fork and stock greedy decoding disagree at a first mismatch whose stock
  teacher-forced bf16 margin is above 0.5 logits (the unchanged Week 2 near-tie threshold).

If neither holds, the result is recorded as generic bf16 drift in cached decoding, and the sweep
continues.

## What the probe measures

`python -m vispec.evaluation.drift_probe` has two parts. It reuses the stock-model loader, the
prompt fingerprint and the teacher-forced top-2 helpers of
`vispec/evaluation/diagnose_divergence.py` by import, and the fork's own model loading and
generation call from `vispec/evaluation/evaluate.py`.

**Part A (100 questions).** For each question of the 100-question TextVQA subset, stock HF greedy
`generate` runs in bf16 with sdpa attention and at most 1,024 new tokens. Its tokens are compared
with the fork's saved AR tokens from the reference run. Each question records exact agreement and
the first differing zero-based output index. Each disagreement also records the stock
teacher-forced logits on the shared prefix: one full forward pass with no KV cache, in bf16, and
again in fp32 with TF32 off. From these come the top-2 margin and which decoder's token the
teacher-forced argmax picks.

**Part B (question 35537, index 162).** The probe records the logits of tokens 13 and 95518, the
full top-2 and the top-2 margin under these conditions:

| Condition | How it is computed |
| --- | --- |
| (i) fork incremental AR, bf16 | A re-run of the fork's AR decode through `vispec/model/autoregressive.py` with the reference run's settings. A forward hook on the base model captures the logits of the step that produces index 162. The model code is not edited, and the re-run must reproduce the saved tokens through index 162 |
| (ii) stock incremental `generate`, bf16 | `generate(do_sample=False, max_new_tokens=1024, output_scores=True)`. Only the scores at index 162 are kept |
| (ii-tf) stock incremental, teacher-forced, bf16 | A manual `DynamicCache` loop that feeds `generate`'s inputs (prefill, then one token per step with cache positions) along the fork's 162-token prefix. It replaces (ii) for condition (a) if stock `generate`'s own tokens leave the fork's prefix before index 162, and it otherwise cross-checks (ii) |
| (iii) stock full recompute, bf16 | One forward pass over the prompt plus the 162-token prefix, with no KV cache (the Week 2 diagnosis pass) |
| (iv) stock full recompute, fp32 | The same pass with fp32 weights and TF32 off |

The probe also reports the largest absolute logit difference over the vocabulary between pairs of
conditions. That separates fork against stock incremental drift from incremental against
full-recompute drift.

## Changes from the reference

The probe answers no benchmark questions and computes no quality scores. Nothing in the model code
changes. The fork side reuses the reference's `runtime.json` `generation_args`, re-validated by
`evaluate.parse_args`: AR, greedy, `--kv-layout compact`, no sparsity flags, slim prefill on, KV
capacity 16,384, and draft settings (unused by AR) depth 3, top-k 8, total-token 30 and `num_q` 2.
The probe refuses a reference run that is not AR, or that has any sparsity setting.

The stock model's `generation_config` samples, with `repetition_penalty` 1.05, and stops on two EOS
IDs (151645 and 151643). The stock `generate` call therefore overrides it with `do_sample=False`,
`temperature=None` and `repetition_penalty=1.0`, and it stops on the fork's single EOS, token 151645.
That makes the two decoders' token sequences comparable. The probe records the overrides and the
model's own generation config in `drift.json`.

## Models

| Role | Model |
| --- | --- |
| Target (fork and stock) | `Qwen/Qwen2.5-VL-3B-Instruct`, local copy `models/Qwen2.5-VL-3B-Instruct` |
| Draft | `JLKang/ViSpec-Qwen2.5-VL-3B-Instruct`, local copy `models/ViSpec-Qwen2.5-VL-3B-Instruct`. The fork's `SpecModel` loads it, and AR decoding does not use it |

## Data and reference

| Item | Value |
| --- | --- |
| Benchmark | TextVQA, the same 100-question seed-42 subset as 02 and 03, `long` prompt style |
| Fork answers | Week 2 `long` 02 compact reference of job 393336, [`20261020T084113Z_66f492a5dc65_long_2P5sjB`](../02_qwen_autoregressive/runs/20261020T084113Z_66f492a5dc65_long_2P5sjB/results/textvqa_test/autoregressive_qwen/test.jsonl). Its tokens equal Week 1's on 100/100 |
| Focus case | Question 35537, output index 162, tokens 13 (".") and 95518 ("**:") |

Before any model loads, every prompt is rebuilt and its input-ID fingerprint is checked against the
saved `input_ids_sha256`. A mismatch stops the probe.

## Settings

The runner fixes bf16, temperature 0.0 (greedy), batch size 1, at most 1,024 new tokens, the Week 1
pixel limits (`min_pixels = 256*28*28`, `max_pixels = 1280*28*28`) and sdpa attention. It reads
`PROMPT_STYLE` (required, `long` only), `ANSWERS`, `BENCH_DIR`, `BASE_MODEL_PATH`, `SPEC_DIR` and
`PYTHON_BIN`. It exits 2 before creating a run folder if the reference answers are missing or empty.
The 04 identity check recorded a missing reference as skipped and still passed, and 07 does not
copy that behaviour.

## Cluster prerequisites

1. Follow the [Ampere setup guide](../AMPERE_SETUP.md). The batch job checks for the pinned 48 GB
   `NVIDIA RTX A6000`.
2. Prepare assets once with `bash prepare.sh --suite qwen`. The runner repeats the offline check.
3. The reference run folder is git-ignored. It must exist on project storage, or `ANSWERS` must
   point at a copy of its `test.jsonl`, with the `test.runtime.json` beside it.

## Run command

From the repository root:

```bash
PROMPT_STYLE=long CLUSTER_PARTITION=preempt SBATCH_EXTRA="--cpus-per-task=4 --mem=22G --time=02:00:00" \
  bash experiment/submit_ampere.sh experiment/07_qwen_decode_drift
```

Inside an existing one-GPU allocation, run `PROMPT_STYLE=long bash experiment/07_qwen_decode_drift/run.sh`.

Job 398844 was submitted from a Git worktree. A worktree has no copy of the git-ignored reference
run, so the same command was prefixed with
`ANSWERS=/home/aeternum/Research2/cv_systems_project/experiment/02_qwen_autoregressive/runs/20261020T084113Z_66f492a5dc65_long_2P5sjB/results/textvqa_test/autoregressive_qwen/test.jsonl`.
`metadata.txt` records that path and its SHA-256.

## Run artifacts

Each launch creates `runs/<UTC>_<commit12>_long_XXXXXX/`. It holds the full commit, the
working-tree status and tracked diff, a runner snapshot, a copy of `drift_probe.py`, the package
list, the device report (`device.txt`), the scheduler job ID, timestamps, the answers path and
SHA-256 in `metadata.txt`, combined stdout and stderr (`output.log`) and `exit-code.txt`. The probe
writes:

```text
drift/drift.json   Part A per question, Part B conditions, stop conditions, settings, environment
drift/drift.md     the same results as tables
```

## Metric definitions

- **Exact agreement.** The fork's and the stock decoder's output token IDs are identical, including
  the EOS token.
- **First mismatch index.** The zero-based output position of the first differing token. If one
  sequence is a prefix of the other, it is the shorter length.
- **Margin.** The top-1 minus top-2 next-token logit, in raw logit units, at the last position of a
  pass. Top-1 is `torch.argmax` (the first maximal index, the decoders' tie-break).
- **Preferred of the pair.** Of tokens 13 and 95518, the one with the higher logit. An exact tie
  goes to the lower ID, as `torch.argmax` does.
- **Max abs logit difference.** The largest `|a - b|` over the vocabulary between two conditions'
  float logits at the focus index.

## Results

### Hardware and run

| Item | Value |
| --- | --- |
| Job | Slurm 398844, `preempt`, 00:23:13, exit code 0, restart count 0 |
| Node and card | `c4-2`, Intel Xeon Silver 4210R, one `NVIDIA RTX A6000` (UUID `GPU-121d8d15-73ed-7503-0def-2b7d7340e02a`, driver 615.71.09) |
| Allocation | 1 GPU, 4 CPUs, 22,528 MB |
| Code | Commit `ebd13d6335c3`, no tracked changes (`tracked-changes.patch` is empty) |
| Run folder | [`runs/20261031T131409Z_ebd13d6335c3_long_P6J18R`](runs/20261031T131409Z_ebd13d6335c3_long_P6J18R/drift/drift.md), [device report](runs/20261031T131409Z_ebd13d6335c3_long_P6J18R/device.txt) |

The job ran from a separate Git worktree, with `models/`, `bench_data/`, `.preparation/`, `.cache/`
and `tools/downloaded/` symlinked to the main checkout. `git-status.txt` therefore lists four
untracked symlink entries and no tracked changes. The link points at a run folder that Git
ignores. It is preserved on project storage and resolves only in a checkout that has it.

The reference fork answers were generated on a different card (`c1-8`, job 393336). For question
35537 the fork re-run on this card reproduced all 1,024 saved tokens. The Week 1 (`c2-2`) and
Week 2 (`c1-8`) 02 runs also match token for token, so the fork's greedy tokens have so far not
depended on which A6000 they ran on.

### Stop conditions

| Condition | Result | Evidence |
| --- | --- | --- |
| (a) stock incremental (ii) prefers 95518 while the fork (i) prefers 13 | **Holds** | (i) gives 13 and 95518 the same bf16 logit, 24.375, and `torch.argmax` takes the lower ID, 13. (ii) prefers 95518 by 0.375 |
| (b) a Part A first mismatch with a stock bf16 teacher-forced margin above 0.5 | **Holds** | Only question 35537 at index 162 (margin 0.625), which is the focus case itself. The other 80 mismatches have margins of at most 0.500 (question 39341, at exactly 0.5) |

By the plan's rule, the result points to a deviation specific to the fork, which every arm
inherits. Condition (b) is not independent evidence, because it is triggered only by the same case
that condition (a) examines.

### Part B: question 35537, output index 162

The prefix is the fork's first 162 output tokens. Stock `generate` followed that prefix through
index 161 and first differed at index 162. So (ii) is measured on the same prefix as (i), and
condition (a) uses (ii). Logits are in raw logit units.

| Condition | Logit of 13 (".") | Logit of 95518 ("**:") | 95518 minus 13 | Preferred | Top-2 margin |
| --- | ---: | ---: | ---: | --- | ---: |
| (i) fork incremental AR, bf16 | 24.375 | 24.375 | 0.000 | 13 (exact tie) | 0.000 |
| (ii) stock incremental `generate`, bf16 | 24.125 | 24.500 | 0.375 | 95518 | 0.375 |
| (ii-tf) stock incremental, teacher-forced, bf16 | 24.125 | 24.500 | 0.375 | 95518 | 0.375 |
| (iii) stock full recompute, bf16 | 24.000 | 24.625 | 0.625 | 95518 | 0.625 |
| (iv) stock full recompute, fp32 | 24.023 | 24.747 | 0.724 | 95518 | 0.724 |

In every condition these two tokens are the top 2. The third-ranked token is at least 4.5 logits
below the top one.
The table below gives the largest absolute logit difference over the whole vocabulary between pairs
of conditions:

| Pair | Max abs logit difference |
| --- | ---: |
| (i) fork vs (ii) stock incremental | 0.594 |
| (ii) vs (ii-tf) | 0.000 |
| (ii-tf) stock incremental vs (iii) stock full recompute | 0.172 |
| (i) fork vs (iii) stock full recompute | 0.625 |
| (iii) bf16 vs (iv) fp32 full recompute | 0.430 |
| (i) fork vs (iv) fp32 full recompute | 0.746 |

Four checks back these numbers:

- The fork re-run reproduced all 1,024 saved tokens.
- The hook saw 1,024 base-model forward calls, one per output token, and its captured argmax is
  the token the re-run emitted, 13.
- The `generate` scores equal its raw logits, so no logits processor was active.
- The manual cached loop (ii-tf) reproduces `generate`'s logits exactly.

### Part A: fork AR tokens against stock greedy `generate` (TextVQA `long`, 100 questions)

| Measure | Value |
| --- | --- |
| Exact agreement | **19/100** |
| First mismatches | 81. Index min 1, median 46, max 314. None at index 0 (the prefill token agrees on all 100), and 7 at index 1 (the first cached decode step). No sequence ended early |
| Stock bf16 teacher-forced margin at the first mismatch | 44 exact ties (0.000), 73 at most 0.125, 78 at most 0.25, 1 at exactly 0.500 (39341), 1 above 0.5 (35537, 0.625) |
| bf16 full-recompute argmax | Stock token 54, fork token 26, neither 1 |
| fp32 full-recompute argmax (TF32 off) | Fork token 41, stock token 40. Median fp32 margin 0.078. Above 0.5: 35537 (0.724, stock), 39158 (0.936, stock) and 35830 (0.500, fork) |
| Stop reasons | Fork: 96 EOS, 4 at the 1,024-token cap. Stock: 94 EOS, 6 at the cap |

The per-question table, with both tokens, the top-2 and both margins, is in the run's
[`drift.md`](runs/20261031T131409Z_ebd13d6335c3_long_P6J18R/drift/drift.md). For scale, fork AR and
ViSpec agreed on 22/100 long-style TextVQA questions in Week 1. The fork and stock HF cached
decoding agree less often than that.

### Reading the results

The measurements show that the fork's cached decode does not reproduce stock HF cached decoding.
At the focus step, the fork's logits are up to 0.59 away from stock incremental decoding, while
stock incremental decoding and stock full recompute are at most 0.17 apart. The 35537 case
therefore goes beyond the cached-against-full drift that stock HF itself shows. Stock's own
cached decode prefers 95518.

The following points are **inferences** and were not measured:

- **Rounding, not wrong arithmetic.** The deviation looks like a different bf16 rounding path,
  not a wrong computation. Of the 81 divergences, 80 are within the 0.5-logit near-tie band, and
  44 are exact bf16 ties. None occurs at the prefill token. In fp32 the full recompute sides with
  the fork about as often as with stock (41 against 40), so neither decoder is closer to the fp32
  reference on these points. In bf16 the full recompute sides with stock 54 to 26, which fits stock
  incremental decoding sharing more of its numeric path with stock full recompute.
- **Weak preference at the focus step.** The fork's preference for 13 is the weakest possible: an
  exact bf16 tie broken by token order. Stock incremental decoding sits 0.375 logits toward 95518,
  and fp32 sits 0.724 toward it.
- **Mechanism not isolated.** Reading the code shows that the fork's decode attention keeps the
  stock SDPA call and the same mask decision. The fork differs from stock in three ways:
  - it stores keys and values in a preallocated buffer (`KVCache.cat` returns narrowed views)
    instead of `DynamicCache`;
  - it passes explicit M-RoPE position IDs in the compact layout;
  - it projects only the last prefill position to the vocabulary (slim prefill).

  The stock model's vision tower was recorded as `eager` attention. The fork's vision-tower
  setting was not recorded. Any of these is a candidate, and no experiment here separates them.

## Limitations

- One focus case. Part B measures one token position of one question, so it cannot estimate how
  often the fork drifts. Condition (b) holds only through that same case.
- Part A covers only TextVQA in the `long` style (100 questions). COCO and the `short` style are not
  probed.
- Part A compares the saved fork tokens (job 393336, card `GPU-8d51152c`) with stock tokens from
  this job's card. Only question 35537 was re-run through the fork on this card. Earlier cross-card
  agreement (Week 1 against Week 2, 100/100) suggests the card does not matter, but this was not
  re-checked for the other 99 questions.
- One run. Repeat variability is not measured, although greedy bf16 decoding on the same software
  stack has reproduced token for token so far.
- The stock reference uses the same transformers 4.51.3 code that the fork was forked from. A bug
  shared by both would be invisible here.
- The probe does not identify which difference between the fork and stock moves the logits.
