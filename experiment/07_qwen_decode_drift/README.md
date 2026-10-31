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

Status: **pending**. The job has not run yet.

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

**Pending.**

## Limitations

- One focus case. Part B measures one token position of one question, so it cannot estimate how
  often the fork drifts.
- Part A covers only TextVQA in the `long` style (100 questions). COCO and the `short` style are not
  probed.
- One run on one card. Repeat variability is not measured, although greedy bf16 decoding on the
  same card and software stack has reproduced token for token so far (Week 1 and Week 2 02 runs).
- The stock reference uses the same transformers 4.51.3 code that the fork was forked from. A bug
  shared by both would be invisible here.
