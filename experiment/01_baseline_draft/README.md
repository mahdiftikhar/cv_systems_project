# 01 — ViSpec draft baseline

This experiment tests whether ViSpec reduces the time needed to answer the same
image questions as the [standard baseline](../00_baseline_original/README.md).
A smaller draft model proposes several candidate tokens, and LLaVA checks them
together. The goal is to advance several tokens per check while preserving the
answer, using existing weights without training.

The current greedy run completed **418/418 questions** (job **950803**). Its mean
generation time is about **62% lower** than the paired baseline, but exact output
equivalence and controlled speedup remain unvalidated. The
[paired review](RESULTS_20261017.md) records the evidence and limitations.

| Component                 | Configuration                                                  |
| ------------------------- | -------------------------------------------------------------- |
| Base model                | `llava-hf/llava-v1.6-vicuna-7b-hf`                             |
| Draft checkpoint          | `JLKang/ViSpec-llava-v1.6-vicuna-7b-hf`                        |
| Change from experiment 00 | Existing `gen_spec_answer_*` evaluators with `--use-ours True` |

## How to read the results

The main timing measures answer different questions. **Mean latency** is the
average time to finish generating a response. **Time to first token (TTFT)** is
the initial wait before an answer token is ready. **p95 latency** shows how long
about 95% of questions take to finish, making slow responses visible.

We also record output length and throughput to check how much text was generated
and how efficiently it was produced. Counts use saved token IDs, including special
tokens. Peak allocated and reserved GPU memory describe live PyTorch allocations
and its memory cache after warmup, respectively.

Answer quality and exact output agreement serve different purposes. Quality
compares predictions with reference answers, while agreement checks whether the
two decoding methods produced identical token sequences. A normal finish is
marked by **EOS**, the end-of-sequence token. A capped answer instead reached a
forced limit and may be incomplete.

The [comparison page](../COMPARISON.md) places both experiments side by side.
The [metric guide](../METRICS.md#what-each-metric-means) explains each measurement
and why it matters, and the [plot guide](../plots/README.md) records figure sources.

## Run on the cluster

### Prerequisites

1. Complete the [Newton setup guide](../NEWTON_SETUP.md).
2. Activate Python 3.10 or later with [`requirements.txt`](../../requirements.txt),
   Transformers 4.51.3, and CUDA-compatible PyTorch.
3. Run `bash prepare.sh` once from the repository root to prepare models,
   datasets, and scoring. Prepared local checkpoints and repository caches are
   detected automatically.
4. Commit code before cluster runs.

### Submit a batch job (recommended)

From the repository checkout on Newton:

```bash
bash experiment/submit.sh experiment/01_baseline_draft
```

See [`run.sbatch`](run.sbatch) for Slurm resources and the job entry point.
Jobs use the prepared repository `.venv` directly. No activation is needed.
Set an absolute `PYTHON_BIN` to use a different environment. `packages.txt`
records installed distribution versions without requiring pip or uv at runtime.

### Run inside an existing GPU allocation

Use an allocated NVIDIA GPU job. The following script does not allocate resources:

```bash
bash experiment/01_baseline_draft/run.sh
```

## Settings

Override model/data and draft-tree settings through environment variables.
Temperature is fixed at **0.0 (greedy)**. `TEMPERATURE` does not override it.
Draft-tree defaults follow `exp.sh`.

| Variable          | Default                                              | Purpose                                                           |
| ----------------- | ---------------------------------------------------- | ----------------------------------------------------------------- |
| `BENCH_DIR`       | `<repo>/bench_data`                                  | Benchmark data root                                               |
| `SPEC_DIR`        | Prepared local draft, then the Hugging Face ID above | Draft checkpoint                                                  |
| `BASE_MODEL_PATH` | Prepared local copy detected automatically           | Optional local copy of the same LLaVA checkpoint                  |
| Temperature | `0.0` (fixed) | Greedy decoding. Same as experiment 00. No environment override |
| `PYTHON_BIN`      | `<repo>/.venv/bin/python`                                             | Python executable. Same as experiment 00                          |
| `DEPTH`           | `3`                                                  | Draft-tree depth                                                  |
| `TOP_K`           | `8`                                                  | Draft-tree width                                                  |
| `TOTAL_TOKEN`     | `30`                                                 | Draft-tree token budget                                           |
| `NUM_Q`           | `2`                                                  | Query vectors. Must match the checkpoint's training configuration |

Both experiments load with `torch_dtype="auto"`. This lets the checkpoint
select its numeric precision. The imported pair used FP16 (16-bit floating
point). Record actual precision because it affects memory, speed and numerics.

## Evaluation

The suite covers three kinds of image questions: **TextVQA** requires reading
text in images, **GQA** tests visual relationships and reasoning, and **MM-Vet**
uses broader, open-ended questions. We evaluate 100, 100 and 218 questions,
respectively. These are the measured subsets, not three full leaderboard scores.


Current runs fix temperature at 0.0 and pass `--max-new-token 1024` for
both warmup and measured questions. The cap counts all returned token IDs,
including EOS. Both methods stop at first EOS or exactly 1,024 output tokens.
ViSpec clips the final accepted prefix before committing it, so it cannot
return tokens beyond EOS or the cap. Prompt construction is unchanged.

Raw generated token IDs and the EOS ID are saved in each turn's `measurement`,
before decoding cleanup. Use the [comparison command](../METRICS.md#compare-methods-and-repeats)
to check exact sequences through first EOS and inspect first mismatch positions.
The imported greedy runs contain token mismatches, so exact equivalence is
not validated. This check compares saved IDs and does not require a judge. Matching capped prefixes do not establish full-answer equivalence,
and truncated runs keep controlled speedup pending.

- **Order:** TextVQA → GQA → MM-Vet. Stop on the first failure.
- **Defaults:** batch size 1, one GPU, three warmups per benchmark.
- **Answers:** one per question, seeded with `torch.manual_seed(0)`.
- **Prompts and data:** same as 00.

| Benchmark | Input                                                         | Subset                     |
| --------- | ------------------------------------------------------------- | -------------------------- |
| TextVQA   | `$BENCH_DIR/textvqa/TextVQA_0.5.1_val.json` + `train_images/` | Shuffle seed 42, first 100 |
| GQA       | `$BENCH_DIR/gqa/llava_gqa_testdev_balanced.jsonl` + `images/` | Shuffle seed 42, first 100 |
| MM-Vet    | `whyu/mm-vet`, test split                                     | Entire split               |

## Run artifacts

Each launch creates a unique folder under `runs/`.

| Evidence    | Contents                                                                 |
| ----------- | ------------------------------------------------------------------------ |
| Logs        | Exact commands and combined stdout/stderr                                |
| Code        | Full Git commit, tracked diff, working-tree status, and runner snapshots |
| Environment | Packages, settings, scheduler job ID, timestamps, and exit status        |
| Hardware    | `device.txt`, collected by [capture_device.sh](../capture_device.sh)     |

Answers are saved at:

```text
results/<benchmark>_test/draft_llava/test-temperature-<T>.jsonl
```

Output fields include answers, references, `wall_time`, `ttft`, `new_tokens`,
`idxs`, `acceptance_length`, and `measurement`.

The current run snapshot is versioned for team review, including raw answers and
launch evidence. Other generated runs stay on project storage and are ignored by
default. Untracked source-file contents are not captured in launch diffs. Exit code `0` means completion. A missing exit-code file means an
unclean finish.

## Metrics and results

### Imported greedy run — job 950803

Run `20261017T211906Z_771911bd8665_ei1koG` completed with evaluation/report
exit codes 0 from clean commit `771911bd8665846e9ab8cb698cb22c2929d296b5`.
Hardware: one H100 PCIe (79.2 GiB), FP16, 8 CPUs and 64 GiB RAM on `evc46`.
See the [device report](runs/20261017T211906Z_771911bd8665_ei1koG/device.txt).

| Benchmark | Turns | Mean latency (s/turn) | Mean TTFT (s/turn) | Capped | Quality / scored |
| --- | ---: | ---: | ---: | ---: | --- |
| TextVQA | 100/100 | 3.3647 | 0.2869 | 11 | 0% consensus (100/100 scored) |
| GQA | 100/100 | 0.6400 | 0.2622 | 0 | 0% exact match (100/100 scored) |
| MM-Vet | 218/218 | 1.6507 | 0.2776 | 3 | Pending judge (0/218 scored) |

Pooled mean latency is 1.8189 s/turn, TTFT 0.2761 s/turn. Throughput is
114.2479 returned IDs/s including special tokens. Means weight by turn count.
The [paired review](RESULTS_20261017.md) records exact agreement (367/418),
all first token mismatches, hardware differences, artifact verification, memory,
and limitations. These measurements do not yet validate greedy equivalence.


Automatic reports are saved under each run:

```text
runs/<run>/reports/<timestamp>_<suffix>/summary.md
runs/<run>/reports/<timestamp>_<suffix>/summary.json
```

| Category   | Reported metrics                                                              |
| ---------- | ----------------------------------------------------------------------------- |
| Speed      | Mean, median, and p95 latency and TTFT. Throughput. Time per output token     |
| Outputs    | Actual token counts, stop reasons, truncation, and post-EOS tokens            |
| Resources  | Peak allocated/reserved GPU memory per visible device, measured after warmup  |
| Quality    | TextVQA consensus, GQA exact match, and imported MM-Vet judge scores          |
| Comparison | Answer agreement, latency speedup, quality deltas, and repeated-run variation |

See the [metrics guide](../METRICS.md) for accuracy prerequisites, definitions,
and commands. Missing scores remain pending. Reports also cover partial runs.
Update the [experiment register](../README.md) with verified per-benchmark results
and hardware links.

## Limitations

### Data and reproducibility

- Scores cover the evaluated subset. Accuracy needs the [scoring prerequisites](../METRICS.md#accuracy-prerequisites).
- Model/data revisions are unpinned. Retain cache revision IDs for replication.

### Generation and timing

- CLI `--max-new-token` is forwarded for warmup and measured generation.
- Experiment runners enforce the same 1,024-output-token cap for both methods.
- ViSpec retains its separate verification-iteration safety budget. An iteration
  stop or token-cap stop keeps controlled speedup pending.
- TTFT is captured before drafting: the initial token is sampled directly from LLaVA.
- TTFT is `null` for other speculative methods (EAGLE/Medusa), outside this experiment.
- Draft `decode_time` covers the verification loop after initial drafting. Use total latency/TTFT for matching comparisons.
- Sampling may produce different answers despite matching seeds.
- Timing synchronizes GPUs. TTFT synchronization overhead is included in total latency.

### Hardware measurements

- `device.txt` captures launch conditions. Runtime allocator peaks are measured separately.
- Energy, utilization, and non-PyTorch GPU memory are not measured.
- CUDA-visible devices may differ from host GPUs listed by `nvidia-smi`.
