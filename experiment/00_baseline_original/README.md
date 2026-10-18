# 00 — Original autoregressive baseline

This experiment measures how long LLaVA takes to answer image questions using
standard autoregressive decoding, which generates one token at a time. It uses
existing weights without training and provides the reference for the
[ViSpec comparison](../01_baseline_draft/README.md).

The current greedy run completed **418/418 questions** on one H100 PCIe
(job **950793**). The paired ViSpec run shows a **2.632× observed latency ratio**,
but exact output equivalence and controlled speedup remain unvalidated. See the
[paired review](../01_baseline_draft/RESULTS_20261017.md) for the evidence.

| Component | Configuration |
| --- | --- |
| Model | `llava-hf/llava-v1.6-vicuna-7b-hf` (LLaVA-1.6 Vicuna 7B) |
| Loader | ViSpec wrapper. The draft checkpoint is loaded but not used for decoding |

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
bash experiment/submit.sh experiment/00_baseline_original
```

See [`run.sbatch`](run.sbatch) for Slurm resources and the job entry point.
Jobs use the prepared repository `.venv` directly. No activation is needed.
Set an absolute `PYTHON_BIN` to use a different environment. `packages.txt`
records installed distribution versions without requiring pip or uv at runtime.

### Run inside an existing GPU allocation

Use an allocated NVIDIA GPU job. The following script does not allocate resources:

```bash
bash experiment/00_baseline_original/run.sh
```

## Settings

Override model/data settings through environment variables. Use absolute paths for local files.
Temperature is fixed at **0.0 (greedy)**. `TEMPERATURE` does not override it.

| Variable | Default | Purpose |
| --- | --- | --- |
| `BENCH_DIR` | `<repo>/bench_data` | Benchmark data root |
| Temperature | `0.0` (fixed) | Greedy decoding. No environment override |
| `PYTHON_BIN` | `<repo>/.venv/bin/python` | Python executable |
| `SPEC_DIR` | Prepared local draft, then its Hugging Face ID | Draft checkpoint required by the loader, even for autoregressive decoding |
| `BASE_MODEL_PATH` | Prepared local copy detected automatically | Optional local copy of the same LLaVA checkpoint |

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

| Benchmark | Input | Subset |
| --- | --- | --- |
| TextVQA | `$BENCH_DIR/textvqa/TextVQA_0.5.1_val.json` + `train_images/` | Shuffle seed 42, first 100 |
| GQA | `$BENCH_DIR/gqa/llava_gqa_testdev_balanced.jsonl` + `images/` | Shuffle seed 42, first 100 |
| MM-Vet | `whyu/mm-vet`, test split | Entire split |

## Run artifacts

Each launch creates a unique folder under `runs/`.

| Evidence | Contents |
| --- | --- |
| Logs | Exact commands and combined stdout/stderr |
| Code | Full Git commit, tracked diff, working-tree status, and runner snapshots |
| Environment | Packages, settings, scheduler job ID, timestamps, and exit status |
| Hardware | `device.txt`, collected by [capture_device.sh](../capture_device.sh) |

Answers, references, and measurements are saved at:

```text
results/<benchmark>_test/baseline_llava/test-temperature-<T>.jsonl
```

The current run snapshot is versioned for team review, including raw answers and
launch evidence. Other generated runs stay on project storage and are ignored by
default. Untracked source-file contents are not captured in launch diffs. Exit code `0` means completion. A missing exit-code file means an
unclean finish.

## Metrics and results

The results below are from the current temperature-0.0 run.

### Greedy run — job 950793

Run `20261017T211853Z_771911bd8665_9RRsjY`, clean commit
`771911bd8665846e9ab8cb698cb22c2929d296b5`, completed 418/418 turns with
exit codes 0 at temperature 0.0 and a 1,024-output-token cap. Hardware: one
H100 PCIe (79.2 GiB), FP16, 8 CPUs and 64 GiB RAM on `evc46`. See the
[device report](runs/20261017T211853Z_771911bd8665_9RRsjY/device.txt).

| Benchmark | Turns | Mean latency (s/turn) | Mean TTFT (s/turn) | Capped | Quality / scored |
| --- | ---: | ---: | ---: | ---: | --- |
| TextVQA | 100/100 | 9.6748 | 0.2857 | 11 | 0% consensus (100/100 scored) |
| GQA | 100/100 | 1.3804 | 0.2729 | 0 | 0% exact match (100/100 scored) |
| MM-Vet | 218/218 | 4.1088 | 0.2766 | 3 | Pending judge (0/218 scored) |

Pooled mean latency is 4.7877 s/turn, TTFT 0.2779 s/turn. Throughput is
43.4600 returned IDs/s including special tokens. Means weight by turn count.
The [paired review](../01_baseline_draft/RESULTS_20261017.md) contains artifact
verification, token mismatches, memory, aggregation definitions and limitations.

Automatic reports are saved under each run:

```text
runs/<run>/reports/<timestamp>_<suffix>/summary.md
runs/<run>/reports/<timestamp>_<suffix>/summary.json
```

| Category | Reported metrics |
| --- | --- |
| Speed | Mean, median, and p95 latency and TTFT. Throughput. Time per output token |
| Outputs | Actual token counts, stop reasons, truncation, and post-EOS tokens |
| Resources | Peak allocated/reserved GPU memory per visible device, measured after warmup |
| Quality | TextVQA consensus, GQA exact match, and imported MM-Vet judge scores |
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

- `wall_time` covers prefill and decoding. `decode_time` covers the decoding loop.
- Both timings exclude model/data loading.
- `new_tokens` counts the returned token IDs saved with each answer.
- `speed.py` re-tokenizes answers, so its token counts and throughput are estimates.
- Current runners enforce the same 1,024-output-token cap for both methods.
- Truncated outputs keep controlled speed comparisons pending.
- Timing synchronizes GPUs. TTFT synchronization overhead is included in total latency.

### Hardware measurements

- `device.txt` captures launch conditions. Runtime allocator peaks are measured separately.
- Energy, utilization, and non-PyTorch GPU memory are not measured.
- CUDA-visible devices may differ from host GPUs listed by `nvidia-smi`.
