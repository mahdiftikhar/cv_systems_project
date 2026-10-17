# 01 — ViSpec draft baseline

Does ViSpec reduce inference latency compared with the
[original autoregressive baseline](../00_baseline_original/README.md)? This
experiment drafts tokens with ViSpec and verifies them with LLaVA, without training.

**Results: pending.** No cluster measurements have been collected.

| Component                 | Configuration                                                  |
| ------------------------- | -------------------------------------------------------------- |
| Base model                | `llava-hf/llava-v1.6-vicuna-7b-hf`                             |
| Draft checkpoint          | `JLKang/ViSpec-llava-v1.6-vicuna-7b-hf`                        |
| Change from experiment 00 | Existing `gen_spec_answer_*` evaluators with `--use-ours True` |

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
Jobs use the prepared repository `.venv` directly; no activation is needed.
Set an absolute `PYTHON_BIN` to use a different environment. `packages.txt`
records installed distribution versions without requiring pip or uv at runtime.

### Run inside an existing GPU allocation

Use an allocated NVIDIA GPU job. The following script does not allocate resources:

```bash
bash experiment/01_baseline_draft/run.sh
```

## Settings

Override settings through environment variables. Defaults follow `exp.sh`.

| Variable          | Default                                              | Purpose                                                           |
| ----------------- | ---------------------------------------------------- | ----------------------------------------------------------------- |
| `BENCH_DIR`       | `<repo>/bench_data`                                  | Benchmark data root                                               |
| `SPEC_DIR`        | Prepared local draft, then the Hugging Face ID above | Draft checkpoint                                                  |
| `BASE_MODEL_PATH` | Prepared local copy detected automatically           | Optional local copy of the same LLaVA checkpoint                  |
| `TEMPERATURE`     | `1.0`                                                | Sampling temperature; same as experiment 00                       |
| `PYTHON_BIN`      | `<repo>/.venv/bin/python`                                             | Python executable; same as experiment 00                          |
| `DEPTH`           | `3`                                                  | Draft-tree depth                                                  |
| `TOP_K`           | `8`                                                  | Draft-tree width                                                  |
| `TOTAL_TOKEN`     | `30`                                                 | Draft-tree token budget                                           |
| `NUM_Q`           | `2`                                                  | Query vectors; must match the checkpoint's training configuration |

Both experiments load with `torch_dtype="auto"`. Record actual precision for comparisons.

## Evaluation

- **Order:** TextVQA → GQA → MM-Vet; stop on the first failure.
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

Keep raw runs on project storage, outside Git. Untracked file contents are not
captured. Exit code `0` means completion; a missing exit-code file means an
unclean finish.

## Metrics and results

Automatic reports are saved under each run:

```text
runs/<run>/reports/<timestamp>_<suffix>/summary.md
runs/<run>/reports/<timestamp>_<suffix>/summary.json
```

| Category   | Reported metrics                                                              |
| ---------- | ----------------------------------------------------------------------------- |
| Speed      | Mean, median, and p95 latency and TTFT; throughput; time per output token     |
| Outputs    | Actual token counts, stop reasons, truncation, and post-EOS tokens            |
| Resources  | Peak allocated/reserved GPU memory per visible device, measured after warmup  |
| Quality    | TextVQA consensus, GQA exact match, and imported MM-Vet judge scores          |
| Comparison | Answer agreement, latency speedup, quality deltas, and repeated-run variation |

See the [metrics guide](../METRICS.md) for accuracy prerequisites, definitions,
and commands. Missing scores remain pending; reports also cover partial runs.
Update the [experiment register](../README.md) with verified per-benchmark results
and hardware links.

## Limitations

### Data and reproducibility

- Scores cover the evaluated subset; accuracy needs the [scoring prerequisites](../METRICS.md#accuracy-prerequisites).
- Model/data revisions are unpinned; retain cache revision IDs for replication.

### Generation and timing

- The CLI's `--max-new-token` is not forwarded to `specgenerate()`.
- Draft generation defaults to `max_new_tokens=512` with a strict `>` stop check;
  a verification iteration can overshoot the limit.
- Its iteration budget is `2048 - total_tokens - 10`; EOS may stop generation earlier.
- 00 instead allows 2048 autoregressive steps with an ineffective 1024-token check.
  These inherited limits differ; truncated outputs prevent a controlled speed comparison.
- TTFT is captured before drafting: the initial token is sampled directly from LLaVA.
- TTFT is `null` for other speculative methods (EAGLE/Medusa), outside this experiment.
- Draft `decode_time` covers the verification loop after initial drafting; use total latency/TTFT for matching comparisons.
- Sampling may produce different answers despite matching seeds.
- Timing synchronizes GPUs; TTFT synchronization overhead is included in total latency.

### Hardware measurements

- `device.txt` captures launch conditions; runtime allocator peaks are measured separately.
- Energy, utilization, and non-PyTorch GPU memory are not measured.
- CUDA-visible devices may differ from host GPUs listed by `nvidia-smi`.
