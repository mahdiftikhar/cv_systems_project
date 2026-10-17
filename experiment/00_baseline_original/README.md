# 00 — Original autoregressive baseline

Measure reference inference latency on the UCF cluster using existing
autoregressive decoding with timing instrumentation. This experiment requires
no training. The [ViSpec draft baseline](../01_baseline_draft/README.md) provides
the speculative decoding comparison.

**Results: pending.** No cluster measurements have been collected.

| Component | Configuration |
| --- | --- |
| Model | `llava-hf/llava-v1.6-vicuna-7b-hf` (LLaVA-1.6 Vicuna 7B) |
| Loader | ViSpec wrapper; the draft checkpoint is loaded but not used for decoding |

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
Jobs use the prepared repository `.venv` directly; no activation is needed.
Set an absolute `PYTHON_BIN` to use a different environment. `packages.txt`
records installed distribution versions without requiring pip or uv at runtime.

### Run inside an existing GPU allocation

Use an allocated NVIDIA GPU job. The following script does not allocate resources:

```bash
bash experiment/00_baseline_original/run.sh
```

## Settings

Override settings through environment variables. Use absolute paths for local files.

| Variable | Default | Purpose |
| --- | --- | --- |
| `BENCH_DIR` | `<repo>/bench_data` | Benchmark data root |
| `TEMPERATURE` | `1.0` | Sampling temperature |
| `PYTHON_BIN` | `<repo>/.venv/bin/python` | Python executable |
| `SPEC_DIR` | Prepared local draft, then its Hugging Face ID | Draft checkpoint required by the loader, even for autoregressive decoding |
| `BASE_MODEL_PATH` | Prepared local copy detected automatically | Optional local copy of the same LLaVA checkpoint |

## Evaluation

- **Order:** TextVQA → GQA → MM-Vet; stop on the first failure.
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

Keep raw runs on project storage, outside Git. Untracked file contents are not
captured. Exit code `0` means completion; a missing exit-code file means an
unclean finish.

## Metrics and results

Automatic reports are saved under each run:

```text
runs/<run>/reports/<timestamp>_<suffix>/summary.md
runs/<run>/reports/<timestamp>_<suffix>/summary.json
```

| Category | Reported metrics |
| --- | --- |
| Speed | Mean, median, and p95 latency and TTFT; throughput; time per output token |
| Outputs | Actual token counts, stop reasons, truncation, and post-EOS tokens |
| Resources | Peak allocated/reserved GPU memory per visible device, measured after warmup |
| Quality | TextVQA consensus, GQA exact match, and imported MM-Vet judge scores |
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

- `wall_time` covers prefill and decoding; `decode_time` covers the decoding loop.
- Both timings exclude model/data loading.
- `new_tokens` now counts returned token IDs; historical runs with zero counters remain unsupported for throughput.
- `speed.py` re-tokenizes answers, so its token counts and throughput are estimates.
- Generation defaults to 2048 steps; the `new_token > 1024` check is ineffective.
- CLI `--max-new-token` is not forwarded to the generation loop.
- Draft decoding has different inherited generation limits; check truncation before comparing latency.
- Timing synchronizes GPUs; TTFT synchronization overhead is included in total latency.

### Hardware measurements

- `device.txt` captures launch conditions; runtime allocator peaks are measured separately.
- Energy, utilization, and non-PyTorch GPU memory are not measured.
- CUDA-visible devices may differ from host GPUs listed by `nvidia-smi`.
