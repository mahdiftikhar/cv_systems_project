# 00 — Original autoregressive baseline

Measure reference inference latency on the UCF cluster using existing
autoregressive decoding with timing instrumentation. This experiment requires
no training. The [ViSpec draft baseline](../01_baseline_draft/README.md) provides
the speculative decoding comparison.

**Baseline generation completed:** 418/418 questions on one H100 PCIe.
TextVQA and GQA scored 0% under the recorded strict scoring protocols; MM-Vet
quality and the ViSpec speed comparison remain pending.

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

Override model/data settings through environment variables. Use absolute paths for local files.
Temperature is fixed at **0.0 (greedy)**; `TEMPERATURE` does not override it.

| Variable | Default | Purpose |
| --- | --- | --- |
| `BENCH_DIR` | `<repo>/bench_data` | Benchmark data root |
| Temperature | `0.0` (fixed) | Greedy decoding; no environment override |
| `PYTHON_BIN` | `<repo>/.venv/bin/python` | Python executable |
| `SPEC_DIR` | Prepared local draft, then its Hugging Face ID | Draft checkpoint required by the loader, even for autoregressive decoding |
| `BASE_MODEL_PATH` | Prepared local copy detected automatically | Optional local copy of the same LLaVA checkpoint |

## Evaluation

Current runs fix temperature at 0.0 and pass `--max-new-token 1024` for
both warmup and measured questions. The cap counts all returned token IDs,
including EOS. Both methods stop at first EOS or exactly 1,024 output tokens.
ViSpec clips the final accepted prefix before committing it, so it cannot
return tokens beyond EOS or the cap. Prompt construction is unchanged.

Raw generated token IDs and the EOS ID are saved in each turn's `measurement`,
before decoding cleanup. Use the [comparison command](../METRICS.md#compare-methods-and-repeats)
to check exact sequences through first EOS and inspect first mismatch positions.
Greedy correctness is pending until matching cluster runs complete; no judge
is needed. Matching capped prefixes do not establish full-answer equivalence,
and truncated runs keep controlled speedup pending.

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

Future runs use temperature 0.0. The completed run below used temperature **1.0**
and must remain separate from the pending greedy baseline/speculative comparison.

### Completed rerun — job 943935

Run `20261017T091226Z_a2ac5b103129_IN6ujH` completed with evaluation and
report exit codes 0, from 2026-10-17 09:12:33 to 10:14:27 UTC (61m 54s).
Launch commit: `a2ac5b1031296589c6714aff2f8e6eb97922f0bf`, with working-tree
changes preserved in the run's status and patch; the commit alone does not
identify all executed code.

Hardware: node `evc38`, one allocated NVIDIA H100 PCIe (79.2 GiB visible to
PyTorch), FP16, 8 allocated logical CPUs and 64 GiB RAM. Host hardware is an
Intel Xeon Gold 6326 with 64 logical CPUs and about 502.7 GiB RAM; host totals
are not the job allocation. PyTorch 2.7.0+cu128, CUDA build 12.8, driver
580.105.08. See the [device report](runs/20261017T091226Z_a2ac5b103129_IN6ujH/device.txt)
and [full metrics report](runs/20261017T091226Z_a2ac5b103129_IN6ujH/reports/20261017T101426Z_c24ebfc7/summary.md).

| Benchmark | Questions | Mean latency (s/turn) | p95 latency (s/turn) | Mean TTFT (s/turn) | Output tokens/s | Mean output tokens | Quality / scored |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| TextVQA | 100/100 | 10.3908 | 18.5753 | 0.2819 | 44.7097 | 464.57 | 0% consensus; 100/100 |
| GQA | 100/100 | 1.7808 | 3.6982 | 0.2566 | 39.0058 | 69.46 | 0% exact match; 100/100 |
| MM-Vet | 218/218 | 4.6926 | 11.5941 | 0.2721 | 43.5750 | 204.48 | Pending official judge; 0/218 |

Pooling all 418 completed turns gives mean latency **5.3592 s/turn**, mean
TTFT **0.2707 s/turn**, and throughput **43.7381 output tokens/s**. Means are
weighted by completed turn count; throughput is total returned token IDs divided
by total generation time. Output counts include special tokens, without
re-tokenizing text. Quality scores are not pooled across benchmarks. Generation
timing excludes loading, prompt preparation, answer decoding, and reporting;
the job's elapsed time includes these costs.

All 418 outputs stopped at EOS, with no recorded truncation, post-EOS tokens,
or token-counter mismatches. Peak PyTorch allocated memory was 20.872 / 20.828 /
20.930 GiB for TextVQA / GQA / MM-Vet; peak reserved memory was 24.709 / 22.725 /
30.959 GiB. These are runtime allocator peaks after warmup, not whole-process
VRAM, energy, or utilization measurements.

The zero scores require diagnosis before treating this as a quality-validated
baseline. Saved outputs are verbose: TextVQA question 34682 includes
“COORS LIGHT”, matching a reference, inside a long explanation that receives
zero consensus credit when scored as a whole. GQA also produces sentences and
explanations that strict exact match can reject. This demonstrates a response
format problem in inspected samples, but does not establish that every zero is
caused by formatting or that the remaining answers are correct. No answer
extraction or prompt changes have been applied to these measurements.

Local verification checked imported artifact SHA-256 hashes against the report,
unique question counts, recomputed latency/TTFT means, EOS stops, and token
counter flags. Scoring annotations and scorer are recorded by hash in the
report but are not imported locally, so quality scores above are the cluster
report's recorded scores, not an independent local re-score. One successful
run provides no repeat variability; experiment 01 has no imported run yet,
so speculative speedup remains pending.

### Earlier failed attempt — job 942134

Run `20261017T042535Z_a2ac5b103129_hISxbf` (job `942134`, commit
`a2ac5b1031296589c6714aff2f8e6eb97922f0bf`) failed with exit code 1 during
TextVQA warmup: CUDA error 101, invalid device ordinal. TextVQA completed 0/100
questions; GQA and MM-Vet were not started. No performance or quality results
are available.

Hardware: node `evc26`, host Intel Xeon Gold 6326 (64 logical CPUs) and about
502.5 GiB RAM; Slurm allocated 8 CPUs, 64 GiB RAM, and one GPU, but no GPU was
visible to PyTorch or `nvidia-smi`. GPU identity/VRAM are unavailable. See the
[device report](runs/20261017T042535Z_a2ac5b103129_hISxbf/device.txt) and
[failure log](runs/20261017T042535Z_a2ac5b103129_hISxbf/output.log), preserved
outside Git on project storage. Batch submissions now exclude `evc26`; runners
check CUDA allocation and computation before model loading.

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
- Current runners enforce the same 1,024-output-token cap for both methods.
- Historical runs used different inherited limits; keep them separate.
- Truncated outputs keep controlled speed comparisons pending.
- Timing synchronizes GPUs; TTFT synchronization overhead is included in total latency.

### Hardware measurements

- `device.txt` captures launch conditions; runtime allocator peaks are measured separately.
- Energy, utilization, and non-PyTorch GPU memory are not measured.
- CUDA-visible devices may differ from host GPUs listed by `nvidia-smi`.
