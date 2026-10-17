# Experiment register

The UCF project compares autoregressive inference with ViSpec speculative decoding
on **TextVQA → GQA → MM-Vet**, in that order. All experiments run on the UCF cluster.

Each numbered directory contains an experiment guide and a runnable shell script.
Every invocation saves its evidence in a unique `runs/` subdirectory.

## Choose an experiment

| Experiment                                                | Question / method                                | Model               | Fixed temperature | Status  |
| --------------------------------------------------------- | ------------------------------------------------ | ------------------- | ------------------- | ------- |
| [00 — Original baseline](00_baseline_original/README.md)  | Reference latency with autoregressive inference  | LLaVA-1.6 Vicuna 7B | 0.0                 | Completed 418/418; quality needs diagnosis |
| [01 — ViSpec draft baseline](01_baseline_draft/README.md) | Does ViSpec speculative decoding reduce latency? | LLaVA-1.6 Vicuna 7B | 0.0                 | Not run |

Both runners fix temperature at **0.0 (greedy)**, including batch submissions;
the `TEMPERATURE` environment variable is ignored. The completed baseline below
used **1.0**. A matching temperature-0 baseline and speculative comparison remain
pending; preserve the historical measurements separately.

Both runners also enforce **1,024 output tokens**, counting EOS, and save raw
output token IDs. Greedy comparisons report exact sequence agreement and first
mismatches; missing token evidence, mismatches, or truncation keep validated
speedup pending. These changes have not yet been evaluated on the cluster.

## Submit a run

Complete the [Newton setup guide](NEWTON_SETUP.md), then submit an experiment from
the repository checkout:

```bash
bash experiment/submit.sh experiment/00_baseline_original
```

For the ViSpec experiment:

```bash
bash experiment/submit.sh experiment/01_baseline_draft
```

Each experiment has its own `run.sbatch`, with independent resource directives and
job name. Jobs reuse the prepared environment.

Both batch files exclude `evc26`, where job `942134` could not access a GPU.
After saving `device.txt`, both runners require a small CUDA allocation,
computation, and synchronization on `cuda:0` before loading models. A failed
check stops the run with a nonzero exit code and preserves the launch evidence.

| Output                       | Location                          |
| ---------------------------- | --------------------------------- |
| Combined scheduler logs      | `$PROJECT_ROOT/logs/`             |
| Detailed experiment evidence | `<experiment>/runs/<unique-run>/` |

## Overall results

**Pending** means no measurements have been collected; it is not a zero result.
Completed measurements below link to their evidence in the experiment README.

| Experiment                                                | Mean wall time (s/turn) | Mean TTFT (s/turn) | Device / allocation report | Commit / run |
| --------------------------------------------------------- | ----------------------: | -----------------: | -------------------------- | ------------ |
| [00 — Original baseline](00_baseline_original/README.md)  | 5.3592 | 0.2707 | H100 PCIe ×1, 79.2 GiB; 8 CPUs, 64 GiB RAM; [device report](00_baseline_original/runs/20261017T091226Z_a2ac5b103129_IN6ujH/device.txt) | `a2ac5b1031296589c6714aff2f8e6eb97922f0bf` + dirty tree / job `943935` |
| [01 — ViSpec draft baseline](01_baseline_draft/README.md) |                 Pending |            Pending | Pending                    | Pending      |

Per-benchmark baseline results (job `943935`, temperature **1.0**, FP16, same H100 allocation):

| Benchmark | Completed questions | Mean latency (s/turn) | Mean TTFT (s/turn) | Quality / scored |
| --- | ---: | ---: | ---: | --- |
| TextVQA | 100/100 | 10.3908 | 0.2819 | 0% consensus; 100/100 |
| GQA | 100/100 | 1.7808 | 0.2566 | 0% exact match; 100/100 |
| MM-Vet | 218/218 | 4.6926 | 0.2721 | Pending judge; 0/218 |

The overall means pool all 418 completed turns, weighted by turn count; quality
scores are not aggregated. All outputs stopped at EOS without recorded
truncation. The 0% scores need diagnosis: inspected answers contain verbose
explanations, sometimes including a correct short answer that strict scoring
rejects. They do not establish zero underlying visual understanding. No ViSpec
run or speedup comparison is available yet. See the
[verified results, limitations, and artifact links](00_baseline_original/README.md#completed-rerun--job-943935).

Earlier baseline job `942134` on `evc26` completed zero questions because no GPU
was accessible. Its evidence remains preserved in the
[run diagnosis](00_baseline_original/README.md#earlier-failed-attempt--job-942134).

### Metrics and comparisons

- **Per-run reports:** latency, time to first token (TTFT), throughput, time per
  output token (TPOT), output lengths, stop reasons, and per-GPU allocator peaks.
- **Quality:** TextVQA consensus, GQA exact match, and imported MM-Vet judge scores.
- **Comparisons:** matching-question speedup, answer agreement, quality deltas,
  and repeated-run variation.

See the [metrics guide](METRICS.md) for definitions, scoring prerequisites, and commands.
Report per-benchmark counts and results before suite aggregation; never average
benchmark quality scores. Keep results from different hardware or settings
separate, and link the device report with verified results.

## Add or update an experiment

Use `NN_descriptive_name/` for each new experiment. Its brief `README.md` must cover:

- Purpose or hypothesis and changes from baseline.
- Models, data, splits, settings, and exact run commands.
- Environment and cluster resource requirements.
- Metric definitions, results, and limitations.

Update this register when adding an experiment and when recording verified results.
Never report paper numbers as measurements from this project.

### Preserve evidence for every run

Keep logs, the full launch commit, working-tree status and diff, exact commands,
environment, device report, and raw outputs. Never overwrite previous runs.

Use [`capture_device.sh`](capture_device.sh) before evaluation to capture:

- GPU identity and VRAM; CPU and RAM.
- OS, CUDA, driver, and PyTorch versions.
- GPU operating state and scheduler allocation.

Optional probes explicitly report missing tools or failures. For each result row,
summarize the allocated GPU model, count, and VRAM, plus CPU/RAM allocation, and
link the run's `device.txt`.

Host totals can exceed job allocations: retain both and use allocated resources
in comparisons. Keep separate rows for different devices. Launch snapshots do
not measure runtime peak memory, energy use, or sustained utilization.
