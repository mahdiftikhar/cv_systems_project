# Experiment register

The UCF project compares autoregressive inference with ViSpec speculative decoding
on **TextVQA → GQA → MM-Vet**, in that order. All experiments run on the UCF cluster.

Each numbered directory contains an experiment guide and a runnable shell script.
Every invocation saves its evidence in a unique `runs/` subdirectory.

## Choose an experiment

| Experiment                                                | Question / method                                | Model               | Default temperature | Status  |
| --------------------------------------------------------- | ------------------------------------------------ | ------------------- | ------------------- | ------- |
| [00 — Original baseline](00_baseline_original/README.md)  | Reference latency with autoregressive inference  | LLaVA-1.6 Vicuna 7B | 1.0                 | Not run |
| [01 — ViSpec draft baseline](01_baseline_draft/README.md) | Does ViSpec speculative decoding reduce latency? | LLaVA-1.6 Vicuna 7B | 1.0                 | Not run |

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

| Output                       | Location                          |
| ---------------------------- | --------------------------------- |
| Combined scheduler logs      | `$PROJECT_ROOT/logs/`             |
| Detailed experiment evidence | `<experiment>/runs/<unique-run>/` |

## Overall results

**Pending** means no measurements have been collected; it is not a zero result.
Only summarize completed runs, and link their evidence in the experiment README.

| Experiment                                                | Mean wall time (s/turn) | Mean TTFT (s/turn) | Device / allocation report | Commit / run |
| --------------------------------------------------------- | ----------------------: | -----------------: | -------------------------- | ------------ |
| [00 — Original baseline](00_baseline_original/README.md)  |                 Pending |            Pending | Pending                    | Pending      |
| [01 — ViSpec draft baseline](01_baseline_draft/README.md) |                 Pending |            Pending | Pending                    | Pending      |

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
