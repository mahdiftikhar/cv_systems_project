# Experiment plots

These figures compare **00: standard autoregressive generation** with
**01: ViSpec speculative generation**. Both answer the same image questions
using LLaVA-1.6 Vicuna 7B. Standard generation produces one token at a time.
ViSpec proposes several using a smaller draft and verifies them together.
The goal is faster answers with the same content and correctness.

Read the [comparison page](../COMPARISON.md) for the embedded plots, an
explanation of every plotted metric, and the benchmark tables. The
[full metric guide](../METRICS.md#what-each-metric-means) covers the remaining
reported measurements and why we collect them.

## Figures

- [Overview](latest_greedy/overview.png): mean generation time, time to first
  token, throughput, time per output token, GPU memory and exact token agreement.
- [Latency distributions](latest_greedy/latency_distributions.png): the fraction
  of questions completed within each generation time, separately by benchmark.
- Individual charts: [latency](latest_greedy/latency.png),
  [time to first token](latest_greedy/ttft.png),
  [throughput](latest_greedy/throughput.png),
  [time per output token](latest_greedy/tpot.png), and
  [GPU memory](latest_greedy/gpu_memory.png).

All figures are PNGs. Source values and source-file SHA-256 hashes are preserved
in [plotted_values.json](latest_greedy/plotted_values.json). A source hash identifies
the exact report used, so the figures can be checked against the saved results.

## Conditions and interpretation

The suite contains 100 TextVQA questions (reading text in images), 100 GQA
questions (visual reasoning) and 218 MM-Vet questions (open-ended visual
understanding). These are single-run subset results, not full leaderboard scores.
Both methods used FP16, greedy decoding and a 1,024-token output cap including
the end-of-sequence token. A token is a model output unit, often part of a word.

Each job received one H100 PCIe with 79.2 GiB visible VRAM, 8 logical CPUs and
64 GiB RAM. The jobs overlapped on `evc46`, using different physical GPUs and
CPU bindings. See the [00 device report](../00_baseline_original/runs/20261017T211853Z_771911bd8665_9RRsjY/device.txt)
and [01 device report](../01_baseline_draft/runs/20261017T211906Z_771911bd8665_ei1koG/device.txt).

The plots show observed timing improvements. They do not establish equivalent
outputs: 51 of 418 token sequences differ, and 14 answers per method were stopped
by the output cap. Official MM-Vet judging and overall quality validation remain
pending. Bar heights summarize measured questions. There are no repeat-run error
bars because we have only one run per method.

## Regenerate from saved reports

Run from the repository root with Python 3.10+ and Matplotlib installed. These
figures were rendered with Matplotlib 3.9.4. The script reads saved JSON reports
without loading a model, accessing a GPU or running evaluation.

```bash
python3 experiment/plot_comparison.py \
  --original experiment/00_baseline_original/runs/20261017T211853Z_771911bd8665_9RRsjY/reports/20261017T220622Z_df807dd5 \
  --draft experiment/01_baseline_draft/runs/20261017T211906Z_771911bd8665_ei1koG/reports/20261017T214508Z_37fa2d97 \
  --output-dir experiment/plots/latest_greedy
```

The command replaces these derived figures. Raw cluster answers and reports
remain unchanged in the versioned current run snapshots. Use a new output directory
to preserve a separate set of figures. This script's captions describe the latest
greedy pair above. Update them when plotting a different experiment or setting.
