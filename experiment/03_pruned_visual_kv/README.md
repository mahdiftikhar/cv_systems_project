# 03 — Pruned visual KV cache

This experiment asks whether the target model can verify draft tokens while keeping
only part of the image in its KV cache, and whether its answers stay the same. Image
positions make up much of a LLaVA prompt, and every verification step attends to all
of them, so removing them could make verification cheaper.

**Hypothesis.** Answers stay close to unpruned ViSpec at moderate pruning and drift
as more of the image is removed, because the text positions after the image already
absorbed image information during prefill. Each verification step gets cheaper, but
the total speedup depends on how much pruning disturbs draft acceptance.

**Status: pending.** No run has been made.

## Changes from experiment 02

The only change to decoding is one pruning step. Prefill runs on the full prompt and
produces the first token as usual. Evenly spaced image positions are then removed
from the target's cache, in every layer, for the rest of the answer.

| Change | 02 | 03 |
| --- | --- | --- |
| Image positions kept after prefill | All | 100%, 50%, 25%, 10%, 0% |
| Prompts | Short | Short and long, as separate sweeps |
| Benchmarks | TextVQA, GQA | Short: TextVQA, GQA. Long: TextVQA, GQA, MM-Vet |
| Comparison | `compare_metrics` | `compare_pruning`, against keep 100% and autoregressive |

Text positions always stay. New tokens keep their absolute position IDs, so pruning
does not shift the positions of what remains. The draft model and its own cache are
not pruned, although the target hidden states it receives can change.

Three code paths support this. `prune_image_positions` in `vispec/model/kv_cache.py`
compacts the cache, `update_inference_inputs` in `vispec/model/utils.py` now separates
cache slots from prompt positions, and the rotary embedding slice in
`vispec/model/modeling_llama_kv.py` now covers absolute positions. The default
`--kv-keep-ratio 1.0` skips compaction.

## Models, data and settings

| Setting | Value |
| --- | --- |
| Target model | `llava-hf/llava-v1.6-vicuna-7b-hf` |
| Draft checkpoint (ViSpec only) | `JLKang/ViSpec-llava-v1.6-vicuna-7b-hf` |
| Data | The same 100 TextVQA, 100 GQA and 218 MM-Vet questions as earlier experiments |
| Decoding | Temperature 0.0 (greedy), at most 1,024 output tokens including EOS |
| ViSpec tree | Depth 3, top-k 8, total tokens 30, query vectors 2 |
| Pruning | Evenly spaced image positions, chosen once per question |
| Resources requested | 1 H100 PCIe, 8 CPUs, 64 GiB RAM, 8 hours |

## Run on Newton

Prepare the models and data with `prepare.sh`, make sure the cleaned MM-Vet cache
exists at `.cache/mmvet_clean`, and commit the code so the run records a clean launch
tree. From the repository root on a login node, first run a short pilot on real
weights:

```bash
mkdir -p logs
export PYTHON_BIN=/absolute/path/to/python
KEEP_PERCENTS="100 25" ANSWER_STYLES="short" sbatch --account=your_slurm_account --time=01:00:00 --export=ALL experiment/03_pruned_visual_kv/run.sbatch
```

If the pilot finishes cleanly, run the full sweep:

```bash
sbatch --account=your_slurm_account --export=ALL experiment/03_pruned_visual_kv/run.sbatch
```

The scheduler log is `logs/llava-prune-JOB_ID.log`. Each launch creates a unique
folder under `runs/`, with `short/` and `long/` sweeps. Each sweep holds an
`autoregressive/` run, one `keep_NNN/` run per ratio and a `comparison/` folder. Each
run folder records the commit, working-tree status and diff, packages, device report,
commands, combined output, answers, reports and exit status. A failed ratio does not
stop the others, but it makes the job's exit code nonzero.

## Metrics

Results are reported per benchmark and never averaged across benchmarks.

- **Identical token sequences** counts questions whose raw output IDs match keep 100%
  through the first EOS. It is the strict test of whether pruning is lossless.
- **Matching prefix** is the share of keep-100% output tokens produced before the
  first difference, pooled across questions. It shows how much of an answer survives
  when the sequences are not identical.
- **Identical answer text** ignores whitespace differences.
- **Autoregressive row.** The same agreement columns, showing how far unpruned ViSpec
  already differs from normal decoding. Pruning differences should be read against it.
- **Quality** is TextVQA consensus or GQA exact match, in percent. **Quality Δ** is
  the change from keep 100%, in percentage points. MM-Vet stays pending without judge
  grades.
- **Speedup** divides the mean latency of keep 100% or autoregressive by this row's,
  on the questions they share. Latency includes prefill and the pruning step.
- **Accepted draft tokens/step** is the mean number of draft tokens kept per
  verification step, pooled across steps.
- **Logical KV removed** is the mean size of the removed cache positions, in GiB per
  question. It is not a measured memory saving.

Latency and TTFT definitions are in the [metric guide](../METRICS.md#what-each-metric-means).

## Results

**Pending.** Hardware and per-benchmark results go here after a verified run, with
links to the device reports.

## Limitations

- **Short answers cannot test pruning well.** The first token comes from the full
  cache, so an answer that is one word plus EOS is largely decided before pruning.
  Short-answer accuracy can therefore look lossless by construction. Long-prompt
  agreement is the main evidence, and short-answer accuracy is supporting evidence.
- **Long answers have no reliable quality score.** The strict scorers mark them wrong,
  so long-prompt results show whether answers change, not whether changes are harmful.
- **Keep 0% is not image-free.** The first token, the text positions after the image
  and the draft's compact image summary all still carry image information.
- **No measured memory saving.** The cache tensor is preallocated and prefill still
  fills it, so peak GPU memory should not fall. A real saving would need a smaller
  allocation after pruning, which this experiment does not implement.
- **Selection is simple.** Evenly spaced positions ignore image content. LLaVA-1.6
  places the base view and high-resolution tiles in one sequence, and this samples
  across both. Smarter selection could do better at the same ratio.
- **One run per setting.** A single sweep gives no measure of variation between
  repeats, and small latency differences may be noise. The scores describe these
  question subsets.
- **Timings across experiments.** The rotary change adds a small operation to every
  attention call, so compare timings within one sweep rather than with 00, 01 or 02.
