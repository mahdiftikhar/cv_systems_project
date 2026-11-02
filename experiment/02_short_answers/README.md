# 02 — Short-answer prompts

This experiment makes answer accuracy measurable. Experiments 00 and 01 ask for
long explanations, so the strict scorers mark almost every answer wrong, even when
the correct answer appears in the text. Later pruning experiments need a real
accuracy number to show whether pruning hurts quality.

**Hypothesis.** With short-answer prompts, TextVQA and GQA scores rise well above
0%, and both decoding methods score about the same. The ViSpec speedup should be
smaller than with long prompts, because short answers leave little text to
speculate on and the first-token wait takes up more of the total time.

**Status: pending.** No run has been made.

## Changes from experiments 00 and 01

| Change | 00 and 01 | 02 |
| --- | --- | --- |
| TextVQA instruction | A detailed OCR explanation | "Answer the question using a single word or phrase." |
| GQA instruction | "Please answer with an explanation." | "Answer the question using a single word or phrase." |
| Benchmarks | TextVQA, GQA, MM-Vet | TextVQA, GQA |
| Scheduling | One job per method | Both methods in one job, so they share a GPU and allocation |

MM-Vet is left out because its questions are open-ended and its scores need
external judge grades, so a word-or-phrase answer does not suit it. The system
prompt is unchanged. The prompts are selected with `--answer-style short`; the
default `long` keeps the original prompts.

`attn_implementation="eager"` was removed from the LLaVA-Next branch of
`vispec/model/spec_model_ours.py`. Only ViSpec loads through that file, so keeping
it would give the two methods different attention settings.

## Models, data and settings

| Setting | Value |
| --- | --- |
| Target model | `llava-hf/llava-v1.6-vicuna-7b-hf` |
| Draft checkpoint (ViSpec only) | `JLKang/ViSpec-llava-v1.6-vicuna-7b-hf` |
| Data | The same 100 TextVQA and 100 GQA questions as 00 and 01 |
| Decoding | Temperature 0.0 (greedy), at most 1,024 output tokens including EOS |
| ViSpec tree | Depth 3, top-k 8, total tokens 30, query vectors 2 |
| Resources requested | 1 H100 PCIe, 8 CPUs, 64 GiB RAM, 2 hours |

## Run on Newton

Prepare the models and data with `prepare.sh` first, and commit the code so the run
records a clean launch tree. From the repository root on a login node, with your
own Slurm account and Python path:

```bash
mkdir -p logs
export PYTHON_BIN=/absolute/path/to/python
sbatch --account=your_slurm_account --export=ALL experiment/02_short_answers/run.sbatch
```

The scheduler log is `logs/llava-short-JOB_ID.log`. Each launch creates a unique
folder under `runs/`, holding `autoregressive/` and `vispec/` run folders and a
`comparison/` folder. Each run folder records the commit, working-tree status and
diff, packages, device report, commands, combined output, answers, reports and exit
status. Inside an existing GPU allocation, `bash experiment/02_short_answers/run.sh`
runs the same pair.

## Metrics

Quality is reported per benchmark and never averaged across them.

- **TextVQA accuracy** uses VQA consensus: an answer earns credit according to how
  many of the human reference answers it matches after normalization.
- **GQA accuracy** is an exact match after stripping whitespace and lowercasing.
- **Exact token agreement** counts questions where both methods produced identical
  token sequences, from the comparison report.
- **Mean latency, TTFT and tokens/s** are defined in the
  [metric guide](../METRICS.md#what-each-metric-means). Both latency and TTFT are in
  seconds per question.
- **Accepted draft tokens** is the mean number of draft tokens kept per verification
  step, found in each `summary.json`.

## Results

**Pending.** Hardware and per-benchmark results go here after a verified run, with
links to the device reports.

## Limitations

- **GQA punctuation.** The scorer does not remove punctuation, so "Yes." would count
  as wrong against "yes". Check `gqa_predictions.json` before trusting the GQA score.
  If answers end in periods, decide on a scoring rule and record it here, since the
  saved answers can be rescored without a GPU.
- **Speed.** These short answers measure quality. Use the long-prompt runs for
  speedup claims, since they exercise far more decoding steps.
- **Subsets and repeats.** The scores describe these 100-question subsets, and a
  single run gives no measure of variation between repeats.
- **Comparison with 00 and 01.** Those runs used different jobs, prompts and possibly
  attention settings, so their timings are not directly comparable with this pair.
