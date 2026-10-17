# Metrics and scoring

- Run `bash prepare.sh` first to prepare repo-local checkpoints, data, and the TextVQA scorer.
- Both runners generate reports under `runs/<run>/reports/<timestamp>_<suffix>/`.
- Reports contain `summary.md`, `summary.json`, `samples.jsonl`, and benchmark prediction exports.
- Re-reporting creates a new folder; raw answers are preserved.
- Missing measurements/scores remain pending, with sample counts and reasons.

## Recorded and calculated metrics

| Metric                  | Definition / units                                                                        |
| ----------------------- | ----------------------------------------------------------------------------------------- |
| Generation latency      | Generation call, including vision/prefill and decoding; seconds                           |
| TTFT                    | Generation entry → first authoritative LLaVA token ready; seconds                         |
| Decode time             | Original: autoregressive loop; draft: verification loop after initial drafting; seconds   |
| Output length           | Returned token IDs, including special tokens and any post-EOS tokens                      |
| Answer length           | Non-special tokens before the first EOS; no re-tokenization                               |
| Throughput              | Total output tokens / total generation seconds, using the same measured turns             |
| TPOT                    | `(latency − TTFT) / (output tokens − 1)`; seconds, only for outputs longer than one token |
| Stop reason             | EOS, token limit, iteration limit, or unknown                                             |
| Truncation rate         | Fraction of known-stop turns ending at a limit; not unknown stops                         |
| Post-EOS tokens         | Extra tokens returned after the first EOS in a speculative block                          |
| Draft acceptance length | Accepted draft tokens per verification iteration; add one for the target token            |
| Peak GPU memory         | PyTorch allocated/reserved bytes, separately for each visible GPU                         |
| Quality score           | Benchmark-specific score on choice 0, turn 0; percent                                     |
| Answer agreement        | Matching answers between methods, ignoring whitespace differences                         |
| Latency speedup         | Original mean latency / draft mean latency on matching turns                              |
| Repeat variation        | Sample standard deviation of independent per-run means; pending for one run               |

- Latency, TTFT, decode time, TPOT, lengths, and acceptance include mean, median, p95, and sample counts.
- Percentiles use linear interpolation at `(n − 1) × quantile`.
- Suite means pool completed turns; benchmark quality scores are never averaged together.
- Throughput is a ratio of totals; per-answer throughput distributions are also saved.
- Token counters are checked against returned IDs. Historical zero counters are not reused.
- TTFT measures token availability, not streamed text; TPOT is not measured inter-token delivery latency.
- Timing excludes model/data loading, prompt preparation, answer decoding, scoring, and report probes.
- Peak memory is reset after warmup, includes resident model/KV memory, and excludes non-PyTorch allocations.
- Peak values are not summed across devices; warmup/model-loading peaks, energy, and utilization are not measured.

## Accuracy prerequisites

- **TextVQA:** ten reference answers are retained from the validation JSON.
- Preparation downloads the scorer to `tools/downloaded/m4c_evaluator.py`; runners discover it automatically.
- For a custom copy, set `TEXTVQA_EVALUATOR` to LLaVA's
  [m4c_evaluator.py](https://github.com/haotian-liu/LLaVA/blob/main/llava/eval/m4c_evaluator.py).
- The scorer uses its `EvalAIAnswerProcessor` normalization with leave-one-out VQA consensus.
- **GQA:** preparation downloads matching testdev answer annotations; runners discover them automatically.
- For custom annotations, set `GQA_ANNOTATIONS` if the input JSONL lacks labels.
- Accepted formats: GQA question JSON keyed by question ID, or JSON/JSONL rows with IDs and `answer`.
- GQA score is exact match after stripping whitespace and lowercasing; explanations are not extracted heuristically.
- **MM-Vet:** `mmvet_predictions.json` is exported for the
  [official judge evaluator](https://github.com/yuweihao/MM-Vet/blob/main/mm-vet_evaluator.py).
- Import its per-question grade JSON and the exact predictions file used for grading.
- Grades must include `score` values in `[0,1]` and a `model` ID for each judge run.
- Imported predictions must match the current answers; grades are averaged per question, then across questions.
- No judge calls or paid API requests run automatically.
- Accuracy is a subset score, not a full-benchmark leaderboard result; partial score coverage is explicit.
- Existing TextVQA prompts request OCR explanations, which may score poorly under strict answer matching.
- Source files and scoring dependencies are recorded with SHA-256 hashes.

## Run with local accuracy scoring

```bash
TEXTVQA_EVALUATOR=/absolute/path/to/m4c_evaluator.py \
GQA_ANNOTATIONS=/absolute/path/to/testdev_balanced_questions.json \
BENCH_DIR=/absolute/path/to/bench_data \
  bash experiment/00_baseline_original/run.sh
```

- Use the same environment variables with `experiment/01_baseline_draft/run.sh`.
- Missing scoring prerequisites do not block performance reports.
- Invalid scoring files fail reporting explicitly; generation failures retain their original exit status.
- A report is attempted after failed/interrupted evaluation to preserve partial coverage.

## Re-score saved answers

```bash
python -m vispec.evaluation.report_metrics \
  --run-dir experiment/01_baseline_draft/runs/<run> \
  --textvqa-evaluator /absolute/path/to/m4c_evaluator.py \
  --gqa-annotations /absolute/path/to/testdev_balanced_questions.json \
  --mmvet-grades /absolute/path/to/model_judge-grade-5runs.json \
  --mmvet-graded-predictions /absolute/path/to/graded/mmvet_predictions.json
```

- Omit unavailable scoring options; those scores stay pending.
- `--textvqa-annotations` supports older outputs without embedded references.
- `--bench-dir` can locate TextVQA annotations automatically.
- `--output-dir` optionally selects a new, unused report folder.

## Compare methods and repeats

```bash
python -m vispec.evaluation.compare_metrics \
  --original /absolute/path/to/original/report1 /absolute/path/to/original/report2 \
  --draft /absolute/path/to/draft/report1 /absolute/path/to/draft/report2 \
  --output-dir /absolute/path/to/new/comparison
```

- Inputs are report folders containing `summary.json` and `samples.jsonl`.
- One report per method is enough; repeat variability requires independent runs.
- Model, temperature, precision, GPU identity, allocation, and prompt-token fingerprints must match.
- Different draft-tree settings are separate experiments, not repeats.
- Ratios use matching question/choice/turn IDs; repeated runs are weighted equally.
- `comparison.md` and `comparison.json` include latency/TTFT ratios, agreement, quality deltas, and repeat variation.
- Quality deltas require complete matching score coverage and scoring protocols/judge settings.
- Controlled speedup remains pending for incomplete runs, unknown stop reasons, or truncation.
- Original and draft generation limits remain inherited and different; inspect lengths and stop reasons.
- Token fingerprints verify prompt IDs; model/data revisions remain unpinned, so retain dataset/cache revision notes.
- Sample token counts, compare answer quality, and repeat runs before drawing conclusions.
- Record verified results in each experiment README and the register, with hardware and artifact links.
