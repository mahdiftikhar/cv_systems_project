# Metrics and scoring

We measure whether ViSpec finishes image-question answers faster, how much GPU
memory it needs, and whether its outputs preserve the baseline answers. Timing,
output agreement and correctness answer different questions, so the reports keep
them separate. The [comparison page](COMPARISON.md) shows the current measured
results and plots.

Each runner writes `summary.md`, `summary.json`, `samples.jsonl` and prediction
exports under a unique `runs/<run>/reports/<report>/` directory. Re-reporting
creates a new folder without replacing raw answers. Missing measurements remain
pending, with their coverage and reason recorded.

## What each metric means

A **token** is a unit of model output, often part of a word. A **turn** is one
answer to one question. **EOS** is the end-of-sequence token that signals a normal
finish. **Prefill** is the initial processing of the image and question before
answer generation. These runs have one measured turn per question.

### Completed questions / score coverage

Completed questions count how many expected questions produced saved answers. Score
coverage counts how many received quality grades. We record both to check that methods
cover the same workload and to make missing judging visible. Saving a response does not
establish that it is correct or complete.

### Mean

The mean is the average across measured questions, with each question contributing
equally to mean latency. It summarizes the average cost of generating an answer. Long
responses can raise it, so we also inspect the median and p95.

### Median

The median is the middle value after sorting the measurements. It describes a typical
question with less influence from unusually slow responses. Comparing it with the mean
helps reveal whether outliers drive the result.

### p95

The 95th percentile is a value at or below which about 95% of measurements fall. We use
it to inspect the slow end of the workload that an average can hide. It is not a
confidence interval or evidence of repeat-run reliability.

### Generation latency

Generation latency measures seconds from entering generation until the returned answer
is ready, including image/question processing and token generation. It excludes loading,
prompt construction, text conversion and scoring. This is our main measure of the wait
for a completed answer, with lower values preferable for comparable work.

### Time to first token (TTFT)

TTFT measures seconds until the first target-model token is ready. We record it to
separate the initial wait from answer completion: a method can finish sooner without
starting sooner. It measures token availability rather than the delay before streamed
text appears on screen.

### Recorded decode-loop time

Decode-loop time records the time spent inside the generation loop. Experiment 00
measures its autoregressive loop, while 01 measures verification after initial drafting.
This helps locate where time is spent, but the intervals cover different work. Use total
generation latency for cross-method speed claims.

### Output throughput

Throughput is total returned token IDs divided by total generation seconds, in tokens
per second. It includes special tokens and uses the same measured turns in both totals.
We use it alongside output length to distinguish efficient generation from earlier
completion caused by shorter answers.

### Time per output token (TPOT)

TPOT estimates the average cost of continuing an answer after its first token: `(latency
− TTFT) / (output tokens − 1)` for answers longer than one token. Raw reports use
seconds and the comparison tables show milliseconds. Lower values indicate faster
continuation, but do not measure individual streamed-token delays.

### Returned output length

Output length counts returned token IDs, including special tokens such as EOS. It comes
from saved IDs rather than re-tokenizing answer text. We record it to check how much
generation each method performed, since different lengths can change latency without
changing decoding efficiency.

### Answer length

Answer length counts non-special tokens before the first EOS. It describes the amount of
answer content rather than control tokens. This helps identify verbosity and explain why
some questions take longer to answer.

### Accepted draft tokens per verification iteration

This counts draft-proposed tokens accepted in each target-model check and applies only
to speculative generation. We measure it to see whether the draft predicts useful
continuations. More accepted tokens can reduce the number of expensive target-model
checks.

### Output tokens per verification iteration

This reports accepted draft tokens plus the target token for each verification step. It
describes how far generation advances per check and connects draft acceptance to
efficiency. Final EOS or output-cap clipping can shorten the block that is actually
committed.

### Stop reason

Stop reason records whether generation ended at EOS, an output-token cap, an iteration
safety limit, or an unknown condition. It distinguishes answers the model chose to
finish from forced stops. Ending at EOS does not establish correctness.

### Truncation rate

Truncation rate is the percentage of known-stop answers that ended at a token or
iteration limit. Unknown stops are excluded from the percentage and remain unresolved.
We track truncation because a cut-off answer cannot establish a fair full-answer
speedup.

### Tokens after EOS

This counts returned tokens after the first end-of-sequence token. The intended value is
zero. It checks whether speculative blocks return extras after the answer should have
ended, which would distort output lengths and throughput.

### Token-counter mismatches

This counts saved token counters that disagree with the raw returned IDs, alongside the
number of turns checked. We use it to verify the counts behind throughput and length
measurements. A counting error could otherwise resemble a performance gain.

### Peak allocated GPU memory

Peak allocated memory is the highest live PyTorch GPU allocation after warmup, including
resident model and generation state. It is reported separately for each visible GPU in
GiB. We measure it to assess the GPU capacity required by each decoding method.

### Peak reserved GPU memory

Peak reserved memory is the highest amount held by the PyTorch allocator after warmup,
including cached blocks available for reuse. It helps explain the memory footprint
beyond live data. Neither reserved nor allocated peaks include all non-PyTorch memory or
model-loading peaks.

### TextVQA consensus score

TextVQA scoring compares a prediction with ten human answers using the benchmark
normalization and leave-one-out consensus rule. We report a percentage and the number of
questions scored. It measures correctness on text-reading questions, but a verbose
response can score zero even when it contains the short reference answer.

### GQA exact-match score

GQA scoring checks whether a prediction matches the reference after trimming surrounding
whitespace and lowercasing. We do not extract short answers from explanations. It
measures correctness under a strict answer format, so formatting failures require
separate interpretation from visual reasoning failures.

### MM-Vet judge score

The official MM-Vet judge gives reference-based grades for open-ended answers, averaged
over judge runs and questions. This supports answers that exact matching cannot assess
well. No official grades have been imported for the current pair, so the score remains
pending rather than zero.

### Exact token sequence agreement

This is the fraction of paired responses with identical generated IDs through first EOS,
including EOS. It tests whether greedy speculative decoding reproduces the baseline
sequence. Matching capped prefixes count as observed agreement, not evidence that
complete answers would match.

### Decoded answer agreement

This is the fraction of paired answers with matching text. The comparison tool ignores
whitespace differences, and our verified counts also hold for exact text matching. It
indicates how often users receive the same written response, but matching responses can
still be incorrect.

### Observed latency speedup

Observed speedup divides standard mean generation latency by ViSpec mean latency on
matching turns. A 2× ratio means half the generation time. It summarizes the timing
benefit, but our raw ratios do not establish equivalent-output speedup because outputs
and hardware bindings differ.

### Quality change

Quality change subtracts the baseline benchmark score from the ViSpec score, in
percentage points. It requires matching questions, scoring rules and complete score
coverage. We use it to check whether faster generation reduces correctness, although
equal zero strict scores do not prove preserved semantic quality.

### Independent runs / repeat variation

Independent runs are separate launches. Repeat variation is the sample standard
deviation of their per-run means, which checks reproducibility. It remains pending with
one run per method, and variation across questions cannot replace it.

### Greedy equivalence / validated speedup

Validation requires matching settings and hardware evidence, complete matching question
coverage, finished outputs and exact greedy token agreement. These conditions
distinguish an observed timing ratio from a demonstrated speedup with equivalent
outputs. They are not all satisfied for the current pair.

### Runtime energy / GPU utilization

Runtime energy measures energy consumed during evaluation, while utilization tracks how
busy the GPU is over time. They would help assess hardware efficiency beyond latency.
Neither was measured in these runs, and launch-time device snapshots cannot supply them.

## Calculation details

Reports include mean, median, p95 and sample counts for latency, TTFT, decode time,
TPOT, lengths and acceptance. Percentiles use linear interpolation at
`(n − 1) × quantile`. Suite statistics pool completed turns, while quality scores
remain separate for each benchmark. Throughput is a ratio of totals, and the
reports also retain per-answer throughput distributions.

Token counters are checked against returned IDs. Timing excludes model and data
loading, prompt preparation, answer text conversion, scoring and report probes.
GPU synchronization is included in timing. TTFT measures token readiness, and
TPOT is an average estimate rather than a streamed inter-token delay.

Memory peaks are reset after warmup. They include resident model and generation
state but exclude non-PyTorch allocations and model-loading peaks. Report peaks
separately for each visible GPU rather than adding them together. Launch reports
describe devices and allocations without measuring runtime energy or utilization.

## Accuracy prerequisites

Run `bash prepare.sh` on the cluster to prepare models, benchmark assets and
scoring dependencies. The benchmark-specific requirements are described below.
No judge calls or paid API requests run automatically.

### TextVQA

The validation data provides ten human reference answers per question. Preparation
downloads LLaVA's [m4c_evaluator.py](https://github.com/haotian-liu/LLaVA/blob/main/llava/eval/m4c_evaluator.py)
to `tools/downloaded/`, and runners discover it automatically. The scorer uses
`EvalAIAnswerProcessor` normalization with leave-one-out VQA consensus. Set
`TEXTVQA_EVALUATOR` to use a custom copy.

The inherited prompt requests OCR explanations. Those responses can fail strict
answer scoring even when they contain a correct short answer. Report these
scores as subset measurements and retain the original predictions.

### GQA

Preparation downloads matching test-dev annotations, which runners discover
automatically. If the input JSONL lacks labels, set `GQA_ANNOTATIONS` to a custom
annotation file. Accepted formats are question JSON keyed by question ID or
JSON/JSONL rows with IDs and an `answer` field. Scoring trims whitespace and
lowercases predictions but does not extract answers from explanations.

### MM-Vet

The report exports `mmvet_predictions.json` for the
[official judge evaluator](https://github.com/yuweihao/MM-Vet/blob/main/mm-vet_evaluator.py).
Import the per-question grade JSON together with the exact predictions file used
for grading. Predictions must match the saved answers. Grade records need `score`
values in `[0,1]` and a `model` ID for every judge run. Scores are averaged over
judge runs per question, then across questions.

All scoring reports retain source-file and dependency SHA-256 hashes. Coverage is
explicit, so an ungraded or partially graded subset is not presented as a complete
benchmark score.

## Run with local accuracy scoring

The word "local" here means reference-based scoring without a judge API call.
The command launches model generation and belongs on a cluster GPU allocation:

```bash
TEXTVQA_EVALUATOR=/absolute/path/to/m4c_evaluator.py \
GQA_ANNOTATIONS=/absolute/path/to/testdev_balanced_questions.json \
BENCH_DIR=/absolute/path/to/bench_data \
  bash experiment/00_baseline_original/run.sh
```

Use the same environment variables with experiment 01. Missing scoring prerequisites
do not block performance reports, but invalid scoring files fail reporting.
Generation failures retain their original exit status, and reporting is attempted
after failed or interrupted evaluation to preserve partial coverage.

## Re-score saved answers

This reads saved outputs without running model generation:

```bash
python -m vispec.evaluation.report_metrics \
  --run-dir experiment/01_baseline_draft/runs/<run> \
  --textvqa-evaluator /absolute/path/to/m4c_evaluator.py \
  --gqa-annotations /absolute/path/to/testdev_balanced_questions.json \
  --mmvet-grades /absolute/path/to/model_judge-grade-5runs.json \
  --mmvet-graded-predictions /absolute/path/to/graded/mmvet_predictions.json
```

Omit unavailable scoring options to leave those scores pending. Additional options
support inputs with missing references or a different report destination:

- `--textvqa-annotations` supplies references when outputs do not embed them.
- `--bench-dir` locates TextVQA annotations automatically.
- `--output-dir` selects a new, unused report folder.

## Compare methods and repeats

Pass report folders containing `summary.json` and `samples.jsonl`. One report per
method produces a comparison, while repeat variation requires independent launches:

```bash
python -m vispec.evaluation.compare_metrics \
  --original /absolute/path/to/original/report1 /absolute/path/to/original/report2 \
  --draft /absolute/path/to/draft/report1 /absolute/path/to/draft/report2 \
  --output-dir /absolute/path/to/new/comparison
```

The tool requires matching model, temperature, precision, GPU identity, allocation
and prompt-token fingerprints. Different draft-tree settings are separate
experiments rather than repeats. Ratios use matching question/choice/turn IDs,
and repeated runs receive equal weight. Quality changes require complete matching
score coverage and the same scoring protocols and judge settings.

The generated `comparison.md` and `comparison.json` retain latency/TTFT ratios,
agreement, quality changes and repeat variation. The `token_agreement` field
records checked, matched and missing pairs. For each mismatching turn it records
the first differing zero-based token index, IDs, lengths and report indices.
A null ID means that sequence ended. Each original/draft run pair is checked on
its common turns.

Token IDs, EOS IDs and prompt fingerprints are stored before text cleanup.
Missing raw IDs leave token validation pending, and re-tokenizing cleaned answers
cannot replace that evidence. Fingerprints verify prompt IDs, while unpinned
model/data revisions still require retained dataset and cache revision notes.

Validated greedy speedup requires complete matching coverage, settings and
hardware evidence, EOS completion and exact token agreement on every pair.
Incomplete runs, unknown stops, truncation, missing or different caps, and missing
or differing token evidence keep validation pending. Raw latency ratios can still
be reported with these limits. Agreement at nonzero temperature is descriptive
and does not validate a sampling distribution.

Record verified per-benchmark results in the experiment guides and register,
with hardware and artifact links. Inspect output counts and quality and repeat
runs before drawing broader conclusions.
