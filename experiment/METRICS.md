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

### Phase memory and live KV (`memory` block)

Each benchmark in `summary.json` has a `memory` block with up to five fields: `prefill_peak_allocated_gib`
(GiB, bytes / 2**30), `decode_peak_allocated_gib` (GiB), `post_compaction_peak_allocated_gib` (GiB,
compact-layout records from Week 3 on, see [Compact KV layout](#compact-kv-layout-experiments-05-and-06)),
`kv_live_mib` (MiB, bytes / 2**20) and `kv_tokens` (tokens). Each field is a distribution over turns
(`count`, `mean`, `median`, `p95`, `stdev`) plus `max`. The three peaks come from the per-device
`gpu_memory` entries and take the largest value across visible devices for a turn. `kv_live_bytes` and `kv_tokens` are
whole-model figures repeated on every device entry, so each is counted once per turn. Turns without
a field (LLaVA runs, older records, or a null phase peak) are excluded and `count` shows how many
remain. A field with no turns is omitted, so a LLaVA report has an empty `memory` block.
`gpu_memory_by_visible_device` is unchanged. `comparison.json` adds, per benchmark,
`memory_ratio_draft_over_original` with `decode_peak_allocated` and `kv_live` entries. Each is the
draft mean divided by the original mean (per-run means over matched turns, averaged across runs),
or null when a side has no data.

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
and prompt-token fingerprints. It also refuses reports whose prompt style or run
settings differ, as described in the [Qwen section](#qwen-experiments-02-and-03).
Different draft-tree settings are separate experiments rather than repeats. Ratios use matching question/choice/turn IDs,
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

### Sparsity mode

`--mode sparsity` compares a sparsity arm (`--draft`, the candidate) with the same
decoding method at 100% retention (`--original`, the reference). The default,
`--mode method`, is the autoregressive vs ViSpec comparison above. The settings
below are read from each benchmark's `runtime.json` `generation_args`. An absent key
and an explicit null mean the same thing, "not set", and that value differs from any
explicit value.

- **Method mode:** `kv_layout`, `kv_retention`, `kv_policy`, `draft_retention` and
  `draft_policy` must match across every compared report. Week 1 reports have none of
  these keys, so they still compare. A 1.0 arm does not pair with a no-flag report here.
- **Sparsity mode:** both groups must have the same `method` (autoregressive or
  `vispec_speculative`). `kv_layout`, the draft-tree settings, `prompt_style` and all
  the other guarded settings must match. The four sparsity settings may differ only
  between the groups, never within one.
- **Reference rule:** the reference must have `kv_retention` and `draft_retention` each
  equal to 1.0 or null (null counts as 1.0, since 1.0 arms are token-identical to the
  no-flag run). The reference may carry an explicit policy, so a 0.5 arm can be
  compared for latency with the same-policy 1.0 arm. The no-flag reference anchors the
  token-identity check.
- **Same Slurm job:** every report in both groups must have a non-empty `slurm_job_id`
  in its `metadata.txt` (the placeholder `unset` counts as missing), all equal, and equal
  `slurm_restart_count` values (absent counts as 0). Otherwise `compare_metrics` raises
  `ValueError`. Latency and memory ratios between arms are only controlled within one
  allocation: the runtime signature (GPU, affinity, revision) can match across jobs.
  Method mode does not apply this check, because the LLaVA 00/01 pair spans two jobs.
- **Runtime evidence (Week 3 on):** for every benchmark that any compared report
  evaluated (it has a runtime file, a manifest or answers), every report must have that
  benchmark's runtime entry with `generation_args`. Otherwise `compare_metrics` raises
  `ValueError` naming the report. Before Week 3 such a side's settings were left null in
  the `sparsity` block. Benchmarks that no report evaluated (the empty `gqa` and `mmvet`
  entries of a Qwen report) are not checked.
- **Requeued jobs:** a requeued preempt job reruns the runner into fresh folders, so one
  `slurm_job_id` can own several folders per arm, one per `slurm_restart_count`. Take the
  final attempt: the folders with the highest `slurm_restart_count` of that job. The
  same-job rule above then holds, because all of them share that restart count.
- **Output:** `comparison.json` gains a `mode` field and a `sparsity` block with each
  side's settings per benchmark. Other keys are unchanged. In sparsity mode the
  `original` columns are the reference and the `draft` columns the candidate.

### Diagnosing divergences

`diagnose_divergence` teacher-forces the stock full-KV Hugging Face model, which
cannot reproduce target-side KV eviction. It refuses any comparison in which either
run has a non-null `kv_retention` below 1.0, in both modes, before it loads a model,
and it labels no divergence. Draft-only arms (`draft_retention` below 1.0 with
`kv_retention` null or 1.0) stay diagnosable because the target model is unchanged.
For a sparsity-mode comparison, `--original-run` is the reference and `--draft-run`
the candidate, and both must share one method; the "ar" and "vispec" labels in the
output then mean reference and candidate.

The refusal fails closed (Week 3 on). `kv_retention` comes from each benchmark's
`runtime.json` `generation_args`, where null or absent means no eviction. When a run has
no `runtime.json`, or an answer file has none beside it, the tool also reads
`kv_retention` from the run's `metadata.txt` (`none` means no eviction). When neither
source gives a value (no `metadata.txt`, or no `kv_retention` key in it, as in a 02 or 03
reference) or the value is not a number, the run is refused as well, because its eviction
setting is unknown.

### Offset check (`offset_check`)

`diagnose_divergence` cannot check an eviction arm, so `offset_check` is the only oracle
for large `kv_offset` values. Greedy ViSpec commits only tokens that the target itself
picks in tree verification, so an autoregressive run and a ViSpec run with the same
target-side eviction must return the same tokens, whatever the draft proposes. Draft-side
pruning changes only the proposals, so the agreement must also hold in a combined arm. In
bf16 a different kernel shape can flip a near-tied argmax. In fp32 with TF32 off those
flips go away (Week 2: 5/5 at 0.5 spatial), so an fp32 mismatch points at the cache
bookkeeping: the offset, the decode positions or the kept rows.

```bash
python -m vispec.evaluation.offset_check --kv-retention 0.25 --kv-policy spatial \
  [--draft-retention 0.25 --draft-policy spatial] --num-samples 5 --dtype float32 \
  --output-dir <new_dir>
```

It loads one `SpecModel` with `--kv-layout compact` in `--dtype` (long TextVQA by
default) and runs `autoregressive_generate` and `specgenerate` on each sample. It writes
`offset_check.json` and `offset_check.md` to the new folder. Per question they record exact
agreement of the raw output IDs through EOS, the first differing zero-based index when the
IDs differ, the eviction counts of both runs (`kv_counts_match`), `kv_offset` and the
draft-pruning counts. The exit code is 0 only when every question agrees and both methods
report the same eviction counts. The fp32 weights of Qwen2.5-VL-3B take about 14 GiB
(twice the 7.0 GiB of bf16 checkpoint), plus the draft and activations, so the check fits
one A6000.

## Qwen experiments (02 and 03)

Experiments 02 and 03 use the unified evaluator `vispec.evaluation.evaluate` and
add the measurements below. The definitions above still apply. These fields are
saved per question in `results/*/test.jsonl`. `report_metrics` aggregates the phase peaks
(GiB, largest over devices per question), `kv_tokens` and live KV (MiB) into a per-benchmark
`memory` block of `summary.json` (distribution plus `max`, over the questions that carry the
field; `count` shows how many). The KV statements below
are scoped to the `fixed` layout (experiments 02 and 03) unless they say `compact`.

### CIDEr-D

COCO caption quality is CIDEr-D multiplied by 100, computed with `pycocoevalcap`
1.2 and its PTB tokenizer, which needs Java. The inverse document frequency comes
from the references of the evaluated 100-image subset, not the full corpus. The
score is therefore a subset measurement and is not comparable to published
full-corpus CIDEr. Without Java the score stays pending.

### Prefill and decode phase peaks

The evaluator splits the allocator peak at the first output token. The
prefill-phase peak (`prefill_peak_allocated_bytes`) runs from the start of the
generation call to that token, and the counters are then reset. The decode-phase
peak (`decode_peak_allocated_bytes`) covers the rest of the call, and
`peak_allocated_bytes` stays the larger of the two. For ViSpec the first output
token comes from the target prefill, before the draft prefill and the first
draft tree, so those count toward the decode-phase peak.

### Live KV (`kv_tokens`, `kv_live_bytes`)

`kv_tokens` is the target KV cache length when generation ends. In the `fixed` layout
it equals input plus output tokens minus one for the autoregressive arm, because the
last generated token is never fed back, and input plus output tokens for ViSpec.
`kv_live_bytes` is `kv_tokens` times the bytes one token occupies across all
decoder layers (layers x 2 x KV heads x head dimension x dtype bytes). It is
arithmetic, not an allocator measurement. It is a whole-model figure, so each
device entry repeats it and it must not be summed across GPUs. In the `fixed` layout
the preallocated buffer holds 16,384 tokens whatever `kv_tokens` is, so live KV does
not change the measured peaks.

### Compact KV layout (experiments 05 and 06)

`--kv-layout compact` sizes the target KV buffer per sample, so live KV can change the
measured peaks. Per sample (`T` = `total_token`; the tree adds `T` slots plus one):

- Prefill buffer: prompt tokens + 10 slots for the autoregressive arm, prompt tokens +
  `T` + 1 + 10 for ViSpec.
- After prefill the cache is compacted into a new buffer of capacity kept prompt tokens
  + `max_new_tokens` + `T` + 1 + 10 slots (`kv_cache.compact_for_decode`). The kept prompt
  is every prompt token minus the evicted image tokens, so a 1.0 arm keeps the whole
  prompt. The same slack applies to both arms, so AR and ViSpec buffers match.
- The `fixed` rule that live KV cannot move the peaks no longer holds.

`kv_tokens` under the compact layout is the physical cache length at the end: kept
prompt tokens + output tokens (minus 1 for the autoregressive arm), not input + output.
Eviction lowers it by `kv_evicted_tokens`, so `kv_tokens` = (input + output [- 1]) -
`kv_evicted_tokens`. For example, long TextVQA ViSpec at 0.5 spatial: 1062.9 - 489.2 +
325.93 = 899.61 (mean input, mean evicted, mean output; see the 06 README).

The decode-phase peak includes the compaction transient, when the prefill-sized
buffer and the new buffer exist together. Steady-state decoding holds only the new
buffer. The transient is most visible for the autoregressive arm, whose decode peak is
then set by compaction and not by decoding. `decode_time` also includes compaction and
selection of the kept positions in these arms.

From Week 3 on, the evaluator also splits the decode phase at the compaction. Right after
the compaction has released the prefill-sized buffer, and before decoding goes on, it keeps
each device's peak so far and resets the peak counters (`on_kv_compacted`, then
`GenerationMeasurement.mark_compaction`). The per-device field
`post_compaction_peak_allocated_bytes` is the allocator peak from that reset to the end of
the call. It measures steady-state decoding with the compacted buffer, without the
transient, so it shows what eviction saves during decoding. The peak kept before the reset
is folded back in at finish. `decode_peak_allocated_bytes` therefore keeps its Week 2
meaning, the peak of the whole decode phase including the transient, and Week 2 and Week 3
decode peaks stay comparable. `peak_allocated_bytes` and `peak_reserved_bytes` stay
whole-call peaks for the same reason. The post-compaction peak is never above the decode
peak, because its window is part of the decode phase.

`post_compaction_peak_allocated_bytes` appears in every compact-layout record, including
the no-flag compact references and the 1.0 arms, because each of them compacts. A
reference and an arm then have the same field, so it can be compared between them.
Fixed-layout records never compact and do not have the field, so they are unchanged.
Records written before Week 3 do not have it either. `report_metrics` reports it as
`post_compaction_peak_allocated_gib` in the `memory` block, over the turns that have it.

Memory ratios stay valid under truncation. The decode buffer is sized from the kept prompt
plus `max_new_tokens` plus slack before the first decode step, so a turn that stops at the
1,024-token cap fills that buffer but cannot grow it. The decode and post-compaction peaks
therefore depend on the kept prompt, not on where the answer stopped. Only the per-layer
attention temporaries grow with the cache length: by shape arithmetic, the repeated K and V
heads take about 8 KiB per cached token, so about 8 MiB over a full 1,024-token answer. A memory ratio between two
arms stays a measurement of eviction even when one arm truncates a turn and the other does
not. `kv_live_mib` and `kv_tokens` do grow with the output length (they include the output
tokens), so read a live-KV ratio together with the output lengths. Latency and TPOT ratios
are the ones that truncation confounds.

The `spatial` policy (both sides) is a 1-D raster stride over each image's merged tokens,
not a 2-D pool. In each image span it keeps `round(linspace(0, n - 1, k))`, so the kept
tokens are evenly spaced in row-major order. At 0.5 it keeps every other token in raster
order: every other column when the merged grid width is even, and a checkerboard when it
is odd. Rounding absorbs the remainder in a few steps that are one token longer (one per
image at 0.5, three at 0.25), which shifts the pattern by a column there. This is the Week 2
selector. Week 3 keeps it unchanged so that Week 2 and Week 3 arms stay comparable.

Per-question fields written by the eviction and draft-pruning arms (all counts of
tokens, no units beyond tokens; absent in other runs):

- `kv_prompt_tokens`: prompt tokens in the target KV cache before eviction (the input
  length, text and image).
- `kv_kept_prompt_tokens`: prompt tokens still in the cache after eviction.
- `kv_evicted_tokens`: `kv_prompt_tokens` - `kv_kept_prompt_tokens`. This is the
  `kv_offset` of ViSpec decoding: the token count minus the physical cache length.
- `kv_score_window`: prompt tokens after the last image token (the chat-template tail);
  0 without image tokens. It is recorded for both policies, but only the `attention`
  policy uses it: that policy scores image tokens by the prefill attention these tokens
  pay them. The `spatial` policy ignores it.
- `kv_image_tokens` (Week 3 on): image tokens in the prompt before eviction, summed over
  the prompt's images.
- `kv_min_kept_per_image` (Week 3 on): the fewest tokens kept in the target cache for any
  one image, counted from the actual kept positions in each image span. Every image keeps
  at least one token. Null for a prompt without image tokens.
- `draft_image_tokens`: image tokens the draft's image adaptor received, summed over
  the prompt's images, before draft-side pruning.
- `draft_kept_image_tokens`: image tokens the adaptor saw after pruning, summed the
  same way. Equal to `draft_image_tokens` at retention 1.0.
- `draft_min_kept_per_image` (Week 3 on): the fewest tokens the adaptor saw for any one
  image, counted from the rows it actually received. Null when no image was adapted.

Records without the eviction or draft flags carry none of these fields. Their other fields
are unchanged, except that compact-layout records gain `post_compaction_peak_allocated_bytes`
(see above). Two guards fail closed: `select_in_span` raises `ValueError` unless
`1 <= k <= n`, and the draft's attention scores raise `RuntimeError` when any score is not
finite. The target's attention scores are checked against their expected sum.

### Decode time

The autoregressive `decode_time` runs from the first output token to the end of
the decoding loop. ViSpec's `decode_time` starts after `initialize_tree` returns,
while TTFT ends at the first target token inside it. The draft prefill and the
first draft tree are therefore in neither TTFT nor `decode_time`; in Week 1 that
window averaged about 12 ms per question. Use latency and TTFT to compare the
arms. In the compact layout `decode_time` also contains compaction and selection
(see above).

### Comparison refusals

`compare_metrics` raises `ValueError`, naming the differing values, when the
compared reports differ in any of these:

- `prompt_style` in the run metadata.
- `slim_prefill`, `kv_capacity`, `num_samples` or `max_new_tokens` in the run
  metadata.
- `slim_prefill` or `kv_capacity` in a benchmark's runtime file, among reports
  that have one. A missing runtime file is a warning instead.
- The per-turn `max_new_tokens` generation limit, among turns that record one.

An absent value counts as its own value, so a report that records a setting
cannot be paired with one that does not. LLaVA reports record only
`max_new_tokens`, so LLaVA pairs still compare. The hardware check also covers
`base_model_revision`. From commit `25bf388` on, the evaluator falls back to the
revision pinned in `.preparation/assets_qwen.json` when the checkpoint is a local
folder, while the Week 1 runtime files hold `null`. A new run therefore cannot be
paired with a Week 1 run, which the GPU and allocation checks would usually
prevent anyway.
