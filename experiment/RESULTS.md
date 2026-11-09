# Results: ViSpec speculative decoding and visual sparsity on Qwen2.5-VL-3B

Status on 2026-11-09: Weeks 1 to 3 are measured. Week 4 (Re:Verse+ manga windows) is partial: 5 of its 14
run folders are complete, and the rest wait for GPU time. Experiments 00 and 01 (LLaVA-1.6) are an
earlier, separate study and are summarised in the [register](README.md), not here.

**How to read this page.**

- **Pending.** "Pending GPU run (job N)" marks a value that waits for queued Slurm job N.
  "Pending GPU run (no job yet)" marks a check that has not been submitted.
- **Speedups** are the reference latency divided by the arm latency, so a value above 1 means the arm (or
  ViSpec) was faster. "Speedup over AR" divides the autoregressive (AR) latency by the ViSpec latency.
  In sweep tables, these are the `*_speedup_vs_*` columns.
- **Memory ratios** are the arm divided by the reference, so a value below 1 means the arm used less
  memory. In sweep tables, these are the `*_arm_over_*` columns.
- **Timing is compared only inside one Slurm job, never across jobs.** Quality scores, output agreement
  and acceptance are compared across jobs.
- **Exact agreement** counts outputs whose generated token IDs are identical. A bf16 near-tie counts as
  a disagreement. Because ViSpec and AR outputs differ in bf16, every speedup here is an observed ratio,
  not a validated equal-output speedup, and its agreement and quality columns sit next to it.
- **Link targets.** Run folders and `experiment/sweeps/` are git-ignored and kept on project storage, so
  their links resolve only in a checkout that has them.

**Setup.** The target is `Qwen/Qwen2.5-VL-3B-Instruct` and the draft is `JLKang/ViSpec-Qwen2.5-VL-3B-Instruct`.

- **Decoding:** bf16, greedy decoding (temperature 0), batch size 1, at most 1,024 new tokens.
- **Draft tree:** depth 3, top-k 8, total-token 30, `num_q` 2.
- **Hardware:** one pinned 48 GB `NVIDIA RTX A6000` per job.
- **Benchmarks:**
  - 100 TextVQA questions and 100 COCO Captions images (Karpathy test), each in two prompt styles. The
    `long` style measures speed. The `short` style measures quality.
  - 55 Re:Verse+ windows of 5 or 6 manga pages, in the `long` style only.
- **Repeats:** one run per arm, so run-to-run spread is unknown.

## 1. The question, and the answer in brief

The project asks: **can dynamic visual sparsity make ViSpec faster and leaner without hurting task quality?**
We answer it in two parts. The first part is what speculative decoding itself delivers. The second part is
what visual sparsity adds on top of it.

1. **ViSpec is 2.13 to 2.32 times faster than AR on long answers** (TextVQA and COCO, `long` style, Weeks 1
   and 3). It is 1.46 to 1.49 times faster on short COCO captions. It is not faster on 5-token TextVQA
   answers (0.96 to 0.98), because prompt processing is most of their time.
2. **ViSpec matched AR in every fp32 check, and in bf16 they differ at near-ties.**
   - **fp32:** ViSpec and AR gave identical tokens in 15 of 15 checks. These are the first 5 long TextVQA
     questions at each of three eviction settings. No fp32 check without eviction is recorded, and the
     Re:Verse+ check is pending.
   - **bf16:** they agree on 22 to 28 of 100 long answers and 88 to 96 of 100 short answers. Every
     AR-against-ViSpec divergence diagnosed so far (Week 1, 166 cases) is a near-tie between the top two
     logits (section 2.2).
3. **On 5- to 6-page manga windows, ViSpec is 1.74 times faster end to end** (section 2.3).
   - **Decode phase:** it gains about 1.9 to 2.0 times when the length cap does not distort the
     comparison. The figure is 1.90 over the 42 windows where neither method hit the 1,024-token cap, and
     1.98 over the 47 windows where the cap was not one-sided.
   - **Against single images:** that is below the 2.28 to 2.48 times on long TextVQA and COCO (the same
     below-cap figure, Week 3 jobs). It fits the lower acceptance: 1.70 against 2.14 and 2.28 accepted
     draft tokens per iteration.
   - **Prompt processing:** it dilutes the gain further. Time to first token (TTFT), which includes encoding
     the pages, is 6.65 to 6.66 s per window. That is 44% of ViSpec's 15.11 s and 25% of AR's 26.24 s, so
     even infinitely fast decoding would give at most about 3.9 times over AR.
4. **Visual sparsity barely helps at batch 1.**
   - **What it does:** it shrinks the live KV cache and, at low budgets, costs some quality and most
     long-answer agreement.
   - **Speed at batch 1:** it did not make decode steps cheaper. It removes little of each step's cost, and
     decoding is host-bound. End-to-end latency ratios against the 1.0 arm ranged from 0.875 (the arm took
     14% longer) to 1.130 (it took 12% less time, on long COCO). These follow output length and, we infer, a
     rise in acceptance.
   - **Why:** section 3 gives the explanation. This is an explanation of the measurements, not a new method.

## 2. Speculative decoding: ViSpec against AR

### 2.1 Every job, per benchmark

Each row compares the AR and ViSpec references of one Slurm job. Speedup over AR is AR latency divided by
ViSpec latency (above 1: ViSpec faster). The TTFT share is ViSpec's mean TTFT divided by its mean latency.
Quality is TextVQA VQA accuracy (%), COCO CIDEr-D x 100, or Re:Verse+ ROUGE-1 F1 (x 100), each as AR / ViSpec.
`long` quality is near zero for TextVQA and COCO, because strict scoring rejects paragraph answers. For those
rows the `long` style is judged by speed, acceptance and agreement.

| Week | Style | Benchmark | Job | n | AR latency (s) | ViSpec latency (s) | Speedup over AR | ViSpec TTFT share | Accepted draft tokens per iteration | Exact agreement | Quality AR / ViSpec |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | long | TextVQA | 392660 | 100 | 11.538 | 5.366 | 2.150 | 17.7% | 2.141 | 22/100 | 0.0 / 0.0 |
| 1 | long | COCO Captions | 392660 | 100 | 4.318 | 2.030 | 2.127 | 11.3% | 2.284 | 28/100 | near zero |
| 1 | short | TextVQA | 392661 | 100 | 1.108 | 1.159 | 0.956 | 84.2% | 0.690 | 96/100 | 71.0 / 70.0 |
| 1 | short | COCO Captions | 392661 | 100 | 0.754 | 0.516 | 1.461 | 43.8% | 1.746 | 88/100 | 121.40 / 119.68 |
| 3 | long | TextVQA | 398883 | 100 | 12.353 | 5.615 | 2.200 | 16.7% | 2.141 | 22/100 | 0.0 / 0.0 |
| 3 | long | TextVQA | 398884 | 100 | 12.485 | 5.700 | 2.190 | 16.5% | 2.141 | 22/100 | 0.0 / 0.0 |
| 3 | long | TextVQA | 398885 | 100 | 12.185 | 5.647 | 2.158 | 16.6% | 2.141 | 22/100 | 0.0 / 0.0 |
| 3 | long | COCO Captions | 398883 | 100 | 4.544 | 1.981 | 2.294 | 11.2% | 2.284 | 28/100 | near zero |
| 3 | long | COCO Captions | 398884 | 100 | 4.697 | 2.039 | 2.304 | 10.9% | 2.284 | 28/100 | near zero |
| 3 | long | COCO Captions | 398885 | 100 | 4.584 | 1.980 | 2.315 | 11.2% | 2.284 | 28/100 | near zero |
| 3 | short | TextVQA | 398923 | 100 | 1.100 | 1.122 | 0.981 | 83.8% | 0.690 | 96/100 | 71.0 / 70.0 |
| 3 | short | COCO Captions | 398923 | 100 | 0.784 | 0.528 | 1.485 | 42.0% | 1.746 | 88/100 | 121.40 / 119.68 |
| 4 | long | Re:Verse+ windows | 398924 | 55 | 26.237 | 15.105 | 1.737 | 44.1% | 1.702 | 0/55 | 22.92 / 22.58 |

Sources: Week 1 is from the [02](02_qwen_autoregressive/README.md#results) and [03](03_qwen_vispec/README.md#results) READMEs.
Week 3 is from the per-job sweep tables `experiment/sweeps/20261105T192610Z_week3_*` (linked from the
[register](README.md#qwen-week-3-results-retention-sweeps-05-06-and-08)). Week 4 is from
`experiment/sweeps/20261109T181008Z_week4_reverse_partial_job398924/sweep.json`. The Week 2 jobs (393336 and
393337) reproduced the Week 1 tokens on 100/100 per benchmark and style, so they are not listed again.
Accepted draft tokens are identical across jobs because greedy outputs are reproduced.

### 2.2 Exactness

- **In fp32, ViSpec matched AR in every check (15 of 15).** Greedy ViSpec commits only tokens that the
  target picks, so it should equal AR.
  - **The checks:** the fp32 offset check (GPU job 402165) decoded both methods in fp32 with TF32 off. It
    covered the first 5 long TextVQA questions at each of three settings: kv 0.25 spatial, kv 0.75 spatial
    and combined 0.25 spatial. AR equalled ViSpec on 5 of 5 in each setting
    ([06](06_qwen_kv_eviction/README.md#fp32-offset-oracle-exactness-check) and
    [08](08_qwen_combined_sparsity/README.md#fp32-offset-oracle-exactness-check) READMEs).
  - **The scope:** every check had eviction on. No fp32 decode without eviction is recorded, and none on
    COCO or on Re:Verse+ windows.
- **In bf16, AR and ViSpec differ at near-ties.** The Week 1 diagnosis found every AR-against-ViSpec
  divergence to be a near-tie, with no exception: 150 of 150 in `long` and 16 of 16 in `short`, with a bf16
  top-2 margin of at most 0.375 logits ([03 README](03_qwen_vispec/README.md#results)). The most likely
  mechanism is not measured: batched tree verification and one-token decoding use different kernel shapes,
  so tied logits can round differently.
- **The draft-pruning arms have one exception, traced to the cached decode.** This is a separate
  comparison: each draft-only arm (05) against the ViSpec reference at retention 1.0.
  - **The exception:** every divergence there is a near-tie except one, question 35537 at output index 162
    (stock bf16 margin 0.625 logits).
  - **Experiment 07:** it compared the fork's cached AR decode with stock Hugging Face greedy decoding. They
    agree on 19 of 100 long TextVQA answers, and 80 of the 81 first mismatches are near-ties. At 35537:162
    the fork's two top logits tie exactly.
  - **Judgement:** it was judged bf16 implementation noise, a known limitation
    ([07 README](07_qwen_decode_drift/README.md#results)).
- **On Re:Verse+ windows**, ViSpec and AR agree on 0 of 55 summaries in bf16. The prompts are identical
  (55 of 55 input fingerprints match), and the first mismatch falls at output index 7 to 443 (median 58).
  The fp32 offset check and the divergence diagnosis for these windows are **Pending GPU run (no job yet)**.

### 2.3 Long multi-page prompts: prompt processing caps the speedup

Re:Verse+ windows are 5 or 6 manga pages given as images, and the model summarises the story. Prompts are
5,805 to 6,967 tokens (mean 6,496.6), of which 6,430.4 on average are image tokens (99%). Long TextVQA
prompts are about 1,063 tokens. The Re:Verse+ row below comes from the run files of job 398924 (restart 1),
and the other rows come from the Week 3 sweep tables. Each ratio is taken inside one job.

| Job | Benchmark | n | TTFT AR / ViSpec (s) | ViSpec TTFT share | Speedup over AR, end to end | Decode-only speedup, below the cap in both (n) | Decode-only speedup, all samples | Per-output-token speedup | Accepted draft tokens per iteration |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 398924 | Re:Verse+ windows | 55 | 6.654 / 6.665 | 44.1% | 1.737 | 1.895 (42) | 2.320 | 1.903 | 1.702 |
| 398883 | TextVQA long | 100 | 0.940 / 0.939 | 16.7% | 2.200 | 2.327 (96) | 2.441 | 2.248 | 2.141 |
| 398884 | TextVQA long | 100 | 0.937 / 0.941 | 16.5% | 2.190 | 2.311 (96) | 2.427 | 2.230 | 2.141 |
| 398885 | TextVQA long | 100 | 0.932 / 0.936 | 16.6% | 2.158 | 2.280 (96) | 2.389 | 2.200 | 2.141 |
| 398883 | COCO long | 100 | 0.223 / 0.221 | 11.2% | 2.294 | 2.457 (100) | 2.457 | 2.445 | 2.284 |
| 398884 | COCO long | 100 | 0.228 / 0.222 | 10.9% | 2.304 | 2.460 (100) | 2.460 | 2.456 | 2.284 |
| 398885 | COCO long | 100 | 0.222 / 0.221 | 11.2% | 2.315 | 2.479 (100) | 2.479 | 2.467 | 2.284 |

Exact agreement and quality for these rows are in the table of section 2.1. For Re:Verse+ they are 0 of 55
and ROUGE-1 22.92 (AR) against 22.58 (ViSpec).

Column definitions:

- **Decode-only speedup:** AR's mean (latency - TTFT) divided by ViSpec's mean (latency - TTFT).
  - **Below the cap in both:** only the samples where neither method stopped at the 1,024-token cap. This
    is the like-for-like figure.
  - **All samples:** includes samples where one method ran to the cap and the other stopped earlier.
- **Per-output-token speedup:** AR's mean time per output token divided by ViSpec's. Time per output token
  is (latency - TTFT) / (output tokens - 1), averaged over samples.
- **Direction:** in all three speedup columns, a value above 1 means ViSpec is faster.

What the Re:Verse+ windows show:

- **The end-to-end speedup is 1.74 times** (AR 26.24 s, ViSpec 15.11 s per window). On long TextVQA and
  COCO answers it is 2.13 to 2.32 times (Weeks 1 and 3).
- **Prompt processing is the cap.** TTFT is 6.65 s for AR and 6.66 s for ViSpec. That is 44% of ViSpec's
  window time and 25% of AR's (the mean of per-window shares is 47% and 30%). TTFT covers everything up to the
  first token, including encoding the 5 or 6 pages. With TTFT fixed, even infinitely fast decoding would bring a window down
  to 6.66 s, at most about 3.9 times faster than AR (26.24 / 6.66 = 3.94).
- **The decode phase gains about 1.9 to 2.0 times, less than on single images.**
  - **Like for like:** over the 42 windows where neither method hit the 1,024-token cap, AR's decode time
    is 1.90 times ViSpec's. Leaving out only the 8 windows where one method hit the cap and the other did
    not gives 1.98 (47 windows).
  - **Single images:** on long TextVQA and COCO the below-cap figure is 2.28 to 2.48 (all samples: 2.39 to
    2.48; Week 3 jobs).
  - **Per output token:** the gain is 1.90 on the windows, against 2.20 to 2.47 on single images.
  - **Acceptance:** the smaller gain fits the lower acceptance, 1.70 accepted draft tokens per iteration
    against 2.14 (TextVQA) and 2.28 (COCO).
  - **End to end:** prompt processing then dilutes the decode gain to 1.74.
- **The all-window decode figure, 2.32, is inflated by a few capped windows.** Over all 55 windows, AR's
  mean decode time is 2.32 times ViSpec's (19.58 s against 8.44 s). The 8 windows where only one method hit
  the cap carry about 95% of the length gap described next. Per window, the decode-only ratio has a median of
  2.02 and ranges from 0.68 to 8.21.
- **The length gap comes mostly from a few windows.** AR's summaries average 497.3 tokens and ViSpec's 424.4.
  - **How many differ:** 49 of the 55 windows differ in length and 6 are equal. AR is longer in 29 windows
    and ViSpec in 20.
  - **The cap:** AR hits the 1,024-token cap in 12 windows and ViSpec in 6. Both hit it in 5, AR alone in
    7 and ViSpec alone in 1.
  - **Where the gap is:** those 8 one-sided windows carry 3,816 of the 4,010-token total gap (95%). In the
    other 47 windows the gap is 194 tokens: AR is longer in 22, ViSpec in 19, and 6 are equal.
  - **Where the outputs split:** they first differ at output index 7 to 443 (median 58), never at the first
    token. That fits the known bf16 divergence between the two methods, which matched in every fp32 check
    so far (section 2.2; the Re:Verse+ fp32 check is pending). After the first difference the two summaries
    can run to different lengths.
  - **Not measured:** why AR reached the cap more often.
- **What is not measured.** These runs do not measure how TTFT splits between the vision encoder and the
  language model's prompt pass. We make no claim about that split.
- **Future work, not done here.** Cutting image tokens before or early in prompt processing is the lever
  that would move this cap. Prior art includes FastV (Chen et al., 2024), VisionZip (Yang et al., 2024) and
  PyramidDrop (Xing et al., 2024). The eviction tested here runs after prompt processing, so it cannot change
  TTFT.

## 3. Visual sparsity: why it barely helps at batch 1

Two forms were tested.

- **Draft pruning (05):** it prunes the image features that the draft's image adaptor reads.
- **Target KV eviction (06):** it drops visual entries from the target's KV cache once, after prompt
  processing.

Experiment 08 applies both at once. At batch 1 neither form made decode steps cheaper, for four reasons.
Each reason says which part is measured and which part is inferred.

1. **The draft already compresses each image to 2 learned queries.**
   - **The mechanism:** ViSpec's image adaptor replaces each image's tokens with `num_q` = 2 query outputs
     before the draft sees them. A 1,148-token page reaches the draft as 2 positions, so draft pruning has
     almost nothing to cut.
   - **Measured, Week 3:** draft pruning kept acceptance within -0.015 to +0.076 of the 1.0 arm. The
     largest rises come from one capped long COCO answer. Latency
     ratios (same-policy 1.0 arm over arm) were 0.976 to 1.037. Live KV did not move (arm over 1.0 arm,
     0.995 to 1.012).
2. **The KV cache is a small part of each decode step's memory traffic.**
   - **Cache size:** the cache holds 36,864 bytes per token (36 layers x 2 for K and V x 2 KV heads x 128 x
     2 bytes).
   - **Weight read:** every decode step reads about 5.75 GiB of language-model weights. This is measured
     from the checkpoint files. The 6.99 GiB checkpoint also holds a 1.25 GiB vision encoder, which decode
     steps do not read.
   - **The ratio:** the cache is 0.6% of the per-step weight read at 1,000 tokens and 4.2% at 7,000 tokens.
     The two are equal only at about 167,000 tokens. On the Re:Verse+ windows the live cache is 243 MiB,
     about 4.1% of the weight read.
   - **Consequence:** at these context lengths, evicting even every image entry removes at most about 4% of
     the bytes a step reads.
3. **Decode is host-bound.**
   - **Measured (experiment 04):** the GPU is busy only 36 to 50% of each decode step (estimated
     unprofiled), and a step costs 21 to 23 µs per kernel launched.
   - **Inference, not measured:** shortening GPU work that is off the critical path barely changes step
     time. A decode profile at Re:Verse+ context length is not measured.
4. **Eviction runs after prompt processing**, so it cannot cut TTFT. TTFT ratios (1.0 arm over arm) stayed
   within 0.986 to 1.013 in Week 3. On Re:Verse+, TTFT is 44% of ViSpec's time, and eviction cannot reach it.

So at batch 1, visual sparsity mainly trades output agreement, and at low budgets quality, for memory.

**End-to-end latency** moved from 0.875 to 1.130 times the same-policy 1.0 arm in Week 3 (1.0 arm latency over
arm latency). The two ends are these:

- **Slow end, 0.875:** long TextVQA kv 0.5 spatial (job 398883). The arm took 14% longer, and its answers were
  17% longer (325.9 against 277.7 tokens).
- **Fast end, 1.130:** long COCO combined 0.25 attention (job 398885). The arm took 12% less time, and its
  answers were 10% shorter (98.6 against 109.9 tokens).
- **Per output token:** in both arms, time per output token was about 2% lower than in the 1.0 arm.

Latency therefore followed output length and, we infer, a rise in acceptance under eviction (2.141 to 2.265
and 2.284 to 2.362 accepted draft tokens per iteration in these two arms), not cheaper steps.

Memory savings are reported as memory and not converted into batched throughput: the ViSpec tree code
supports batch 1 only.

### 3.1 Sparsity results, Weeks 2 to 4

Week 3 swept budgets 1.0, 0.75 (`short` only), 0.5 and 0.25 with both policies, ViSpec arms only. The ranges
cover the arms below 1.0.

- **Latency ratio:** the same-policy 1.0 arm's latency divided by the arm's (above 1: the arm was faster).
- **Live KV and decode peak ratios:** arm divided by the same-policy 1.0 arm (below 1: the arm used less).
- **Live KV:** arithmetic, `kv_tokens` x 36,864 bytes, output tokens included.
- **Exact agreement:** with the same-job ViSpec reference.

| Style | Benchmark | Experiment (job) | Latency ratio | Live KV ratio | Decode peak ratio | Exact agreement | Quality, reference to arms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| long | TextVQA | 05 draft (398884) | 0.992 to 1.037 | 0.995 to 1.003 | 1.000 | 40 to 56 of 100 | near zero |
| long | TextVQA | 06 KV (398883) | 0.875 to 0.983 | 0.470 to 0.671 | 0.997 to 0.998 | 0 to 6 of 100 | near zero |
| long | TextVQA | 08 both (398885) | 0.902 to 0.986 | 0.467 to 0.673 | 0.997 to 0.998 | 0 to 7 of 100 | near zero |
| short | TextVQA | 05 draft (398923) | 0.997 to 1.002 | 1.000 | 1.000 | 100 of 100 | 70.0 to 70.0 |
| short | TextVQA | 06 KV (398923) | 0.955 to 1.002 | 0.298 to 0.766 | 0.997 to 0.999 | 69 to 96 of 100 | 70.0 to 60.8 to 71.0 |
| short | TextVQA | 08 both (398923) | 0.956 to 1.001 | 0.298 to 0.766 | 0.997 to 0.999 | 69 to 96 of 100 | 70.0 to 60.8 to 71.0 |
| long | COCO | 05 draft (398884) | 0.976 to 1.008 | 1.001 to 1.012 | 1.000 | 58 to 77 of 100 | near zero |
| long | COCO | 06 KV (398883) | 0.997 to 1.106 | 0.468 to 0.659 | 0.999 | 0 to 2 of 100 | near zero |
| long | COCO | 08 both (398885) | 1.001 to 1.130 | 0.465 to 0.659 | 0.999 | 0 to 2 of 100 | near zero |
| short | COCO | 05 draft (398923) | 0.991 to 1.001 | 1.000 | 1.000 | 99 to 100 of 100 | 119.68 to 119.68 to 119.93 |
| short | COCO | 06 KV (398923) | 0.985 to 1.034 | 0.371 to 0.791 | 0.999 to 1.000 | 10 to 77 of 100 | 119.68 to 104.98 to 122.86 |
| short | COCO | 08 both (398923) | 0.988 to 1.027 | 0.371 to 0.791 | 0.999 to 1.000 | 10 to 77 of 100 | 119.68 to 104.98 to 122.86 |
| long | Re:Verse+ | 09 KV 0.75 spatial (398924) | 0.985 | 0.770 | 1.000 | 0 of 55 | ROUGE-1 22.58 to 22.32 |

In the Quality column, "70.0 to 60.8 to 71.0" means: reference 70.0, arms from 60.8 to 71.0.

- **Quality in `short`.**
  - **Spatial policy at 0.25:** lost 9.2 VQA points on TextVQA (70.0 to 60.8) and 14.70 CIDEr-D on COCO
    (119.68 to 104.98).
  - **Attention policy at 0.25:** lost nothing measurable on 100 samples.
  - **Attention policy at 0.5:** lost 8.85 CIDEr-D on COCO (in 06, Week 2 and Week 3).
- **Acceptance rose under eviction.** The cause is an inference and is not tested.
  - **Long TextVQA:** from 2.141 to between 2.255 and 2.324.
  - **Re:Verse+ at 0.75:** from 1.702 to 1.760.
  - **Per step:** decode time per iteration stayed at 52.1 to 54.3 ms in the long KV job, so part of the
    small per-token gain comes from the rise and not from cheaper steps.
- **Week 2** ran budget 0.5 only. Its results are in line with the Week 3 0.5 arms. For example, the long TextVQA
  eviction arms had latency ratios of 0.890 to 0.935 and live KV ratios of 0.661 to 0.671. See the
  [register](README.md#qwen-week-2-results-05-and-06).
- **Re:Verse+ kv 0.75 spatial, the one finished eviction arm.**
  - **Live KV:** fell from 243.3 MiB to 187.4 MiB (ratio 0.770). Image tokens are 99% of the prompt, so the
    ratio follows the budget closely.
  - **Memory peaks:** the post-compaction peak fell by 0.8% (8.489 to 8.421 GiB, ratio 0.992). The decode
    peak did not move (ratio 1.000). That peak also covers the draft prefill and the compaction transient.
  - **TTFT:** unchanged, 6.656 s against 6.643 s for the 1.0 spatial arm.
  - **Latency:** the arm was 1.5% slower than the 1.0 arm (ratio 0.985). Its summaries were 4% longer
    (442.7 against 424.4 tokens), while time per output token was 0.2% lower.
  - **Quality:** ROUGE-1 fell 0.26 points. BERTScore fell 0.37 points. BERTScore is truncated to the opening
    pages; see section 8.
  - **Arms still pending:** KV 0.5 and 0.25 are Pending GPU run (jobs 402641 and 402642).

## 4. Week 4: Re:Verse+ status

**What finished.** Slurm job 398924, attempt `slurm_restart_count=1`, ran on 2026-11-04 on one
`NVIDIA RTX A6000` (`GPU-eb032c66-7f3f-d1b6-96cf-c9db5600d457`, node `c1-8`) with 4 CPUs and 22,528 MB, at
commit `f90aa6a8f19f` with a clean tree. Five run folders completed, each with 55 of 55 windows and exit 0:

- references `ar` and `vispec`;
- kv `1.0:spatial`, kv `1.0:attention` and kv `0.75:spatial`.

Preemption requeued the job (its folders carry restart counts 0 to 3), and it was then cancelled. Folders from
its other attempts are not used. The
[09 README](09_qwen_reverse_longcontext/README.md#results) lists the folders and the checks.

| Check | Result |
| --- | --- |
| Completion: exit codes, one GPU UUID, 55/55 windows, clean tree, expected metadata | Passed for all five folders |
| Identity gate: each 1.0 arm token-identical to the same-job ViSpec reference | **Passed**: kv 1.0 spatial 55/55, kv 1.0 attention 55/55, with identical prompts (55/55 fingerprints) |
| Smallest number of image tokens kept per image (`kv_min_kept_per_image`) | 1,148 at 1.0 (both policies), 861 at 0.75 spatial |
| fp32 offset check (AR equals ViSpec under the same eviction) | Pending GPU run (no job yet) |
| Divergence diagnosis (AR against ViSpec, and the draft-only arms) | Pending GPU run (no job yet) |

**Pending arms.** Each queued job runs both references first, so its ratios stay inside that job.

| Arms | Job |
| --- | --- |
| KV 0.5 spatial and 0.5 attention | Pending GPU run (job 402641) |
| KV 0.25 spatial and 0.25 attention | Pending GPU run (job 402642) |
| KV 0.5 spatial on AR; draft pruning 0.5 and 0.25 spatial | Pending GPU run (job 402643) |
| Combined (draft and KV) 0.5 and 0.25 spatial | Pending GPU run (job 402644) |

**Analysis files** (git-ignored):

- sweep table: `experiment/sweeps/20261109T181008Z_week4_reverse_partial_job398924/`;
- comparisons: `experiment/sweeps/20261109T181008Z_week4_reverse_partial_comparisons_job398924/`. It holds
  ViSpec against AR, the two identity comparisons, and kv 0.75 spatial against both the ViSpec reference and
  the 1.0 spatial arm.

## 5. Trade-off figures

The figures are in `experiment/sweeps/20261109T181249Z_tradeoffs_weeks3_4/`, one PNG and SVG per benchmark
and style. The files are:

- [textvqa_long](sweeps/20261109T181249Z_tradeoffs_weeks3_4/textvqa_long.png)
- [textvqa_short](sweeps/20261109T181249Z_tradeoffs_weeks3_4/textvqa_short.png)
- [coco_caption_long](sweeps/20261109T181249Z_tradeoffs_weeks3_4/coco_caption_long.png)
- [coco_caption_short](sweeps/20261109T181249Z_tradeoffs_weeks3_4/coco_caption_short.png)
- [reverse_summary_long](sweeps/20261109T181249Z_tradeoffs_weeks3_4/reverse_summary_long.png)

`figures.json` lists every plotted point and the sha256 of each source table.

**Panels.** Each figure has four panels. Each panel puts one ratio on the x axis and the quality delta from
the ViSpec reference on the y axis. The four ratios are:

- latency speedup against the ViSpec reference;
- live KV against the same-policy 1.0 arm;
- post-compaction peak against the same-policy 1.0 arm;
- speedup against AR.

The Re:Verse+ figure adds a BERTScore panel. Every series stays inside one job's references.

**Not embedded.** Each PNG is 293 to 425 KB, over the 200 KB limit for a committed copy, so the page links
the files instead.

The Re:Verse+ figure holds only the five finished folders. It will be redrawn when jobs 402641 to 402644 finish.

Command, run from the main checkout (whose `experiment/sweeps/` holds the tables), with the script from this
commit:

```bash
python experiment/plot_tradeoffs.py \
  --sweeps experiment/sweeps/20261105T192610Z_week3_long_kv_job398883/sweep.json \
           experiment/sweeps/20261105T192610Z_week3_long_draft_job398884/sweep.json \
           experiment/sweeps/20261105T192610Z_week3_long_combined_job398885/sweep.json \
           experiment/sweeps/20261105T192610Z_week3_short_all_job398923/sweep.json \
           experiment/sweeps/20261109T181008Z_week4_reverse_partial_job398924/sweep.json \
  --output-dir experiment/sweeps/20261109T181249Z_tradeoffs_weeks3_4
```

The command was re-run into a second folder to check that the output reproduces.

- **PNG files:** byte-identical (same sha256).
- **SVG files:** they differ only in the embedded creation date and in matplotlib's randomly salted clip-path
  and marker IDs. They are identical once those are normalised.
- **`figures.json`:** it differs only in the recorded `--output-dir`.

## 6. Which predictions held

Each prediction was written in an experiment README before its runs. The outcomes are from the measurements
above.

| Prediction (where it was written) | Outcome | Evidence |
| --- | --- | --- |
| Greedy ViSpec reproduces AR tokens (design spec, Week 1) | **Held in all 15 fp32 checks. Not held in bf16, explained** | fp32: 15 of 15 checks identical (5 long TextVQA questions at three eviction settings; none without eviction or on Re:Verse+). bf16: 22 to 28 of 100 (`long`), 88 to 96 of 100 (`short`), 0 of 55 (Re:Verse+). Every diagnosed AR-against-ViSpec divergence is a near-tie, and the Re:Verse+ ones are not yet diagnosed (section 2.2) |
| Decode is host-bound, so visual KV eviction barely moves decode latency ([04](04_qwen_decode_profile/README.md)) | **Held (inference from a profile)** | GPU busy 36 to 50% of a step. TTFT and per-iteration time did not change under eviction |
| Draft pruning costs acceptance at most slightly ([05](05_qwen_draft_pruning/README.md)) | **Held** | Acceptance within -0.015 to +0.076 of the 1.0 arm |
| Draft pruning barely moves latency (05) | **Held** | Latency ratios 0.976 to 1.037 |
| Greedy output matches the 1.0 reference up to bf16 near-ties under draft pruning (05) | **Held for `short`; held for `long` apart from one known case** | 9 of 10 Week 3 draft-only arms pass the diagnosis. Long 0.5 attention fails on question 35537, the exact-tie case of experiment 07 |
| Eviction lowers live KV ([06](06_qwen_kv_eviction/README.md)) | **Held** | Live KV ratio 0.298 to 0.791 (Week 3), 0.770 at Re:Verse+ 0.75 |
| Eviction lowers the decode peak (06) | **Partly held** | About 0.3% lower (ratio 0.997 to 1.000), and unchanged at Re:Verse+ 0.75 |
| Eviction barely moves latency (06) | **Held for TTFT and per-iteration time, partly end to end** | End-to-end latency ratios (1.0 arm over arm) of 0.875 to 1.106 follow output length and acceptance |
| Eviction can change the output (06) | **Held, strongly** | Long agreement 0 to 6 of 100; Re:Verse+ 0.75 agrees on 0 of 55 |
| A combined arm tracks the matching eviction arm ([08](08_qwen_combined_sparsity/README.md)) | **Held** | Live KV within 0.005, agreement within 1 of 100, `short` quality within 0.58 CIDEr-D |
| Re:Verse+ live KV is about 6 times the TextVQA size: at least 4 times the long TextVQA 06 figure, at least 200 MiB ([09](09_qwen_reverse_longcontext/README.md)) | **Held** | 243.3 MiB at kv 1.0 spatial, 5.2 times the 47.1 MiB of long TextVQA at 1.0 spatial (job 398883) |
| Re:Verse+ eviction saves hundreds of MiB (09; cutoffs at 0.5 and 0.25) | **Pending GPU run (jobs 402641, 402642)** | Only 0.75 has run: 55.9 MiB saved, ratio 0.770 |
| Re:Verse+ eviction at 0.25 makes some arm at least 3% faster than its same-policy 1.0 arm (09) | **Pending GPU run (job 402642)** | At 0.75 the arm was 1.5% slower (ratio 0.985) |

## 7. Anti-goal checklist (from the design spec)

| Anti-goal | Status | Evidence |
| --- | --- | --- |
| Reporting a speedup or memory saving while outputs changed, without agreement and quality next to it | Avoided | Every speedup and memory result on this page is given with its exact agreement and quality, in the same table or in the table it extends (section 2.3 extends 2.1). Speedups are called observed ratios where outputs differ |
| Comparing runs from different GPU models, settings or prompts as if matched | Avoided | Every Qwen run is on a pinned RTX A6000 with its device report. `compare_metrics` refuses pairs whose model, settings, GPU identity, allocation or prompt fingerprints differ. `sweep_table` and `plot_tradeoffs` compute and plot every ratio inside one (style, job) group. Timing is never compared across jobs |
| Claiming KV-eviction memory savings from a measurement that cannot see them | Avoided | All sparsity arms use the compact cache layout, sized per sample. Live KV is labelled arithmetic. The decode and post-compaction allocator peaks are reported separately. The fixed 16,384-token buffer of Week 1 is never used for an eviction memory claim |
| Committing benchmark images or manga pages, or ignoring the licence terms | Avoided | No page image, narration or generated summary text is in the repository or on this page, only scores, counts and token IDs. Re:Verse+ is reached through the git-ignored link `bench_data/reverse_plus`, and the dataset folder was left unchanged (no file newer than its `FREEZE.json`) |
| Substituting paper numbers for measurements | Avoided | Every measurement on this page comes from this cluster's run files, sweep tables or comparison files. The weight sizes come from the checkpoint file headers, and the prior-art methods are named as citations only |

## 8. Open issues

- **Re:Verse+ work that waits for GPUs.**
  - KV 0.5 spatial and attention: Pending GPU run (job 402641).
  - KV 0.25 spatial and attention: Pending GPU run (job 402642).
  - KV 0.5 spatial on AR, and draft pruning 0.5 and 0.25: Pending GPU run (job 402643).
  - Combined 0.5 and 0.25: Pending GPU run (job 402644).
  - fp32 offset check and divergence diagnosis: Pending GPU run (no job yet).
- **The cached-decode drift probe (07).** The fork's cached AR decode matches stock Hugging Face greedy
  decoding on only 19 of 100 long TextVQA answers. 80 of the 81 first mismatches are near-ties. In the
  remaining case (35537:162) the fork's two top logits tie exactly, while stock decoding prefers the other
  token by 0.375. This was judged bf16 implementation noise and is a known limitation. The mechanism is not
  isolated.
- **bf16 agreement is low on long answers.** ViSpec and AR agree on 22 to 28 of 100 long TextVQA and COCO
  answers and on 0 of 55 Re:Verse+ summaries. In exact arithmetic they would agree on all of them. Equal
  outputs are shown only in fp32 (15 checks on long TextVQA, all with eviction), so the speedups are observed
  ratios. The Re:Verse+ divergences
  are not yet diagnosed.
- **One run per arm.** Run-to-run spread is unknown. In job 398924 the two 1.0 arms, whose tokens are
  identical to the ViSpec reference, still ran 0.9% and 0.5% faster than it. Latency differences of about 1%
  between arms are therefore within timing noise.
- **Re:Verse+ scoring limits.**
  - **BERTScore:** it truncates at 512 word pieces, so it compares only the opening of 50 of the 55
    references. It cannot show whether later pages are covered. ROUGE-1 is the primary score.
  - **Absolute scores:** both scores are low (ROUGE-1 about 22.6, BERTScore about 46.6), because a summary
    is compared with page-by-page narration. Only differences between arms are meaningful.
- **Re:Verse+ input handling differs from TextVQA and COCO.**
  - **Pixel limit:** pages are capped at 1200 x 28 x 28 pixels, not the 1280 x 28 x 28 of TextVQA and COCO, so
    that the largest window fits the draft's 8,192-position limit with 1,024 new tokens.
  - **Prompt layout:** the images come before the instruction.
  - **Vision encoding:** each page goes through the vision encoder in its own call, because one call over 6
    pages ran out of GPU memory. This is mathematically the same as one batched call, but not bit-identical
    (about 1e-4 absolute in an fp32 check).
- **Prompt processing dominates multi-page inputs.** Cutting image tokens before prompt processing (FastV,
  VisionZip, PyramidDrop) is future work. The vision-encoder and language-model shares of TTFT are not
  measured.
- **HiViS on Qwen2.5-VL-7B.** HiViS is a second speculative decoder. It is built and checked on CPU, and its
  GPU runs are pending. See `experiment/10_qwen7b_methods/README.md` on branch `qwen-hivis-7b`.

## Data credits and terms

The manga pages and narration come from Re:Verse+, a re-aligned derivative of Re:Verse (arXiv:2508.08508).
The pages are credited to Yen Press, the official English publisher. The data is used for non-commercial
research only and is not redistributed. No page, narration or generated summary text is stored in this
repository.
