# Results: ViSpec speculative decoding and visual sparsity on Qwen2.5-VL-3B, with a 7B method comparison

Status on 2026-11-21: Weeks 1 to 4 are measured. All 14 Re:Verse+ manga arms are complete (five Slurm jobs),
with their fp32 check and divergence diagnosis. The ViSpec and HiViS comparison on Qwen2.5-VL-7B is complete (section 2.4). Experiments 00 and 01 (LLaVA-1.6) are an
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

**Setup.** The target is `Qwen/Qwen2.5-VL-3B-Instruct` and the draft is `JLKang/ViSpec-Qwen2.5-VL-3B-Instruct`,
except in section 2.4, which uses the 7B target.

- **Decoding:** bf16, greedy decoding (temperature 0), batch size 1, at most 1,024 new tokens.
- **Draft tree:** depth 3, top-k 8, total-token 30, `num_q` 2.
- **Hardware:** one pinned 48 GB `NVIDIA RTX A6000` per job.
- **Benchmarks:**
  - 100 TextVQA questions and 100 COCO Captions images (Karpathy test), each in two prompt styles. The
    `long` style measures speed. The `short` style measures quality.
  - 55 Re:Verse+ windows of 5 or 6 manga pages, in the `long` style only.
- **Repeats:** one run per arm. The Re:Verse+ AR and ViSpec references ran in five jobs with identical
  outputs, which gives the only measure of timing spread between jobs (section 2.3).

## 1. The question, and the answer in brief

The project asks: **can dynamic visual sparsity make ViSpec faster and leaner without hurting task quality?**
We answer it in two parts. The first part is what speculative decoding itself delivers. The second part is
what visual sparsity adds on top of it.

1. **ViSpec is 2.13 to 2.32 times faster than AR on long answers** (TextVQA and COCO, `long` style, Weeks 1
   and 3). It is 1.46 to 1.49 times faster on short COCO captions. It is not faster on 5-token TextVQA
   answers (0.96 to 0.98), because prompt processing is most of their time.
2. **ViSpec matched AR in every fp32 check, and in bf16 they differ at near-ties.**
   - **fp32:** ViSpec and AR gave identical tokens in 24 of 24 checks: the first 5 long TextVQA questions at
     three eviction settings, and the first 3 manga windows at kv 1.0 (no eviction), kv 0.25 spatial and
     combined 0.25.
   - **bf16:** they agree on 22 to 28 of 100 long answers, 88 to 96 of 100 short answers and 0 of 55 manga
     summaries. Every AR-against-ViSpec divergence diagnosed (221 cases: Week 1 and the manga windows) is a
     near-tie between the top two logits (section 2.2).
3. **On 5- to 6-page manga windows, ViSpec is 1.74 to 1.87 times faster end to end** in five jobs with
   identical outputs (mean 1.79, section 2.3).
   - **Decode phase:** it gains about 1.9 to 2.1 times when the length cap does not distort the
     comparison. The figure is 1.90 to 2.01 over the 42 windows where neither method hit the 1,024-token
     cap, and 1.98 to 2.10 over the 47 windows where the cap was not one-sided.
   - **Against single images:** that is below the 2.28 to 2.48 times on long TextVQA and COCO (the same
     below-cap figure, Week 3 jobs). It fits the lower acceptance: 1.70 against 2.14 and 2.28 accepted
     draft tokens per iteration.
   - **Prompt processing:** it dilutes the gain further. Time to first token (TTFT), which includes encoding
     the pages, is 6.64 to 6.93 s per window, 41% to 44% of ViSpec's time. Even infinitely fast decoding
     would give at most 3.94 to 4.59 times over AR.
4. **Visual sparsity barely helps at batch 1.**
   - **What it does:** it shrinks the live KV cache and, at low budgets, costs some quality and most
     long-answer agreement.
   - **Speed at batch 1:** it did not make decode steps cheaper. It removes little of each step's cost, and
     decoding is host-bound. End-to-end latency ratios against the 1.0 arm ranged from 0.863 (the arm took
     16% longer, manga combined 0.25) to 1.130 (it took 12% less time, long COCO). These follow output length
     and acceptance. On manga windows, spatial eviction made summaries longer and more repetitive.
   - **Why:** section 3 gives the explanation. This is an explanation of the measurements, not a new method.
5. **On Qwen2.5-VL-7B, a drafter that sees no image tokens is at least as fast as ViSpec.** HiViS was 1.93 to
   2.34 times faster than AR on long answers, against 1.85 to 2.07 for ViSpec in the same jobs. It was 1.11
   times faster than ViSpec on long COCO and 1.01 to 1.04 times on long TextVQA (section 2.4). The two drafters
   were trained separately, so this compares two checkpoints, not the visual input alone.

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
| 4 | long | Re:Verse+ windows | 402641 | 55 | 31.802 | 17.027 | 1.868 | 40.7% | 1.702 | 0/55 | 22.92 / 22.58 |
| 4 | long | Re:Verse+ windows | 402642 | 55 | 28.016 | 15.512 | 1.806 | 43.6% | 1.702 | 0/55 | 22.92 / 22.58 |
| 4 | long | Re:Verse+ windows | 402643 | 55 | 26.593 | 15.211 | 1.748 | 43.9% | 1.702 | 0/55 | 22.92 / 22.58 |
| 4 | long | Re:Verse+ windows | 402644 | 55 | 27.199 | 15.247 | 1.784 | 44.3% | 1.702 | 0/55 | 22.92 / 22.58 |

Sources: Week 1 is from the [02](02_qwen_autoregressive/README.md#results) and [03](03_qwen_vispec/README.md#results) READMEs.
Week 3 is from the per-job sweep tables `experiment/sweeps/20261105T192610Z_week3_*` (linked from the
[register](README.md#qwen-week-3-results-retention-sweeps-05-06-and-08)). Week 4 is from
`experiment/sweeps/20261121T212041Z_week4_reverse_full/sweep.json`. Its AR and ViSpec references are
token-identical across the five jobs (55 of 55 windows each). The Week 2 jobs (393336 and
393337) reproduced the Week 1 tokens on 100/100 per benchmark and style, so they are not listed again.
Accepted draft tokens are identical across jobs because greedy outputs are reproduced.

### 2.2 Exactness

- **In fp32, ViSpec matched AR in every check (24 of 24).** Greedy ViSpec commits only tokens that the
  target picks, so it should equal AR.
  - **The checks:** the fp32 offset check (GPU job 402165) decoded both methods in fp32 with TF32 off. It
    covered the first 5 long TextVQA questions at each of three settings: kv 0.25 spatial, kv 0.75 spatial
    and combined 0.25 spatial. AR equalled ViSpec on 5 of 5 in each setting
    ([06](06_qwen_kv_eviction/README.md#fp32-offset-oracle-exactness-check) and
    [08](08_qwen_combined_sparsity/README.md#fp32-offset-oracle-exactness-check) READMEs).
  - **The scope:** every single-image check had eviction on, and none covers COCO.
  - **Re:Verse+ windows (job 412902):** AR equalled ViSpec in fp32 on the first 3 windows at kv 1.0 spatial
    (no eviction), kv 0.25 spatial and combined 0.25 spatial, 9 of 9. The prompt fingerprints equal the
    runs'. This also confirms the draft's multi-image branch and the cache bookkeeping with 5,166 of 6,955
    prompt positions evicted
    ([09 README](09_qwen_reverse_longcontext/README.md#fp32-offset-check-and-divergence-diagnosis)).
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
  The diagnosis (job 412902, comparison from job 402643) finds all 55 first differences to be near-ties, with
  a bf16 top-2 margin of at most 0.375 logits (median 0.125). In fp32 the replay picks AR's token in 21 cases
  and ViSpec's in 34. The draft-only arms at 0.5 and 0.25 are near-ties too, 39 of 39 and 37 of 37, with
  margins of at most 0.25.

### 2.3 Long multi-page prompts: prompt processing caps the speedup

Re:Verse+ windows are 5 or 6 manga pages given as images, and the model summarises the story. Prompts are
5,805 to 6,967 tokens (mean 6,496.6), of which 6,430.4 on average are image tokens (99%). Long TextVQA
prompts are about 1,063 tokens. The Re:Verse+ rows below come from the run files of jobs 398924 (restart 1) and
402641 to 402644, and the other rows come from the Week 3 sweep tables. Each ratio is taken inside one job.

| Job | Benchmark | n | TTFT AR / ViSpec (s) | ViSpec TTFT share | Speedup over AR, end to end | Decode-only speedup, below the cap in both (n) | Decode-only speedup, all samples | Per-output-token speedup | Accepted draft tokens per iteration |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 398924 | Re:Verse+ windows | 55 | 6.654 / 6.665 | 44.1% | 1.737 | 1.895 (42) | 2.320 | 1.903 | 1.702 |
| 402641 | Re:Verse+ windows | 55 | 6.901 / 6.927 | 40.7% | 1.868 | 2.009 (42) | 2.466 | 2.024 | 1.702 |
| 402642 | Re:Verse+ windows | 55 | 6.793 / 6.762 | 43.6% | 1.806 | 1.979 (42) | 2.425 | 1.995 | 1.702 |
| 402643 | Re:Verse+ windows | 55 | 6.636 / 6.680 | 43.9% | 1.748 | 1.925 (42) | 2.339 | 1.935 | 1.702 |
| 402644 | Re:Verse+ windows | 55 | 6.771 / 6.750 | 44.3% | 1.784 | 1.967 (42) | 2.404 | 1.977 | 1.702 |
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

- **The end-to-end speedup is 1.74 to 1.87 times** over five jobs (mean 1.79; job 398924: AR 26.24 s,
  ViSpec 15.11 s per window). On long TextVQA and COCO answers it is 2.13 to 2.32 times (Weeks 1 and 3).
- **Timing spread between jobs.** The five jobs produced identical outputs, so their differences are timing
  alone. AR's mean latency ranged from 26.24 to 31.80 s (21%) and ViSpec's from 15.11 to 17.03 s (13%),
  while the speedup ranged from 1.74 to 1.87. Job 402641 was slow for both methods on the same node and GPU as
  two of the others. A busier host fits host-bound decoding (section 3), but host load was not recorded.
  The rest of this list uses job 398924.
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
    token. That fits the known bf16 divergence between the two methods, which matched in every fp32 check,
    including the three manga windows (section 2.2). After the first difference the two summaries
    can run to different lengths.
  - **Not measured:** why AR reached the cap more often.
- **What is not measured.** These runs do not measure how TTFT splits between the vision encoder and the
  language model's prompt pass. We make no claim about that split.
- **Future work, not done here.** Cutting image tokens before or early in prompt processing is the lever
  that would move this cap. Prior art includes FastV (Chen et al., 2024), VisionZip (Yang et al., 2024) and
  PyramidDrop (Xing et al., 2024). The eviction tested here runs after prompt processing, so it cannot change
  TTFT.

### 2.4 Qwen2.5-VL-7B: ViSpec and HiViS

Experiment 10 (`experiment/10_qwen7b_methods/README.md` on branch `qwen-hivis-7b`) moves to
`Qwen/Qwen2.5-VL-7B-Instruct` and adds HiViS (arXiv 2509.23928) next to ViSpec. The ViSpec draft sees each image
as 2 compressed tokens, and the HiViS draft sees no image tokens. The samples, decoding settings and the 30-3-8
tree are those of Weeks 1 to 3, the drafts are `JLKang/ViSpec-Qwen2.5-VL-7B-Instruct` and
`Irisssme/HiViS-Qwen2.5-VL-7B-Instruct`, and each job holds its own AR reference. Seven jobs (412893 to 412899)
ran 31 arms with 100 of 100 samples each.

| Style | Benchmark | Job | AR latency (s) | ViSpec speedup over AR | HiViS speedup over AR | HiViS over ViSpec | Accepted draft tokens per iteration, ViSpec / HiViS | Exact agreement with AR, ViSpec / HiViS | Quality AR / ViSpec / HiViS |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| long | TextVQA | 412893 | 15.832 | 1.972 | 1.989 | 1.009 | 2.181 / 2.282 | 2/100 / 3/100 | 0.0 / 0.0 / 0.0 |
| long | COCO Captions | 412894 | 4.878 | 1.980 | 2.203 | 1.113 | 2.287 / 2.724 | 18/100 / 9/100 | near zero |
| short | TextVQA | 412895 | 1.108 | 0.973 | 0.966 | 0.993 | 0.846 / 0.549 | 100/100 / 100/100 | 75.6 / 75.6 / 75.6 |
| short | COCO Captions | 412895 | 0.824 | 1.401 | 1.559 | 1.113 | 1.794 / 2.345 | 87/100 / 84/100 | 54.63 / 54.50 / 54.43 |

- **Repeated jobs.** Each primary `long` arm ran in three jobs per benchmark, on three different GPUs, with
  identical output token IDs. Its speedup over AR ranged from 1.85 to 2.00 (ViSpec) and 1.93 to 2.01 (HiViS) on
  TextVQA, and from 1.98 to 2.07 and 2.20 to 2.34 on COCO. HiViS over ViSpec ranged from 1.008 to 1.044 on TextVQA
  and from 1.113 to 1.127 on COCO, so only the COCO advantage is clear of the spread between jobs.
- **Why HiViS is faster on COCO.** It accepts more draft tokens per iteration at the same step time (46.1 ms
  against 46.3 ms, job 412894), so its time per output token is lower (12.5 ms against 14.3 ms).
- **Larger tree (60-5-10).** Acceptance rose by 0.31 to 0.61 for both methods on both benchmarks, but each step
  took 11.5% to 13.8% longer. Latency ranged from 3.2% slower (ViSpec, TextVQA) to 2.3% faster (HiViS, COCO)
  than at 30-3-8.
- **Prompt order.** Placing the image first changed the HiViS minus ViSpec acceptance difference by -0.004
  (TextVQA) and +0.013 (COCO). It changed AR's own TextVQA answers more (mean output 388 to 225 tokens). In job
  412898 both `image-first` TextVQA speedups were about 1.56, but a rerun with the arms in reverse order (job
  412937, token-identical outputs) gave 1.885 (ViSpec) and 1.913 (HiViS) against 1.849 and 1.860 with the image
  last. The slow steps of 412898 came from when its last two arms ran, not from the prompt order, so timing
  inside one job can shift by up to 21% between arms.
- **Against 3B (context only).** ViSpec's `long` acceptance is almost the same on 7B (2.181 against 2.141 on
  TextVQA, 2.287 against 2.284 on COCO), and its speedup is somewhat lower. Exact agreement on long TextVQA is
  lower (2 of 100 against 22 of 100). The 7B `short` COCO CIDEr-D (54.63 for AR) is less than half of the 3B
  value (121.40). The 7B captions are longer and more descriptive: 15.2 words against 12.1, 13 of 100 wrapped
  in quotation marks, and 54% of their words found in a reference caption against 76% for 3B. That fits the
  lower consensus score, and the three 7B methods are within 0.20 CIDEr-D of each other.
- **Exactness (job 412936).** All 397 bf16 differences from AR in the primary jobs are near-ties (0 unexplained).
  In fp32 re-runs of 229 of them, the stock model prefers AR's token 112 times and the speculative token 111 times.
  In fp32, the 7B ViSpec draft matches AR on 5 of 5 long TextVQA questions and HiViS on 5 of 5 image-first COCO
  images, which with job 402297 makes 20 of 20.
- **Parity with HiViS's own code (job 412935).** On 20 TextVQA questions at 60-5-10 in fp16, our port matches
  HiViS's code exactly in pruning, positions and the first draft tree (6 trees up to fp16 rounding ties), and
  its acceptance on the identical outputs is equal (2.8646). The token-count gate did not pass as written: 9 of
  20 outputs are identical, against at least 18, and all 11 differences are near-ties. HiViS's own speculative
  and AR paths agree on 0 of these 20.
- **Limits.** The two drafters are separately trained checkpoints, so the acceptance difference compares the
  checkpoints and cannot be attributed to the draft's visual input alone. Every arm ran once, and every job but
  412937 ran its arms in the order AR, ViSpec, HiViS.

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
     time. A decode profile at Re:Verse+ context length is not measured. There, time per verification step
     stayed within -2.9% and +3.9% of the reference under eviction, inside the cache's 4.1% share of the
     bytes read.
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
| long | Re:Verse+ | 09 KV (398924, 402641, 402642) | 0.868 to 1.055 | 0.300 to 0.770 | 1.000 | 0 of 55 | ROUGE-1 22.58 to 20.41 to 22.50 |
| long | Re:Verse+ | 09 KV on AR, 0.5 spatial (402643) | 1.051 | 0.536 | 0.987 | 0 of 55 | ROUGE-1 22.92 to 21.52 |
| long | Re:Verse+ | 09 draft (402643) | 1.010 to 1.024 | 1.000 | 1.000 | 16 to 18 of 55 | ROUGE-1 22.58 to 22.58 to 22.64 |
| long | Re:Verse+ | 09 both (402644) | 0.863 to 0.923 | 0.331 to 0.553 | 1.000 | 0 of 55 | ROUGE-1 22.58 to 20.69 to 22.01 |

In the Quality column, "70.0 to 60.8 to 71.0" means: reference 70.0, arms from 60.8 to 71.0. The Re:Verse+
rows except 0.75 come from jobs without a same-policy 1.0 arm, so their ratios and agreement are against the
same-job reference of the same method. In job 398924 the 1.0 arms reproduce that reference token for token,
with the same live KV.

- **Quality in `short`.**
  - **Spatial policy at 0.25:** lost 9.2 VQA points on TextVQA (70.0 to 60.8) and 14.70 CIDEr-D on COCO
    (119.68 to 104.98).
  - **Attention policy at 0.25:** lost nothing measurable on 100 samples.
  - **Attention policy at 0.5:** lost 8.85 CIDEr-D on COCO (in 06, Week 2 and Week 3).
- **Acceptance rose under eviction.** The cause is an inference and is not tested.
  - **Long TextVQA:** from 2.141 to between 2.255 and 2.324.
  - **Re:Verse+:** from 1.702 to 1.758 under attention eviction at 0.5 and 0.25, and to 1.760, 1.984 and
    2.237 under spatial eviction at 0.75, 0.5 and 0.25. The large rises go with repetitive summaries (next
    list).
  - **Per step:** decode time per iteration stayed at 52.1 to 54.3 ms in the long KV job, so part of the
    small per-token gain comes from the rise and not from cheaper steps.
- **Week 2** ran budget 0.5 only. Its results are in line with the Week 3 0.5 arms. For example, the long TextVQA
  eviction arms had latency ratios of 0.890 to 0.935 and live KV ratios of 0.661 to 0.671. See the
  [register](README.md#qwen-week-2-results-05-and-06).
- **Re:Verse+ windows, all budgets** (details in the
  [09 README](09_qwen_reverse_longcontext/README.md#what-visual-sparsity-did-on-manga-windows)).
  - **Live KV:** it follows the budget closely, because image tokens are 99% of the prompt. It fell from
    243.3 MiB to about 0.55 of that at 0.5 and 0.30 to 0.34 at 0.25, saving 109 to 114 MiB and 162 to
    170 MiB. The post-compaction peak fell by 1.5% and about 2.1%. The ViSpec decode peak did not move.
  - **TTFT:** unchanged, because eviction runs after prompt processing.
  - **Per step:** time per verification step stayed within -2.9% and +3.9% of the reference.
  - **End to end:** latency followed output length. Spatial eviction lengthened the summaries by 20% at 0.5
    and 52% at 0.25, and those arms took 6% and 15% longer. The attention arms wrote 3% and 5% shorter
    summaries and took 2% and 5% less time.
  - **Repetition:** spatial eviction pushed summaries into loops. At 0.25 the share of distinct token
    4-grams fell from 0.86 to 0.63, and 23 windows ran to the 1,024-token cap against 6 for the reference.
    The attention policy did not do this (0.88, 6 windows). Repetition was measured from the saved token
    IDs only.
  - **Quality:** at 0.25 the attention policy lost 0.59 ROUGE-1 and 0.66 BERTScore points, and the spatial
    policy 2.17 and 2.42. BERTScore is truncated to the opening pages; see section 8.
  - **Draft pruning:** acceptance moved by -0.010 and +0.038 and latency by 1.0% and 2.3%, as in Week 3.

## 4. Week 4: Re:Verse+ status

**What finished.** Five Slurm jobs ran all 14 arms on pinned `NVIDIA RTX A6000` cards, each arm with 55 of 55
windows and exit 0. Job 398924 (restart 1, 2026-11-04, node `c1-8`, commit `f90aa6a8f19f`) ran the references,
kv 1.0 spatial and attention, and kv 0.75 spatial. Jobs 402641 to 402644 (2026-11-10 and 11, nodes `c1-8` and
`c2-1`, commit `27405e81224d`) each ran both references and then the remaining arms. Job 402642 is used at its
final attempt (restart 2). Its restart 1 also finished, with token-identical outputs. The
[09 README](09_qwen_reverse_longcontext/README.md#results) lists every folder and check.

| Check | Result |
| --- | --- |
| Completion: exit codes, one GPU UUID per job, 55/55 windows, clean tree, expected metadata | Passed for all 22 folders |
| Identity gate: each 1.0 arm token-identical to the same-job ViSpec reference | **Passed**: kv 1.0 spatial 55/55, kv 1.0 attention 55/55, with identical prompts (55/55 fingerprints) |
| References across jobs | AR and ViSpec references token-identical across all five jobs (55/55 each) |
| Smallest number of image tokens kept per image (`kv_min_kept_per_image`) | 1,148 at 1.0, 861 at 0.75, 574 at 0.5 and 287 at 0.25 (draft side: 574 and 287) |
| fp32 offset check (AR equals ViSpec under the same eviction, 3 windows at kv 1.0, kv 0.25 spatial and combined 0.25) | **Passed**: 3/3 at each setting, job 412902 |
| Divergence diagnosis (AR against ViSpec, and the draft-only arms) | **Passed**: all near-ties (55/55, 39/39, 37/37), job 412902 |

**Dataset freeze.** The Re:Verse+ folder was re-frozen on 2026-11-15 (new `FREEZE.json` digest). The volume 1
page images and texts used here are dated 2026-07-13 and 2026-08-30, before the first freeze, so they are
unchanged. The checks of job 412902 ran against a re-prepared asset record, and every window's prompt fingerprint
equals the runs'. This project never writes to the dataset folder.

**Analysis files** (git-ignored, under the main checkout's `experiment/sweeps/`):

- sweep table for all five jobs, with the repetition script and its output:
  `20261121T212041Z_week4_reverse_full/`;
- comparisons: `20261109T181008Z_week4_reverse_partial_comparisons_job398924/` and
  `20261121T211835Z_week4_reverse_comparisons/`.

## 5. Trade-off figures

The figures are in `experiment/sweeps/20261121T212041Z_tradeoffs_weeks3_4_final/`, one PNG and SVG per benchmark
and style. The files are:

- [textvqa_long](sweeps/20261121T212041Z_tradeoffs_weeks3_4_final/textvqa_long.png)
- [textvqa_short](sweeps/20261121T212041Z_tradeoffs_weeks3_4_final/textvqa_short.png)
- [coco_caption_long](sweeps/20261121T212041Z_tradeoffs_weeks3_4_final/coco_caption_long.png)
- [coco_caption_short](sweeps/20261121T212041Z_tradeoffs_weeks3_4_final/coco_caption_short.png)
- [reverse_summary_long](sweeps/20261121T212041Z_tradeoffs_weeks3_4_final/reverse_summary_long.png)

`figures.json` lists every plotted point and the sha256 of each source table.

**Panels.** Each figure has four panels. Each panel puts one ratio on the x axis and the quality delta from
the ViSpec reference on the y axis. The four ratios are:

- latency speedup against the ViSpec reference;
- live KV against the same-policy 1.0 arm;
- post-compaction peak against the same-policy 1.0 arm;
- speedup against AR.

The Re:Verse+ figure adds a BERTScore panel. Every series stays inside one job's references.

**Not embedded.** Each PNG is 352 to 425 KB, over the 200 KB limit for a committed copy, so the page links
the files instead.

The Re:Verse+ figure holds all five jobs. Its two memory panels show only job 398924, because they divide by
the same-policy 1.0 arm, which ran only there. The memory ratios of the other jobs are in section 3.1.

Command, run from the main checkout (whose `experiment/sweeps/` holds the tables), with the script from this
commit:

```bash
python experiment/plot_tradeoffs.py \
  --sweeps experiment/sweeps/20261105T192610Z_week3_long_kv_job398883/sweep.json \
           experiment/sweeps/20261105T192610Z_week3_long_draft_job398884/sweep.json \
           experiment/sweeps/20261105T192610Z_week3_long_combined_job398885/sweep.json \
           experiment/sweeps/20261105T192610Z_week3_short_all_job398923/sweep.json \
           experiment/sweeps/20261121T212041Z_week4_reverse_full/sweep.json \
  --output-dir experiment/sweeps/20261121T212041Z_tradeoffs_weeks3_4_final
```

The command was re-run into a second folder (`..._final_regen`) to check that the output reproduces.

- **PNG files:** byte-identical (same sha256), all five.
- **SVG files:** they differ only in the embedded creation date and in matplotlib's randomly salted clip-path
  and marker IDs. They are identical once those are normalised.
- **`figures.json`:** it differs only in the recorded `--output-dir`.

## 6. Which predictions held

Each prediction was written in an experiment README before its runs. The outcomes are from the measurements
above.

| Prediction (where it was written) | Outcome | Evidence |
| --- | --- | --- |
| Greedy ViSpec reproduces AR tokens (design spec, Week 1) | **Held in all 24 fp32 checks. Not held in bf16, explained** | fp32: 24 of 24 checks identical (5 long TextVQA questions at three eviction settings, 3 manga windows at three settings including no eviction). bf16: 22 to 28 of 100 (`long`), 88 to 96 of 100 (`short`), 0 of 55 (Re:Verse+). Every diagnosed AR-against-ViSpec divergence is a near-tie, 221 of 221 (section 2.2) |
| Decode is host-bound, so visual KV eviction barely moves decode latency ([04](04_qwen_decode_profile/README.md)) | **Held (inference from a profile)** | GPU busy 36 to 50% of a step. TTFT and per-iteration time did not change under eviction |
| Draft pruning costs acceptance at most slightly ([05](05_qwen_draft_pruning/README.md)) | **Held** | Acceptance within -0.015 to +0.076 of the 1.0 arm |
| Draft pruning barely moves latency (05) | **Held** | Latency ratios 0.976 to 1.037 |
| Greedy output matches the 1.0 reference up to bf16 near-ties under draft pruning (05) | **Held for `short` and on manga; held for `long` apart from one known case** | 9 of 10 Week 3 draft-only arms pass the diagnosis, and both manga draft arms pass (39/39 and 37/37 near-ties). Long 0.5 attention fails on question 35537, the exact-tie case of experiment 07 |
| Eviction lowers live KV ([06](06_qwen_kv_eviction/README.md)) | **Held** | Live KV ratio 0.298 to 0.791 (Week 3), 0.300 to 0.770 on Re:Verse+ |
| Eviction lowers the decode peak (06) | **Partly held** | About 0.3% lower (ratio 0.997 to 1.000). Unchanged in every Re:Verse+ ViSpec arm, and 1.3% lower in the Re:Verse+ AR arm |
| Eviction barely moves latency (06) | **Held for TTFT and per-iteration time, partly end to end** | End-to-end latency ratios of 0.863 to 1.130 follow output length and acceptance. Per verification step, Re:Verse+ arms stayed within -2.9% and +3.9% |
| Eviction can change the output (06) | **Held, strongly** | Long agreement 0 to 6 of 100; every Re:Verse+ eviction arm agrees on 0 of 55, and spatial eviction made summaries longer and more repetitive |
| A combined arm tracks the matching eviction arm ([08](08_qwen_combined_sparsity/README.md)) | **Held** | Live KV within 0.005, agreement within 1 of 100, `short` quality within 0.58 CIDEr-D |
| Re:Verse+ live KV is about 6 times the TextVQA size: at least 4 times the long TextVQA 06 figure, at least 200 MiB ([09](09_qwen_reverse_longcontext/README.md)) | **Held** | 243.3 MiB at kv 1.0 spatial, 5.2 times the 47.1 MiB of long TextVQA at 1.0 spatial (job 398883) |
| Re:Verse+ eviction saves hundreds of MiB (09; cutoffs at 0.5 and 0.25) | **Held, every cutoff** | Spatial 0.548 and 110.0 MiB at 0.5, 0.335 and 161.8 MiB at 0.25 (sum 271.8). Attention 0.534 and 113.4 MiB, 0.300 and 170.2 MiB (sum 283.6) |
| Re:Verse+ eviction at 0.25 makes some arm at least 3% faster than its same-policy 1.0 arm (09) | **Held as stated, mostly for another reason** | kv 0.25 attention took 5.2% less time than the same-job reference. About two thirds of its decode saving came from fewer verification steps (shorter outputs, higher acceptance) and one third from 2.9% cheaper steps. kv 0.25 spatial took 15% longer |
| On 7B, ViSpec and HiViS both beat AR in the `long` style, on both benchmarks (10, branch `qwen-hivis-7b`) | **Held** | Speedup over AR 1.972 and 1.989 on TextVQA, 1.980 and 2.203 on COCO in the primary jobs. Every `long` speculative arm was faster than its same-job AR reference (1.550 to 2.392) |
| HiViS's acceptance relative to ViSpec depends on prompt order, a change of at least 0.1 (10) | **Not held** | The HiViS minus ViSpec difference changed by -0.004 on TextVQA and +0.013 on COCO |
| The 60-5-10 tree raises acceptance for both methods on both benchmarks (10) | **Held** | +0.31 (ViSpec) and +0.47 (HiViS) on TextVQA, +0.38 and +0.61 on COCO. Latency, left open, ranged from 3.2% slower to 2.3% faster |

## 7. Anti-goal checklist (from the design spec)

| Anti-goal | Status | Evidence |
| --- | --- | --- |
| Reporting a speedup or memory saving while outputs changed, without agreement and quality next to it | Avoided | Every speedup and memory result on this page is given with its exact agreement and quality, in the same table or in the table it extends (section 2.3 extends 2.1). Speedups are called observed ratios where outputs differ |
| Comparing runs from different GPU models, settings or prompts as if matched | Avoided | Every Qwen run is on a pinned RTX A6000 with its device report. `compare_metrics` refuses pairs whose model, settings, GPU identity, allocation or prompt fingerprints differ. `sweep_table` and `plot_tradeoffs` compute and plot every ratio inside one (style, job) group. Timing is never compared across jobs |
| Claiming KV-eviction memory savings from a measurement that cannot see them | Avoided | All sparsity arms use the compact cache layout, sized per sample. Live KV is labelled arithmetic. The decode and post-compaction allocator peaks are reported separately. The fixed 16,384-token buffer of Week 1 is never used for an eviction memory claim |
| Committing benchmark images or manga pages, or ignoring the licence terms | Avoided | No page image, narration or generated summary text is in the repository or on this page, only scores, counts and token IDs. Re:Verse+ is reached through the git-ignored link `bench_data/reverse_plus`, and this project never writes to the dataset folder. The folder was re-frozen on 2026-11-15, and the volume 1 pages and texts used here predate both freezes (section 4) |
| Substituting paper numbers for measurements | Avoided | Every measurement on this page comes from this cluster's run files, sweep tables or comparison files. The weight sizes come from the checkpoint file headers, and the prior-art methods are named as citations only |

## 8. Open issues

- **The cached-decode drift probe (07).** The fork's cached AR decode matches stock Hugging Face greedy
  decoding on only 19 of 100 long TextVQA answers. 80 of the 81 first mismatches are near-ties. In the
  remaining case (35537:162) the fork's two top logits tie exactly, while stock decoding prefers the other
  token by 0.375. This was judged bf16 implementation noise and is a known limitation. The mechanism is not
  isolated.
- **bf16 agreement is low on long answers.** ViSpec and AR agree on 22 to 28 of 100 long TextVQA and COCO
  answers and on 0 of 55 Re:Verse+ summaries. In exact arithmetic they would agree on all of them. Equal
  outputs are shown only in fp32 (24 checks on long TextVQA and manga windows), so the speedups are observed
  ratios. Every diagnosed divergence is a near-tie (221 of 221).
- **One run per arm.** Inside one job, the two 1.0 arms of job 398924, whose tokens are identical to the
  ViSpec reference, still ran 0.9% and 0.5% faster than it, so latency differences of about 1% between arms
  are within timing noise. Between jobs the spread is much larger: the Re:Verse+ references, identical in
  five jobs, took 21% (AR) and 13% (ViSpec) longer in the slowest job than in the fastest, and the speedup
  ranged from 1.74 to 1.87. No other benchmark has repeated jobs on identical outputs, so their spread
  between jobs is unknown.
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
- **HiViS on Qwen2.5-VL-7B.** The comparison, its divergence diagnosis and its fp32 checks are complete
  (section 2.4). Two points remain. The two drafters are separately trained, so the acceptance difference
  cannot be attributed to the visual input alone. The parity check against HiViS's own code did not pass its
  token-count gate as written (9 of 20 identical, all differences near-ties). The experiment README is on
  branch `qwen-hivis-7b`.

## Data credits and terms

The manga pages and narration come from Re:Verse+, a re-aligned derivative of Re:Verse (arXiv:2508.08508).
The pages are credited to Yen Press, the official English publisher. The data is used for non-commercial
research only and is not redistributed. No page, narration or generated summary text is stored in this
repository.
