# 11 — Qwen2.5-VL-7B on Re:Verse+ page windows (autoregressive, ViSpec, HiViS)

## Purpose and hypothesis

Experiment 09 measured ViSpec on the Re:Verse+ manga windows with the 3B model, and experiment 10 compared
ViSpec and HiViS on single images with the 7B model. This experiment completes the pair. It runs
autoregressive (AR) decoding, ViSpec and HiViS on `Qwen/Qwen2.5-VL-7B-Instruct` over the same 55 page
windows of 5 to 6 pages as experiment 09. The question is whether the two speculative methods keep their
single-image speedups when the prompt holds several pages, and how the HiViS draft, which sees no image
tokens, compares with ViSpec's draft, which sees each page as 2 compressed tokens, on long multi-page prompts.

At 3B, ViSpec was 1.74 to 1.87 times faster than AR on these windows, against 2.13 to 2.32 times on long
TextVQA and COCO answers. Time to first token (TTFT) was about 44% of ViSpec's window time, so prompt
processing capped the gain, and acceptance fell to 1.70 accepted draft tokens per iteration.

Predictions, written before any run:

1. **Both speculative methods beat AR on the windows, by less than on single images.** Cutoff: in every job,
   the mean latency of ViSpec and of HiViS is below the same-job AR mean, and each method's speedup over AR
   is below the lowest long TextVQA speedup of experiment 10 (1.85 for ViSpec and 1.93 for HiViS).
2. **HiViS accepts at least as many draft tokens per iteration as ViSpec.** The summaries are long
   free-form text, as are the COCO captions, where HiViS accepted 2.72 against ViSpec's 2.29 on 7B. Cutoff:
   HiViS's mean accepted draft tokens per iteration is at least ViSpec's, in every job.
3. **TTFT is the same for the three methods.** Both drafts process a short prompt next to the target's
   pass over about 6,500 tokens. Cutoff: the ViSpec and HiViS mean TTFT are each within 5% of the same-job
   AR mean.
4. **Greedy decoding gives AR's tokens in fp32.** Cutoff: AR equals HiViS and AR equals ViSpec on 3 of 3
   windows in the fp32 offset check.

These predictions were written before any run. The [Results](#results) section scores them.

Status: **complete** (jobs 413077 to 413085, 2026-11-22). The port checks passed, and all 9 arms of the three
main jobs ran 55 of 55 windows, with identical outputs for each method in every job. All four predictions held.
ViSpec was 1.19 to 1.28 times and HiViS 1.33 to 1.43 times faster than AR, and HiViS was 1.12 times faster
than ViSpec in each job. The divergence diagnosis did not pass its gate as written: of 102 bf16 differences
from AR, 100 are near-ties, and the other two are one token, shared by both methods, that is a near-tie in fp32.

## Changes from the reference

- **HiViS on several images.** HiViS's own `build_mask` finds one `<|vision_start|>` with `.item()` and
  removes that image's span from the draft's input: `<|vision_start|>`, the image pads,
  `<|vision_end|>` and the one token after it. `multi_image_mask` in `vispec/model/cnets_hivis.py` removes
  the same span for every image. A prompt with one image goes to HiViS's `build_mask` unchanged, so
  experiment 10 is not affected, and the port checks below test that on GPU. The pages of a window follow
  one another, so the token after each page is the next page's `<|vision_start|>`, and after the last page
  it is the first instruction token, as when one image is placed first. The HiViS draft's prompt is then the
  system turn, `<|im_start|>user\n`, the instruction without its first token "Sum", and the assistant
  header, 54 tokens in all. HiViS was trained and released for one image per prompt, so this is an
  extension of the method, not HiViS as published.
- **`evaluate`** no longer refuses `--method hivis` with `--benchmark reverse_summary`.
- **Runner.** `run.sh` uses the `methods` sweep of `experiment/qwen_arm_runner.sh` with the 7B models, the
  `qwen7b` asset suite and `reverse_summary`. An arm is `<method>[:<tree>]`. The page windows have one fixed
  layout, so an arm takes no prompt order. Each arm also saves the Re:Verse+ asset record as
  `prepared-assets-reverse.json`.
- **Checks.** `checks/run.sh` runs the port checks and, after the main runs, the divergence diagnosis.

## Models

| Role | Model | Revision |
| --- | --- | --- |
| Target | `Qwen/Qwen2.5-VL-7B-Instruct` | `cc594898137f460bfe9f0759e9844b3ce807cfb5` |
| ViSpec draft | `JLKang/ViSpec-Qwen2.5-VL-7B-Instruct` | `e06635b54327df78a5eb82361791f2ccec9f41ab` |
| HiViS draft | `Irisssme/HiViS-Qwen2.5-VL-7B-Instruct` | `6dd8717d912a5626d6ad4e81b124c54bb8a08ef2` |

The revisions are the pinned ones of the `qwen7b` asset suite, as in experiment 10. The ViSpec draft has
8,192 positions, the same as the 3B draft, and the HiViS draft has 128,000.

## Data, splits and subsets

The 55 windows of Re:Verse+ volume 1, with the loader, references and pixel limit of experiment 09
(see the [09 README](../09_qwen_reverse_longcontext/README.md#data-splits-and-subsets)). Pages are capped at
1200 x 28 x 28 pixels, the largest round cap at which every window fits the ViSpec draft's 8,192 positions with
1,024 new tokens. The 7B model uses the same processor, so the prompts are those of experiment 09.

**Dataset terms.** The data is for non-commercial research only. It must not be redistributed. Credit
Yen Press, and cite it as a re-aligned derivative of Re:Verse (arXiv:2508.08508). No page image or text is
committed to this repository, and result files keep only model outputs, scores and token counts. The dataset
folder is read-only for this project.

## Settings

- Greedy decoding (temperature 0.0), at most 1,024 new tokens, bf16, batch size 1.
- `long` style, the only style of `reverse_summary`: the pages in reading order, then "Summarize the story
  told in these manga pages, in order."
- Draft tree 30-3-8 (total tokens, depth, top-k). ViSpec uses `num_q` 2.
- `--kv-layout compact`, KV capacity 16,384 as an upper bound, no sparsity flag. Slim prefill is on.
- Each page goes through the vision encoder in its own call, as in experiment 09.
- Three jobs run all three arms, with the arm order rotated (`ar vispec hivis`, `vispec hivis ar`,
  `hivis ar vispec`), so that a slow stretch on the node does not always fall on the same method.

## Cluster prerequisites

The pinned `NVIDIA RTX A6000` nodes through `experiment/submit_ampere.sh`, as in
[AMPERE_SETUP.md](../AMPERE_SETUP.md). The `qwen7b` and `reverse` asset suites must be prepared
(`bash prepare.sh --check-only --suite qwen7b` and `--suite reverse` print their completion lines). Each arm
and each check job checks them again before evaluation.

## Exact commands

Port checks first, then the three main jobs once the checks pass:

```bash
CHECK_SET=port PROMPT_STYLE=long CLUSTER_PARTITION=preempt \
  SBATCH_EXTRA="--requeue --job-name=m11-port --cpus-per-task=4 --mem=48G --time=02:00:00" \
  bash experiment/submit_ampere.sh experiment/11_qwen7b_reverse/checks

for arms in "ar vispec hivis" "vispec hivis ar" "hivis ar vispec"; do
  METHOD_ARMS="$arms" PROMPT_STYLE=long CLUSTER_PARTITION=preempt \
    SBATCH_EXTRA="--requeue --job-name=m11-methods --cpus-per-task=4 --mem=48G --time=05:00:00" \
    bash experiment/submit_ampere.sh experiment/11_qwen7b_reverse
done
```

After the main runs, `CHECK_SET=diagnosis DIAG_LIST=<file> ORACLE_LIST=<file>` with the same submit line
runs `diagnose_divergence` on every AR against method comparison. A requeued job reruns its runner into fresh
run folders, and the final attempt is the one with the highest `slurm_restart_count` and exit code 0.

## Metric definitions

As in the [09 README](../09_qwen_reverse_longcontext/README.md#metric-definitions): latency and TTFT in
seconds per window, time per output token in milliseconds, accepted draft tokens per iteration (accepted draft
tokens, without the target's own token), exact token-ID agreement, and memory from `kv_live_bytes` and the
prefill and decode peaks. Task quality is ROUGE-1 F1 (the primary score) and BERTScore F1 (`bert-base-uncased`,
which truncates at 512 word pieces), each averaged over windows. Speedup is the same-job AR mean latency
divided by the method's mean latency. A decode-only speedup uses latency minus TTFT.

## Results

### Hardware and run folders

Every job ran on node `c4-3` (Intel Xeon Silver 4210R, 40 CPUs in two NUMA nodes) with one `NVIDIA RTX A6000`
(48 GB, driver 620.70.06), at commit `4d3aaf8`, with a clean working tree, `torch` 2.7.0+cu128 and
`transformers` 4.51.3. Each job ran at its first attempt (restart 0) and exited 0, except the diagnosis job,
which exits 1 when its gate does not pass. Run folders are under `runs/` and named
`20261122T<time>_4d3aaf84b0fe_<arm>_<suffix>`. The device report of each is `runs/<folder>/device.txt`.

| Job | Purpose | Arm order | GPU UUID | Run folders (suffix) | Elapsed |
| --- | --- | --- | --- | --- | ---: |
| 413077 | Port checks | | `GPU-55a11ed4` | `checks_port_vS8sY0` | 12.6 min |
| 413078 | Main job A | AR, ViSpec, HiViS | `GPU-55a11ed4` | AR `SBXRZy`, ViSpec `LcC0yh`, HiViS `7QLKo9` | 48.3 min |
| 413082 | Main job B | ViSpec, HiViS, AR | `GPU-f5bb2497` | AR `EpZ01R`, ViSpec `QiS5zB`, HiViS `suIPqX` | 45.9 min |
| 413084 | Main job C | HiViS, AR, ViSpec | `GPU-55a11ed4` | AR `a86C6d`, ViSpec `4giqvu`, HiViS `hWijqG` | 48.0 min |
| 413085 | Divergence diagnosis | | `GPU-f5bb2497` | `checks_diagnosis_CW5BgZ` | 25.6 min |

Jobs A and B overlapped from 10:03 to 10:38 UTC, and jobs B and C from 10:39 to 10:49 UTC, on two GPUs of the
same node, which also ran another user's job.

### Completion checks

- **Coverage.** All 9 arms completed 55 of 55 windows with exit code 0.
- **Same prompts as experiment 09.** The input-ID fingerprints of all 55 windows equal those of the 3B AR
  reference of job 402641.
- **Same outputs in every job.** For each method, the output token IDs of all 55 windows are identical in the
  three jobs. The differences between jobs below are therefore timing alone.

### Port checks (job 413077)

| Check | Result |
| --- | --- |
| HiViS on one image is unchanged: 5 long TextVQA questions against experiment 10 (job 412896) | **Passed**, 5 of 5 identical in output token IDs and input-ID fingerprint |
| fp32 exactness on the first 3 page windows, 30-3-8 tree | **Passed**, AR equals HiViS on 3 of 3 and AR equals ViSpec on 3 of 3, model on `cuda:0` with no offload |
| bf16 smoke on the first 2 page windows | **Passed**, AR, ViSpec and HiViS 2 of 2 each, with the pinned revisions and bf16 parameters |

### Primary comparison

Speedups inside each job, where a value above 1 means the method is faster than AR. Decode time is latency
minus TTFT. "Below the cap in both" keeps only the windows where neither method stopped at the 1,024-token cap.

| Job | Method | Speedup over AR, end to end | Decode-only, below the cap in both (n) | Decode-only, all windows | Per output token | TTFT method / AR |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 413078 | ViSpec | 1.284 | 1.815 (44) | 1.584 | 1.861 | 0.999 |
| 413078 | HiViS | 1.434 | 1.938 (47) | 2.018 | 2.059 | 0.999 |
| 413082 | ViSpec | 1.187 | 1.581 (44) | 1.387 | 1.622 | 1.001 |
| 413082 | HiViS | 1.329 | 1.692 (47) | 1.778 | 1.805 | 1.001 |
| 413084 | ViSpec | 1.258 | 1.745 (44) | 1.530 | 1.790 | 1.001 |
| 413084 | HiViS | 1.410 | 1.873 (47) | 1.963 | 2.000 | 1.001 |

The outputs are the same in every job, so the following figures hold for all three. Latency and TTFT are given
as their range over the jobs.

| Method | Mean latency (s) | Mean TTFT (s) | TTFT share of latency | Output tokens | At the cap | Accepted draft tokens per iteration | ROUGE-1 | BERTScore | Same tokens as AR |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AR | 16.41 to 18.21 | 7.11 to 7.30 | 40.1% to 43.3% | 275.1 | 4 | | 21.36 | 47.93 | |
| ViSpec | 13.82 to 14.24 | 7.11 to 7.29 | 51.2% to 51.5% | 352.9 | 9 | 1.984 | 20.52 | 47.69 | 5 of 55 |
| HiViS | 12.34 to 12.70 | 7.11 to 7.29 | 57.4% to 57.6% | 310.5 | 6 | 2.333 | 20.77 | 47.92 | 3 of 55 |

What the windows show:

- **Both methods are faster than AR, by less than on single images.** ViSpec is 1.19 to 1.28 times and HiViS
  1.33 to 1.43 times faster end to end. On long TextVQA in experiment 10 they were 1.85 to 2.00 and 1.93 to
  2.01 times faster.
- **Prompt processing is the cap.** TTFT is 7.11 to 7.30 s for every method and within 0.1% of AR's in each
  job, so both drafts add nothing measurable to it. It is 40% to 43% of AR's time, 51% of ViSpec's and 57% of
  HiViS's. TTFT covers everything up to the first token, including encoding the 5 or 6 pages.
- **HiViS accepts more draft tokens than ViSpec.** It accepts 2.33 draft tokens per iteration against
  ViSpec's 1.98. Its draft prompt is the same 54 tokens for every window (the system turn, the instruction
  without its first token and the assistant header). What differs between windows is the target's hidden
  states for those tokens, which after the pages have attended to them.
- **HiViS is 1.12 times faster than ViSpec in each job** (1.117, 1.120 and 1.121). Per output token it is 1.11
  times faster (1.106 to 1.117). Over the 41 windows where neither of the two stopped at the cap, its decode
  time is 1.06 times shorter (1.058 to 1.063). The rest of the end-to-end difference comes from length:
  ViSpec's summaries average 352.9 tokens against HiViS's 310.5, and ViSpec stops at the cap in 9 windows
  against HiViS's 6.
- **ViSpec's length gap to AR comes from a few windows.** ViSpec's summaries are 4,278 tokens longer than AR's
  in total. The 9 windows where only one of the two stopped at the cap (7 ViSpec only, 2 AR only) carry 4,253
  of them. ViSpec is longer in 26 windows and shorter in 22. For HiViS the total gap is 1,943 tokens, of which
  1,516 come from the 6 windows with a one-sided cap (4 HiViS only, 2 AR only).
- **Quality is close.** ROUGE-1 is 21.36 for AR, 20.52 for ViSpec and 20.77 for HiViS. BERTScore is 47.93,
  47.69 and 47.92. ViSpec gives AR's exact tokens in 5 of 55 windows and HiViS in 3. The outputs first
  differ at output index 1 to 519 (median 36.5) for ViSpec and 1 to 290 (median 40) for HiViS.
- **Timing spread between jobs.** With identical outputs, AR's mean latency ranged from 16.41 to 18.21 s
  (11%), ViSpec's from 13.82 to 14.24 s (3.0%) and HiViS's from 12.34 to 12.70 s (2.9%). AR's time per output
  token ranged from 33.8 to 39.9 ms. Job B, on the node's other GPU, was the fastest for all three methods,
  and AR gained most. The cause is not measured. The three jobs had similar CPU allocations (four cores
  across both NUMA nodes), and host load was not recorded. The ratio of HiViS to ViSpec moved by 0.004 across
  the jobs, much less than either speedup over AR.

### Against the 3B windows (experiment 09)

The models differ, so this is context, not a controlled comparison. At 3B, ViSpec was 1.74 to 1.87 times
faster than AR on the same windows. At 7B it is 1.19 to 1.28 times faster. The 7B AR summaries are shorter
(275.1 tokens against 497.3), so decoding is a smaller part of each window, and TTFT (7.11 to 7.30 s against
6.64 to 6.93 s) is 40% to 43% of AR's time against 25% at 3B (job 398924). The decode-only gain below the cap
is also lower, 1.58 to 1.82 against 1.90 to 2.01, although ViSpec's 7B draft accepts more draft tokens per
iteration than the 3B draft (1.98 against 1.70).

### Divergence diagnosis (job 413085)

`diagnose_divergence` recomputes the target's next-token logits at the first differing token of every window,
in bf16 and, for up to 50 windows per pair, in fp32. A difference is a near-tie when the bf16 margin between the
top two logits is at most 0.5, and the gate passes when no difference is unexplained. It ran on job 413078,
whose outputs equal those of the other two jobs.

| Pair | bf16 differences | Near-ties | Unexplained | fp32 favours AR / the method / neither / not run | Median bf16 margin |
| --- | ---: | ---: | ---: | --- | ---: |
| AR against ViSpec | 50 | 49 | 1 | 22 / 26 / 2 / 0 | 0.125 |
| AR against HiViS | 52 | 51 | 1 | 26 / 22 / 2 / 2 | 0.0625 |

**The gate did not pass as written.** The one unexplained difference is the same in both pairs: window
`v1_ch09_p030-034` at output index 37, where AR chose token 7149 and both ViSpec and HiViS chose token 19029. The
bf16 recompute favours the methods' token by 0.75 (14.625 against 13.875). The fp32 recompute favours AR's
token by 0.16 (14.639 against 14.480). So it is a near-tie in fp32, where AR's token is the fp32 choice, and bf16
rounding over this window's 5,805-token prompt moves the two logits 0.9 apart, past the 0.5 threshold. The fp32
checks of the port job found AR's tokens in both methods on the first 3 windows. Speculative decoding is
lossless in exact arithmetic, and this case does not contradict that.

### Predictions

| Prediction | Outcome | Evidence |
| --- | --- | --- |
| 1. Both methods beat AR on the windows, by less than their lowest long TextVQA speedup in experiment 10 (1.85 ViSpec, 1.93 HiViS) | **Held** | ViSpec 1.187 to 1.284 and HiViS 1.329 to 1.434 in the three jobs |
| 2. HiViS accepts at least as many draft tokens per iteration as ViSpec, in every job | **Held** | 2.333 against 1.984 in each job |
| 3. ViSpec and HiViS TTFT within 5% of AR's | **Held** | TTFT method / AR 0.999 to 1.001 |
| 4. AR equals HiViS and AR equals ViSpec in fp32 on 3 of 3 windows | **Held** | 3 of 3 for each method (job 413077) |

### Analysis files

Git-ignored, under the main checkout's `experiment/sweeps/`:

- `20261122T103901Z_qwen7b_reverse_jobA/` holds the `compare_metrics --mode method` comparisons of job 413078
  (`ar_vs_vispec/`, `ar_vs_hivis/`) and the diagnosis lists (`diag_list.txt`, `oracle_list.txt`).
- `20261122T112816Z_qwen7b_reverse_full/` holds `arm_table.py` with its output `arm_table.md` (every arm and
  every ratio above), `extras.py` with `extras.txt` (HiViS against ViSpec, output lengths and cap counts), and
  `run_dirs.txt` (the 9 run folders).

## Limitations

- **HiViS on several images is an extension.** The HiViS draft was trained on one image per prompt. Its
  draft prompt on these windows is 54 tokens, and the kept tokens keep their original positions, so the
  instruction sits about 6,500 positions after the system turn, a wider spread than single-image prompts
  give. A low HiViS acceptance here would describe this extension, not HiViS on the prompts it was trained
  for.
- **The two drafters are separately trained checkpoints**, as in experiment 10, so a difference between them
  is a difference between checkpoints, not between visual inputs alone.
- **Timing spread.** With identical outputs, AR's mean latency varied by 11% across the three jobs and the
  speculative methods' by about 3%, so each speedup over AR is a range, not a single value. The jobs shared a
  node with each other and with another user's job, and host load was not recorded.
- **The diagnosis gate.** Its 0.5 near-tie threshold is a bf16 margin. On these long prompts one difference
  exceeds it in bf16 (0.75) although it is a near-tie in fp32 (0.16), so the gate did not pass as written.
- **TTFT split.** The runs do not measure how TTFT splits between the vision encoder and the language
  model's prompt pass, and no claim is made about it.
- **Scores.** BERTScore truncates at 512 word pieces and compares only the opening of most references, so
  ROUGE-1 is the primary score. Both scores are low in absolute terms, because a summary is compared with
  page-by-page narration, and only differences between arms are meaningful.
