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

Pending GPU run (no job yet).

## Limitations

- **HiViS on several images is an extension.** The HiViS draft was trained on one image per prompt. Its
  draft prompt on these windows is 54 tokens, and the kept tokens keep their original positions, so the
  instruction sits about 6,500 positions after the system turn, a wider spread than single-image prompts
  give. A low HiViS acceptance here would describe this extension, not HiViS on the prompts it was trained
  for.
- **The two drafters are separately trained checkpoints**, as in experiment 10, so a difference between them
  is a difference between checkpoints, not between visual inputs alone.
- **Timing spread.** At 3B, identical outputs took up to 21% longer in one job than in another. The three
  jobs here give the spread for each method, and only differences larger than it are read as real.
- **TTFT split.** The runs do not measure how TTFT splits between the vision encoder and the language
  model's prompt pass, and no claim is made about it.
- **Scores.** BERTScore truncates at 512 word pieces and compares only the opening of most references, so
  ROUGE-1 is the primary score. Both scores are low in absolute terms, because a summary is compared with
  page-by-page narration, and only differences between arms are meaningful.
