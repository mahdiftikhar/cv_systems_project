# Weeks 4–5: Re:Verse Long-Context Stress Test and Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Two things.
- **Week 4:** run the sparsity methods on multi-page Re:Verse manga windows. Each window is
  summarised and scored against its own light-novel segments with ROUGE-1 F1 and BERTScore F1.
- **Week 5:** turn the Week 2–4 measurements into trade-off curves, tables and one results page.

**Architecture:**
1. Fix the upstream multi-image bug in the draft's image branch before any multi-page prompt runs.
2. Add a `reverse` asset suite, a `reverse_summary` benchmark (page windows plus aligned
   references) and two scorers.
3. Run every Re:Verse arm in one job through the existing arm runner, with a benchmark-list
   override.
4. Week 5 reads only the existing `sweep.json` tables and run reports. It runs no models.

**Tech Stack:** Python 3.11 (`vipsec`), torch 2.7.0+cu128, transformers 4.51.3,
`rouge-score`, `bert-score`, matplotlib, and Slurm on the pinned RTX A6000s.

**Spec:** [`docs/superpowers/specs/2026-10-18-qwen-dynamic-visual-sparsity-design.md`](../specs/2026-10-18-qwen-dynamic-visual-sparsity-design.md)
(Goals 4–5, the metrics, and hazards 5, 5a and 6). This plan follows
[`2026-10-20-week3-retention-sweeps.md`](2026-10-20-week3-retention-sweeps.md). It reuses the
arm runner, `sweep_table`, `offset_check` and the comparison guards that plan produced.

## Global Constraints

- **No test files. Commit messages are plain.** No `Co-Authored-By`, no `Claude-Session`, no
  Claude attribution of any kind.
- **Data (the user's decision, replacing the spec's `sochastic/Re-Verse`).** Use volume 1 of
  **Re:Verse+**, the user's frozen local copy at `/home/aeternum/Research/Comic/M3/reverse_plus`
  (`FREEZE.json` date 2026-10-16, digest `7b71fc7af18aad41341d7d0c2ad656fb5ce7f8fb2d0d35a97d3a5ff88fa4650b`).
  - It has 11 chapters and 308 pages (18–41 per chapter). Images are at
    `volume_1/images/chapter_<n>/NNN.jpg`, and each page's narration is at
    `volume_1/text/chapter_<n>/NNN.txt`. `NNN` is 1-based.
  - Every volume 1 chapter passes the dataset's own sync check (`manifest.csv`
    `block_page_ok=True`, including chapter 9 at 34/34).
  - `page_exclusions.json` lists no volume 1 page.
  - The text is the hand-made gold standard: Yen Press-style prose checked against the art.
  - **Terms (`TERMS.txt`):** non-commercial research only, with no redistribution. Credit Yen
    Press, and cite the data as a re-aligned derivative of Re:Verse (arXiv:2508.08508). Never
    commit or copy pages or text into the repository. Reference them through a git-ignored
    symlink, `bench_data/reverse_plus`.
- **Windows:**
  - Non-overlapping and consecutive within a chapter.
  - Each chapter of `n` pages is split into `ceil(n/6)` windows whose sizes differ by at most
    1, with the larger windows first. That gives 55 windows of 5–6 pages over the 11 chapters,
    and no window exceeds 6 pages.
  - Every window's full prompt must be under **8,192 positions**, the draft's RoPE table. The
    loader computes this per window from the processor's `image_grid_thw` plus the text tokens,
    and raises if a window exceeds it.
  - If the Week 1 pixel limit (`max_pixels = 1280*28*28`) cannot fit 6 pages, the per-page
    `max_pixels` for this benchmark is lowered once, uniformly, and recorded. It is never set
    per window.
- **References:** each window's reference is its pages' `NNN.txt` texts, joined in page order.
  Strip any BOM, and strip the `<D>…</D>` and `<T>…</T>` markup while keeping the inner text.
- **Prompt:** one style, `long`. The instruction is "Summarize the story told in these manga
  pages, in order." Pages are given as images only, with no OCR. The same prompt and windows
  are used in every arm.
- **Scoring:**
  - ROUGE-1 F1: `rouge-score`, stemming on, mean over windows.
  - BERTScore F1: `bert-score` with `bert-base-uncased` at revision
    `86b5e0934494bd15c9632b12f734a8a67f723594`, offline, no baseline rescaling, mean over windows.
  - Both packages are pinned in `requirements.txt`, and the weights live under `models/`.
- **Fixed settings:** the Week 3 settings: bf16, temperature 0.0, 1,024 new tokens, depth 3,
  top-k 8, total-token 30, `num_q` 2, batch 1, `--kv-layout compact` in every arm, and
  same-job references.
- **Gates carried forward:**
  - At retention 1.0, every arm is token-identical to its reference.
  - `diagnose_divergence` runs only on draft-only arms, with the 0.5-logit threshold unchanged.
  - `offset_check` is the fp32 oracle for eviction.
- **Manga109 is out of scope** unless the user provides the licensed zip. The spec makes it
  optional.

## Review Focus

1. **Multi-image draft correctness.**
   - A 6-page window is the first prompt with more than one image.
   - The fix must leave single-image prompts byte-identical, which means Week 2 tokens are
     reproduced.
   - On multi-image prompts, ViSpec must equal AR in fp32 (`offset_check` on Re:Verse at 1.0)
     and accept well above 1 token per iteration. *(Task 0.)*
2. **Position budget.** A window just over 8,192 positions would silently index past the draft's
   RoPE table, so the loader must raise rather than truncate. *(Task 2 check: report the
   largest window's positions.)*
3. **Reference alignment.** An off-by-one between pages and texts would pair summaries with the
   wrong prose. For every chapter, the loader asserts that the image and text file sets match one
   to one by page number. *(Task 2.)*
4. **Eviction at long context.** Per-image budgets must apply to each of the 6 images. Every image
   keeps at least 1 token, and text is never evicted. `kv_min_kept_per_image` is reported per
   arm. *(Task 4.)*
5. **Offline scoring.** BERTScore must not reach the network inside a job: it loads from the
   local `models/` path with `HF_HUB_OFFLINE=1`. *(Task 1 check.)*

---

### Task 0: Fix the multi-image draft branch

**Files:**
- Modify: `vispec/model/cnets_ours.py` (image branch of `Model.forward`, about :1009-1011)

**Interfaces:**
- **The defect.** Inside the per-image loop, `t_m` slices `eye_m` using `h_s[0].shape[0]` and
  `h_s[1].shape[0]`, which are the first image's text segment and adapted rows, for every
  image.
- **The fix.** Use the current image's own two entries, `h_s[-2]` and `h_s[-1]`, as appended
  just above. With one image these are the same objects, so single-image behaviour is unchanged.

- [ ] **Step 1:** Implement the fix. Add a shape assertion before `torch.cat`: the rows of
  `t_m` must equal the rows of `h_s` for every batch element.
- [ ] **Step 2: Verify on CPU.**
  - Build a tiny random draft config.
  - Feed it a synthetic 3-image sequence in which the text segments before each image have
    different lengths.
  - Show that the old code produces mismatched `t_m` rows (or raises), and that the new code
    matches.
  - Show that a 1-image input gives a bit-identical `trans_mat` before and after.
- [ ] **Step 3: Run the GPU smoke** in one `gpu_check.sh` allocation, writing to `.smoke/w4_t0/`:
  - 5 long TextVQA ViSpec compact no-flag questions. Tokens must equal the Week 2 03 long
    compact run on 5/5.
  - A 3-image prompt of three consecutive COCO subset images, with the question "Describe each
    image in order.", for both AR and ViSpec in bf16. Record the ViSpec acceptance and whether
    the run completes, plus fp32 AR == ViSpec through `offset_check`. Extend it with
    `--images` only if needed.
- [ ] **Step 4: Commit.** `git commit -m "Fix per-image row selection in the draft's multi-image branch"`

### Task 1: Re:Verse and scorer assets

**Files:**
- Modify: `experiment/prepare_assets.py` (`--suite reverse`), `prepare.sh` (pass-through if
  needed), `requirements.txt` (`rouge-score`, `bert-score`, exact versions)

**Interfaces:**
- **Local data, no download.** `--suite reverse` validates the local Re:Verse+ copy.
  - The location comes from `REVERSE_PLUS_DIR`, defaulting to the path above.
  - It checks the sha256 of every `volume_1/images/**` and `volume_1/text/chapter_*/NNN.txt`
    file against `FREEZE_manifest.tsv`, and fails on any mismatch or missing file.
  - It creates the git-ignored symlink `bench_data/reverse_plus`.
- **Weights.** It fetches `bert-base-uncased` at the pinned revision into
  `models/bert-base-uncased/`. This is the only download.
- **Manifest.** It writes `.preparation/assets_reverse.json` with:
  - the `FREEZE.json` digest and date;
  - per-chapter page and text counts;
  - the BERT revision;
  - a hash of `TERMS.txt`.
- **Check-only.** `--check-only --suite reverse` validates the manifest offline. It uses the
  same lock as the other suites.

- [ ] **Step 1:** Implement the suite and install the two packages into `vipsec`. Use the
  `--extra-index-url` pattern from `CLAUDE.md`, or plain `pip` if both are pure Python.
- [ ] **Step 2: Run** inside a CPU allocation on partition `normal` (downloads belong on compute
  nodes): `srun -p normal --cpus-per-task=4 --mem=8G --time=01:00:00 bash prepare.sh --suite reverse`,
  after clearing `SLURM_*`. Then `bash prepare.sh --check-only --suite reverse` must print the
  completion line.
- [ ] **Step 3: Verify offline scoring.** With `HF_HUB_OFFLINE=1`, score one synthetic pair with
  BERTScore from `models/bert-base-uncased` and ROUGE-1. Both must succeed with networking
  unavailable to Hugging Face.
- [ ] **Step 4: Commit.** `git commit -m "Prepare Re:Verse and summary-scoring assets"`

### Task 2: `reverse_summary` benchmark and scorers

**Files:**
- Modify: `vispec/evaluation/benchmarks.py`, `vispec/evaluation/evaluate.py` (benchmark
  choice), `vispec/evaluation/scoring.py`, `vispec/evaluation/report_metrics.py` (`BENCHMARKS`),
  `vispec/evaluation/diagnose_divergence.py` (prompt rebuild for multi-image samples),
  `experiment/METRICS.md`

**Interfaces:**
- **The benchmark.** `BENCHMARK_NAMES` gains `reverse_summary`. `load_samples(...)` returns
  one `Sample` per window:
  - `question_id = "v1_ch{n:02d}_p{first:03d}-{last:03d}"`;
  - `images`, a list of page paths in order;
  - `references`, a list with one joined reference string.
- **Prompts.** `build_messages` puts the images first, then the instruction, matching the
  single-image layout. It handles a list of images.
- **Scoring.** `Scorer.score_corpus` gives `rouge1_f1` and `bertscore_f1` (both 0–100) for
  `reverse_summary`. `report_metrics` reports them as this benchmark's quality.
- **Windows file.** `load_samples` raises if any window reaches 8,192 positions. It writes the
  per-window position counts to the manifest the evaluator already writes.

- [ ] **Step 1:** Implement.
- [ ] **Step 2: Verify on CPU** with the prepared data and the processor, without loading a
  model. Report:
  - the number of windows;
  - the page count per window;
  - the min, median and max positions per window;
  - every included chapter's segment == page check;
  - a decoded example prompt.

  Also score a hand-made pair, and check that `diagnose_divergence`'s rebuilt prompt
  fingerprint equals the evaluator's for 2 windows.
- [ ] **Step 3: Commit.** `git commit -m "Add the Re:Verse window summary benchmark and scorers"`

### Task 3: Experiment 09 (Re:Verse long-context)

**Files:**
- Create: `experiment/09_qwen_reverse_longcontext/{run.sh,README.md}`
- Modify: `experiment/qwen_arm_runner.sh` (benchmark list from `QWEN_BENCHMARKS`, default
  `textvqa coco_caption`, and a `reference` sweep type whose arms are `ar` and `vispec`, with no
  sparsity flags), `experiment/README.md`, `CLAUDE.md` (one line) and `AGENTS.md` (scope 02–09)

**Interfaces:**
- **Default arm list.** `09/run.sh` runs `QWEN_BENCHMARKS=reverse_summary` and
  `PROMPT_STYLE=long`, with these sweeps in order:
  - `reference`: `ar vispec`;
  - `kv`: `1.0:spatial 1.0:attention 0.75:spatial 0.5:spatial 0.5:attention 0.25:spatial 0.25:attention 0.5:spatial:autoregressive`;
  - `draft`: `0.5:spatial 0.25:spatial`;
  - `combined`: `0.5:spatial 0.25:spatial`.

  Each list can be overridden through `REF_ARMS`, `KV_ARMS`, `DRAFT_ARMS` or `COMBINED_ARMS`.
- **Runner behaviour.** Folders and metadata follow the Week 3 runner. Default 02–08 behaviour
  must not change: show it with the stub command diff.

- [ ] **Step 1:** Implement. The README states the predictions:
  - At about 7–8k-token prompts, live KV is about 6× the TextVQA size.
  - So eviction is predicted to save hundreds of MiB.
  - It may also move latency slightly more than at TextVQA scale (attention share grows with
    context).
- [ ] **Step 2: Verify.**
  - Run `bash -n` and the invalid-arm exit 2.
  - Diff the stub command output for 05/06/08 before and after.
  - Run `sbatch --test-only`.
- [ ] **Step 3: Commit.** `git commit -m "Add the Re:Verse long-context experiment"`

### Task 4: Re:Verse runs and gates

- [ ] **Step 1: Submit.** Submit one job, from a clean tree:
  `CLUSTER_PARTITION=preempt SBATCH_EXTRA="--cpus-per-task=4 --mem=22G --time=05:00:00" bash experiment/submit_ampere.sh experiment/09_qwen_reverse_longcontext`.
  Record the job ID.
- [ ] **Step 2: Verify completion** the same way as in Week 3:
  - exit codes are 0;
  - one UUID;
  - full window counts;
  - a clean tree;
  - the expected metadata.
- [ ] **Step 3: Identity gate.** Each 1.0 arm matches the ViSpec reference on all windows,
  token for token.
- [ ] **Step 4: Run the oracles and diagnosis** in one `gpu_check.sh` allocation:
  - `offset_check` on 3 windows in fp32 at kv 1.0, 0.25 spatial, and combined 0.25. AR must
    equal ViSpec on 3/3 each;
  - `diagnose_divergence` on the draft-only arms.
- [ ] **Step 5:** Build the sweep table for the job group and report `kv_min_kept_per_image` per
  arm.
- [ ] **Step 6:** Make no commit. Record the job IDs and every path.

### Task 5: Week 5 trade-off curves and results page

**Files:**
- Create: `experiment/plot_tradeoffs.py`, `experiment/RESULTS.md`
- Modify: the 05, 06, 08 and 09 READMEs, and `experiment/README.md`

**Interfaces:**
- **Plot script.** `python experiment/plot_tradeoffs.py --sweeps <sweep.json>... --output-dir experiment/sweeps/<UTC>_tradeoffs`
  writes PNG and SVG files. Each benchmark and style gets one panel, with every method as a
  series and the budget as point labels. The panels are:
  - latency speedup against ViSpec at 100% vs quality delta;
  - live-KV `arm_over` vs quality delta;
  - post-compaction peak vs quality delta;
  - speedup against AR.

  The script reads only `sweep.json`, and it refuses to plot ratios across groups.
- **`experiment/RESULTS.md`** is the comparison page. It covers:
  - the project question;
  - per-benchmark tables of Weeks 1–4 (counts, references and gates);
  - the trade-off figures. Their image files stay out of Git, so the page links their path and
    embeds a small committed copy only if it is under 200 KB;
  - which predictions held;
  - the anti-goal checklist from the spec, with evidence for each;
  - open issues, including the cached-decode drift probe result.

- [ ] **Step 1:** Implement the script. Write the Week 4 results into the 09 README, then write
  `RESULTS.md`.
- [ ] **Step 2: Verify.** Spot-check at least fifteen values on the page against `sweep.json`,
  `summary.json` and comparison or diagnosis JSON. Regenerate the figures from the recorded
  command, and show that the files match.
- [ ] **Step 3: Commit.** `git commit -m "Add Week 2-4 trade-off curves and the results page"`
