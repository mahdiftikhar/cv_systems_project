# Design: dynamic visual sparsity for ViSpec on Qwen2.5-VL-3B

Date: 2026-10-18. Source plan: [`CV_Sys_Project_Proposal.md`](../../../CV_Sys_Project_Proposal.md)
(Topic 14, "Speculative Decoding for Vision-Language Models: Accelerated Inference via Dynamic
Visual Sparsity"). This document turns that five-week plan into something buildable in this
repository. Where it records a choice the user did not make, the choice is marked **builder's call**.

## Intent

Speculative decoding speeds up a VLM by letting a small draft propose tokens that the full model
verifies in one pass. ViSpec is a speculative decoder built for VLMs. During verification, the
target model still attends over its full visual KV cache. The project asks one question:
**can dynamic visual sparsity make ViSpec faster and leaner without hurting task quality?** It
tests two forms of sparsity. One prunes the visual features the draft receives. The other evicts
visual entries from the target's KV cache during verification. The stress case is long sequential
visual input (whole manga chapters), where the visual KV cache should matter most.

The answer has to stand on measured evidence from this cluster. A clear negative result counts as
an answer, for example "eviction saves memory but costs quality beyond 50% retention" or "draft
pruning has nothing to cut". A speedup claim built on outputs that silently changed does not.

## Starting point

The repository already contains:

- the upstream ViSpec code (`vispec/`), including a KV-cache fork of Qwen2.5-VL
  (`vispec/model/modeling_qwen2_5_vl_kv.py`), a Qwen branch in `SpecModel.specgenerate`, and Qwen
  handling in the TextVQA and GQA prompt builders. These Qwen paths have never been run here. A
  teammate tried Qwen on their own machine and failed; the cause is unknown.
- `gen_{baseline,spec}_answer_coco_caption.py` evaluators. They load `HuggingFaceM4/COCO` (test
  split), filter by image ID, shuffle with seed 42 and keep 100 images.
- a complete evidence harness from experiments 00 and 01 (LLaVA-1.6 7B on Newton H100s). It
  covers unique run folders, Git and device capture, a CUDA preflight, `report_metrics`,
  `compare_metrics`, plots, and the exact token-agreement checks. Those two experiments stay in
  the register as an earlier, separate result. They are not redone.
- the `vipsec` conda env on this cluster (Python 3.11, pinned `requirements.txt`, torch
  2.7.0+cu128, transformers 4.51.3). A CUDA preflight passed on an RTX 2080 Ti.

## Decisions made by the user

| Topic | Decision |
| --- | --- |
| Target model | `Qwen/Qwen2.5-VL-3B-Instruct` with the released draft `JLKang/ViSpec-Qwen2.5-VL-3B-Instruct`. No training. |
| Scope | Follow the plan across all five weeks. Manga is one part (Week 4), not the focus. |
| Who builds | The whole pipeline is built here, including the parts assigned to Mahd and Yusra in the plan. |
| Standard benchmarks | COCO Captions (CIDEr) and TextVQA (VQA accuracy), as subsets. GQA and MM-Vet are dropped for the Qwen work. |
| Manga task | Re:Verse story summary against the light-novel text. |
| Manga data | Get it from the original sources: Re:Verse from `sochastic/Re-Verse` (gated HF dataset, the user's account has access), and Manga109 from manga109.org under its academic licence. Do not use the copies inside `~/Research/Comic/Re-ID.zip`. |
| Compute | This Slurm cluster's Ampere GPU nodes, not Newton. |
| LLaVA work | Experiments 00 and 01 stay as recorded results. |
| Prompt styles | Run both styles as separate run groups. The inherited long-answer prompts measure speed and acceptance. Short-answer prompts measure CIDEr and VQA accuracy. |
| Prefill memory | Slim prefill the same way in every arm, baselines included: logits only where they are used, and only the hidden states the draft needs. This is output-preserving. Then report decode-phase memory and live KV size. Eviction stays after prefill, as the plan describes. |
| Manga prompts | Windows of about 6 pages at full resolution, each under the draft's 8,192-position limit. Each window's summary is scored against its own page-aligned light-novel segments. No whole-chapter prompts, no RoPE extension. |

## Goals

1. **Week 1, baselines.** Qwen2.5-VL-3B standard autoregressive decoding and unmodified ViSpec,
   on the COCO Captions and TextVQA subsets, with the full evidence harness. Each runs in two
   prompt groups: long-answer prompts for speed, short-answer prompts for quality. Greedy ViSpec
   should reproduce the autoregressive tokens. Any disagreement is measured and explained before
   moving on. The output-preserving prefill slimming is in place before the baselines are
   recorded, so every later arm compares against the same implementation.
2. **Week 2, the two sparsity methods,** each behind one switch with a retention budget. They must
   be correct at the identity setting: at 100% retention, outputs match the unmodified method
   token for token under greedy decoding.
   - *Draft-side compression*: spatial pooling or attention-guided pruning of the visual features
     the draft consumes.
   - *Target-side KV eviction*: after prefill, drop a share of the target's visual KV entries and
     keep every text entry, then run verification against the reduced cache.
3. **Week 3, retention sweeps.** Budgets of 25%, 50% and 75% (plus 100% as the control), for each
   method alone and for both combined, on COCO and TextVQA. Each run logs latency, VRAM and
   acceptance.
4. **Week 4, long-context stress test.** Re:Verse chapters are split into windows of about 6
   consecutive pages at full resolution, each under the draft's 8,192-position limit. The model is
   asked for a story summary of each window. This runs under autoregressive decoding, unmodified
   ViSpec and the sweep settings, scoring each summary against the light-novel segments for its
   pages, alongside the efficiency metrics. Manga109 is added as an efficiency and agreement stress
   set once licensed data is available (it has no light-novel reference).
5. **Week 5, analysis.** Speedup-versus-quality and memory-versus-quality trade-off curves and
   per-benchmark tables, written into the experiment READMEs, the register and the comparison page.

### Metrics

Each metric is reported per benchmark, with sample counts, before any aggregation:

- end-to-end latency, ms per output token and TTFT
- wall-clock speedup against autoregressive decoding and against unmodified ViSpec
- throughput
- peak GPU memory, and the live KV-cache size (see constraints)
- draft acceptance α (accepted tokens per verification step)
- output agreement with the uncompressed target, exact token and decoded text
- task quality: CIDEr for COCO, VQA accuracy for TextVQA, and ROUGE-1 plus BERTScore F1
  (bert-base-uncased, as in the Re:Verse paper) for the manga summaries

## Non-goals

- Training or fine-tuning any draft or target. The ViSpec adaptor has a fixed `num_q`, so query
  counts outside the released checkpoint are out of scope.
- Re-running the LLaVA experiments, or restoring the upstream eight-benchmark suite.
- Other Re:Verse tasks (character grounding, next-page prediction, VQA) and other VLM families.
- The plan's demo or UI.
- Paid judge calls. The Re:Verse paper also used a GPT-4o judge. It stays off by default, and its
  grades could be imported later the way MM-Vet grades are.

## Anti-goals

These would count as failure even if everything runs:

- Reporting a speedup or memory saving while outputs changed, without the agreement and quality
  numbers next to it.
- Comparing runs from different GPU models, settings or prompts as if they were matched.
- Claiming KV-eviction memory savings from a measurement that cannot see them (see the
  preallocation hazard below).
- Committing benchmark images or manga pages, or ignoring the Re:Verse or Manga109 licence terms.
- Substituting paper numbers for measurements.

## Constraints

- **AGENTS.md workflow applies.** That means experiments under `experiment/NN_descriptive_name/`,
  each with a run script and a README. Every launch captures the device report, Git state,
  environment and exit status. The register in `experiment/README.md` stays current, results go
  in `pending` until measured, and no test files are created. Correctness checks are lightweight
  scripts or recorded runs. AGENTS.md and CLAUDE.md currently pin LLaVA, TextVQA/GQA/MM-Vet and
  Newton. They must be updated to this scope as part of the work, while keeping 00/01 described
  correctly.
- **Hardware pinning.** The `gpu:ampere` GRES covers several GPU models: 48 GB cards on c1-8,
  c2-1, c2-2, c4-2 and c4-3, 80 GB cards on c4-4 and c4-5, and a mixed node c5-2. Every
  comparison is pinned to one GPU model, starting with the 48 GB nodes (**builder's call**), and
  the model is checked in each device report. torch 2.7.0+cu128 has no Pascal or Volta kernels.
- **Environment.** Jobs use `PYTHON_BIN=/home/aeternum/anaconda3/envs/vipsec/bin/python`.
  `experiment/submit.sh` and the run scripts are written for Newton (`NEWTON_*` variables, a
  `.venv` default). They need a path for this cluster that leaves 00/01 reproducible. A shell
  running inside the VS Code Slurm job must clear inherited `SLURM_*` variables before
  `srun`/`sbatch`.
- **Assets.** `prepare.sh`/`prepare_assets.py` is extended to fetch and validate both Qwen
  checkpoints at pinned revisions, the COCO subset with its references, TextVQA (already handled)
  and Re:Verse (gated, so it needs the user's HF token and accepted terms). Manga109 is placed by
  hand from the licensed zip. Downloads run on cluster nodes and stay out of Git.
- **Matched settings across a sweep:** greedy decoding (temperature 0), the same prompts, the same
  image resolution limits, the same output cap, and the same draft-tree settings (depth 3, top-k
  8, total tokens 30, `num_q` 2, unless a check shows the Qwen checkpoint needs others).

## Hazards found in the current code

These come from reading the code and checking the released configs. The builder should treat
them as known, not rediscover them.

1. **ViSpec's draft already compresses images.** `cnets_ours.Model.forward` replaces each image's
   token run with `num_q`=2 learned query outputs (`ImgAdaptor`) at prefill. The draft's
   per-step cost therefore does not grow with the number of visual tokens. Draft-side pruning can
   only change what the adaptor pools over and the target text hidden states the draft
   receives. Expect little speed effect; still measure it as the plan asks, and report it
   honestly. Most of the opportunity is on the target side.
2. **The KV cache is preallocated at `max_position_embeddings`.** `kv_cache.initialize_past_key_values`
   allocates every layer to that length, which is 128,000 for Qwen2.5-VL-3B: 36 layers × 2 KV
   heads × 128 dims in bf16, about 4.4 GiB. Evicting entries will not lower PyTorch peak memory
   while that buffer exists. The design needs a right-sized or compacting cache, plus a reported
   live-KV byte count (about 36 KiB per token for this model), so memory results are real.
3. **Positions vs. cache length.** Tree verification computes positions from `input_ids` length
   plus `rope_deltas`, not from the cache length. That is good for eviction, but cache slot
   bookkeeping (`current_length_data`), attention masks and the tree's cache-update paths must stay
   consistent after compaction. The 100%-retention identity check is the guard.
3a. **The visual KV cache is not what dominates memory.** At prefill, the target computes
   logits for every position (`lm_head` over all hidden states, about 297 KiB per token in bf16
   with the 151,936-token vocabulary). `SpecModel.forward` also keeps all 37 hidden-state layers
   (`output_hidden_states=True`, about 148 KiB per token). The KV cache itself is only about
   36 KiB per token. Evicting 50% of a 1,280-token image frees about 23 MiB. Peak VRAM and TTFT
   happen at prefill, before any eviction after prefill can act. Resolved: slim prefill in every
   arm, then report decode-phase memory and live KV.
4. **Inherited prompts are verbose.** The TextVQA prompt asks for an OCR explanation and the COCO
   prompt asks for a detailed description. On LLaVA this gave 0% strict TextVQA accuracy.
   Verbose prompts suit speed measurement but defeat CIDEr and VQA accuracy. Resolved: both styles
   run as separate groups.
5. **Context length for manga.** The prompt builders cap images at 1280×28×28 pixels, which is at
   most 1,280 visual tokens each. Re:Verse chapters have 18 to 41 pages, so a whole chapter is up
   to about 52k tokens. Two limits apply:
   - The target is designed for about 32k tokens natively. Its M-RoPE positions grow by image
     grid size, not token count.
   - **The draft is the tighter limit.** `cnets_ours.Model.topK_genrate` gives each draft token
     its index in the full, uncompressed sequence (`len_posi = input_ids.shape[1]`). The draft's
     RoPE table is a fixed buffer sized by the released checkpoint's `max_position_embeddings`
     (8,192). `LlamaRotaryEmbedding.forward` only slices that buffer, so unmodified ViSpec will
     index past it on any prompt longer than about 8k tokens.

   The draft's prefill also builds a `torch.eye(seq_length)` matrix, about 2.4 GiB at 36k tokens.
   Resolved: windows of about 6 pages, each under 8,192 positions.
5a. **Re:Verse text looks page-aligned.** `data/text/chapter_001.txt` has 23 `________________`
   separators, and chapter 1 has 24 page images (`001_0006` to `001_0029`). Before relying on it,
   the builder checks that every chapter's segment count matches its page count. If it does,
   page windows can be scored against their own light-novel segments.
6. **Scoring dependencies.** The usual CIDEr implementation (pycocoevalcap PTBTokenizer) needs Java,
   and none is on the login/VS Code node. ROUGE and BERTScore need new packages, and BERTScore
   needs the bert-base-uncased weights. All of these must be added to the pinned environment and
   prepared ahead of time, so offline jobs can score.
7. **COCO loader.** `HuggingFaceM4/COCO` is a dataset loading script. It needs remote-code trust
   and may pull full COCO 2014 archives. A lighter path, such as val2014 plus the Karpathy split
   JSON, is the **builder's call** if it yields the same image subset and references.

## Left to the builder

- Experiment numbering and names after 01, as long as each method and setting is its own
  experiment or clearly separated run group.
- The exact pruning selectors: one structured, spatial policy and one attention-guided policy for
  each side is the default (**builder's call**). Attention-guided scoring needs attention weights
  that SDPA does not return, so the builder chooses how to compute them and records that cost
  inside the timing.
- How evaluators are extended (new flags vs. new modules), given that each benchmark script is a
  near-copy and changes must reach every in-scope copy.
- The COCO and TextVQA subset size. The default is to keep the existing 100-sample seed-42 subsets
  (**builder's call**). Sample counts are always reported.
- The plot set, beyond the trade-off curves named in the goals.

## Calls made without the user (check these)

- **Precision is bf16,** Qwen2.5-VL's native dtype, for every Qwen arm. The 00/01 LLaVA runs used
  FP16. They are a separate result, so this does not break a comparison.
- **Week 1 gate.** Exact greedy agreement between autoregressive decoding and ViSpec is measured
  but not required on every sample (LLaVA showed 51/418 divergences that were never explained).
  Every divergence is diagnosed at its first differing token, using the margin between the top
  two target logits, plus an fp32 spot check on a sample of cases. The gate closes when the
  divergences are explained as near-ties, or a bug is found and fixed. Week 1 is not extended
  for open-ended exact-match work.
- **Agreement reference.** Each sparsity arm is compared against the same decoding method at 100%
  retention (ViSpec arms against unpruned ViSpec), so baseline noise does not leak into
  sparsity agreement. Agreement with autoregressive decoding is reported as a secondary column.
- **Throughput is single-stream (batch size 1).** The ViSpec tree code only supports batch 1.
  Memory savings are reported as memory, not converted into batched throughput.
- **Manga109 is optional.** It joins Week 4 only if licensed data is available. Re:Verse is the
  primary manga set.

## Resolved after review

The builder check raised three questions only the user could answer. The user accepted the
recommended answer for each; they are recorded in the decisions table above.

1. **Prompt format** (hazard 4): both styles, as separate run groups.
2. **Eviction savings** (hazard 3a): slim prefill identically in every arm, then report
   decode-phase memory and live KV. Eviction stays after prefill.
3. **Manga context** (hazard 5): page windows of about 6 pages, scored against aligned
   light-novel segments.

Short-answer prompt wording and window boundaries are the **builder's call**. The defaults are a
standard short-answer instruction per benchmark, and non-overlapping windows. The same prompts
and windows are used in every compared arm.
