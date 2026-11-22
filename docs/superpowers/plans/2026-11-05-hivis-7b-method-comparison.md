# HiViS on Qwen2.5-VL-7B: Method Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add HiViS as a second speculative-decoding method next to ViSpec, on Qwen2.5-VL-7B. Then
measure autoregressive (AR) decoding, ViSpec and HiViS on the same TextVQA and COCO subsets as Weeks
1–3, in both prompt styles. Two short secondary arms, in the `long` style only, measure HiViS's own
tree setting and an image-first prompt order.

**Architecture:**
1. Vendor the HiViS drafter with a small adapter, so that the unchanged tree loop in
   `vispec/model/utils.py` drives it. Select it in `SpecModel` with a default-preserving
   `draft_method`.
2. Add `--method hivis` and `--prompt-order` to the evaluator, a pinned `qwen7b` asset suite, and
   three-method support in the comparison tools. Every 3B default path stays byte-identical.
3. Verify the port in three stages: on the CPU (strict load, synthetic adapter checks), on the GPU
   (fp32 losslessness, acceptance, 3B identity), and against HiViS's own code, run once in a
   separate environment.
4. Run experiment 10 through the existing arm runner with a new `methods` sweep type. Each
   (method, tree, prompt order) gets one run folder, and every job carries its own AR reference.

**Tech Stack:** Python 3.11 (`vipsec`), torch 2.7.0+cu128, transformers 4.51.3, bash, and Slurm on
the pinned RTX A6000s. The reference run alone also uses a separate conda environment: Python 3.9,
torch 2.6.0 (cu124) and transformers 4.54.0.

**Spec:** the integration analysis
`/home/aeternum/Research2/cv_systems_project/.superpowers/recon/hivis-integration.md` (untracked, on
project storage). It holds the file:line evidence for the HiViS algorithm, our fork, the defaults,
the 4.54 against 4.51.3 differences, the change list, the verification strategy and the risks. The
asset note is `.superpowers/recon/assets-7b.md`. **The user's decisions of 2026-11-05, listed as
fixed decisions under Global Constraints, are the binding spec.** Where the analysis and a decision
differ, the decision wins. For example, the analysis proposed `experiment/10_qwen_hivis/` with the 02
and 03 runners reused, while the decision is one experiment, `experiment/10_qwen7b_methods/`, that
runs all methods. This plan follows
[`2026-10-31-weeks4-5-longcontext-and-analysis.md`](2026-10-31-weeks4-5-longcontext-and-analysis.md).
It reuses that plan's arm runner, `sweep_table`, `offset_check` and `diagnose_divergence`.

## Global Constraints

**Fixed decisions (do not reopen):**

- **Target and drafts.**
  - Target: `Qwen/Qwen2.5-VL-7B-Instruct` @ `cc594898137f460bfe9f0759e9844b3ce807cfb5`.
  - Drafts: `JLKang/ViSpec-Qwen2.5-VL-7B-Instruct` @ `e06635b54327df78a5eb82361791f2ccec9f41ab`
    and `Irisssme/HiViS-Qwen2.5-VL-7B-Instruct` @ `6dd8717d912a5626d6ad4e81b124c54bb8a08ef2`.
  - All three are already in `models/` (CPU job 402272).
  - The 3B work stays as it is. Every 3B default path stays byte-identical: ViSpec and AR tokens,
    and the input fingerprints.
- **Port HiViS into our harness.**
  - Vendor the drafter as `vispec/model/cnets_hivis.py`, with an attribution header naming the
    source repository, commit `6db36b2`, Apache-2.0 and the paper (arXiv 2509.23928, CVPR 2026
    Findings).
  - Add an adapter so that our unchanged `utils.py` tree functions drive the drafter.
  - Add a default-preserving `draft_method` in `SpecModel`, and `--method hivis` in `evaluate.py`.
  - HiViS refuses the draft-pruning flags. The compact layout and slim prefill apply to it as they
    do to ViSpec.
  - KV eviction on 7B may be technically allowed, but it is out of scope and is not run.
- **Assets.** A `qwen7b` suite in `experiment/prepare_assets.py` and `prepare.sh` verifies the
  three pinned checkpoints, and downloads any that are missing. Existing suites are unchanged.
- **Primary comparison (experiment 10, `experiment/10_qwen7b_methods/`).**
  - AR, ViSpec and HiViS on Qwen2.5-VL-7B, on TextVQA and COCO: the same 100 samples each as in
    Weeks 1–3, in both prompt styles.
  - Our standard settings: bf16, greedy, a 1,024-token cap, tree total 30 / depth 3 / top-k 8, our
    prompts with the image last, the compact layout and slim prefill.
  - One Slurm job per style, with all methods in the same job.
  - Each job should finish in about 2 h or less, because preemptible jobs here are killed after
    1–2 h. A style may be split into more jobs, but every job that reports speedups contains its
    own AR reference.
- **Secondary arms (long style only, both benchmarks; separate short jobs, each with its own
  references).**
  - Tree setting: ViSpec and HiViS at total 60 / depth 5 / top-k 10. These are HiViS's CLI values
    (`ge_hivis_answer.py:263-265`, analysis section 3; the paper calls this "top-10, depth 6, top-60
    paths" because it also counts the first draft step). The jobs also run a same-job AR reference
    and same-job ViSpec and HiViS at 30/3/8.
  - Prompt order: `--prompt-order {image-last,image-first}`. `image-last` is the default and stays
    byte-identical. AR, ViSpec and HiViS run with `image-first` at 30/3/8, next to a same-job
    `image-last` reference set.
- **Comparison tools.** `compare_metrics`, `sweep_table`, `diagnose_divergence` and `offset_check`
  handle HiViS and the 7B model. Every existing behaviour and output stays unchanged for the current
  runs, and a re-reported Week 3 run is byte-identical.
- **Port verification**, in this order:
  1. CPU: a strict load, and synthetic adapter checks.
  2. GPU smoke on 5 long TextVQA samples. It covers fp32 HiViS == AR, bf16 acceptance in a sane
     range, a ViSpec-7B smoke, and 3B ViSpec and AR byte-identity against the Week 2 job 393336
     runs.
  3. A reference run of HiViS's own code. **The user approved it on 2026-11-05 ("you have my go
     ahead"), so it is a regular task**, isolated as described in Task 4.
- **Write-up framing.**
  - Lead with each method's speedup against AR, then acceptance, time per output token and TTFT.
  - Then describe the draft's visual input as a spectrum: an EAGLE-style draft sees every image
    token, ViSpec compresses each image to 2 tokens, and HiViS sees none. Relate this to the Week 3
    finding that pruning ViSpec's draft input changed nothing.
  - Be honest about prompt order and tree size.
- **Out of scope:**
  - Re:Verse+ on 7B (revisit after the 3B Re:Verse run lands);
  - KV eviction on 7B;
  - training any draft;
  - LLaVA targets;
  - other methods (MSD, EAGLE-3).

**Working rules:**

- **No test files** (`AGENTS.md`). Verify with heredoc checks, `py_compile`, `bash -n`,
  `sbatch --test-only`, recorded GPU runs, re-reports and comparisons.
- **Plain wording.** READMEs, `METRICS.md` and commit messages contain no task numbers, decision
  codes or "controller". Commit messages carry no `Co-Authored-By`, no `Claude-Session` and no Claude
  attribution of any kind.
- **Never write, link or run anything inside** `/home/aeternum/Research/Comic/M3/reverse_plus`.
- **Evidence.** Every runner, the new check and reference runners included, follows the capture
  pattern of `experiment/03_qwen_vispec/run.sh` and `experiment/qwen_arm_runner.sh`. That means:
  - the commit, `git status` and the diff;
  - a copy of the runner, the package list and the device report;
  - the asset check and the CUDA check;
  - `metadata.txt` with the Slurm job ID and restart count;
  - `output.log`, the exit code, and an EXIT-trap report.
- **GPU submission.** Submit with this prefix, from a clean tree of the merged branch:

  ```bash
  CLUSTER_PARTITION=preempt CLUSTER_EXCLUDE=c4-4,c4-5,c5-2,c2-2 \
    SBATCH_EXTRA="--requeue --cpus-per-task=4 --mem=<32G|48G> --time=<per job>" \
    bash experiment/submit_ampere.sh <dir>
  ```

  - `run_sequence.sbatch` already sets `--requeue` and checks the GPU name (`NVIDIA RTX A6000`).
    The explicit flag records it in `submit_command`.
  - Use `--mem=48G` for jobs that load the 7B target in fp32 (port checks, diagnosis), and
    `--mem=32G` otherwise. Loading a 7B draft holds about 10–11 GB of host memory for a moment
    (analysis risk 8).
  - Hold at most two experiment 10 GPU jobs at once, because the user's own jobs share these
    nodes. Chain further jobs with `--dependency=afterany:<id>` in `SBATCH_EXTRA`.
- **CPU allocations** (downloads, environment build, CPU checks):
  `source experiment/ampere_env.sh && clear_inherited_slurm && srun -p normal --cpus-per-task=4 --mem=<n>G --time=<t> <cmd>`.
- **Storage.** Runs, comparisons and sweep outputs stay out of Git. Never modify existing files in
  `experiment/*/runs/`. Week 2 and Week 3 evidence is read from the main checkout,
  `/home/aeternum/Research2/cv_systems_project/experiment/...`. Scratch output goes to
  `.smoke/hivis_*` (git-ignored).
- **Branch.** `qwen-hivis-7b` at `dfb2e48`. It contains the per-image vision encoding and the
  multi-image draft fix, so it is the base the analysis asks for.
- **Job size rule.** Project each job from measured latencies before submitting it. Task 3 gives
  5-sample latencies, and Task 7 gives full per-arm durations from `metadata.txt`
  `started_utc`/`finished_utc`. If a projection exceeds 110 min:
  1. split by benchmark first;
  2. then split by method;
  3. every split job keeps an AR reference, one per prompt order present.

**Sanity range: HiViS's own numbers for Qwen2.5-VL-7B on TextVQA at T = 0 (arXiv 2509.23928 v3,
fetched 2026-11-05).** τ is the paper's "average number of drafter tokens accepted per verification
step", and SR is the speedup ratio.

| Paper setting | Our code equivalent | EAGLE-2 τ / SR | ViSpec τ / SR | HiViS τ / SR |
| --- | --- | --- | --- | --- |
| ViSpec's settings (Table 5: 30 draft tokens, depth 4, 8 per step) | 30/3/8 | 2.44 / 2.34× | 3.22 / 2.94× | 3.34 / 3.04× |
| HiViS's settings (Table 2: top-10, depth 6, top-60 paths) | 60/5/10 | 2.78 / 2.31× | 3.60 / 2.87× | 3.95 / 3.15× |

- **The paper's harness.** A single A800 at batch size 1, an 80-sample subset, and per the code: the
  first 80 TextVQA *test* rows, fp16, at most 500 new tokens, an image-first prompt and processor
  default pixel limits.
- **τ and our metrics.** HiViS's code averages `accept_length`, which counts accepted draft tokens
  without the target's own token. That matches our "accepted draft tokens per verification
  iteration" (`METRICS.md`). The paper does not say whether its τ adds 1, so the mapping is
  uncertain by up to one token.
- **The HiViS–ViSpec gap.** In the paper, HiViS leads ViSpec by 0.12 τ at 30/3/8 and by 0.35 τ at
  60/5/10. This is why the tree arm matters.
- **Our own reference point.** 3B ViSpec on long TextVQA at 30/3/8 accepted 2.141 draft tokens per
  iteration at a 2.158–2.200 speedup (Week 3 jobs 398883–398885).
- **The smoke range that follows.** Expect roughly 1.8–3.4 accepted draft tokens per iteration for
  both 7B drafts at 30/3/8 on our image-last prompts. **Gate: at least 1.5 for each.** A value near
  1 or below means a broken drafter. These numbers are context only, never results (`AGENTS.md`).

## Review Focus

1. **Silent acceptance loss in the port.** Greedy speculative decoding is lossless, so a drafter
   bug changes no output and only lowers acceptance. The likely faults are:
   - the positions or `add` offset;
   - the mask span (the `numel - 1` clamp, and the extra removed token);
   - the residual index;
   - the embedding source (analysis risk 2).

   *(Task 1: equality with HiViS's own recipe on synthetic input. Task 3: the position ablation.
   Task 4: exact prune and position equality, and first-tree equality against HiViS's code on
   identical inputs.)*
2. **3B default paths stay byte-identical.** The prompt builder, `SpecModel` and the arm runner are
   shared with every 3B run. *(Task 2: 400 prompt fingerprints. Task 3: 5/5 tokens against Week 2.
   Task 6: runner command diffs. Task 5: re-report, comparison and sweep-table diffs.)*
3. **Comparison confounds.** Every ratio is computed with one tree, one prompt order and one job.
   The tree and order arms carry their own references. The paper's numbers come from a different
   harness and are cited as context only. *(Tasks 7–9.)*
4. **Guards are loosened only where intended.** AR may be compared with HiViS although the recorded
   draft paths differ. Nothing else is relaxed: base model, prompts, hardware, layout or settings.
   *(Task 5 negative checks.)*
5. **Job length against preemption.** *(The GPU Budget section and the job size rule.)*
6. **Isolation of third-party code.** HiViS's code runs only in its own environment, from a
   read-only checkout outside `/tmp`, offline and without credentials. Only safetensors and JSON
   files cross between the two environments. *(Task 4.)*

## GPU Budget

The estimates use the analysis's 7B figures (section 9):

- AR: 13–16 s per long TextVQA answer and 5–6 s per long COCO answer.
- ViSpec: 6–8 s and 2.5–3 s.
- HiViS: within ±20% of ViSpec. The 60/5/10 arms are assumed to land in the same range.
- Every evaluator call adds 2–4 min for loading the 16.6 GB target and three warm-ups.

They are estimates, not measurements, and the job size rule replaces them with measured values.

| Job | Contents (100 samples per benchmark unless stated) | Estimate | `--time` | `--mem` |
| --- | --- | ---: | --- | --- |
| Port checks | 3B identity (2 × 5), fp32 oracle (2 × 5), bf16 smoke (3 × 5), 60/5/10 (2 × 2), position ablation (5) | 40–60 min | 01:30:00 | 48G |
| HiViS reference | 20 samples, fp16, max 500 new tokens: HiViS's speculative and AR paths, then ours plus the near-tie margins | 30–45 min | 01:30:00 | 48G |
| Primary, long | AR, ViSpec, HiViS × TextVQA, COCO | 67–103 min | 02:30:00 | 32G |
| Primary, short | AR, ViSpec, HiViS × TextVQA, COCO | 27–39 min | 01:00:00 | 32G |
| Diagnosis | 4 divergence diagnoses (AR against each method, both styles) and 2 fp32 oracles | 50–70 min | 01:30:00 | 48G |
| Tree, TextVQA, ViSpec | AR, ViSpec 30/3/8, ViSpec 60/5/10 | 46–70 min | 02:00:00 | 32G |
| Tree, TextVQA, HiViS | AR, HiViS 30/3/8, HiViS 60/5/10 | 44–73 min | 02:00:00 | 32G |
| Tree, COCO | AR, ViSpec and HiViS at both trees | 30–54 min | 01:30:00 | 32G |
| Order, TextVQA, ViSpec | AR and ViSpec, each image-last and image-first | 72–98 min | 02:30:00 | 32G |
| Order, TextVQA, HiViS | AR and HiViS, each image-last and image-first | 68–104 min | 02:30:00 | 32G |
| Order, COCO | AR, ViSpec and HiViS, each order | 42–66 min | 01:30:00 | 32G |

**Total: about 9–13 GPU-hours over 11 jobs.** About 1 CPU-hour more goes to the asset check, the CPU
checks and the reference environment build.

**Why the TextVQA secondary arms are split by method.** With all five tree arms, or all six order
arms, one TextVQA job would take 98–143 min. Each split job still holds AR and both settings of its
method, so each method's tree or order effect is measured within one job. COCO fits in one job each.
A requeued job reruns its whole runner into fresh folders, and the tables take the final attempt.

## Task Order and Parallel Work

- **Parallel from the start**, in separate worktrees:
  - Task 0 (assets);
  - Task 1 (drafter);
  - Task 5 (comparison tools);
  - Task 6 (experiment 10 runner);
  - Task 4 Steps 1–3: the environment build on a CPU node, and the reference code written against
    Task 1's adapter interface.
- **Task 2** needs only the `draft_method` keyword from Task 1. Write it in parallel against the
  interface below, and merge it after Task 1.
- **Task 3** is the first GPU step. It needs Tasks 0, 1 and 2 merged.
- **Then:**
  - Task 4's GPU run follows a passing Task 3.
  - Task 7 submits after Tasks 3 and 4 pass and Task 6 is merged. Its gates need Task 5.
  - Task 8 follows the primary long job, whose durations it uses for projection.
  - Task 9 comes last.
- **Merge order:** 0, 1, 2, 3, then 4, 5 and 6 in any order.

---

### Task 0: Pinned `qwen7b` asset suite

**Files:**
- Modify: `experiment/prepare_assets.py` (`QWEN7B_MODELS`, `run_qwen7b`, `--suite qwen7b`),
  `prepare.sh` (usage comment only; arguments already pass through)

**Interfaces:**
- **Pinned models.** `QWEN7B_MODELS` maps a key to `(repo_id, folder, revision)`:
  - `base`: `Qwen/Qwen2.5-VL-7B-Instruct`, `Qwen2.5-VL-7B-Instruct`,
    `cc594898137f460bfe9f0759e9844b3ce807cfb5`;
  - `vispec_draft`: `JLKang/ViSpec-Qwen2.5-VL-7B-Instruct`, `ViSpec-Qwen2.5-VL-7B-Instruct`,
    `e06635b54327df78a5eb82361791f2ccec9f41ab`;
  - `hivis_draft`: `Irisssme/HiViS-Qwen2.5-VL-7B-Instruct`, `HiViS-Qwen2.5-VL-7B-Instruct`,
    `6dd8717d912a5626d6ad4e81b124c54bb8a08ef2`.

  The revisions are pinned in code. The suite never resolves the latest revision; the 3B suite
  does, and stays as it is.
- **Full preparation** (compute node only):
  1. A folder counts as present when `validate_model` passes and the first line of every
     `.cache/huggingface/download/*.metadata` file equals the pinned revision. A present folder is
     not downloaded. Otherwise the suite runs
     `snapshot_download(repo_id, revision=<pin>, local_dir=models/<folder>)`.
  2. It then hashes every `*.safetensors` file with sha256. The hash must equal the LFS etag on the
     second line of that file's metadata.
  3. The three folders already exist, so the expected run downloads nothing.
- **Manifest** `.preparation/assets_qwen7b.json`:
  - top-level fields: `schema_version`, `suite: "qwen7b"`, `ready`, `finished_utc`;
  - `models.<base|vispec_draft|hivis_draft>` with `repo_id`, `revision`, `path` and
    `files: {name: {bytes, sha256}}`;
  - `data_manifest` with the path and `finished_utc` of `.preparation/assets_qwen.json`.

  The `base` key keeps the 3B manifest's shape, so the evaluator reads both manifests the same way.
- **Data.** The 7B runs use the 3B suite's TextVQA and COCO files. The suite requires a ready
  `assets_qwen.json` and runs the existing `validate_textvqa` and `validate_coco`. It never
  re-downloads or rewrites those files.
- **Check-only** (`--check-only --suite qwen7b`). It is offline and runs in every experiment 10
  runner. It checks:
  - the manifest is ready;
  - the folders validate;
  - the metadata revisions equal the pins;
  - every weight file's size equals the manifest (no hashing, so the check takes seconds);
  - the data validators pass.

  It uses the same shared lock as the other suites.

- [ ] **Step 1:** Implement.
- [ ] **Step 2: Run** in a CPU allocation:
  `srun -p normal --cpus-per-task=4 --mem=8G --time=00:45:00 bash prepare.sh --suite qwen7b`
  (with `PYTHON_BIN` set by `ampere_env.sh`).
  - Expected: no download, three ready lines with the pinned revisions, three hash matches on the
    weight files, and the manifest.
  - Then `bash prepare.sh --check-only --suite qwen7b` prints the completion line.
- [ ] **Step 3: Verify that the existing suites are unchanged.**
  - Run `bash prepare.sh --check-only --suite qwen` and `--suite reverse` before and after the
    change, and diff their output (ignoring the `Preparation log:` path line).
  - `sha256sum .preparation/assets_qwen.json .preparation/assets_reverse.json` must not change.
  - Negative check: use `--repo-dir .smoke/hivis_assets/root`, a scratch root whose `models`,
    `bench_data` and `tools` are symlinks and whose `.preparation` holds copied manifests with one
    `bytes` value changed. Check-only must fail and name that file.
- [ ] **Step 4: Commit.** `git commit -m "Add a pinned Qwen2.5-VL-7B asset suite"`

### Task 1: Vendor the HiViS drafter and select it in `SpecModel`

**Files:**
- Create: `vispec/model/cnets_hivis.py`
- Modify: `vispec/model/spec_model_ours.py` (`draft_method` in `__init__`, `from_pretrained` and
  `specgenerate`)

**Interfaces:**
- **Vendored code.** `HiViS` below means the read-only checkout at
  `/home/aeternum/Research2/cv_systems_project/.cache/ext/HiViS` (Step 1).
  - Copy `hivis/model/cnets_hivis.py` lines 1–1000 verbatim: the rotary classes, the Llama layer
    and `Model`. Leave out `acc`, `Vhead` and `__main__` (lines 1002–1084).
  - Append `build_mask` and `prune_image_tokens` from `hivis/model/utils_hivis.py:21-105`
    verbatim.
  - The only edit inside copied code is the import: `from .configs import EConfig` replaces the
    try/except import. Our `EConfig` parses the HiViS config (analysis section 5).
- **Header.** Keep the original Apache-2.0 notice, and add:
  - the source, `https://github.com/lnn-ops/HiViS`, at commit
    `6db36b241eaf733ec313cbc9b86206c0f20762c9`, with both source files and their line ranges;
  - the licence, Apache-2.0. The repository's own `LICENSE` is already Apache-2.0;
  - the paper: "HiViS: Hiding Visual Tokens from the Drafter for Speculative Decoding in
    Vision-Language Models", arXiv:2509.23928, CVPR 2026 Findings;
  - a list of what was changed, as Apache-2.0 §4(b) requires.
- **Adapter** `HiViSDraft(Model)`:
  - **Constructor.** `__init__(config, *, bias, total_tokens, depth, top_k, threshold,
    residual_count, vision_start_id, image_token_id, video_token_id)`. It asserts that the three
    IDs equal HiViS's literals, 151652, 151655 and 151656. `SpecModel` passes them from the target
    config.
  - **Signature.** `topK_genrate(hidden_states, input_ids, head, logits_processor,
    inputs_embeds=None, embed_weights=None, image_mask=None)`. This is the signature our
    `initialize_tree` and `update_inference_inputs` already call.
  - **First call** (`stable_kv is None`), from the unchanged `initialize_tree`:
    1. The inputs are `input_ids` (prompt plus first token, `L + 1`), `hidden_states` and
       `inputs_embeds` (the full prompt, `L`).
    2. Mask `input_ids[0, :-1]` with `build_mask`. Masking the prompt length makes the `numel - 1`
       clamp match `utils_hivis.py:359-365`.
    3. Keep the IDs, target embeds and hidden states under the mask. The positions are the kept
       indices of `arange(L)`.
    4. Build `draft_embeds = cat(kept_embeds, self.embed_tokens(token))`.
    5. Store `prompt_len = L`, `pruned_prompt_ids` and `prompt_positions`.
    6. Call the parent with `cat(kept_ids, token)`, `input_embeds=draft_embeds` and
       `position_ids=positions`. The parent then sets `self.add = L - P`, where `P` is the kept
       length.
  - **Later calls**, from the unchanged `update_inference_inputs`, which passes the full IDs plus
    the new token. Build `draft_ids = cat(pruned_prompt_ids, input_ids[:, prompt_len:])` and call
    the parent. This is HiViS's own `cat(draft_input_ids, token)` (analysis 7.1).
  - **What the drafter keeps of our prompts.**
    - Image last: it drops `<|vision_start|>` … `<|vision_end|>` and `<|im_end|>`. That leaves the
      4-token tail `\n<|im_start|>assistant\n` as the only prompt text after the image.
    - Image first: it drops the span plus the first question token, as in HiViS's own evaluation.
  - **Guards** (each raises `ValueError`):
    - more than one `<|vision_start|>`. HiViS's `.item()` would crash, and multi-image input is
      outside its training (analysis 7.4);
    - any video pad;
    - an `image_mask` position outside the removed span.

    A prompt without an image keeps every token (`add = 0`), as HiViS does.
  - **Reset.** `reset_kv()` also clears the prompt state and sets `add = 0`.
  - **Ablation switch.** The class attribute `position_mode = "original"` can be set to `"pruned"`,
    which passes `arange(P)` as the positions, so `add` is 0. Only the Task 3 position ablation
    sets it.
  - **Draft pruning.** `draft_sparsity = None`, so `draft_pruning` stays inert.
- **`SpecModel`:**
  - **The new argument.** `DRAFT_METHODS = ("vispec", "hivis")`. `__init__(..., kv_layout="fixed",
    draft_method="vispec")` and `from_pretrained(..., kv_layout="fixed", draft_method="vispec",
    **kwargs)` take it. An unknown value raises before any weights load, and `self.draft_method`
    records the value.
  - **Checkpoint check.** Before the target loads, `from_pretrained` reads the draft's tensor names
    with `safetensors.safe_open`, when `model.safetensors` exists.
    - A `residual` tensor marks a HiViS checkpoint, and `imadpt.q` a ViSpec one.
    - A mismatch with `draft_method` raises.
    - Drafts stored as `pytorch_model.bin` (the LLaVA path) skip this check.
  - **The `"vispec"` branch** is the existing code, byte for byte:
    `Model(config, bias, total_tokens, depth, top_k, threshold, num_q)` and `strict=False`, with the
    missing and unexpected keys printed.
  - **The `"hivis"` branch.** `load_hivis_draft(config_path, state_dict, total_token, depth, top_k,
    threshold, base_config)` builds `HiViSDraft` with
    `residual_count = state_dict["residual"].shape[0]` and loads it with `strict=True`. The shared
    `diff_device` setup, `.to(dtype).to(device)` and `init_tree()` follow.
  - **Refusals in `specgenerate`.** It raises `ValueError` for `draft_sparsity` when `draft_method ==
    "hivis"`, because the HiViS draft sees no image tokens. Target KV eviction stays technically
    allowed, because it acts on the target side only (analysis 7.4), but no experiment here runs it.
- **Unchanged:** `utils.py`, `cnets_ours.py`, `autoregressive.py`, `kv_cache.py`,
  `modeling_qwen2_5_vl_kv.py` and `sparsity.py`.

- [ ] **Step 1: Copy the HiViS checkout** to a location that compute nodes can see and Git ignores
  (`.cache/` is ignored):

  ```bash
  git clone --no-hardlinks /tmp/claude-1016/-home-aeternum-Research2-cv-systems-project/22915a62-b721-40aa-bed5-c56e919050a1/scratchpad/ext/HiViS /home/aeternum/Research2/cv_systems_project/.cache/ext/HiViS
  git -C .cache/ext/HiViS checkout --detach 6db36b241eaf733ec313cbc9b86206c0f20762c9
  ```

  - `git rev-parse HEAD` must equal the pin, and `git status --porcelain` must be empty.
  - Then run `chmod -R a-w .cache/ext/HiViS`.
  - This is a local clone, with no network. If the scratch source is gone, stop and ask the user,
    because a clone from GitHub is network use beyond package installs.
- [ ] **Step 2:** Implement.
- [ ] **Step 3: Verbatim check** (read-only, login node). Diff `sed -n 1,1000p` of the HiViS file,
  and `sed -n 21,105p` of `utils_hivis.py`, against the matching ranges of
  `vispec/model/cnets_hivis.py`. Only the header and the import line may differ.
- [ ] **Step 4: Verify on CPU** with heredocs, in one CPU allocation
  (`--mem=24G --time=00:30:00`):
  - **Strict load.** `load_hivis_draft` on `models/HiViS-Qwen2.5-VL-7B-Instruct` in fp32.
    - It loads 15 tensors with the shapes in analysis section 4: `fc.weight [3584, 10752]`,
      `residual [4, 3584]`, `k_proj [512, 3584]` and the others.
    - `residual_count` is 4.
    - `EConfig` gives `num_key_value_heads=4`, `max_position_embeddings=128000` and `qkv_bias=True`.
  - **ViSpec-7B.** `cnets_ours.Model` with `num_q=2`, loaded with `strict=False` from
    `models/ViSpec-Qwen2.5-VL-7B-Instruct`, reports no missing and no unexpected keys.
  - **Method mismatch.** `SpecModel.from_pretrained` raises before the target loads for:
    - the HiViS checkpoint with `draft_method="vispec"`;
    - the ViSpec-7B checkpoint with `"hivis"`;
    - `"eagle"`.
  - **Synthetic adapter checks.** These use a tiny config (hidden size 64, 4 heads, 1 KV head, the
    real vocabulary size), random weights, fp32 and `.eval()`:
    - **Mask.** Take an image-last prompt with `L = 20`, `<|vision_start|>` at 5, six image pads
      and `<|vision_end|>` at 12. The kept indices must be `0–4` and `14–19`. Check that by hand,
      and also against the analysis's rule (remove `start … end + 1`, with
      `end = min(start + count + 1, L - 1)`). Then check an image-first prompt, and a text-only
      prompt (everything kept, `add = 0`). Two vision starts, a video pad, or an `image_mask`
      outside the span each raise.
    - **First call.** The adapter's four outputs equal, under `torch.equal`, those of the vendored
      parent called with HiViS's own recipe: `prune_image_tokens(..., text_position_ids=arange(L))`
      and the concatenated draft embeds (`utils_hivis.py:349-420`). Also
      `prompt_positions == kept indices` and `add == L - P`.
    - **Second call.** Commit 3 accepted tokens and 1 new token. The outputs equal HiViS's recipe
      `cat(draft_input_ids, accepted, token)`. A forward pre-hook on `layers[0].self_attn` records
      the positions:
      - the accepted pass uses the logical indices `L … L + 3`;
      - tree depth `i` uses its parent's logical index `+ i`.
    - **Reset.** After `reset_kv()`, a second prompt of a different length starts from fresh
      positions.
    - **Ablation.** With `position_mode = "pruned"`, the prompt positions are `arange(P)` and `add`
      is 0.
- [ ] **Step 5: Commit.** `git commit -m "Add the HiViS drafter and select it in SpecModel"`

### Task 2: The `hivis` method and prompt order in the evaluator and the fp32 oracle

**Files:**
- Modify: `vispec/evaluation/benchmarks.py`, `vispec/evaluation/evaluate.py`,
  `vispec/evaluation/offset_check.py`, `vispec/evaluation/profile_generation.py`

**Interfaces:**
- **Prompt order in `benchmarks`.**
  - New names: `PROMPT_ORDERS = ("image-last", "image-first")` and `check_prompt_order(benchmark,
    order)`.
  - A trailing keyword `prompt_order="image-last"` is added to `build_messages(benchmark, sample,
    prompt_style, ...)`, `build_inputs(processor, benchmark, sample, prompt_style, device, ...)` and
    `prompt_layout(...)`. Existing positional calls are unaffected.
  - `image-last` runs the existing code path unchanged.
  - `image-first` moves the single `{"type": "image"}` item to the front of the user content:
    `[image, question, instruction]` for TextVQA and `[image, instruction]` for COCO. The system
    text, the instructions and the join without a separator stay ours.
  - `reverse_summary` keeps its fixed pages-first layout and refuses `image-first`, because the flag
    selects a single-image layout.
- **`evaluate`:**
  - **Methods.** `METHODS = ("autoregressive", "vispec", "hivis")`. `generate()` sends `hivis` to
    `specgenerate`, the same call ViSpec makes.
  - **The new flag.** `--prompt-order {image-last,image-first}`, default `image-last`. It is added
    last in `add_generation_arguments`, so the earlier `generation_args` keys keep their order. Every
    `build_inputs` call passes it.
  - **Draft pruning.** `draft_sparsity_config()` refuses `hivis` with the message that its draft
    sees no image tokens.
  - **Draft method.** `load_model()` passes `draft_method="hivis" if method == "hivis" else
    "vispec"`. AR keeps loading the ViSpec draft, whose `total_tokens` sets AR's compact-layout
    slack (analysis 7.2).
  - **`runtime.json`** gains `draft_method`, `spec_model_revision` and `spec_model_revision_source`.
  - **Revision lookup.** `prepared_base_revision` and the new `prepared_spec_revision` search
    `.preparation/assets_qwen.json` and then `assets_qwen7b.json` for an entry whose resolved path
    equals the given path. The 3B values and source strings are unchanged. The 7B runs now record a
    non-null `base_model_revision`, which closes the evidence gap the analysis found.
- **`offset_check`.** New flags: `--draft-method {vispec,hivis}` (default `vispec`) and
  `--prompt-order` (default `image-last`). `hivis` refuses the draft-pruning flags. The default 3B
  paths are unchanged. The output labels say "HiViS" for `hivis` and keep their current wording for
  ViSpec. The inputs block records `draft_method` and `prompt_order`.
- **`profile_generation`.** `--method hivis` is a parser error. Profiling HiViS is out of scope, and
  the tool's step counting knows only AR and ViSpec.

- [ ] **Step 1:** Implement.
- [ ] **Step 2: Verify on CPU**, in a CPU allocation for the image work:
  - **image-last byte identity.** Use `load_processor("models/Qwen2.5-VL-3B-Instruct")` and
    `build_inputs(..., device="cpu")` for all 100 TextVQA and 100 COCO samples in both styles. Each
    `diagnose_divergence.fingerprint(input_ids)` must equal the `input_ids_sha256` recorded by the
    Week 3 ViSpec references:
    - long: `.../experiment/03_qwen_vispec/runs/20261031T172630Z_c3a36ac25f95_long_Z1rgED`;
    - short: `.../20261101T122530Z_c3a36ac25f95_short_ePf9XE`.

    Required: 400/400.
  - **image-first.** With the 7B processor, every fingerprint differs from image-last. Print one
    decoded TextVQA prompt: the question must follow `<|vision_end|>` directly. Run `build_mask` on
    it and show that it removes exactly the span plus the first question token.
  - **Parser checks.**
    - `--method hivis --draft-retention 0.5` fails with the HiViS message.
    - `--benchmark reverse_summary --prompt-order image-first` fails.
    - `profile_generation --method hivis` fails.
    - The `vars(args)` keys equal the `dfb2e48` list with `prompt_order` appended.
  - **Revision lookups.** They return the 3B revisions, with unchanged source strings, for the 3B
    paths, and the pinned 7B revisions for the three 7B paths.
  - **Help and imports.** `evaluate --help` and `offset_check --help` show the new flags, and the
    LLaVA evaluator imports still succeed.
- [ ] **Step 3: Commit.** `git commit -m "Add the HiViS method and a prompt-order option to the evaluator"`

### Task 3: Port checks on the GPU

**Files:**
- Create: `experiment/10_qwen7b_methods/checks/run.sh`

**Interfaces:**
- **The runner.** It captures evidence in full, including `--check-only` for `--suite qwen` and
  `--suite qwen7b`.
  - It writes to `experiment/10_qwen7b_methods/runs/<UTC>_<commit12>_checks_<set>_XXXXXX/`, the
    parent experiment's ignored `runs/` folder.
  - It reads `CHECK_SET`, which is `port` or `diagnosis`. Any other value exits 2 before a folder
    is created.
  - It writes `checks.json`, with one entry per check: the command, exit code, pass or fail, and
    key numbers. It exits 1 if any gate fails.
- **`CHECK_SET=port`** runs, in order:
  1. **3B identity.** `evaluate` on 3B, for `autoregressive` and `vispec`: `--benchmark textvqa
     --prompt-style long --num-samples 5 --kv-layout compact`, with the default tree. These are
     compared with Week 2 job 393336:
     - AR with `/home/aeternum/Research2/cv_systems_project/experiment/02_qwen_autoregressive/runs/20261020T084113Z_66f492a5dc65_long_2P5sjB`;
     - ViSpec with `/home/aeternum/Research2/cv_systems_project/experiment/03_qwen_vispec/runs/20261020T090922Z_66f492a5dc65_long_wDElc3`.
  2. **fp32 losslessness.** `offset_check --draft-method hivis --kv-retention 1.0 --kv-policy
     spatial --num-samples 5 --dtype float32`, with the 7B paths. It runs twice:
     - image-last at 30/3/8;
     - image-first at 60/5/10. This covers the other mask case and the large tree.

     Retention 1.0 keeps every position and is token-identical to no eviction (the Week 3 gate).
  3. **bf16 smoke.** `evaluate` on 7B, long TextVQA, 5 samples, compact layout, for `autoregressive`
     (ViSpec-7B path), `vispec` and `hivis` at 30/3/8.
  4. **Large tree.** `vispec` and `hivis` at `--total-token 60 --depth 5 --top-k 10` on 2 samples.
  5. **Position ablation.** A heredoc reruns step 3's HiViS samples with
     `HiViSDraft.position_mode = "pruned"`.
- **`CHECK_SET=diagnosis`** is used by Task 7. It runs `diagnose_divergence` for each line of
  `DIAG_LIST` (`<comparison.json> <original run> <draft run> <base model path>`), then
  `offset_check` for each line of `ORACLE_LIST`. Both list files are copied into the run folder.
- **Pass rules** for `port` (each one recorded in `checks.json`):

  | Check | Pass |
  | --- | --- |
  | 3B identity | 5/5 AR and 5/5 ViSpec equal Week 2 in `output_token_ids` and `input_ids_sha256` |
  | fp32 losslessness | `offset_check` exits 0 (AR == HiViS on 5/5) in both runs. The runtime devices show every parameter on `cuda:0`, with no CPU offload |
  | bf16 smoke | all three runs exit 0. HiViS and ViSpec-7B each average ≥ 1.5 accepted draft tokens per iteration. The ViSpec-7B log has no `missing_keys` or `unexpected_keys` line. `runtime.json` shows `draft_method`, both revisions and bf16 |
  | Large tree | both complete. Acceptance is recorded |
  | Position ablation | mean acceptance below the normal HiViS mean on the same 5 samples |

  Recorded but not gated: AR against HiViS and AR against ViSpec token agreement on the 5 samples,
  and the mean latency per method.

- [ ] **Step 1:** Implement. Then check:
  - `bash -n`;
  - a dry run with a stub `PYTHON_BIN` that prints every evaluator and oracle command;
  - `sbatch --test-only`.

  Remove the stub run folders afterwards.
- [ ] **Step 2: Commit.** `git commit -m "Add GPU port checks for the Qwen2.5-VL-7B methods"`
- [ ] **Step 3: Submit** from a clean tree:

  ```bash
  PROMPT_STYLE=long CHECK_SET=port CLUSTER_PARTITION=preempt CLUSTER_EXCLUDE=c4-4,c4-5,c5-2,c2-2 \
    SBATCH_EXTRA="--requeue --cpus-per-task=4 --mem=48G --time=01:30:00" \
    bash experiment/submit_ampere.sh experiment/10_qwen7b_methods/checks
  ```

- [ ] **Step 4: Verify** that the job exits 0 and every check in `checks.json` passes. Record the
  acceptance and latency numbers. Apply the job size rule: project the primary long job from these
  5-sample latencies.
- [ ] **Step 5: Stop condition.** Stop the plan and report to the user, with the case details, if
  any of these occurs:
  - an fp32 mismatch;
  - a 3B identity miss;
  - acceptance below 1.5;
  - an ablation that does not lower acceptance.

  Make no commit, because runs are ignored.

### Task 4: Reference run of HiViS's own code

**Files:**
- Create: `experiment/10_qwen7b_methods/hivis_reference/{setup_env.sh,run.sh,hivis_side.py}`,
  `vispec/evaluation/hivis_parity.py`

**Interfaces:**
- **Checkout.** `/home/aeternum/Research2/cv_systems_project/.cache/ext/HiViS`, from Task 1 Step 1:
  pinned to `6db36b2`, clean and read-only. `.cache/` is git-ignored and sits on project storage
  that compute nodes can see.
- **Environment** (`setup_env.sh`, run in a CPU allocation; network is used only for these
  installs). It runs:

  ```bash
  conda create -y -p .cache/envs/hivis-ref python=3.9
  pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0 torchvision==0.21.0
  pip install transformers==4.54.0 accelerate==1.9.0 datasets==4.0.0 safetensors pillow numpy
  ```

  - It sets `PIP_CACHE_DIR=.cache/pip` and `PYTHONNOUSERSITE=1`.
  - These are exactly the packages that inference needs. Not installed: the training, demo and API
    packages in HiViS's `requirements.txt` (deepspeed, fschat, gradio, openai, anthropic, protobuf,
    sentencepiece), and ray and flash-attn.
  - It writes `.cache/envs/hivis-ref.freeze.txt`.
  - It runs an import check of `hivis.model.model_hivis` and `hivis.evaluation.benchmark_data`,
    with the checkout on `sys.path`.
- **Isolation in `run.sh`.**
  - **Evidence.** The 03-runner capture, written to
    `experiment/10_qwen7b_methods/runs/<UTC>_<commit12>_hivis_reference_XXXXXX/`. It also records
    the checkout's `git rev-parse HEAD`, its `git status --porcelain` (which must be empty) and the
    freeze file.
  - **Offline and without credentials.** The HiViS step runs with `HF_HUB_OFFLINE=1`,
    `TRANSFORMERS_OFFLINE=1` and `HF_DATASETS_OFFLINE=1`. `HF_HOME` points to an empty folder inside
    the run folder, so no stored Hugging Face token is read. `HF_TOKEN` and `HUGGING_FACE_HUB_TOKEN`
    are unset.
  - **No writes.** `PYTHONNOUSERSITE=1` and `PYTHONDONTWRITEBYTECODE=1` are set. The working
    directory is inside the run folder, so nothing is written to the checkout.
  - **Exchange format.** Only safetensors and JSON files pass between the environments. Neither
    side unpickles the other's output.
- **Samples.** The first 20 questions of our seed-42 TextVQA val subset, in evaluation order, from
  `load_samples("textvqa", ..., 20)`. These are the first 20 of the 100 used in every run, with
  images from `bench_data/textvqa/train_images`.
  - `hivis_parity samples` writes `samples.json`, with the question ID, question, absolute image
    path and image sha256 for each.
  - HiViS's own TextVQA rows are not used. They come from the test split, whose images we have not
    prepared, and that split has no answers.
- **Feeding the samples to HiViS's harness** (`hivis_side.py`, in the HiViS environment). Only
  HiViS's data loader is replaced:
  1. **Samples.** Check each image's sha256. Build
     `datasets.Dataset.from_list([{question_id, question, image: path}])` and apply
     `.cast_column("image", Image())`. This is the row shape `_load_textvqa` produces.
  2. **Model.** Load
     `EaModel.from_pretrained(base_model_path=models/Qwen2.5-VL-7B-Instruct, ea_model_path=models/HiViS-Qwen2.5-VL-7B-Instruct, total_token=60, depth=5, top_k=10, draft_method="hivis", torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto")`,
     the same call as `get_model_answers`.
  3. **Seed and warm-up.** Seed as HiViS's `setup_seed(42)` does. Warm up 3 times on indices 0–2
     with `prepare_inputs(model, ds, i, "textvqa", truncation=True)` and `eagenerate(...,
     temperature=0.0, max_new_tokens=500, log=True)`, as `get_model_answers` does.
  4. **Inputs per sample.**
     - Build `inputs = prepare_inputs(model, ds, i, "textvqa", truncation=False)`. This is HiViS's
       own prompt: no system message (so the template inserts "You are a helpful assistant."), the
       user content `[image, question + " Perform an OCR task…"]`, and the processor defaults.
     - Assert that the prompt is at most 4,000 tokens, which is HiViS's skip rule: no sample may be
       skipped.
     - Assert that prompt + 500 + 60 + 10 < 4,027, the `max_length` stop in `eagenerate`.
     - Save `input_ids`, `pixel_values` and `image_grid_thw` to `bundle/<qid>.safetensors`.
  5. **Speculative path.** Run `eagenerate(inputs, temperature=0.0, max_new_tokens=500, log=True)`.
     Two recording wrappers are active only around the first draft call of each sample:
     - `hivis.model.utils_hivis.prune_image_tokens`, recording the full IDs, embeds and hidden
       states in, and the pruned IDs and positions out;
     - `model.ea_layer.topK_genrate`, recording its arguments, its four outputs, and
       `ea_layer.add` afterwards.

     The records go to `first_call/<qid>.safetensors`. Then save the raw output IDs after the prompt,
     `accept_lengths`, `new_token` and `idx`.
  6. **AR path.** Run `naivegenerate(inputs, temperature=0.0, max_new_tokens=500, log=True)` for
     HiViS's own AR tokens.
  7. **Outputs.** Write `hivis_side.jsonl` and `hivis_side_env.json`. The latter records the torch
     and transformers versions, the processor and image-processor classes, the attention
     implementation, the dtype and the GPU.
- **Our side.** `python -m vispec.evaluation.hivis_parity ours --run-dir R --base-model-path ...
  --spec-model-path ...`, in the `vipsec` environment.
  - It loads `SpecModel.from_pretrained(draft_method="hivis", total_token=60, depth=5, top_k=10,
    torch_dtype=torch.float16, slim_prefill=True, kv_layout="fixed", kv_capacity=16384,
    device_map="auto")`.
  - Each sample uses the bundle tensors. The pixel values are cast to bf16 first, as `eagenerate`
    does, and `specgenerate` then casts them to the fp16 vision dtype, as HiViS's vision tower does.
  - **Drafter isolation.** After `reset_kv()`, make the adapter's first call on HiViS's recorded
    full hidden states, embeds, prompt IDs and first token, with our target's head. Record the kept
    IDs, the positions, `add` and the first tree.
  - **End to end.** Run `specgenerate(temperature=0.0, max_new_tokens=500,
    return_acceptance_len=True)` and `autoregressive_generate(..., 500)`.
  - **Margins.** Where our HiViS tokens first differ from HiViS's, compute the stock model's fp16
    top-2 margin at that index. Use `diagnose_divergence.load_model` and `last_logits`, loaded after
    `SpecModel` is freed.
  - It writes `ours_side.jsonl`.
- **The verdict.** `hivis_parity compare --run-dir R` writes `parity.json` and `parity.md` and exits
  1 if any gate fails:

  | Gate | Check | Pass |
  | --- | --- | --- |
  | Inputs | bundle and sample digests; 20 samples, none skipped | all |
  | Prune and positions (exact integers) | the adapter's kept IDs, kept positions and `add` against HiViS's recorded values for the same prompt | 20/20 |
  | First tree, drafter isolated (fp16, identical inputs) | `draft_tokens`, `retrieve_indices`, `tree_mask`, `tree_position_ids` | 20/20 |
  | Tokens end to end (both greedy, both fp16) | output IDs through the first EOS, cut at 500 | ≥ 18/20 identical, and every mismatch a near-tie: stock fp16 top-2 margin ≤ 0.5 logits at the first differing index |
  | Acceptance | mean accepted draft tokens per iteration over the samples with identical tokens, leaving out each sample's last iteration on both sides | within 3% relative |

- **Unavoidable differences** (`parity.md` lists them):
  - **Target code and kernels.** HiViS runs its Transformers 4.54 model files on torch 2.6, and we
    run our 4.51.3 fork on torch 2.7; the attention dispatch and vision attention differ too. The
    prefill hidden states can therefore differ in the low bits. That is why the token gate tolerates
    near-ties, and why the drafter is also checked in isolation on HiViS's own hidden states.
  - **Drafter kernels.** Both sides run the same vendored code, but on torch 2.6 and torch 2.7. An
    isolated first-tree mismatch is not accepted by default: it stops the plan. It is accepted only
    if rerunning that sample's isolated drafter in fp32 on both sides makes the trees equal, which
    would make it a kernel rounding tie. `parity.md` then records it.
  - **Stop rule.** HiViS can overshoot `max_new_tokens` by up to one tree, because it stops at
    `new_token > 500`, while ours clamps at 500. Hence the cut at 500, and leaving out the last
    iteration.
  - **KV cache and prefill.** The cache has 128,000 slots there and 16,384 here. Here the prefill
    is slim; there the prefill logits are full. Neither changes a value that is used.
  - **Reported, not gated:**
    - HiViS's `eagenerate` against its `naivegenerate` tokens. Its AR path does not round the pixel
      values through bf16, so its vision inputs differ slightly;
    - our fp16 HiViS against our fp16 AR;
    - HiViS-side mean acceptance next to the paper's 3.95 at 60/5/10, which used different samples.

- [ ] **Step 1: Check the checkout** from Task 1 Step 1: the pinned commit, a clean status and
  read-only files.
- [ ] **Step 2: Build the environment** in a CPU allocation, in parallel with Tasks 0–3:
  `srun -p normal --cpus-per-task=4 --mem=16G --time=01:00:00 bash experiment/10_qwen7b_methods/hivis_reference/setup_env.sh`.
  Expected: a freeze file with `torch==2.6.0+cu124` and `transformers==4.54.0`, and a passing
  import check.
- [ ] **Step 3: Implement and check on CPU.**
  - `bash -n`, and `hivis_parity --help`.
  - The 20 IDs from `hivis_parity samples` equal the first 20 question IDs of the Week 3 long 03
    manifest.
  - Run `compare` on hand-made JSONL in `.smoke/hivis_ref_dry/`, with one identical sample, one
    near-tie mismatch and one non-tie mismatch. It must give the expected gate results.
- [ ] **Step 4: Commit.** `git commit -m "Add a parity check against the HiViS reference code"`
- [ ] **Step 5: Submit** after Task 3 passes:

  ```bash
  PROMPT_STYLE=long CLUSTER_PARTITION=preempt CLUSTER_EXCLUDE=c4-4,c4-5,c5-2,c2-2 \
    SBATCH_EXTRA="--requeue --cpus-per-task=4 --mem=48G --time=01:30:00" \
    bash experiment/submit_ampere.sh experiment/10_qwen7b_methods/hivis_reference
  ```

- [ ] **Step 6: Verify** that the job exits 0 and that the gates in `parity.json` pass.
  - Any failure stops the plan before Task 7. Report to the user with `parity.md`.
  - Record the run path and the HiViS-side and our-side mean acceptance.
  - Make no commit.

### Task 5: Comparison tools for three methods and the 7B model

**Files:**
- Modify: `vispec/evaluation/compare_metrics.py`, `vispec/evaluation/sweep_table.py`,
  `vispec/evaluation/diagnose_divergence.py`, `experiment/METRICS.md` (short notes; the full section
  is written in Task 9)

**Interfaces:**
- **`compare_metrics`:**
  - **Methods.** `METHODS = ("autoregressive", "vispec_speculative", "hivis_speculative")`.
  - **Method mode.** The original group is `autoregressive`. The draft group uses one speculative
    method, either one. The title is `# Original vs ViSpec` (unchanged) or `# Original vs HiViS`.
  - **The `spec_model` guard.** When no report is HiViS, the guard is unchanged: one value across
    all reports. When the draft group is HiViS, each group must share one `spec_model`. The AR
    reports record the ViSpec draft they load and do not use.
  - **Unchanged guards:** `base_model`, temperature, prompt style, prompt fingerprints, the hardware
    signature, the layout guards and the sparsity guards.
  - **New prompt-order guard.** Every report's runtime `generation_args.prompt_order` must match. An
    absent value counts as `image-last`. Nothing is added to the output files.
- **`sweep_table`:**
  - **Layout flag.** `--layout {sweep,methods}`, default `sweep`. The `sweep` output for existing
    runs is byte-identical. The `sweep` layout refuses HiViS runs and any non-default tree or prompt
    order, with a message naming `--layout methods`.
  - **The `methods` layout.**
    - It groups runs by `(prompt_style, slurm_job_id)`, with the existing final-attempt, single-UUID,
      allocation and hardware checks.
    - `spec_model` is shared within AR and ViSpec together, and within HiViS.
    - Prompt fingerprints are checked per benchmark, prompt order and sample.
    - It refuses any run with sparsity settings, and more than one completed folder per method, tree
      and prompt order.
  - **Rows and columns.** One row per group, method, tree, prompt order and benchmark. The columns:
    `prompt_style, slurm_job_id, method, tree, prompt_order, role, benchmark, n, latency_mean_s,
    ttft_mean_s, tpot_mean_s, throughput_tokens_per_s, output_tokens_mean, acceptance_mean,
    output_tokens_per_iteration_mean, decode_peak_mean_gib, post_compaction_peak_gib, quality,
    token_matched_vs_ar, token_checked_vs_ar, token_agreement_vs_ar, answer_agreement_vs_ar,
    speedup_vs_ar, speedup_vs_vispec, speedup_vs_tree_30_3_8, note`.
  - **Ratios.** As in the existing column notes, each ratio is the reference's mean latency over the
    row's mean latency, and above 1 means the row is faster. Each ratio uses the same job and the
    same prompt order. The reference is:
    - for `speedup_vs_ar`, the AR run;
    - for `speedup_vs_vispec`, the ViSpec run at the same tree (HiViS rows);
    - for `speedup_vs_tree_30_3_8`, the same method at 30-3-8 (60-5-10 rows).

    A missing reference gives null, with a note.
- **`diagnose_divergence`.**
  - Method mode accepts `hivis_speculative` as the draft run's method.
  - It rebuilds the prompts with the runs' `prompt_order`. An absent value means image-last, and the
    two runs must match.
  - Labels that name ViSpec use the draft method's name; for ViSpec runs the text is unchanged.
  - The 0.5-logit threshold and the classes are unchanged.

- [ ] **Step 1:** Implement.
- [ ] **Step 2: Byte identity**, on the CPU from saved runs only.
  - Extract the pre-change code with `git archive dfb2e48 vispec | tar -x -C .smoke/hivis_tools/base`.
  - Run each command below from both trees, into fresh folders, and compare the two outputs with
    `diff -r`:
    1. Re-report Week 3 job 398883's ViSpec reference,
       `/home/aeternum/Research2/cv_systems_project/experiment/03_qwen_vispec/runs/20261031T172630Z_c3a36ac25f95_long_Z1rgED`,
       with `report_metrics --output-dir` and `TEXTVQA_EVALUATOR` set.
    2. `compare_metrics` in method mode on that job's 02 and 03 reports, and in sparsity mode on its
       06 `0.5:spatial` arm against the 03 reference.
    3. `sweep_table` with the default layout on every job-398883 run folder, found with
       `grep -l slurm_job_id=398883 /home/aeternum/Research2/cv_systems_project/experiment/0[236]_*/runs/*/metadata.txt`.
  - Expected: no differences.
- [ ] **Step 3: Check the new behaviour on synthetic copies** in `.smoke/hivis_tools/`.
  - **Setup.** Copy the short 02 and 03 runs of job 398923. In the 03 copy, set the metadata to
    `method=hivis_speculative`, set `spec_model` to the HiViS path, drop `num_q`, and re-report it.
  - **Method mode.**
    - AR against the HiViS copy succeeds, titled "Original vs HiViS".
    - The same pair with a changed `base_model` is still refused.
    - Two AR reports with different `spec_model` values are still refused.
  - **Methods layout.**
    - `sweep_table --layout methods` gives one row per method and benchmark.
    - Its `speedup_vs_ar` equals `compare_metrics`' latency ratio.
    - The default layout refuses, with the `--layout methods` message.
  - **Prompt order.** Make a copy with `prompt_order=image-first` in `generation_args` and a changed
    fingerprint.
    - `compare_metrics` refuses to mix it with an image-last report.
    - The methods layout keys it separately and needs its own AR reference.
  - **`diagnose_divergence`.**
    - `--help` works.
    - Its argument checks accept a `hivis_speculative` draft run.
    - They refuse a prompt-order mismatch before any model loads.
- [ ] **Step 4: Commit.** `git commit -m "Compare HiViS and the 7B model in the comparison tools"`

### Task 6: Experiment 10 runner

**Files:**
- Create: `experiment/10_qwen7b_methods/{run.sh,README.md}`
- Modify: `experiment/qwen_arm_runner.sh` (sweep type `methods`, `QWEN_ASSET_SUITE`, `HIVIS_DIR`),
  `experiment/README.md` (register row, status pending), `CLAUDE.md` (one line in the Qwen
  section), `AGENTS.md` (scope: experiments 02–09 on 3B, experiment 10 on 7B, the user's decision of
  2026-11-05)

**Interfaces:**
- **Arms.** `METHOD_ARMS` is a space-separated list of `<method>[:<tree>][:<order>]`.
  - `method` is `ar`, `vispec` or `hivis`.
  - `tree` is `30-3-8` (the default) or `60-5-10`, written as total-depth-top-k in the code
    convention.
  - `order` is `image-last` (the default) or `image-first`.
  - `ar` takes no tree.
  - `qwen_validate_method_arms` exits 2, before any folder is created, on an unknown value, a
    repeated field, a tree on `ar`, a duplicate arm or an empty list.
- **Per arm.** Each method maps to an evaluator method, a method folder and a metadata method:

  | Arm | Evaluator method | Method folder | Metadata method | Draft path |
  | --- | --- | --- | --- | --- |
  | `ar` | `autoregressive` | `autoregressive_qwen` | `autoregressive` | `SPEC_DIR` |
  | `vispec` | `vispec` | `vispec_qwen` | `vispec_speculative` | `SPEC_DIR` |
  | `hivis` | `hivis` | `hivis_qwen` | `hivis_speculative` | `HIVIS_DIR` |

  - Speculative arms pass `--depth`, `--top-k` and `--total-token` from their tree. ViSpec arms
    alone also pass `--num-q 2`.
  - Every arm passes `--prompt-order` and `--kv-layout compact`, and no sparsity flag.
- **Folder:** `runs/<UTC>_<commit12>_<style>_<method>[_t<tree>]_<order>_XXXXXX`.
- **Metadata additions** (the `methods` sweep only): `prompt_order`, `draft_method` (`none` for AR)
  and `asset_suite`. Speculative arms record depth, top-k and total-token, and ViSpec arms also
  `num_q`. Existing keys and the other sweep types are unchanged.
- **Assets.** `QWEN_ASSET_SUITE` (default `qwen`) selects the `--check-only --suite` and the copied
  `.preparation/assets_<suite>.json`. The default reproduces today's command and copy.
- **`10/run.sh`.**
  - It sources the arm runner.
  - Defaults: `QWEN_ASSET_SUITE=qwen7b`, `BASE_MODEL_PATH=$repo_dir/models/Qwen2.5-VL-7B-Instruct`,
    `SPEC_DIR=$repo_dir/models/ViSpec-Qwen2.5-VL-7B-Instruct`,
    `HIVIS_DIR=$repo_dir/models/HiViS-Qwen2.5-VL-7B-Instruct`, `METHOD_ARMS="ar vispec hivis"` and
    `QWEN_BENCHMARKS="textvqa coco_caption"`.
  - It requires `PROMPT_STYLE` and sets `sweep=methods`.
- **README skeleton.** The `AGENTS.md` sections: purpose and hypothesis, changes from the baseline,
  models with revisions, data, settings, cluster prerequisites, exact commands, metric definitions,
  results (pending) and limitations. The hypotheses:
  1. Both speculative methods beat AR in the long style.
  2. HiViS's acceptance relative to ViSpec depends on prompt order, because image-last leaves its
     drafter a 4-token tail after the image.
  3. 60/5/10 raises acceptance for both methods. Its effect on latency is open, because the tree
     verification is larger.

- [ ] **Step 1:** Implement.
- [ ] **Step 2: Verify.**
  - Run `bash -n`.
  - Each of these exits 2 with no folder created: `ar:60-5-10`, `hivis:30-3-8:image-first:x`,
    `eagle`, `vispec:40-4-8`, an empty list, and `vispec vispec`.
  - The stub-Python command output for 05, 06, 08 and 09 is identical before and after, by diff.
  - The stub output for 10 with `METHOD_ARMS="ar vispec hivis:60-5-10:image-first"` shows the
    expected flags, paths, metadata and folder names.
  - Run `sbatch --test-only` for 10.
  - Delete the stub run folders.
- [ ] **Step 3: Commit.** `git commit -m "Add the Qwen2.5-VL-7B method comparison experiment"`

### Task 7: Primary runs and gates

- [ ] **Step 1: Submit both styles** from a clean tree, recording the job IDs:

  ```bash
  PROMPT_STYLE=long CLUSTER_PARTITION=preempt CLUSTER_EXCLUDE=c4-4,c4-5,c5-2,c2-2 \
    SBATCH_EXTRA="--requeue --cpus-per-task=4 --mem=32G --time=02:30:00" \
    bash experiment/submit_ampere.sh experiment/10_qwen7b_methods
  PROMPT_STYLE=short CLUSTER_PARTITION=preempt CLUSTER_EXCLUDE=c4-4,c4-5,c5-2,c2-2 \
    SBATCH_EXTRA="--requeue --cpus-per-task=4 --mem=32G --time=01:00:00" \
    bash experiment/submit_ampere.sh experiment/10_qwen7b_methods
  ```

  If the Task 3 projection for the long job exceeds 110 min, submit `long` as two jobs, with
  `QWEN_BENCHMARKS=textvqa` and `QWEN_BENCHMARKS=coco_caption`. Each runs all three methods and so
  keeps its own AR. Record the split in the README.
- [ ] **Step 2: Verify completion,** as in Week 3:
  - every arm exits 0, and so does its report;
  - one GPU UUID per job;
  - 100/100 per benchmark;
  - a clean tree;
  - the expected metadata: method, tree, prompt order, paths and `asset_suite=qwen7b`;
  - both revisions in `runtime.json`;
  - no `missing_keys` or `unexpected_keys` line in the ViSpec-7B logs.

  After a requeue, use the final attempt.
- [ ] **Step 3: Compare.** For each job, run `compare_metrics` in method mode twice: AR against
  ViSpec, and AR against HiViS. Write each comparison into the speculative run's
  `comparisons/<UTC>_vs_ar/`.
- [ ] **Step 4: Diagnose and run the fp32 oracles** in one job with `CHECK_SET=diagnosis`, using the
  long-job values from Step 1's submission (`--mem=48G --time=01:30:00`).
  - `DIAG_LIST` holds the four comparisons (two methods by two styles), each with
    `models/Qwen2.5-VL-7B-Instruct`.
  - `ORACLE_LIST` holds two fp32 checks on 5 samples each:
    - `--draft-method vispec`, long TextVQA, ViSpec-7B;
    - `--draft-method hivis --benchmark coco_caption --prompt-order image-first`.
  - **Gates:**
    - every diagnosis passes, with no `unexplained` and no `both_wrong` case at the unchanged
      0.5-logit threshold;
    - both oracles give AR == speculative on 5/5.

    Report each failure with its case details, as Week 3 did.
- [ ] **Step 5: Build the tables.** For each job, run
  `python -m vispec.evaluation.sweep_table --layout methods --run-dirs <the job's runs> --output-dir experiment/sweeps/<UTC>_10_<style>_job<id>`.
- [ ] **Step 6:** Make no commit. Record the job IDs and every run, comparison, diagnosis and sweep
  path.

### Task 8: Secondary runs (tree setting and prompt order)

- [ ] **Step 1: Project** every job below from the primary long job's per-arm durations. Then apply
  the job size rule:
  - The COCO jobs may split by method, each part keeping AR.
  - The TextVQA order jobs cannot split further. If one projects above 110 min, ask the user before
    submitting it.
- [ ] **Step 2: Submit,** all with `PROMPT_STYLE=long` and the GPU prefix. Hold at most two at once
  and chain the rest with `--dependency=afterany:<id>`:

  | Job | `QWEN_BENCHMARKS` | `METHOD_ARMS` | `--time` |
  | --- | --- | --- | --- |
  | Tree, TextVQA, ViSpec | `textvqa` | `ar vispec vispec:60-5-10` | 02:00:00 |
  | Tree, TextVQA, HiViS | `textvqa` | `ar hivis hivis:60-5-10` | 02:00:00 |
  | Tree, COCO | `coco_caption` | `ar vispec hivis vispec:60-5-10 hivis:60-5-10` | 01:30:00 |
  | Order, TextVQA, ViSpec | `textvqa` | `ar vispec ar:image-first vispec:image-first` | 02:30:00 |
  | Order, TextVQA, HiViS | `textvqa` | `ar hivis ar:image-first hivis:image-first` | 02:30:00 |
  | Order, COCO | `coco_caption` | `ar vispec hivis ar:image-first vispec:image-first hivis:image-first` | 01:30:00 |

- [ ] **Step 3: Verify completion** as in Task 7 Step 2.
- [ ] **Step 4: Build the tables.** Make one methods-layout sweep table per job. Run
  `compare_metrics` in method mode for AR against each speculative arm, within one prompt order, to
  get the agreement counts.
- [ ] **Step 5:** Make no commit. Record the job IDs and paths.

### Task 9: Write-up

**Files:**
- Modify: `experiment/10_qwen7b_methods/README.md`, `experiment/METRICS.md` (new section "Method
  comparison on Qwen2.5-VL-7B (experiment 10)"), `experiment/README.md` (register status and a
  "Qwen2.5-VL-7B method comparison (10)" results summary)

- [ ] **Step 1: Write the results.** Use the primary jobs per style and benchmark.
  1. **Speculative decoding first.** For each method: speedup against AR, accepted draft tokens per
     iteration and output tokens per iteration, TPOT, TTFT, and token agreement with AR (with the
     bf16 note). Quality comes from the `short` style.
  2. **Hardware.** A hardware summary per job, with `device.txt` links.
  3. **The diagnosis and oracle gates.**
  4. **The reference parity result**, with the HiViS harness acceptance next to ours.
- [ ] **Step 2: Describe the spectrum of the draft's visual input.**
  - An EAGLE-style draft reads every image token. That end is described here, not measured.
  - ViSpec compresses each image to 2 learned queries (`num_q = 2`) and mixes them into every draft
    step.
  - HiViS sees no image token. Visual information reaches it only through the target's hidden states
    after the image, and for the accepted tokens.
  - **Relate this to Week 3.** Pruning ViSpec's adaptor input down to 25% left acceptance at the 1.0
    level: 2.129–2.145 against 2.141 on long TextVQA. That fits a draft that relies little on
    per-token image detail, and HiViS is the end point of that direction. State causes as
    inferences.
- [ ] **Step 3: Be honest about prompt order and tree size.**
  - **Image-last** leaves HiViS a 4-token tail after the image. The image-first arm shows the
    effect.
  - **Tree size.** 30/3/8 is ViSpec's setting, and 60/5/10 is HiViS's. In the paper, the gap between
    HiViS and ViSpec is larger at 60/5/10. The TextVQA tree and order effects are within-job per
    method, and HiViS is compared with ViSpec there only through each job's speedup against AR.
  - **The paper's numbers** come from a different harness (A800, fp16, test split, image-first,
    80 samples) and are context only.
  - **Draft RoPE tables** are bf16 here and fp16 in HiViS's harness.
  - **Separate rows.** The 7B rows stay separate from the 3B and LLaVA rows.
- [ ] **Step 4: Write the `METRICS.md` section,** covering:
  - the methods-layout columns and ratio directions;
  - our acceptance against the paper's τ;
  - prompt order and the mask on each order;
  - the reference gates;
  - the comparison-guard exception for AR against HiViS.
- [ ] **Step 5: Verify.** Spot-check at least fifteen values against `sweep.json`,
  `comparison.json`, the diagnosis JSON and `parity.json`, and show the commands in the task report.
- [ ] **Step 6: Commit.** `git commit -m "Record the Qwen2.5-VL-7B method comparison results"`

---

## Self-Review Against the Fixed Decisions

| Decision | Where it is met | Status |
| --- | --- | --- |
| Target and drafts, pinned; 3B stays byte-identical | Global Constraints, Task 0 (pins); Tasks 2, 3, 5 and 6 (fingerprints, Week 2 tokens, re-report, comparison, sweep and runner diffs) | Met |
| Port: vendored drafter with attribution, adapter, default-preserving `draft_method`, `--method hivis`, draft flags refused, compact layout and slim prefill, no 7B KV eviction | Tasks 1 and 2; experiment 10 always passes `--kv-layout compact` and no sparsity flag | Met |
| `qwen7b` asset suite; existing suites unchanged | Task 0, Steps 2–3 | Met |
| Primary comparison: three methods, both benchmarks, 100 samples, both styles, standard settings, one job per style, about 2 h or less, AR in every job | Tasks 6–7, GPU Budget | Met. The long job (67–103 min estimated) splits by benchmark only if the measured projection exceeds 110 min |
| Secondary: tree 60/5/10 and prompt order, long style only, both benchmarks, same-job references | Tasks 6 and 8 | **Partly met.** The TextVQA tree and order arms are split by method into two jobs each, because one job would take 98–143 min. Each job holds AR and both settings of its method, so each method's tree and order effect is within one job. But HiViS against ViSpec at 60/5/10, and with image-first prompts, on TextVQA is compared across jobs, through each job's speedup against AR. COCO keeps all arms in one job |
| Comparison tools handle HiViS and 7B; outputs unchanged for current runs; Week 3 re-report byte-identical | Task 5 | Met |
| Port verification in order: CPU, GPU smoke (fp32, bf16 acceptance, ViSpec-7B, 3B identity), reference run | Tasks 1, 3 and 4 | Met. The 3B AR identity uses the job 393336 02 run, the AR partner of the named 03 run. The reference run is a regular task, as the user approved |
| Sanity range from HiViS's own numbers | Global Constraints, sanity table | Met: Tables 2 and 5 found for Qwen2.5-VL-7B TextVQA |
| Write-up: experiment README, METRICS section, register and summary; framing; honesty on prompt order and tree size | Tasks 6 and 9 | Met |
| Out of scope respected | No task touches Re:Verse+ on 7B, 7B eviction, training, LLaVA or other methods | Met |
| Working rules: plain wording, no attribution, no test files, no writes to `reverse_plus`, evidence capture, A6000 pin, exclusions, `preempt` with `--requeue` through `submit_ampere.sh` | Global Constraints; every GPU step uses the prefix | Met. CPU-only steps use `srun -p normal` |
| Task granularity: reviewable alone, parallel where possible, GPU smoke early | Task Order and Parallel Work | Met |
