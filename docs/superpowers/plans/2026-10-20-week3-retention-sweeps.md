# Week 3: Retention Sweeps Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Sweep retention budgets of 25%, 50% and 75% (with 100% as the control) for draft
pruning alone, KV eviction alone and both combined, on TextVQA and COCO in both prompt styles.
Each arm logs latency, VRAM and acceptance. Before the sweep, settle the open question from Week 2
about drift between cached decoding and full recompute.

**Architecture:** The plan has four parts, run in this order:

1. A drift probe (experiment 07) runs first and ends at a conditional checkpoint.
2. A combined-sparsity experiment (08) reuses the Week 2 arm runner with one new sweep type.
3. A sweep-table tool reads the per-run reports directly. It groups runs by Slurm job and
   computes every ratio within one job, so arms from different cards are never mixed.
4. The runs go in four jobs: one for short style, and three for long style, one per sweep.
   Each job carries its own references.

**Tech Stack:** Python 3.11 (`vipsec`), torch 2.7.0+cu128, transformers 4.51.3, bash, and Slurm on
the pinned RTX A6000s.

**Spec:** [`docs/superpowers/specs/2026-10-18-qwen-dynamic-visual-sparsity-design.md`](../specs/2026-10-18-qwen-dynamic-visual-sparsity-design.md)
(Goal 3 and Metrics). Week 2 is described in
[`2026-10-19-week2-visual-sparsity.md`](2026-10-19-week2-visual-sparsity.md) and its results in
`experiment/05_qwen_draft_pruning/README.md` and `experiment/06_qwen_kv_eviction/README.md`.
Week 3 starts from the merged Week 2 branch.

## Global Constraints

- **No test files** (`AGENTS.md`). Verify directly with heredoc checks, `py_compile`, `bash -n`,
  `sbatch --test-only` and recorded GPU runs.
- **Commit messages are plain.** No `Co-Authored-By`, no `Claude-Session`, no Claude attribution
  of any kind.
- **Fixed settings:** bf16, temperature 0.0, 1,024 new tokens, depth 3, top-k 8, total-token 30,
  `num_q` 2, batch 1, `--kv-layout compact` in every arm and reference, the Week 1 pixel limits,
  and the same 100-sample TextVQA and COCO subsets in the `long` and `short` styles.
- **Budgets and policies:** budgets are 0.25, 0.5, 0.75 and 1.0; policies are `spatial` and
  `attention`. Combined arms apply the same budget and the same policy to both sides (the
  diagonal only). This is the plan's call, recorded in the 08 README.
- **References come from the same job.** The 02 and 03 runs with `KV_LAYOUT=compact` and no
  sparsity flags are the AR and ViSpec references. Agreement and quality are compared against the
  ViSpec reference. Latency and TTFT are also compared against the same-policy 1.0 arm (Week 2
  ruling R6). Speedup against AR uses the same job's 02 run. Ratios never span two jobs.
- **Diagnosis:** `diagnose_divergence` runs only on draft-only arms. It refuses eviction and
  combined arms (Week 2 ruling R8). The 0.5-logit near-tie threshold is not changed.
- **Jobs:** submit with `experiment/submit_ampere.sh` using `CLUSTER_PARTITION=preempt` and
  `SBATCH_EXTRA="--cpus-per-task=4 --mem=22G --time=<per job>"`. The user's own jobs share these
  nodes, so hold at most two Week 3 GPU jobs at once (see Task 4).
- **Storage:** generated runs and sweep outputs stay out of Git. Never modify existing files in
  `experiment/*/runs/`; only create new folders.

## Review Focus

1. **Cross-job mixing in sweep tables.** Long-style arms are spread over three jobs. The table tool
   must refuse to compute a ratio between runs whose GPU UUID or allocation differ, and it must
   label every row with its job. *(Task 3 check.)*
2. **Requeue duplicates.** A preempted job reruns every runner into fresh folders. The table tool
   must take the final attempt (highest `slurm_restart_count` with `exit_code=0`) and refuse two
   completed folders for one arm. *(Task 3 check.)*
3. **Composing the combined arm.** Both flags together have only had a CPU check. The combined
   1.0 arm must match the reference on 100/100 for both policies. *(Tasks 2 and 4.)*
4. **The 0.25 budget on small images.** COCO images can have as few as 256 merged tokens, which
   leaves 64 kept. Every image keeps at least 1 token, and the counts are logged. *(Task 4: report
   the minimum kept per arm.)*
5. **A fork-specific decode deviation.** If the Task 1 probe shows that the KV-cache fork departs
   from stock HF incremental decoding, every arm inherits the deviation. That is a stop condition.
   *(Task 1 checkpoint.)*

---

### Task 0: Harden the Week 2 code before the sweep

The Week 2 final review sorted these items as "fix in Week 3". They have to land before any
Week 3 GPU run. Each item is small, so they share one task.

**Files:**
- Modify: `vispec/model/sparsity.py`, `vispec/model/cnets_ours.py`, `vispec/model/utils.py`,
  `vispec/model/spec_model_ours.py`, `vispec/model/autoregressive.py`,
  `vispec/evaluation/measurement.py`, `vispec/evaluation/compare_metrics.py`,
  `vispec/evaluation/diagnose_divergence.py`, `experiment/METRICS.md`
- Create: `vispec/evaluation/offset_check.py`, which promotes `.smoke/w2_t5/diag_agreement.py` to a
  tracked module

**Interfaces:**
- **Selection guards.**
  - `select_in_span` raises `ValueError` unless `1 <= k <= n`.
  - The draft attention score raises on a non-finite value. The target side is already covered
    by the score-sum check.
- **New measurement fields.** These are additive and appear only with eviction.
  - `kv_image_tokens`: the total image tokens before eviction.
  - `kv_min_kept_per_image`: the smallest number of tokens kept for any one image.
  - The draft side gets the same pair: `draft_image_tokens` (which exists already) and
    `draft_min_kept_per_image`.
  - `gpu_memory[dev]["post_compaction_peak_allocated_bytes"]` is new. Reset the peak counter
    straight after compaction, then read the peak at finish. `decode_peak_allocated_bytes` keeps
    its Week 2 meaning, so the Week 2 numbers stay comparable. `report_metrics` reports the new
    field in the `memory` block as `post_compaction_peak_allocated_gib`.
- **Narrow the bare `except:`** around `topK_genrate` in `utils.py` (about :302-326). Pick the head
  explicitly with `hasattr(model.base_model, "language_model")`, as `update_inference_inputs`
  already does, so a draft-pruning error raises its own traceback. The LLaVA path must still
  select the same head.
- **Fail closed in `diagnose_divergence`.** If a run has no `*.runtime.json`, read
  `kv_retention` from `metadata.txt`. If neither source exists, refuse. Do not pass silently.
- **Fail closed in `compare_metrics --mode sparsity`.** Every report needs a runtime entry for
  every compared benchmark. A missing entry raises instead of leaving `sparsity` null.
- **`offset_check.py`.** It runs `python -m vispec.evaluation.offset_check --kv-retention R
  --kv-policy P [--draft-retention R --draft-policy P] --num-samples 5 --dtype float32 --output-dir D`.
  - It runs fp32 AR and fp32 ViSpec under the same eviction setting on long TextVQA, and
    reports exact AR == ViSpec agreement.
  - Greedy verification does not depend on the draft, so agreement must hold even with draft
    pruning.
  - This is the only oracle for large `kv_offset`.
- **`METRICS.md` additions.**
  - Memory ratios stay valid under truncation.
  - `spatial` is a 1-D raster stride over each image's merged tokens, not a 2-D pool. At 0.5 it
    keeps every other column. This is the Week 2 selector, kept so Week 2 and Week 3 stay
    comparable.
  - Requeued jobs leave duplicate folders. Take the final attempt.
  - The template-tail score window applies to the `attention` policy.

- [ ] **Step 1:** Implement.
- [ ] **Step 2: Verify on CPU.**
  - The `k` and NaN guards raise.
  - On a synthetic two-image mask, `kv_min_kept_per_image` equals the hand-computed minimum.
  - Head selection gives the same head object as before for both LLaVA (`spec_model_ours` with
    a LLaVA config) and Qwen. Inspect it on a meta or tiny model.
  - Both fail-closed paths refuse on `.smoke` copies with `runtime.json` removed.
  - `offset_check --help` works.
  - The LLaVA evaluator imports succeed.
- [ ] **Step 3: Run the GPU smoke** in one `gpu_check.sh` allocation, into `.smoke/w3_t0/`:
  - `offset_check` at 0.25 spatial, and at 0.25 spatial combined with draft 0.25 spatial. Both
    must give AR == ViSpec on 5/5 in fp32.
  - One 5-sample `evaluate` run at 0.25 spatial. Show the new fields, and show that
    `post_compaction_peak` ≤ `decode_peak`.
- [ ] **Step 4: Commit.** `git commit -m "Harden selection, diagnosis and measurement for the retention sweep"`

### Task 1: Cached-decode drift probe (experiment 07), with a CONDITIONAL CHECKPOINT

**Files:**
- Create: `vispec/evaluation/drift_probe.py`, `experiment/07_qwen_decode_drift/{run.sh,README.md}`
- Modify: `experiment/README.md` (register row)

**Interfaces:**
- Consumes:
  - from `diagnose_divergence.py`: the stock `Qwen2_5_VLForConditionalGeneration` loader, the
    prompt rebuild with fingerprint check, and the teacher-forced top-2 logits. Import these;
    do not copy them.
  - `benchmarks.build_inputs`.
- Produces: `python -m vispec.evaluation.drift_probe --answers <AR test.jsonl> --data-dir bench_data
  --base-model-path models/Qwen2.5-VL-3B-Instruct --num-samples 100 [--question-ids ...]
  --focus 35537:162 --output-dir D`. It writes `drift.json` and `drift.md`.
  - **Part A:** for each question, run stock HF greedy `generate` in bf16 (sdpa,
    `max_new_tokens=1024`). Compare its tokens with the fork's AR tokens from `--answers`. Record
    exact agreement, the first-mismatch index, and the stock teacher-forced top-2 margin at that
    index.
  - **Part B (focus question, at index 162):** record the logits of tokens 13 and 95518 under
    four conditions:
    - (i) fork incremental AR, bf16;
    - (ii) stock incremental `generate` with `output_scores`, bf16;
    - (iii) stock full recompute, bf16;
    - (iv) stock full recompute, fp32.
- `run.sh` follows the 04 evidence pattern. It probes long-style TextVQA against the Week 2 02
  compact run of job 393336:
  `experiment/02_qwen_autoregressive/runs/20261020T084113Z_66f492a5dc65_long_2P5sjB`. That run's
  tokens equal Week 1's on 100/100.

- [ ] **Step 1:** Implement the probe and the runner. The README sets out the question, the
  decision rule below and the limits. The runner must fail if `--answers` is missing or empty;
  the 04 identity check passed silently in that case, and this runner must not copy that.
- [ ] **Step 2: Verify on CPU.** Run `py_compile` and `bash -n`, a `--help` check, and a heredoc
  that imports the reused diagnose helpers.
- [ ] **Step 3: Run.** `PROMPT_STYLE=long CLUSTER_PARTITION=preempt SBATCH_EXTRA="--cpus-per-task=4 --mem=22G --time=02:00:00" bash experiment/submit_ampere.sh experiment/07_qwen_decode_drift`.
  Expected: exit 0, and `drift.json` holds Part A for 100 questions and Part B for 35537.
- [ ] **Step 4:** Write the findings into the 07 README and the register. Label causes as
  inferences.
- [ ] **Step 5: Commit.** `git commit -m "Probe drift between cached decoding and full recompute"`
- [ ] **Step 6: CONDITIONAL CHECKPOINT.** Stop and report to the user if either condition holds:
  - in Part B, stock incremental (ii) prefers 95518 while the fork (i) prefers 13;
  - in Part A, fork and stock greedy disagree at a first mismatch whose stock margin is above
    0.5.

  Either result points to a deviation specific to the fork, which every arm inherits. If
  neither holds, record the result as generic bf16 drift in cached decoding and continue.

### Task 2: Combined sparsity experiment (08)

**Files:**
- Create: `experiment/08_qwen_combined_sparsity/{run.sh,README.md}`
- Modify: `experiment/qwen_arm_runner.sh` (sweep type `combined`), `experiment/README.md`,
  `CLAUDE.md` (one line listing 07–08 next to 04–06), and `AGENTS.md` (scope sentence 02–08)

**Interfaces:**
- Consumes: the Week 2 arm runner (`qwen_validate_arms`, `qwen_run_arms`, `sweep=draft|kv`).
- Produces:
  - **Sweep dispatch:** both `sweep` branches in `qwen_arm_runner.sh` (about :94 and :208) become
    a `case` statement with a `*) exit 2` branch. An unknown sweep type must never run as a draft
    sweep.
  - **Arm list:** 08 reads `COMBINED_ARMS`, a space-separated list of `<retention>:<policy>`. A
    method suffix is rejected, because 08 runs ViSpec only.
  - **Flags:** each arm passes `--kv-retention R --kv-policy P --draft-retention R --draft-policy P`.
    Metadata records `sweep=combined` and all four settings.
  - **Folder name:** `runs/<UTC>_<commit12>_<style>_r<R>_<P>_XXXXXX`.

- [ ] **Step 1:** Implement the sweep type and the runner. The README states the diagonal-only
  call, a hypothesis framed from the Week 2 results, and results as `pending`.
- [ ] **Step 2: Verify.**
  - Run `bash -n`.
  - `COMBINED_ARMS="0.5:spatial:vispec"` and `COMBINED_ARMS="0.3:spatial"` must each exit 2,
    with no run folder created.
  - Print the evaluator command for one combined arm with a stub Python, the way Week 2 Task 8
    did, and show that all four flags are present.
  - Run `sbatch --test-only` for `experiment/08_qwen_combined_sparsity`.
- [ ] **Step 3: Commit.** `git commit -m "Add the combined draft and KV sparsity experiment"`

### Task 3: Sweep table tool

**Files:**
- Create: `vispec/evaluation/sweep_table.py`
- Modify: `experiment/METRICS.md` (sweep table section), `.gitignore` (add `experiment/sweeps/*`)

**Interfaces:**
- Consumes:
  - each run's newest `reports/*/summary.json`, `metadata.txt` and `device.txt`;
  - from `compare_metrics`: `runtime_signature`, `generation_setting` and the token-agreement
    helper. Import these; do not copy them.
- Produces: `python -m vispec.evaluation.sweep_table --run-dirs <dir>... --output-dir experiment/sweeps/<UTC>_<label>`.
  It writes `sweep.md`, `sweep.csv` and `sweep.json`.
  - **Grouping.** Runs are grouped by `(prompt_style, slurm_job_id)`. Within a group, every run
    must share the GPU UUID, allocation and prompt fingerprints, or the tool raises
    `ValueError`. Inside each group it takes the final attempt for each arm, and raises if two
    completed folders exist for one arm.
  - **Reference roles inside a group.** The AR reference is metadata `method=autoregressive` with
    no sparsity settings. The ViSpec reference is `vispec_speculative` with no sparsity settings.
    Exactly one of each is required; a missing AR reference leaves that column null, with a
    note.
  - **One row per (group, arm, benchmark):** job, sweep, the kv and draft settings, `n`, latency
    mean, TTFT mean, TPOT mean, throughput, acceptance mean, decode-peak mean and max (GiB),
    live-KV mean (MiB), quality, minimum kept image tokens per image
    (from `kv_min_kept_per_image` / `draft_min_kept_per_image`), post-compaction peak, exact-token and
    decoded-text agreement with the ViSpec reference, exact-token
    agreement with the AR reference (the spec's secondary column, computed by the tool, because
    method mode refuses sparsity-arm-vs-AR pairs), speedup against AR, latency ratio against the ViSpec reference, latency ratio
    against the same-policy 1.0 arm of the same sweep, and quality delta against the ViSpec
    reference.

- [ ] **Step 1:** Implement.
- [ ] **Step 2: Verify on CPU with the Week 2 short job 393337.** It has 02, 03, six 06 arms and
  four 05 arms.
  - At least six cells must equal the matching Task 9 `compare_metrics` outputs: agreement
    counts, latency ratios and memory ratios.
  - Passing one Week 2 long-style run together with the short job's runs makes the tool raise
    on the mixed group, because the GPU UUID differs.
  - A copied run folder that duplicates an arm (in `.smoke/w3_t3/`) is refused.
- [ ] **Step 3: Commit.** `git commit -m "Add a per-job sweep table for retention sweeps"`

### Task 4: Week 3 sweep runs

- [ ] **Step 1: Submit four jobs.** Clear the tree first: `git status --short` must be empty.
  - **Arm list.** Every sweep uses `ARMS="1.0:spatial 1.0:attention 0.75:spatial 0.75:attention 0.5:spatial 0.5:attention 0.25:spatial 0.25:attention"`.
    Pass it as `KV_ARMS`, `DRAFT_ARMS` and `COMBINED_ARMS`, with `KV_LAYOUT=compact`.
  - **Short style.** One job runs `02 03 06 05 08` with `--time=03:00:00`.
  - **Long style.** Three jobs run `02 03 06`, `02 03 05` and `02 03 08`, each with
    `--time=04:00:00`.
  - **Footprint.** Submit the short job and the long-06 job first. Submit long-05 with
    `--dependency=afterany:<short id>` and long-08 with `--dependency=afterany:<long-06 id>`,
    added through `SBATCH_EXTRA`.
  - Record all job IDs.
- [ ] **Step 2: Verify completion,** the same way as Week 2 Task 9 Step 2:
  - exit 0 and report exit 0;
  - the pinned GPU, with one UUID per job;
  - 100/100 per benchmark;
  - a clean tree;
  - the expected metadata.
- [ ] **Step 3: Identity gate.** Compare every 1.0 arm in sparsity mode against its same-job
  ViSpec reference: 06 ×2, 05 ×2 and 08 ×2 in each style. Each must match exactly on 100/100,
  on both benchmarks. Any mismatch blocks.
- [ ] **Step 4: Diagnose the draft-only arms** (05 at 0.75, 0.5 and 0.25, for both policies and
  both styles) in one `gpu_check.sh` allocation. Report the class counts and `gate_passes`. The
  threshold stays at 0.5. Report each gate failure with its case details, as Week 2 did.
- [ ] **Step 4b: Offset oracle.** Run `offset_check` in fp32 on 5 samples at 0.25 and 0.75
  spatial, and on the combined 0.25 arm, in the same allocation as Step 4. AR must equal ViSpec
  on 5/5. A miss blocks.
- [ ] **Step 5: Build the sweep tables.** Make one per job group: four groups, written to
  `experiment/sweeps/`. Also report the minimum kept image tokens per arm (Review Focus 4).
- [ ] **Step 6:** Make no commit, because the runs are git-ignored. Record the job IDs, run paths,
  comparison paths and sweep paths for Task 5.

### Task 5: Record Week 3 results

**Files:**
- Modify: the 05, 06 and 08 READMEs and `experiment/README.md`

- [ ] **Step 1:** Write the sweep results per method, style and benchmark:
  - quality and agreement against the ViSpec reference;
  - latency against the same-policy 1.0 arm and against AR;
  - acceptance;
  - decode peak and live KV;
  - the gates.

  Say which Week 2 prediction held. Carry forward the Week 2 review's write-up items:
  - mark live-KV savings as arithmetic (`kv_tokens` × bytes) wherever they are stated;
  - put a bf16 note beside every exact-agreement column;
  - explain each latency ratio that runs against the trend;
  - show the R6 and null-reference readings without preferring one after the fact;
  - footnote output-length artefacts in live-KV ratios;
  - use the post-compaction peak for steady-state decode memory;
  - give the 05, 06 and 08 READMEs the `AGENTS.md` cluster-prerequisites section;
  - report AR agreement as the secondary column. Add a hardware summary per job, with `device.txt` links.
  Long-style ratios are computed within each job; state that absolute times across the three
  long jobs are not compared. Leave the trade-off curves to Week 5.
- [ ] **Step 2: Verify.** Spot-check at least twelve values against `sweep.json` and the
  comparison or diagnosis JSON, and show the commands in the task report.
- [ ] **Step 3: Commit.** `git commit -m "Record Week 3 retention sweep results"`
