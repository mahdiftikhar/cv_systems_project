# Week 2: Visual Sparsity Methods Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Profile where Qwen decoding time goes, make KV memory visible to the measurements, and
build the two sparsity methods (draft-side visual pruning and target-side visual KV eviction).
Each method must be token-identical to its unpruned reference at 100% retention, and the eviction
bookkeeping must be shown correct at 50% retention.

**Architecture:** A decode profile (experiment 04) runs first, with a stop for the user's
go-ahead. Then the harness gains:

- report-level memory metrics;
- a `compact` KV layout, which right-sizes the target cache per sample after prefill and
  decouples the cache length from the token count;
- one selection module shared by both methods.

Target eviction runs once after prefill, in both decoders. Draft pruning runs on the draft's image
adaptor input. New experiments 05 (draft pruning) and 06 (KV eviction) sweep their arms inside one
allocation, beside a same-job reference.

**Tech Stack:** Python 3.11 (`vipsec`), torch 2.7.0+cu128, transformers 4.51.3, `torch.profiler`,
Slurm on the pinned RTX A6000s.

**Spec:** [`docs/superpowers/specs/2026-10-18-qwen-dynamic-visual-sparsity-design.md`](../specs/2026-10-18-qwen-dynamic-visual-sparsity-design.md)
(Goal 2, hazards 1–3a). The Week 1 results are in `experiment/02_qwen_autoregressive/README.md`,
`experiment/03_qwen_vispec/README.md` and `experiment/METRICS.md`. The Week 3 sweeps come in a
later plan.

## Global Constraints

These hold for every task:

- **No test files** (`AGENTS.md`). Verification is direct: heredoc checks, `py_compile`, `bash -n`,
  `sbatch --test-only`, and recorded GPU runs.
- **Commits carry plain messages.** No `Co-Authored-By`, no `Claude-Session`, no Claude
  attribution of any kind.
- **Models:** `Qwen/Qwen2.5-VL-3B-Instruct` and `JLKang/ViSpec-Qwen2.5-VL-3B-Instruct` at the
  pinned revisions. No training.
- **Fixed settings:** bf16, temperature 0.0, 1,024 new tokens, depth 3, top-k 8, total-token 30,
  `num_q` 2, batch 1, `min_pixels = 256*28*28`, `max_pixels = 1280*28*28`, and the same 100-sample
  TextVQA and COCO subsets in `long` and `short` styles.
- **Retention budgets:** 0.25, 0.50, 0.75 and 1.00. Week 2 runs only 1.00 and 0.50.
- **Policies:** `spatial` and `attention`. At retention 1.00 every policy must yield every index in
  ascending order, so the selection path runs but changes nothing.
- **Eviction:** target-side eviction happens once, after prefill. It removes only image-token
  positions and never text positions. Each image keeps at least 1 token.
- **Arms and references:** each sparsity arm is compared with the same decoding method at
  retention 1.00, in the same Slurm job, on the same GPU and harness settings. Comparison with the
  other method is a secondary column.
- **00/01 unchanged.** The LLaVA paths in experiments 00 and 01 keep their exact behaviour. The
  Week 1 experiments 02 and 03 stay reproducible with their default settings (`kv_layout=fixed`).
- **Jobs:** submit with `experiment/submit_ampere.sh` using `CLUSTER_PARTITION=preempt` and
  `SBATCH_EXTRA="--cpus-per-task=4 --mem=22G --time=<per task>"`. Clear inherited `SLURM_*`
  variables first (the helper does this). Use
  `PYTHON_BIN=/home/aeternum/anaconda3/envs/vipsec/bin/python` and `PYTHONNOUSERSITE=1`.
- **Generated runs stay out of Git.** Never modify `experiment/*/runs/` contents except by
  creating new folders.

## Review Focus

1. **Retention 1.00 must be bit-identical** to the unpruned reference for both policies, both
  methods and both decoders. The selection code runs but must not change any value or order.
  *(Tasks 5, 6 and 9.)*
2. **Multi-image prompts.** Week 4 manga windows carry about 6 images per prompt. The budget
  applies per image, each image keeps at least 1 token, and text is never evicted.
  *(Task 4 check.)*
3. **Bookkeeping drift after eviction.** Errors would surface over long generations of up to 1,024
  tokens and hundreds of verification steps. At retention 0.50, AR-with-eviction and
  ViSpec-with-eviction must agree as closely as unpruned AR and ViSpec do: near-tie divergences
  only, none at index 0. *(Tasks 5 and 9.)*
4. **Comparisons that silently mix settings.** Arms with different retention, policy or KV layout
  must never be compared as repeats. Each arm gets its own run folder. *(Tasks 7 and 8.)*
5. **Memory that hides eviction.** At retention below 1.00 with the `compact` layout, live KV and
  the decode-phase allocated peak must fall measurably. *(Tasks 3, 5 and 9.)*

---

### Task 1: Decode profile (experiment 04) — CHECKPOINT

**Files:**
- Create: `vispec/evaluation/profile_generation.py`, `experiment/04_qwen_decode_profile/{run.sh,README.md}`
- Modify: `experiment/README.md` (register row, pending until measured)

**Interfaces:**
- Consumes: `evaluate.py` model loading and input building (refactor shared helpers out of
  `evaluate.py` into importable functions if needed, keeping its CLI and outputs identical),
  `autoregressive_generate`, `SpecModel.specgenerate`.
- Produces: `python -m vispec.evaluation.profile_generation --method {autoregressive,vispec}
  --benchmark textvqa --prompt-style long --num-samples 3 --output-dir P` (plus the model, data
  and tree arguments, with `evaluate.py` defaults). It writes:
  - `profile_summary.json`. For each sample it records: wall time; prefill and decode wall time;
    the sum of CUDA kernel time during decode; the GPU-busy fraction (kernel time over decode wall
    time); kernel launches per decode step or per verification iteration; host CPU time inside
    `torch` ops; and the top 15 ops by self-CPU time and by self-CUDA time.
  - a Chrome trace for sample 1.
- After 3 untimed warmups, a fourth untimed run is profiled with `torch.profiler` (activities CPU
  and CUDA, `record_shapes=False`). The decode window is marked with `torch.profiler.record_function`.
- `run.sh` follows the 02/03 runner evidence pattern: unique run folder, metadata, device report
  and CUDA preflight. It profiles both methods on long TextVQA, then COCO, writing `profile/<method>_<bench>/`.

- [ ] **Step 1:** Implement the profiler module and the runner. Write the README with purpose,
  hypothesis and limits. The hypothesis is that Week 1 decode is host-bound, inferred from about
  38 ms/token against a weight-bandwidth floor of about 9 ms. Profiler overhead inflates absolute
  times, so read the ratios, not the absolutes.
- [ ] **Step 2: Verify on CPU.** Run `py_compile`, `bash -n`, and `python -m
  vispec.evaluation.evaluate --help` unchanged. Run a heredoc that imports the shared helpers.
- [ ] **Step 3: Run.** `PROMPT_STYLE=long CLUSTER_PARTITION=preempt SBATCH_EXTRA="--cpus-per-task=4 --mem=22G --time=01:00:00" bash experiment/submit_ampere.sh experiment/04_qwen_decode_profile`.
  Expected: exit 0, and a `profile_summary.json` for each method and benchmark.

  If Step 1 refactored `evaluate.py`, the same job must also run `evaluate.py --method vispec
  --benchmark textvqa --prompt-style long --num-samples 2` into the run folder. Its
  `output_token_ids` must equal the first 2 questions of the Week 1 03 long run.
- [ ] **Step 4:** Write the measured findings into the 04 README and the register. Report the
  GPU-busy fraction, launches per token or iteration, and the top CPU ops, per method.
- [ ] **Step 5: Commit.** `git commit -m "Profile Qwen decoding on the pinned A6000"`
- [ ] **Step 6: CHECKPOINT — stop here.** The controller reports the profile to the user and waits
  for an explicit go-ahead before Task 2. The user may re-scope Tasks 5–6, for example to weight
  memory and long contexts if eviction cannot move latency.

### Task 2: Memory metrics in reports and comparisons

**Files:**
- Modify: `vispec/evaluation/report_metrics.py` (`summarize`, summary markdown),
  `vispec/evaluation/compare_metrics.py` (per-benchmark memory section), `experiment/METRICS.md`

**Interfaces:**
- Produces, per benchmark in `summary.json`, a `memory` block:

  | Field | Unit | Aggregate | Source |
  | --- | --- | --- | --- |
  | `prefill_peak_allocated_gib` | GiB | `distribution()` (mean/median/p95/count) | `gpu_memory[dev]` |
  | `decode_peak_allocated_gib` | GiB | `distribution()` | `gpu_memory[dev]` |
  | `kv_live_mib` | MiB | `distribution()` | `kv_live_bytes` |
  | `kv_tokens` | tokens | `distribution()` | `kv_tokens` |

  Turns without the fields are excluded, and `count` reflects that. The existing
  `gpu_memory_by_visible_device` output is unchanged.
- `comparison.json` gains, per benchmark, `memory_ratio_draft_over_original` for the decode-peak
  mean and the `kv_live` mean.

- [ ] **Step 1:** Implement.
- [ ] **Step 2: Verify.** Re-report the four Week 1 Qwen runs into `.smoke/w2_rereport/` with
  `--output-dir`. For every run, benchmark and field in the Week 1 memory tables of the 02 and 03
  READMEs, the new `memory` means must equal the table values at the table's rounding. The tables
  were computed from the same per-question fields. Print the side-by-side list.

  Re-reporting the 00 snapshot must yield an empty `memory` block with no error.
- [ ] **Step 3: Commit.** `git commit -m "Report phase memory and live KV per benchmark"`

### Task 3: Compact KV layout and decoupled cache offset

**Files:**
- Modify:
  - `vispec/model/kv_cache.py`
  - `vispec/model/spec_model_ours.py` (`specgenerate`)
  - `vispec/model/utils.py` (`update_inference_inputs`)
  - `vispec/model/autoregressive.py`
  - `vispec/evaluation/evaluate.py` (`--kv-layout`)

**Interfaces:**
- Produces:
  - `allocate_past_key_values(model, capacity: int) -> None` in `kv_cache.py`. It (re)builds
    `model.past_key_values`, `model.past_key_values_data` and `model.current_length_data` at exactly
    `capacity` slots, after releasing the previous buffers.
  - `compact_past_key_values(model, keep: torch.LongTensor | None, capacity: int) -> int` in
    `kv_cache.py`. It copies the cached positions in `keep` (ascending; `None` means all
    `current_length` positions) into a fresh buffer of `capacity` slots, drops the old buffer,
    sets the length, and returns the kept length.
  - `update_inference_inputs(..., kv_offset: int = 0)`. Cache writes use `prev_kv_len =
    input_ids.shape[1] - kv_offset` wherever the code now uses `prev_input_len` for cache
    indexing; token bookkeeping still uses `prev_input_len`. With the default 0, behaviour is
    unchanged, so LLaVA is unaffected.
  - `SpecModel.from_pretrained(..., kv_layout: str = "fixed")`. `compact` is allowed only for
    Qwen2.5-VL and raises `ValueError` otherwise.
- **`compact` layout in `specgenerate` and `autoregressive_generate`:**
  1. Before prefill, allocate exactly `prompt_len + total_token + 10` slots for ViSpec, or
     `prompt_len + 10` for AR.
  2. After prefill (for ViSpec, after `initialize_tree`), call `compact_past_key_values(model,
     keep, prompt_kept + max_new_tokens + total_token + 10)`.
  3. Pass `kv_offset = prompt_len - prompt_kept` into every `update_inference_inputs` call.
  4. In AR decode steps, pass explicit `position_ids`: the logical index `input_ids.shape[1]` plus
     the base model's `rope_deltas`, expanded for M-RoPE as `tree_decoding` does. Positions then
     never derive from the cache length.

  In Task 3 itself, `keep` is always `None` (no eviction yet).
- `evaluate.py`: add `--kv-layout {fixed,compact}` (default `fixed`). Record it in `runtime.json`
  `generation_args`. The `fixed` path stays byte-identical to Week 1.

- [ ] **Step 1:** Implement.
- [ ] **Step 2: Verify on CPU.** `py_compile` the changed files. Run
  `inspect.signature` checks for the new defaults, and the LLaVA evaluator imports.
- [ ] **Step 3: Run the GPU identity check.** Use `CLUSTER_PARTITION=preempt
  SRUN_EXTRA="--mem=22G --cpus-per-task=4 --time=01:00:00" bash experiment/gpu_check.sh ...`.
  For each method in {autoregressive, vispec}, run `--benchmark textvqa --prompt-style long
  --num-samples 5` twice, once with `--kv-layout fixed` and once with `compact`, into `.smoke/w2_t3/`.
  Expected, checked by heredoc: `output_token_ids` are identical per question, and the max
  `decode_peak_allocated_bytes` is lower under `compact`. That checks a memory reduction, not an
  exact value.
- [ ] **Step 4: Commit.** `git commit -m "Add a compact per-sample KV layout with a decoupled cache offset"`

### Task 4: Shared image-token selection

**Files:**
- Create: `vispec/model/sparsity.py`

**Interfaces:**
- Produces:
  ```python
  POLICIES = ("spatial", "attention")
  @dataclass(frozen=True)
  class SparsityConfig:
      retention: float = 1.0      # one of 0.25, 0.5, 0.75, 1.0
      policy: str = "spatial"
  def image_spans(image_mask: torch.BoolTensor) -> list[tuple[int, int]]   # [start, end) per contiguous image run, 1-D mask
  def keep_count(n: int, retention: float) -> int                          # max(1, round(retention * n)); n when retention == 1.0
  def select_in_span(n: int, k: int, policy: str, scores: torch.Tensor | None) -> torch.LongTensor
  #   spatial: torch.linspace(0, n - 1, k).round().long().unique(); if fewer than k remain, add the
  #   smallest unused indices until there are k; return ascending.
  #   attention: topk(scores, k).indices, sorted ascending. scores has shape [n] and is required.
  def keep_positions(image_mask: torch.BoolTensor, config: SparsityConfig,
                     scores: torch.Tensor | None = None) -> torch.LongTensor
  #   Returns all non-image positions plus the selected positions of every image span, ascending.
  #   With config.retention == 1.0 it returns torch.arange(len(image_mask)).
  ```
- `spatial` means uniform raster-stride subsampling over each image's merged tokens. Both methods
  use it. This is the plan's concrete choice for the spec's "spatial pooling": it hits budgets
  exactly and stays a subset, so identity at 1.00 holds.

- [ ] **Step 1:** Implement.
- [ ] **Step 2: Verify on CPU.** Build a synthetic mask:
  `[text×5, img×100, text×3, img×37, text×4]`. Check that:
  - (a) for every policy at retention 1.0, the result equals `arange(149)`;
  - (b) at 0.5 spatial, each span keeps 50 and 18 positions, all 12 text positions are present,
    and the result is strictly ascending;
  - (c) at 0.25 attention with random scores, each span keeps 25 and 9, the kept positions are the
    top scores, and the result is ascending;
  - (d) a 1-token image at 0.25 keeps 1;
  - (e) `attention` without scores raises `ValueError`.
- [ ] **Step 3: Commit.** `git commit -m "Add shared image-token selection for visual sparsity"`

### Task 5: Target-side visual KV eviction

**Files:**
- Modify:
  - `vispec/model/modeling_qwen2_5_vl_kv.py` (score capture in `Qwen2_5_VLSdpaAttention.forward`)
  - `vispec/model/spec_model_ours.py`
  - `vispec/model/autoregressive.py`
  - `vispec/evaluation/evaluate.py`
  - `vispec/evaluation/measurement.py` (additive fields)

**Interfaces:**
- Consumes: Task 3 (`compact_past_key_values`, `kv_offset`) and Task 4 (`keep_positions`, `SparsityConfig`).
- **Score capture.** When `module.kv_score_window` is a positive int and the call is a prefill
  (`q_len > 1`), each language-model attention layer adds an importance score to a shared
  `model.kv_scores` tensor of shape `[prompt_len]`. The score is the sum over heads and window
  queries of `softmax(Q_win · Kᵀ / √d)`, with causal masking. The window is the queries after the
  last image token. With no capture flag the module is unchanged.
- **One kept set for all layers.** Scores are summed over layers, and one set of positions applies
  to every layer. This makes the evicted model equal to "hide these image keys from all later
  queries".
- **Eviction.** With `SparsityConfig` `kv_sparsity` passed through `specgenerate(...,
  kv_sparsity=None)` and `autoregressive_generate(..., kv_sparsity=None)`, and the `compact` layout
  required:
  1. Prefill runs with capture on, for the `attention` policy only.
  2. Compute `keep = keep_positions(image_mask[0], kv_sparsity, scores)`.
  3. Call `compact_past_key_values(model, keep, ...)`.
  4. Set `kv_offset`.

  `kv_sparsity=None` means no eviction and no capture, which is the Task 3 behaviour.
- **CLI.** `--kv-retention {0.25,0.5,0.75,1.0}` (default 1.0) and `--kv-policy {spatial,attention}`
  (default `spatial`). Retention below 1.0 requires `--kv-layout compact`. Both are recorded in
  `generation_args`.
- **Measurement.** Additive fields: `kv_prompt_tokens`, `kv_kept_prompt_tokens`,
  `kv_evicted_tokens`.

- [ ] **Step 1:** Implement.
- [ ] **Step 2: Verify on CPU.** `py_compile` the changed files. Run a heredoc that builds a tiny
  random fake KV of 2 layers with 2 heads, compacts it with a known `keep`, and asserts the gathered
  keys and values equal the source rows.
- [ ] **Step 3: Run the GPU smoke** in one `gpu_check.sh` allocation, outputs in `.smoke/w2_t5/`,
  long TextVQA, 5 samples, `compact` layout, both methods:
  - (a) Arms at 1.0 with spatial, and at 1.0 with attention, against the Task 3 `compact`
    reference. `output_token_ids` must be identical.
  - (b) An arm at 0.5 with spatial. `kv_evicted_tokens > 0` on every sample, and the decode peak
    max must be below the 1.0 arm's.
  - (c) AR against ViSpec, both at 0.5 spatial: report the exact agreement count and the first
    mismatch indices. Index 0 must never be the first mismatch.
- [ ] **Step 4: Commit.** `git commit -m "Add target-side visual KV eviction after prefill"`

### Task 6: Draft-side visual pruning

**Files:**
- Modify: `vispec/model/cnets_ours.py` (image branch of `Model.forward`, around lines 896–930),
  `vispec/model/spec_model_ours.py`, `vispec/evaluation/evaluate.py`, `vispec/evaluation/measurement.py`

**Interfaces:**
- Consumes: Task 4.
- **Configuration.** `spec_layer.draft_sparsity: SparsityConfig | None` is set by
  `specgenerate(..., draft_sparsity=None)` for the duration of a call. When it is set, the image
  embeddings for each span (`img_emd`, shape `[1, n, d]`) are reduced to the kept indices before
  `self.imadpt(...)`:
  - `spatial` uses `select_in_span(n, keep_count(n, r), "spatial", None)`;
  - `attention` uses the adaptor's own attention. The score is the sum over the `num_q` queries
    and the heads of `softmax(q · k_projᵀ / √d)` over the span. The selected keys are then fed back
    through the unchanged `imadpt`.

  The draft position ids stay as they are, because the adaptor output length is still `num_q`.
- **CLI.** `--draft-retention {0.25,0.5,0.75,1.0}` (default 1.0) and `--draft-policy
  {spatial,attention}`. These apply to `--method vispec` only, and AR rejects a retention below
  1.0. Both are recorded in `generation_args`.
- **Measurement.** Additive fields: `draft_image_tokens` and `draft_kept_image_tokens`.

- [ ] **Step 1:** Implement.
- [ ] **Step 2: Verify on CPU.** `py_compile`. Run a heredoc on a randomly initialised
  `ImgAdaptor` with a fixed seed. At retention 1.0, both policies must produce outputs that are
  bit-identical (`torch.equal`) to the unpruned adaptor.
- [ ] **Step 3: Run the GPU smoke.** ViSpec only, long TextVQA, 5 samples, `compact` layout, into
  `.smoke/w2_t6/`:
  - at 1.0 with both policies, tokens must be identical to the Task 3 `compact` reference;
  - at 0.5 with spatial, it must run, with acceptance recorded and `draft_kept_image_tokens` about
    half of `draft_image_tokens`.
- [ ] **Step 4: Commit.** `git commit -m "Add draft-side visual pruning of the image adaptor input"`

### Task 7: Sparsity comparison mode

**Files:**
- Modify: `vispec/evaluation/compare_metrics.py`, `vispec/evaluation/diagnose_divergence.py` (accept the new mode's comparisons), `experiment/METRICS.md`

**Interfaces:**
- **Option.** `compare_metrics --mode {method,sparsity}`, default `method`, which is today's
  behaviour.
- **Sparsity mode.** `--original` is the reference and `--draft` is the candidate.
  - Both groups must have the same `method`, either AR or ViSpec.
  - The reference must have `kv_retention == 1.0` and `draft_retention == 1.0`.
  - Every guarded setting must match, plus `kv_layout`, `prompt_style` and the tree settings.
  - Sparsity settings (`kv_retention`, `kv_policy`, `draft_retention`, `draft_policy`) may differ
    only between the groups, never within one.
  - These settings are read from `runtime.json` `generation_args`.
  - In `method` mode, all four sparsity settings and `kv_layout` must match across every compared
    report. An absent value counts as its own value, so Week 1 pairs still compare.
  - The output schema is unchanged, plus a `mode` field and a `sparsity` block naming each side's
    settings.
- **Diagnosis.** In sparsity mode, `diagnose_divergence` refuses comparisons whose candidate has
  `kv_retention < 1.0`. Its stock-model reference cannot reproduce eviction, so it must say so and
  must not mislabel the divergences. Diagnosis stays valid for draft-only arms, because the target
  is unchanged.

- [ ] **Step 1:** Implement.
- [ ] **Step 2: Verify on CPU** using `.smoke/` copies of the Task 5 and 6 smoke reports:
  - the 1.0 arm against the reference compares, with 5/5 agreement;
  - a reference whose copy is edited to `kv_retention=0.5` is refused;
  - mixing AR and ViSpec in sparsity mode is refused;
  - method mode on the Week 1 long pair still reproduces its `comparison.json` headline values.
- [ ] **Step 3: Commit.** `git commit -m "Add a sparsity comparison mode against the 100% reference"`

### Task 8: Experiments 05 and 06, reference layout, docs

**Files:**
- Create: `experiment/05_qwen_draft_pruning/{run.sh,README.md}`, `experiment/06_qwen_kv_eviction/{run.sh,README.md}`
- Modify: `experiment/02_qwen_autoregressive/run.sh`, `experiment/03_qwen_vispec/run.sh` (add
  `KV_LAYOUT`, default `fixed`, recorded in metadata), `experiment/README.md`, `CLAUDE.md`,
  `AGENTS.md` (Qwen scope sentence lists 04–06)

**Interfaces:**
- **Arm list.** 05 reads `DRAFT_ARMS` and 06 reads `KV_ARMS`. Each is a space-separated list of
  `<retention>:<policy>[:<method>]`. 06's method defaults to `vispec`, and 06 also accepts
  `autoregressive`.
- **Per arm.** Each arm writes its own run folder, `runs/<UTC>_<commit12>_<style>_r<ret>_<policy>[_<method>]_XXXXXX`,
  using the same evidence capture as 03. Both experiments force `--kv-layout compact`. Metadata
  adds `kv_layout`, the four sparsity settings and `arm=<spec>`.
- **Order inside a job.** References come first (02 and 03 with `KV_LAYOUT=compact`), then the
  arms.
- **Docs.** The READMEs follow `AGENTS.md`, with results `pending`. `CLAUDE.md` gains the new flags,
  `kv_layout`, and the rule that eviction runs are not diagnosable against the stock model.

- [ ] **Step 1:** Implement.
- [ ] **Step 2: Verify.**
  - `bash -n` on every runner.
  - An invalid arm such as `KV_ARMS="0.3:spatial"` exits 2 before creating a run folder.
  - `PROMPT_STYLE=long KV_LAYOUT=compact KV_ARMS="1.0:spatial" SBATCH_EXTRA=--test-only CLUSTER_PARTITION=preempt bash experiment/submit_ampere.sh experiment/03_qwen_vispec experiment/06_qwen_kv_eviction`
    prints an estimate.
- [ ] **Step 3: Commit.** `git commit -m "Add Qwen draft-pruning and KV-eviction experiments"`

### Task 9: Week 2 correctness runs

- [ ] **Step 1: Submit one job per style.** Use `SBATCH_EXTRA="--cpus-per-task=4 --mem=22G --time=06:00:00"`
  and these experiments in order: `02` and `03` with `KV_LAYOUT=compact`, then
  `06 KV_ARMS="1.0:spatial 1.0:attention 0.5:spatial 0.5:attention 0.5:spatial:autoregressive"`,
  then `05 DRAFT_ARMS="1.0:spatial 1.0:attention 0.5:spatial 0.5:attention"`. Record the job IDs.
- [ ] **Step 2: Verify completion.** Every run must show exit 0, the pinned GPU, one shared UUID
  per job, 100/100 per benchmark, and a clean tree.
- [ ] **Step 3: Identity gate.** Compare each 1.0 arm with its reference in sparsity mode. It must
  match exactly on 100/100 for both benchmarks and both styles. Any mismatch blocks the task;
  investigate with superpowers:systematic-debugging.
- [ ] **Step 4: Bookkeeping check at 0.50.**
  - Compare AR-with-eviction against ViSpec-with-eviction at 0.5 spatial in method mode.
  - Report the agreement and the first-mismatch index distribution next to the unpruned 02-vs-03
    compact pair from the same job. Index 0 must never be the first mismatch.
  - If agreement falls more than 10 percentage points below the unpruned pair, stop and report.
  - Run `diagnose_divergence` on the draft-only 0.5 arms.
- [ ] **Step 5:** No commit, because the runs are git-ignored. Record the job IDs and paths for
  Task 10.

### Task 10: Record Week 2 results

**Files:**
- Modify: the 04/05/06 READMEs and `experiment/README.md`

- [ ] **Step 1:** Write per-benchmark, per-style tables for the 1.0 identity outcome and the 0.5
  arms. Cover:
  - latency against the reference;
  - acceptance;
  - quality, which is primary for short style;
  - live KV and decode peak, with the eviction counts;
  - agreement;
  - the bookkeeping-check result.

  Add a hardware summary that links each `device.txt`. Mark 0.25/0.75 and the combined arms as
  Week 3, pending. Frame causal explanations as inference, not measurement.
- [ ] **Step 2: Verify.** Spot-check at least eight values against their source JSON, with the
  commands in the report.
- [ ] **Step 3: Commit.** `git commit -m "Record Week 2 visual-sparsity correctness results"`
