# Week 1: Qwen2.5-VL-3B Baselines Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Record measured Qwen2.5-VL-3B standard-decoding and unmodified-ViSpec baselines on the
TextVQA and COCO Captions subsets, in both prompt styles, on one pinned Ampere GPU model. The
recording uses the output-preserving memory harness that every later sparsity arm will share.
Every greedy divergence between the two methods is explained.

**Architecture:** The existing per-benchmark evaluator copies stay untouched for 00/01. Qwen work
goes through one new evaluator, `vispec/evaluation/evaluate.py`. It takes benchmark adapters from
`vispec/evaluation/benchmarks.py`, and loads one `spec_model_ours.SpecModel` for both methods (the
autoregressive method decodes with the base model only). Its answer files use the existing JSONL
schema, so `report_metrics` and `compare_metrics` keep working. A new submission path for this
cluster runs both methods in one allocation, on the same GPU.

**Tech Stack:** Python 3.11 (`vipsec` conda env), torch 2.7.0+cu128, transformers 4.51.3, datasets
3.5.0, pycocoevalcap 1.2 with OpenJDK 17, Slurm.

**Spec:** [`docs/superpowers/specs/2026-10-18-qwen-dynamic-visual-sparsity-design.md`](../specs/2026-10-18-qwen-dynamic-visual-sparsity-design.md).
Read it first. This plan covers its Goal 1 (Week 1). The roadmap at the end lists the later plans.

## Global Constraints

These apply to every task:

- **No test files.** `AGENTS.md` forbids creating or modifying test files. Each "verify" step is a
  direct check: an inline `python - <<'PY'` heredoc with `assert`s, `bash -n`,
  `sbatch --test-only`, or a recorded GPU smoke run. Never add `tests/`, pytest files or a test
  framework.
- **Models:** target `Qwen/Qwen2.5-VL-3B-Instruct`, draft `JLKang/ViSpec-Qwen2.5-VL-3B-Instruct`.
  Revisions observed on 2026-10-18 were `66285546d2b821cf421d4f5eb2576359d3770cd3` (target) and
  `7dd32bc8a7010ed1efc16d17118fa7bdb3f29fe9` (draft). Preparation pins and records whatever it
  resolves.
- **Settings shared by all arms:** bf16 (`torch_dtype="auto"`), temperature 0.0, at most 1,024
  new tokens, depth 3, top-k 8, total-token 30, `num_q` 2, batch size 1.
- **Image limits:** `min_pixels = 256*28*28`, `max_pixels = 1280*28*28`, for every benchmark and
  style.
- **Subsets:** 100 TextVQA questions from the same seed-42 shuffle as experiment 00, and 100 COCO
  Karpathy-test images with a seed-42 shuffle. Sample counts are reported per benchmark.
- **Python for jobs:** `PYTHON_BIN=/home/aeternum/anaconda3/envs/vipsec/bin/python`. Always use
  `PYTHONNOUSERSITE=1`.
- **Hardware:** comparison runs only use the 48 GB Ampere GPU model pinned in Task 1. Before
  `srun`/`sbatch`, clear the shell's inherited `SLURM_*` variables (this shell runs inside VS Code
  job 391999).
- **Commits:** plain messages with no `Co-Authored-By`, `Claude-Session` or other Claude
  attribution. Work on branch `qwen-visual-sparsity`.
- **Evidence:** follow the `AGENTS.md` capture rules. Generated runs stay out of Git. The 00/01
  LLaVA code paths and their versioned snapshots must keep behaving exactly as before.
- **Benchmark order:** TextVQA runs first, then COCO Captions.

## Review Focus

These are the failure modes most likely to bite, and the task that pins each one:

1. **Qwen special tokens leaking into answers.** EOS `<|im_end|>` (151645) is above
   `tokenizer.vocab_size` (151643). The inherited `output_ids[ids > vocab_size] = 0` turns it
   into `!`. Answers must be cut at the first EOS and decoded with special tokens skipped, with
   raw IDs saved unchanged. *(Task 7 check.)*
2. **A job landing on a different GPU model.** It must stop before evaluation with a clear
   message, not produce an unmatched row. *(Task 1 check.)*
3. **Mixing prompt styles.** A long-style run and a short-style run must never share a run folder,
   and `compare_metrics` must refuse to pair them. *(Task 9 and Task 10 checks.)*
4. **A missing offline asset or scorer inside a batch job.** Missing processor files, COCO
   images, the pycocoevalcap jar or `java` must fail at preflight. If only a scorer is missing,
   the score stays `pending`; the run must not crash mid-evaluation. *(Task 3, Task 8 and Task 9
   checks.)*
5. **Slimming that changes outputs, or breaks LLaVA.** With slimming on and off, outputs must be
   token-identical, and LLaVA `SpecModel` calls must be unaffected. *(Task 5 and Task 7 checks.)*

---

### Task 1: Ampere submission path and GPU pin

**Files:**
- Create: `experiment/ampere_env.sh`, `experiment/gpu_check.sh`, `experiment/submit_ampere.sh`,
  `experiment/run_sequence.sbatch`, `experiment/AMPERE_SETUP.md`
- Modify: `.gitignore`, `.rsyncignore` (add `/.smoke/`)

**Interfaces:**
- Produces:
  - `source experiment/ampere_env.sh` exports `CLUSTER_PARTITION` (default `gpu`),
    `CLUSTER_GPU_GRES` (`gpu:ampere:1`), `CLUSTER_EXCLUDE` (`c4-4,c4-5,c5-2`, extended by the probe),
    `CLUSTER_EXPECTED_GPU` (the exact `nvidia-smi` name from the probe) and `PYTHON_BIN` (vipsec
    python). It also defines `clear_inherited_slurm`.
  - `bash experiment/gpu_check.sh CMD...` clears `SLURM_*` and runs
    `srun --partition --gres --exclude --cpus-per-task=4 --mem=32G --time=01:00:00 $SRUN_EXTRA CMD...`.
  - `PROMPT_STYLE=long|short bash experiment/submit_ampere.sh EXP_DIR [EXP_DIR...]` submits
    `run_sequence.sbatch` once with `EXPERIMENTS` set to the given directories (colon-separated),
    `REPO` set to the repository root, `--export=ALL`, logs under `logs/%x-%j.log`, and any extra
    `sbatch` options from `$SBATCH_EXTRA`.
  - `run_sequence.sbatch` checks that `nvidia-smi --query-gpu=name --format=csv,noheader` equals
    `CLUSTER_EXPECTED_GPU`, exiting with status 3 before any evaluation if not. It then runs each
    `EXP_DIR/run.sh` in order in the same allocation. It continues past a failed experiment and
    exits non-zero if any failed. Resources: 8 CPUs, 64G, 08:00:00, `HF_HUB_OFFLINE=1`,
    `HF_DATASETS_OFFLINE=1`.

- [ ] **Step 1:** Write `ampere_env.sh` and `gpu_check.sh`. Then probe each 48 GB node:
  `for n in c1-8 c2-1 c2-2 c4-2 c4-3; do SRUN_EXTRA="-w $n --time=00:05:00" bash experiment/gpu_check.sh nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader; done`.
  Expected: one line per node. Set `CLUSTER_EXPECTED_GPU` to the majority model name, and add any
  node with a different name to `CLUSTER_EXCLUDE`.
- [ ] **Step 2:** Write `submit_ampere.sh` and `run_sequence.sbatch`.
- [ ] **Step 3: Verify.** Run `bash -n` on all four scripts. Then
  `PROMPT_STYLE=long SBATCH_EXTRA=--test-only bash experiment/submit_ampere.sh experiment/00_baseline_original`.
  Expected: `sbatch: Job ... to start at ...` and no submission. Then
  `CLUSTER_EXPECTED_GPU=nonexistent EXPERIMENTS=experiment/00_baseline_original REPO=$PWD SRUN_EXTRA="--time=00:05:00" bash experiment/gpu_check.sh bash experiment/run_sequence.sbatch`.
  Expected: exit 3 with a GPU-mismatch message, and no new folder under
  `experiment/00_baseline_original/runs/`.
- [ ] **Step 4:** Write `AMPERE_SETUP.md`: the probe results, the pin, how to submit, and how to
  clear `SLURM_*`. It points to `NEWTON_SETUP.md` for 00/01.
- [ ] **Step 5: Commit.** `git commit -m "Add Ampere cluster submission path with GPU pin"`

### Task 2: CIDEr scoring dependencies

**Files:**
- Modify: `requirements.txt` (add `pycocoevalcap==1.2`), `experiment/AMPERE_SETUP.md` (environment section)

- [ ] **Step 1:** Install with
  `~/.local/bin/uv pip install --python "$PYTHON_BIN" pycocoevalcap==1.2` and
  `conda install -n vipsec -y -c conda-forge openjdk=17`. Document both commands, and the need to
  put `$(dirname "$PYTHON_BIN")` on `PATH` so `java` resolves without activating the env.
- [ ] **Step 2: Verify.**
  ```bash
  PATH="$(dirname "$PYTHON_BIN"):$PATH" PYTHONNOUSERSITE=1 "$PYTHON_BIN" - <<'PY'
  from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
  from pycocoevalcap.cider.cider import Cider
  refs = {1: [{"caption": "a man riding a red moped on a dirt road"}, {"caption": "a man rides a motor bike on a dirt road"}],
          2: [{"caption": "two dogs playing in the snow"}, {"caption": "dogs run through deep snow"}]}
  good = {1: [{"caption": "a man riding a moped on a dirt road"}], 2: [{"caption": "two dogs playing in snow"}]}
  bad = {1: [{"caption": "a plate of food"}], 2: [{"caption": "a plate of food"}]}
  tok = PTBTokenizer()
  r = tok.tokenize(refs)
  sg, _ = Cider().compute_score(r, tok.tokenize(good))
  sb, _ = Cider().compute_score(r, tok.tokenize(bad))
  assert sg > sb >= 0, (sg, sb)
  print("cider ok", sg, sb)
  PY
  ```
  Expected: `cider ok` with the first value greater than the second.
  `uv pip check --python "$PYTHON_BIN"` must still report everything compatible.
- [ ] **Step 3: Commit.** `git commit -m "Add pycocoevalcap and Java prerequisite for CIDEr"`

### Task 3: Qwen asset preparation

**Files:**
- Modify: `experiment/prepare_assets.py`, `prepare.sh` (pass-through only), `experiment/AMPERE_SETUP.md`

**Interfaces:**
- Consumes: `vispec.evaluation.benchmarks.coco_subset` (Task 4). Insert the repository root on
  `sys.path` before importing it. Tasks 3 and 4 can be built together; Task 3 is verified after
  Task 4.
- Produces:
  - `--suite {llava,qwen}`, default `llava`, which keeps today's behaviour byte-for-byte.
  - The `qwen` suite prepares `models/Qwen2.5-VL-3B-Instruct/` and
    `models/ViSpec-Qwen2.5-VL-3B-Instruct/` (pinned via `snapshot_download`, processor required on
    the base), the existing TextVQA assets including `tools/downloaded/m4c_evaluator.py`,
    `bench_data/coco/karpathy_test.parquet` (from
    `yerevann/coco-karpathy` file `data/test-00000-of-00001.parquet`, revision pinned in the
    manifest, observed `448fdb1bc7b2d09e46881c4541a14d796a3d41e8`), and the 100 subset images
    under `bench_data/coco/val2014/<filename>`, each fetched from its row's `url` with the existing
    resumable `download_http`.
  - It writes `.preparation/assets_qwen.json`. The existing `assets.json` is left alone.
  - `--check-only --suite qwen` validates model shards and processor files, the parquet SHA-256,
    and that all 100 images exist and decode with PIL.

- [ ] **Step 0:** Before editing, save the default-suite behaviour:
  `mkdir -p .smoke && PYTHON_BIN=$PYTHON_BIN bash prepare.sh --check-only > .smoke/prepare_llava_before.txt 2>&1; echo $? >> .smoke/prepare_llava_before.txt`.
- [ ] **Step 1:** Implement the suite option, reusing the existing download, validate and lock
  helpers.
- [ ] **Step 2: Verify (after Task 4).** Run `PYTHON_BIN=$PYTHON_BIN bash prepare.sh --suite qwen`
  inside job 391999 (a CPU node with network). Then run
  `PYTHON_BIN=$PYTHON_BIN bash prepare.sh --check-only --suite qwen`. Expected: exit 0,
  `assets_qwen.json` lists both revisions, and 100 COCO files are present. Then repeat Step 0's
  command into `.smoke/prepare_llava_after.txt`, and `diff` the two files. Expected: no difference
  other than the log-folder path line.
- [ ] **Step 3: Commit.** `git commit -m "Prepare Qwen checkpoints and COCO Karpathy subset"`

### Task 4: Benchmark samples and Qwen prompt inputs

**Files:**
- Create: `vispec/evaluation/benchmarks.py`

**Interfaces:**
- Produces:
  ```python
  BENCHMARK_NAMES = ("textvqa", "coco_caption")
  PROMPT_STYLES = ("long", "short")
  QWEN_MIN_PIXELS = 256 * 28 * 28
  QWEN_MAX_PIXELS = 1280 * 28 * 28

  @dataclass(frozen=True)
  class Sample:
      question_id: str
      image_paths: tuple[str, ...]   # opened by build_inputs, outside timing
      question: str | None           # None for captioning
      references: tuple[str, ...]

  def coco_subset(parquet_path: Path, num_samples: int = 100, seed: int = 42) -> "datasets.Dataset"
  def load_samples(benchmark: str, data_dir: Path, num_samples: int = 100) -> list[Sample]
  def build_messages(benchmark: str, sample: Sample, prompt_style: str) -> list[dict]
  def load_processor(model_path: str) -> "transformers.ProcessorMixin"
  def build_inputs(processor, benchmark: str, sample: Sample, prompt_style: str, device: str = "cuda:0") -> "BatchFeature"
  ```
- **TextVQA:** `TextVQA_0.5.1_val.json`, `qid = question_id`, references are the 10 `answers`,
  and the image is `train_images/<image_id>.jpg`. Subset order: `Dataset.from_list(rows)` over
  image-free dicts in file order, then `.shuffle(seed=42).select(range(num_samples))`, which is
  the same permutation as experiment 00.
- **COCO:** `question_id = str(cocoid)`, references are `sentences`, and the image is
  `bench_data/coco/val2014/<filename>`.
- **Messages** keep the inherited layout: the inherited system text verbatim, then a user turn
  with `[question text (TextVQA only), instruction text, {"type": "image"}]`. Instruction text:
  - TextVQA long: the inherited OCR instruction from `textvqa_prompt.py`, verbatim.
  - TextVQA short: `Answer the question using a single word or phrase.`
  - COCO long: `Please provide a detailed description of the given image.`
  - COCO short: `Provide a one-sentence caption for the provided image.`
- `build_inputs` applies `processor.apply_chat_template(..., add_generation_prompt=True)`, then
  `processor(images=..., text=..., return_tensors="pt")`. The processor is built once with the
  pixel limits above.

- [ ] **Step 1:** Implement `benchmarks.py`.
- [ ] **Step 2: Verify (CPU, after Task 3 has fetched the processor files).**
  ```bash
  PYTHONNOUSERSITE=1 "$PYTHON_BIN" - <<'PY'
  import json, glob
  from pathlib import Path
  from vispec.evaluation.benchmarks import load_samples, load_processor, build_inputs
  snap = glob.glob("experiment/00_baseline_original/runs/20261017T211853Z_771911bd8665_9RRsjY/results/textvqa_test/*/*.manifest.json")[0]
  old = [q["question_id"] for q in json.loads(Path(snap).read_text())["questions"]]
  tv = load_samples("textvqa", Path("bench_data"))
  assert [s.question_id for s in tv] == old, "TextVQA subset differs from experiment 00"
  coco = load_samples("coco_caption", Path("bench_data"))
  assert len(coco) == 100 and len(set(s.question_id for s in coco)) == 100
  assert all(len(s.references) >= 5 for s in coco)
  proc = load_processor("models/Qwen2.5-VL-3B-Instruct")
  for bench, s in (("textvqa", tv[0]), ("coco_caption", coco[0])):
      lens = {}
      for style in ("long", "short"):
          x = build_inputs(proc, bench, s, style, device="cpu")
          n_img = int((x["input_ids"] == proc.tokenizer.convert_tokens_to_ids("<|image_pad|>")).sum())
          assert n_img == int(x["image_grid_thw"].prod()) // 4 <= 1280
          lens[style] = x["input_ids"].shape[1]
      assert lens["long"] > lens["short"], lens
  print("benchmarks ok")
  PY
  ```
  Expected: `benchmarks ok`. If the 00 manifest's IDs differ only in type, compare them as strings.
  Any other difference means the TextVQA subset or its order differs from 00; fix that.
- [ ] **Step 3: Commit.** `git commit -m "Add Qwen benchmark adapters for TextVQA and COCO"`

### Task 5: Fixed KV capacity, output-preserving prefill slimming, phase memory

**Files:**
- Modify: `vispec/model/kv_cache.py` (`initialize_past_key_values`),
  `vispec/model/modeling_qwen2_5_vl_kv.py` (`Qwen2_5_VLForConditionalGeneration.forward`, class at
  line 1752), `vispec/model/spec_model_ours.py` (`from_pretrained`, `__init__`, `forward`,
  `specgenerate`), `vispec/model/utils.py` (`initialize_tree`), `vispec/evaluation/timing.py`,
  `vispec/evaluation/measurement.py`

**Interfaces:**
- `initialize_past_key_values(model, max_length: int | None = None)`. `None` keeps
  `config.max_position_embeddings`, so LLaVA is unchanged.
- `Qwen2_5_VLForConditionalGeneration.forward(..., logits_to_keep: int = 0, last_hidden_only: bool = False)`.
  When `logits_to_keep > 0`, `lm_head` runs on `hidden_states[:, -logits_to_keep:]` only. When
  `last_hidden_only` is set, the text model runs with `output_hidden_states=False` and the output's
  `hidden_states` is `(final_normed_hidden,)`.
- `SpecModel.from_pretrained(..., slim_prefill: bool = False, kv_capacity: int | None = None)`
  stores `self.slim_prefill` (forced to `False` for any architecture other than
  `Qwen2_5_VLForConditionalGeneration`) and `self.kv_capacity`.
- `SpecModel.forward(..., logits_to_keep: int = 0)` passes `last_hidden_only=True` (and
  `logits_to_keep`, when non-zero) to the base model only when `self.slim_prefill` is set.
- `initialize_tree` passes `logits_to_keep=1` only when `getattr(model, "slim_prefill", False)`.
- `specgenerate` allocates with `initialize_past_key_values(self.base_model, max_length=self.kv_capacity)`.
  Before prefill it raises `ValueError("KV capacity ...")` if
  `prompt_len + max_new_tokens + total_token + 10 > kv_capacity`.
- `FirstTokenTimer.mark` additionally records `prefill_peak_allocated_bytes` per visible device,
  then calls `torch.cuda.reset_peak_memory_stats`. `GenerationMeasurement.finish` adds
  `decode_peak_allocated_bytes`, `kv_tokens` and `kv_live_bytes` to `gpu_memory`. Here
  `kv_live_bytes = kv_tokens * num_layers * 2 * kv_heads * head_dim * dtype_bytes`, and `kv_tokens`
  is the cache length at finish. `finish` gains an optional `kv_tokens: int | None = None`
  argument; when it is `None` these fields are `None`. The schema stays version 1 with additive
  fields.

- [ ] **Step 1:** Implement. Default arguments must leave every existing call site's behaviour
  identical.
- [ ] **Step 2: Verify that LLaVA is unchanged (CPU, no weights).** Check that `inspect.signature`
  of `initialize_past_key_values`, `SpecModel.forward` and `from_pretrained` show the new
  parameters with the defaults above. Check `git diff --stat` touches only the listed files.
  `PYTHONNOUSERSITE=1 "$PYTHON_BIN" -c "import vispec.evaluation.gen_spec_answer_textvqa, vispec.evaluation.gen_baseline_answer_textvqa"`
  must succeed. Output identity is verified in Task 7.
- [ ] **Step 3: Commit.** `git commit -m "Add KV capacity, slim Qwen prefill and phase memory capture"`

### Task 6: Autoregressive generation on the shared model

**Files:**
- Create: `vispec/model/autoregressive.py`

**Interfaces:**
- Consumes: `SpecModel` attributes from Task 5 (`slim_prefill`, `kv_capacity`).
- Produces:
  ```python
  @torch.inference_mode()
  def autoregressive_generate(model, model_inputs: dict, max_new_tokens: int,
                              on_first_token=None, generation_stats: dict | None = None
                              ) -> tuple[torch.Tensor, int, int]
  # returns (prompt+output ids [1, L], new_token_count, steps)
  ```
- Greedy decoding, ported from `baseline_forward` in `gen_baseline_answer_textvqa.py`, using the
  same KV-fork base model and `initialize_past_key_values(model.base_model, max_length=model.kv_capacity)`
  (reused across calls like `specgenerate`, with `current_length_data.zero_()`). The tree buffers
  are dropped. `logits_to_keep=1` is passed when `model.slim_prefill`. It stops on
  `tokenizer.eos_token_id` (`stop_reason="eos"`) or at `max_new_tokens` (`"token_limit"`), and
  fills `generation_stats` with the same keys as `baseline_forward`.

- [ ] **Step 1:** Implement.
- [ ] **Step 2: Verify.** The import check
  `PYTHONNOUSERSITE=1 "$PYTHON_BIN" -c "from vispec.model.autoregressive import autoregressive_generate"`
  runs now. The GPU equivalence check is in Task 7.
- [ ] **Step 3: Commit.** `git commit -m "Add greedy autoregressive generation for SpecModel"`

### Task 7: Unified Qwen evaluator

**Files:**
- Create: `vispec/evaluation/evaluate.py`

**Interfaces:**
- Consumes: Tasks 4, 5 and 6; `GenerationMeasurement`, `write_runtime` and `write_manifest` from
  `measurement.py`.
- CLI: `python -m vispec.evaluation.evaluate`
  `--method {autoregressive,vispec} --benchmark {textvqa,coco_caption} --prompt-style {long,short}`
  `--base-model-path P --spec-model-path P --data-dir P --answer-file P --model-id S`
  `[--num-samples 100] [--max-new-token 1024] [--temperature 0.0] [--depth 3] [--top-k 8]`
  `[--total-token 30] [--num-q 2] [--kv-capacity 16384] [--no-slim-prefill]`.
  Booleans use `action="store_true"`, never `type=bool`.
- Behaviour:
  - It loads `spec_model_ours.SpecModel` once for either method (`torch_dtype="auto"`,
    `device_map="auto"`).
  - It runs 3 untimed warmups on sample 0 with `torch.manual_seed(0)`.
  - For each sample it builds the inputs outside timing. It then runs `specgenerate(...,
    return_acceptance_len=True, on_first_token=..., generation_stats=..., return_decode_time=True)`
    or `autoregressive_generate(...)` inside `GenerationMeasurement`, and passes
    `kv_tokens=int(model.current_length_data[0])` to `finish`.
  - The answer text is the output IDs up to the first EOS, decoded with
    `skip_special_tokens=True` and stripped. The raw IDs go into `measurement` unchanged.
  - Records use the existing schema (`question_id`, `reference_answers`, `answer_id`, `model_id`,
    `choices[{index, turns, idxs, new_tokens, wall_time, ttft, measurement, decode_time,
    acceptance_length}]`). The autoregressive method writes `acceptance_length: []` and
    `decode_time` from its loop.
  - It writes `<answer>.manifest.json` and `<answer>.runtime.json`. The runtime gains
    `prompt_style`, `slim_prefill`, `kv_capacity`, `method` and `benchmark`. Records are sorted by
    `question_id` at the end, as `reorg_answer_file` does.

- [ ] **Step 1:** Implement.
- [ ] **Step 2: Smoke run (GPU).** For each method in `autoregressive vispec`, run
  `bash experiment/gpu_check.sh env PYTHONNOUSERSITE=1 "$PYTHON_BIN" -m vispec.evaluation.evaluate --method $m --benchmark textvqa --prompt-style long --num-samples 5 --base-model-path models/Qwen2.5-VL-3B-Instruct --spec-model-path models/ViSpec-Qwen2.5-VL-3B-Instruct --data-dir bench_data --model-id smoke --answer-file .smoke/$m/textvqa.jsonl`.
  Then repeat the vispec run with `--no-slim-prefill` into `.smoke/vispec_noslim/`.
- [ ] **Step 3: Verify the smoke outputs.**
  ```bash
  PYTHONNOUSERSITE=1 "$PYTHON_BIN" - <<'PY'
  import json
  rd = lambda p: [json.loads(l) for l in open(p)]
  ids = lambda rows: {r["question_id"]: r["choices"][0]["measurement"][0]["output_token_ids"] for r in rows}
  m = lambda rows: [r["choices"][0]["measurement"][0] for r in rows]
  from transformers import AutoTokenizer
  tok = AutoTokenizer.from_pretrained("models/Qwen2.5-VL-3B-Instruct")
  slim, noslim = rd(".smoke/vispec/textvqa.jsonl"), rd(".smoke/vispec_noslim/textvqa.jsonl")
  ar = rd(".smoke/autoregressive/textvqa.jsonl")
  assert ids(slim) == ids(noslim), "slim prefill changed ViSpec output tokens"
  assert len(slim) == len(ar) == 5
  for row in slim + ar:
      t = row["choices"][0]["turns"][0]
      raw = row["choices"][0]["measurement"][0]["output_token_ids"]
      cut = raw[: raw.index(tok.eos_token_id)] if tok.eos_token_id in raw else raw
      assert t == tok.decode(cut, skip_special_tokens=True).strip(), (t, raw[-5:])
      assert "<|im_end|>" not in t and "<|endoftext|>" not in t, t
  ps = max(x["gpu_memory"]["0"]["prefill_peak_allocated_bytes"] for x in m(slim))
  pn = max(x["gpu_memory"]["0"]["prefill_peak_allocated_bytes"] for x in m(noslim))
  assert ps < pn, (ps, pn)
  assert all(x["gpu_memory"]["0"]["kv_live_bytes"] > 0 for x in m(slim))
  print("smoke ok; AR-vs-ViSpec token agreement:", sum(ids(slim)[q] == ids(ar)[q] for q in ids(ar)), "/ 5")
  PY
  ```
  Expected: `smoke ok`. The agreement count is informational; it is diagnosed in Task 11. If the
  Qwen path fails to load or run, debug it here (a teammate could not run Qwen before). Record
  the root cause and fix in the commit message.
- [ ] **Step 4: Commit.** `git commit -m "Add unified Qwen evaluator for both decoding methods"`

### Task 8: COCO reporting and CIDEr

**Files:**
- Modify: `vispec/evaluation/report_metrics.py` (`BENCHMARKS`, prediction export, the CLI flag
  `--coco-references`, which is optional because references are embedded),
  `vispec/evaluation/scoring.py`, `vispec/evaluation/measurement.py` (`write_manifest` ID key for
  `coco_caption`)

**Interfaces:**
- `BENCHMARKS = ("textvqa", "gqa", "mmvet", "coco_caption")`. `compare_metrics` inherits this.
- `Scorer.score_corpus(benchmark: str, predictions: dict[str, str], references: dict[str, list[str]]) -> dict`
  returns `{"score": float | None, "scored": int, "method": "cider_d_pycocoevalcap_1.2", "pending_reason": str | None}`.
  It is used for `coco_caption`. It is `pending` with a reason when `java` is not on `PATH` or
  pycocoevalcap is missing, and it never raises for a missing scorer. Per-sample CIDEr values are
  kept in `samples.jsonl`.
- The report writes `coco_caption_predictions.json` as `[{image_id, caption}]`, and the summary
  shows CIDEr ×100 with its scored count.

- [ ] **Step 1:** Implement.
- [ ] **Step 2: Verify.** Produce a 5-sample COCO answer file with the Task 7 smoke command,
  changed to `--benchmark coco_caption --num-samples 5 --answer-file .smoke/report/results/coco_caption_test/vispec_qwen/test.jsonl`.
  Then, on CPU, run
  `PATH="$(dirname "$PYTHON_BIN"):$PATH" "$PYTHON_BIN" -m vispec.evaluation.report_metrics --run-dir .smoke/report`.
  Expected: the new report's `summary.md` shows COCO CIDEr with `5` scored. Run the same command
  with `PATH=/usr/bin:/bin`. Expected: CIDEr `pending` with a Java reason, and exit 0.
- [ ] **Step 3: Verify the old benchmarks are unchanged.** Re-report the versioned 00 snapshot
  with `--output-dir .smoke/report00`, then compare it with the snapshot's
  `reports/20261017T220622Z_df807dd5/summary.json`. Expected: for textvqa, gqa and mmvet, the
  latency, TTFT and output-token distributions are equal. Quality scores may be `pending` here
  because their annotations are not prepared on this cluster.
- [ ] **Step 4: Commit.** `git commit -m "Report COCO captions with CIDEr-D"`

### Task 9: Experiments 02 and 03, register and scope docs

**Files:**
- Create: `experiment/02_qwen_autoregressive/{run.sh,README.md}`, `experiment/03_qwen_vispec/{run.sh,README.md}`
- Modify: `experiment/README.md` (register rows, pending), `AGENTS.md`, `CLAUDE.md`

**Interfaces:**
- Each `run.sh` follows the 00/01 runner structure (unique run folder, commit/status/diff,
  packages, device report, CUDA preflight, combined log, report in the EXIT trap) with these
  changes:
  - It requires `PROMPT_STYLE` to be `long` or `short`, exiting with status 2 otherwise.
  - Its run-folder suffix includes the style: `runs/<UTC>_<commit12>_<style>_XXXXXX`.
  - It prepends `$(dirname "$PYTHON_BIN")` to `PATH`, and exports `TEXTVQA_EVALUATOR` when
    `tools/downloaded/m4c_evaluator.py` exists, as 00 does.
  - After the device capture and before the CUDA preflight, it runs
    `"$python_bin" experiment/prepare_assets.py --repo-dir "$repo_dir" --check-only --suite qwen`.
    A failure stops the run before any model loads.
  - `metadata.txt` adds `prompt_style`, `slim_prefill=true`, `kv_capacity=16384`, and
    `method=autoregressive` (02) or `method=vispec_speculative` (03) with the tree settings.
  - It runs `evaluate.py` for `textvqa` then `coco_caption` into
    `results/<bench>_test/<method>_qwen/test.jsonl`.
  - Python defaults to the vipsec path.
- READMEs cover every `AGENTS.md` item, with results `pending`.
- `AGENTS.md`: the Qwen experiments (02+) use Qwen2.5-VL-3B on TextVQA and COCO Captions (plus
  Re:Verse windows in Week 4) on the Ampere cluster. The 00/01 rules stay as written for those
  experiments. `CLAUDE.md`: add the Qwen evaluator, `benchmarks.py`, the Ampere submission path and
  the Qwen hazards (special tokens, draft 8,192-position limit). Remove "Only … LLaVA are in scope".

- [ ] **Step 1:** Implement the runners and docs.
- [ ] **Step 2: Verify.** Run `bash -n` on both runners. `PROMPT_STYLE=medium bash experiment/02_qwen_autoregressive/run.sh`
  must exit 2 before creating a run folder (`git status --short experiment/02_qwen_autoregressive` shows
  no `runs/`).
- [ ] **Step 3: Commit.** `git commit -m "Add Qwen baseline experiments 02 and 03"`

### Task 10: Launch the four baseline runs

- [ ] **Step 1:** Run `PROMPT_STYLE=long bash experiment/submit_ampere.sh experiment/02_qwen_autoregressive experiment/03_qwen_vispec`
  and the same with `PROMPT_STYLE=short`: two jobs, each running 02 then 03 on one GPU. Record the
  job IDs.
- [ ] **Step 2: Verify completion.** For each of the four run folders: `exit-code.txt` is `0`,
  `device.txt` shows `CLUSTER_EXPECTED_GPU`, the summary shows 100/100 for TextVQA and for COCO,
  and the two runs in each job share one GPU UUID.
- [ ] **Step 3:** Compare within each style with `"$PYTHON_BIN" -m vispec.evaluation.compare_metrics --original <02 report> --draft <03 report> --output-dir <03 run>/comparisons/<UTC>`.
  Then confirm that pairing a long 02 report with a short 03 report is refused (expect a
  fingerprint or prompt-mismatch error).
- [ ] **Step 4:** No commit (runs stay out of Git). Note the job IDs and run paths for Task 12.

### Task 11: Divergence diagnosis

**Files:**
- Create: `vispec/evaluation/diagnose_divergence.py`

**Interfaces:**
- CLI: `--comparison <comparison.json> --original-run <02 run> --draft-run <03 run> --data-dir bench_data --base-model-path P --output-dir <new dir> [--fp32-limit 50]`.
- For every mismatching pair in `comparison.json`'s `token_agreement`, it:
  - rebuilds that sample's inputs with `benchmarks.build_inputs` (the style is read from
    metadata);
  - appends the common output prefix (the AR IDs before the first differing index);
  - runs one forward pass of the stock `transformers.Qwen2_5_VLForConditionalGeneration` in bf16;
  - records the top-2 tokens and their logit margin at the last position, and whether bf16 argmax
    equals the AR or the ViSpec token;
  - for up to `--fp32-limit` cases (sorted by question ID), repeats the pass in fp32.
- Classification: `near_tie` if the bf16 margin is ≤ 0.5. Otherwise `unexplained`. `fp32_agrees`
  is `ar`, `vispec` or `neither`.
- Writes `diagnosis.json` and `diagnosis.md`: counts per class and per benchmark, the margin
  distribution (min, median, max), and one row per case.

- [ ] **Step 1:** Implement.
- [ ] **Step 2: Run** on both comparisons via `gpu_check.sh` (`SRUN_EXTRA="--mem=48G --time=02:00:00"`).
  Expected: `diagnosis.md` per style.
- [ ] **Step 3: Gate.** The gate passes when `unexplained` is 0. Otherwise, take the first
  unexplained case, find the cause in the tree code or the KV fork (use the
  superpowers:systematic-debugging skill), fix it, rerun Task 10 for the affected style, and
  diagnose again. Stop and report to the user if the cause is not found after one investigation
  pass.
- [ ] **Step 4: Commit.** `git commit -m "Add greedy divergence diagnosis for baseline pairs"`

### Task 12: Record Week 1 results

**Files:**
- Modify: `experiment/02_qwen_autoregressive/README.md`, `experiment/03_qwen_vispec/README.md`, `experiment/README.md`

- [ ] **Step 1:** Fill the per-benchmark tables for each style:
  - sample counts, mean, median and p95 latency, TTFT, TPOT, throughput;
  - prefill and decode peaks, live KV;
  - acceptance (03 only), truncation;
  - CIDEr and VQA accuracy, token and text agreement, and the diagnosis class counts.

  Add a hardware summary that links each `device.txt`. Label long-style quality and short-style
  speed as secondary, per the spec. Use measured values only; anything missing stays `pending`.
- [ ] **Step 2: Verify.** Every number in the READMEs appears in a `summary.json`,
  `comparison.json` or `diagnosis.json` under the cited run paths (spot-check five values by
  `grep`).
- [ ] **Step 3: Commit.** `git commit -m "Record Week 1 Qwen baseline results"`

---

## Roadmap: later plans

Each later plan is written when the previous one has landed, because its design depends on what
the measurements show (for example, whether divergences or the slim harness change anything).

| Plan | Spec goal | Deliverable |
| --- | --- | --- |
| Week 2: sparsity methods | Goal 2 | Draft-side pooling/pruning before `ImgAdaptor`, and target-side visual KV eviction after prefill with cache compaction. Each has a spatial and an attention-guided selector behind `--draft-retention` / `--kv-retention`. At 100% retention, outputs are token-identical to 03. `compare_metrics` gains a mode with ViSpec at 100% as the reference. |
| Week 3: retention sweeps | Goal 3 | 25/50/75/100% budgets for each method and for both combined, in both prompt styles, on one pinned GPU. Each arm is compared against ViSpec at 100% retention, with autoregressive decoding as a second column. |
| Week 4: Re:Verse windows | Goal 4 | Re:Verse prepared from `sochastic/Re-Verse`. A check that each chapter's segment count matches its page count. Non-overlapping windows of about 6 pages under the draft's 8,192-position limit. ROUGE-1 and BERTScore F1 (bert-base-uncased, prepared offline). Manga109 only if licensed data is provided. |
| Week 5: analysis | Goal 5 | Trade-off curves of speedup, memory and quality, per-benchmark tables, the register and the comparison page. |
