# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

This repository compares standard autoregressive decoding (experiment 00) with ViSpec speculative
decoding (experiment 01) on LLaVA-1.6 Vicuna 7B over TextVQA, GQA and MM-Vet. It builds on the
upstream ViSpec code in `vispec/` and adds an evidence-capturing experiment harness in `experiment/`.
`AGENTS.md` holds the binding workflow rules: experiment layout, launch evidence, device capture, no
test files, cluster-only compute, and which run snapshots are versioned. Follow it alongside this file.

## Environment

There is no build, lint config or test suite, and AGENTS.md forbids adding test files. Verify changes
with import checks, `bash prepare.sh --check-only`, or by re-reporting saved runs. Everything runs
from the repository root, because evaluators are invoked as `python -m vispec...` modules and the
draft config falls back to the relative path `./vispec/train/llava_1.6_7B_config.json`.

Scripts default to `<repo>/.venv/bin/python` (a uv environment on Newton). Set an absolute
`PYTHON_BIN` to use another interpreter. `experiment/prepare_assets.py` needs Python 3.11+.
`requirements.txt` pins `torch==2.7.0+cu128` and `transformers==4.51.3`. Keep the Transformers pin,
because `vispec/model/modeling_*_kv.py` are forks of that version's model code. The cu128 wheel only
contains sm_75+ kernels, so Pascal and Volta GPUs cannot run it.

```bash
uv venv --python 3.11 .venv
UV_CACHE_DIR="$PWD/.cache/uv" uv pip install --python .venv/bin/python \
  --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match \
  -r requirements.txt
```

These packages are imported only by code outside the current scope and are not in requirements:
`ray` (multi-GPU splitting), `flash_attn` (Qwen), `fastchat` and `wandb` (training/data generation),
`torch_npu` (Ascend), and `matplotlib` (only for `experiment/plot_comparison.py`).

## Commands

Model downloads and evaluation belong on a cluster compute node. `GETTING_STARTED.md` and
`experiment/NEWTON_SETUP.md` give the full Newton workflow.

```bash
bash prepare.sh                 # inside a Slurm allocation: models, data, scorer -> models/, bench_data/, tools/downloaded/
bash prepare.sh --check-only    # offline validation of prepared assets

# Submit from a login node (needs NEWTON_ACCOUNT, NEWTON_PARTITION, NEWTON_GPU_GRES)
bash experiment/submit.sh experiment/00_baseline_original
bash experiment/submit.sh experiment/01_baseline_draft

# Or run directly inside a one-GPU allocation. 01 also reads DEPTH, TOP_K, TOTAL_TOKEN, NUM_Q.
bash experiment/00_baseline_original/run.sh
```

To run a single benchmark by hand, call one evaluator module. This is what the runners do per
benchmark (01 uses `gen_spec_answer_*` plus `--use-ours True --num-q 2 --depth 3 --top-k 8 --total-token 30`):

```bash
python -m vispec.evaluation.gen_baseline_answer_textvqa \
  --base-model-path models/llava-v1.6-vicuna-7b-hf \
  --spec-model-path models/ViSpec-llava-v1.6-vicuna-7b-hf \
  --model-id test --bench-name /tmp/out/textvqa_test/baseline_llava \
  --max-new-token 1024 --temperature 0.0 --data-folder bench_data/textvqa
```

The following commands read saved outputs only. They load no models and can run without a GPU
(details in `experiment/METRICS.md`):

```bash
python -m vispec.evaluation.report_metrics --run-dir experiment/<exp>/runs/<run> [--bench-dir bench_data]
python -m vispec.evaluation.compare_metrics --original <report>... --draft <report>... --output-dir <new_dir>
python experiment/plot_comparison.py --original <report> --draft <report> --output-dir <dir>
```

## Architecture

**Run pipeline.** `experiment/NN_*/run.sbatch` sets offline Hugging Face mode and calls `run.sh`.
`run.sh` creates a unique `runs/<UTC>_<commit12>_<rand>/` directory. It snapshots the commit,
`git status`, the diff, the runner, the package list (`capture_packages.py`) and the device report
(`capture_device.sh`), then gates on `check_cuda.sh`. It then runs the evaluators in the order
TextVQA → GQA → MM-Vet, and an EXIT trap always runs `report_metrics`, even after failure or
interruption. Runners hard-code temperature 0.0 and a 1,024-token cap, and they ignore `TEMPERATURE`.
They also point `HF_HOME` at `<repo>/.cache/huggingface`, and they prefer local checkpoints in
`models/` over Hub IDs.

**Evaluators** (`vispec/evaluation/gen_{baseline,spec}_answer_<bench>.py`) are standalone,
near-duplicate scripts, one per benchmark and method. Prompts come from `<bench>_prompt.py`, and
timing comes from `measurement.py`/`timing.py`. Each script repeats its own data loading, warmup
(3 untimed runs), generation loop and output schema. A change to generation, measurement or output
fields therefore has to be made in all six in-scope files (textvqa, gqa, mmvet × baseline, spec).
TextVQA and GQA use `Dataset.shuffle(seed=42).select(range(100))`, and MM-Vet uses all 218 questions.
Outputs save raw token IDs, an input-ID fingerprint and per-turn measurements; downstream agreement
checks depend on these.

**Models** (`vispec/model/`). Each `spec_model*.py` defines a `SpecModel` wrapper.
`SpecModel.from_pretrained` dispatches on `config.architectures[0]` to a KV-cache-enabled fork of the
base model (`modeling_*_kv.py`, with a preallocated cache from `kv_cache.py`), and loads the draft
weights into a `cnets*.py` network.

- `spec_model.py` / `cnets.py` is the EAGLE-style draft.
- `spec_model_ours.py` / `cnets_ours.py` is ViSpec. Its `ImgAdaptor` compresses image hidden states
  into `num_q` learned queries for the draft.
- `spec_model_medusa.py` is Medusa.

`specgenerate` runs the tree loop from `utils.py`: `initialize_tree` → `tree_decoding` →
`evaluate_posterior` → `update_inference_inputs`. The draft tree's shape comes from `depth`, `top_k`
and `total_token`. The 00 baseline still loads `spec_model.SpecModel` with draft weights, so
`--spec-model-path` is required. It decodes greedily through each script's own `baseline_forward`
loop on the base model.

**Scoring and reports.** `report_metrics.py` + `scoring.py` write a new `reports/<timestamp>/`
folder (`summary.{md,json}`, `samples.jsonl`, prediction exports) and never overwrite raw answers.
Each benchmark has its own scorer:

- TextVQA uses the downloaded LLaVA `m4c_evaluator.py` (`TEXTVQA_EVALUATOR`).
- GQA uses exact match against the test-dev annotations (`GQA_ANNOTATIONS`).
- MM-Vet requires importing external judge grades. It stays pending otherwise.

`compare_metrics.py` refuses to compare reports unless model, temperature, precision, GPU identity,
allocation and prompt fingerprints all match.

## Gotchas

- Evaluator flags such as `--use-ours`, `--use-medusa` and `--image-fc` use argparse `type=bool`.
  Any non-empty value, including `False`, enables them, so omit a flag to disable it.
- The top-level `baseline.sh`, `exp.sh`, `exp_eagle.sh` and `exp_medusa.sh` are the upstream
  eight-benchmark scripts. The project runners are in `experiment/`. Don't restore the eight-benchmark
  suite.
- `.gitignore` ignores `experiment/*/runs/*`, except the two snapshots for jobs 950793 and 950803.
  Treat those as immutable evidence and write reruns and re-reports to new directories.
- Only TextVQA, GQA and MM-Vet with LLaVA are in scope. The Qwen, Mixtral, Medusa, training
  (`vispec/train`) and data-generation (`vispec/ge_data`) paths are inherited and unexercised.
