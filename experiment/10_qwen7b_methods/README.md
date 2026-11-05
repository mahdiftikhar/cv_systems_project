# 10 — Qwen2.5-VL-7B method comparison (autoregressive, ViSpec, HiViS)

## Purpose and hypothesis

Experiments 02 to 09 use the 3B Qwen model. This experiment moves to `Qwen/Qwen2.5-VL-7B-Instruct` and
adds HiViS, a second speculative-decoding method, next to ViSpec. It measures autoregressive (AR)
decoding, ViSpec and HiViS on the same 100 TextVQA questions and 100 COCO Captions images as Weeks 1 to 3,
in both prompt styles. Two secondary comparisons run in the `long` style only: a larger draft tree, and the
image placed before the text in the prompt.

The write-up leads with each method's speedup against AR, then acceptance, time per output token and
time to first token. The draft's visual input then reads as a spectrum: an EAGLE-style draft sees every
image token, ViSpec compresses each image to 2 tokens, and HiViS sees none.

Predictions, written before any run:

1. **Both speculative methods beat AR in the `long` style.** Cutoff: the mean latency of ViSpec and of
   HiViS is below the same-job AR mean on both benchmarks.
2. **HiViS's acceptance relative to ViSpec depends on prompt order.** With the image last, the prompt ends
   with a 4-token tail after the image, which is all of the HiViS drafter's context about the prompt
   end. Image-first gives it a longer text tail. Cutoff: the difference between HiViS and ViSpec in accepted
   draft tokens per iteration changes by at least 0.1 between `image-last` and `image-first`.
3. **The 60/5/10 tree raises acceptance for both methods.** Its effect on latency is open, because the
   tree verification is larger. Cutoff for the acceptance part: accepted draft tokens per iteration at
   60/5/10 exceed the 30/3/8 value for both methods on both benchmarks.

These are predictions, not results. The Results section is `pending`.

Status: **pending**. No 10 run has been made.

## Changes from the baseline

- New runner `run.sh`, which uses the `methods` sweep type of `experiment/qwen_arm_runner.sh`. Each arm is
  one run folder with its own launch evidence and EXIT report, as in 05 and 06. The command lines of
  experiments 05, 06, 08 and 09 are unchanged (checked by diffing a dry run before and after).
- New evaluator method `hivis` and flag `--prompt-order {image-last,image-first}` (`image-last` is the
  default and unchanged).
- New asset suite `qwen7b`, selected by `QWEN_ASSET_SUITE`, which checks the three pinned checkpoints.
  The default suite stays `qwen`.
- Every arm runs `--kv-layout compact` and passes no sparsity flag. Slim prefill is on. ViSpec's draft
  prunes nothing, and HiViS refuses the draft-pruning flags.
- Each job holds its own AR reference, so speedups never span two jobs.

## Models

| Role | Model | Local copy |
| --- | --- | --- |
| Target | `Qwen/Qwen2.5-VL-7B-Instruct` @ `cc594898137f460bfe9f0759e9844b3ce807cfb5` | `models/Qwen2.5-VL-7B-Instruct` |
| ViSpec draft | `JLKang/ViSpec-Qwen2.5-VL-7B-Instruct` @ `e06635b54327df78a5eb82361791f2ccec9f41ab` | `models/ViSpec-Qwen2.5-VL-7B-Instruct` |
| HiViS draft | `Irisssme/HiViS-Qwen2.5-VL-7B-Instruct` @ `6dd8717d912a5626d6ad4e81b124c54bb8a08ef2` | `models/HiViS-Qwen2.5-VL-7B-Instruct` |

The HiViS drafter code is vendored in `vispec/model/cnets_hivis.py` with an attribution header (source
repository, Apache-2.0, arXiv 2509.23928).

## Data, splits and subsets

The same samples as experiments 02 and 03: 100 TextVQA questions (seed-42 shuffle) and 100 COCO Karpathy
test images, loaded by `vispec/evaluation/benchmarks.py`. Scores are subset measurements, and the CIDEr-D
IDF is computed over the 100-image subset.

## Settings

Greedy decoding (temperature 0.0, fixed), bf16, batch size 1, at most 1,024 new tokens, slim prefill on,
`--kv-layout compact`, KV capacity 16,384 as an upper bound, no sparsity flag.

An arm is `<method>[:<tree>][:<order>]`:

| Field | Values | Default | Meaning |
| --- | --- | --- | --- |
| method | `ar`, `vispec`, `hivis` | required | Decoding method |
| tree | `30-3-8`, `60-5-10` | `30-3-8` | Total tokens, depth and top-k of the draft tree (`--total-token`, `--depth`, `--top-k`). `ar` takes none |
| order | `image-last`, `image-first` | `image-last` | Position of the image in the prompt (`--prompt-order`) |

ViSpec arms also pass `--num-q 2`. HiViS arms do not.

| Variable | Default | Purpose |
| --- | --- | --- |
| `PROMPT_STYLE` | required | `long` or `short` |
| `METHOD_ARMS` | `ar vispec hivis` | Space-separated arms |
| `QWEN_BENCHMARKS` | `textvqa coco_caption` | Space-separated benchmark names |
| `QWEN_ASSET_SUITE` | `qwen7b` | Suite for the asset check and the copied `.preparation/assets_<suite>.json` |
| `BASE_MODEL_PATH` | `models/Qwen2.5-VL-7B-Instruct` | Target |
| `SPEC_DIR` | `models/ViSpec-Qwen2.5-VL-7B-Instruct` | ViSpec draft (also passed to AR arms) |
| `HIVIS_DIR` | `models/HiViS-Qwen2.5-VL-7B-Instruct` | HiViS draft |
| `BENCH_DIR`, `PYTHON_BIN` | as in 03 | Data location and interpreter |

`run.sh` validates every arm before it creates the first run folder. An unknown value, a repeated field,
a tree on `ar`, a duplicate arm (after the defaults are filled in) or an empty list exits with status 2
and creates no folder.

The run folder is `runs/<UTC>_<commit12>_<style>_<method>[_t<tree>]_<order>_XXXXXX/`. The tree appears only
when the arm wrote it out. `metadata.txt` keeps the keys of the other sweeps and adds `prompt_order`,
`draft_method` (`none`, `vispec` or `hivis`) and `asset_suite`. Speculative arms record `depth`, `top_k`
and `total_token`, and ViSpec arms also `num_q`. The metadata `method` is `autoregressive`,
`vispec_speculative` or `hivis_speculative`.

A failed arm is recorded and the remaining arms still run. A failed benchmark stops that arm's remaining
benchmarks. The runner exits 1 if any arm failed.

## Cluster prerequisites

1. Follow the [Ampere setup guide](../AMPERE_SETUP.md). Jobs run on the pinned 48 GB `NVIDIA RTX A6000`
   nodes through `submit_ampere.sh`, which fails with status 3 on any other GPU.
2. Prepare the 7B assets once with `bash prepare.sh --suite qwen7b` inside a Slurm allocation. The runner
   repeats the offline check and stops before any model loads if an asset is missing.
3. Commit code before launching, so the run records a clean commit.
4. Loading a 7B draft holds about 10 to 11 GB of host memory for a moment, so ask for `--mem=32G`.
   Pass `--time` in `SBATCH_EXTRA`. On `preempt`, jobs are killed after 1 to 2 hours, and a requeued job
   reruns the runner into fresh run folders, so give `sweep_table` and `compare_metrics` only the final
   attempt's folders (highest `slurm_restart_count`, `exit_code=0`).

## Exact commands

Primary comparison, one job per style:

```bash
PROMPT_STYLE=long CLUSTER_PARTITION=preempt CLUSTER_EXCLUDE=c4-4,c4-5,c5-2,c2-2 \
  SBATCH_EXTRA="--requeue --cpus-per-task=4 --mem=32G --time=<per job>" \
  bash experiment/submit_ampere.sh experiment/10_qwen7b_methods
```

Secondary arms, `long` style only, each job with its own references:

```bash
# Tree size
PROMPT_STYLE=long METHOD_ARMS="ar vispec hivis vispec:60-5-10 hivis:60-5-10" ...
# Prompt order
PROMPT_STYLE=long METHOD_ARMS="ar vispec hivis ar:image-first vispec:image-first hivis:image-first" ...
```

(Each uses the same `submit_ampere.sh` command as above.)

Verification without a GPU: `bash -n experiment/10_qwen7b_methods/run.sh`, and
`PROMPT_STYLE=long SBATCH_EXTRA=--test-only bash experiment/submit_ampere.sh experiment/10_qwen7b_methods`,
which schedules nothing.

## Metric definitions

As in the [03 README](../03_qwen_vispec/README.md#metric-definitions) and the [metrics guide](../METRICS.md):
latency and time to first token in seconds per sample, time per output token in milliseconds, accepted draft
tokens per verification iteration (accepted draft tokens, without the target's own token), exact token-ID
agreement, and memory from `kv_live_bytes` and the prefill and decode peaks. Speedup is the same-job AR
mean latency divided by the method's mean latency.

## Results

**Pending.** No run exists yet. When the runs finish, this section reports hardware and run folders, per-method
speedup, acceptance, time per output token and time to first token, and a table scoring each prediction above
against the measurements with its cutoff.

## Limitations

- One run per arm, 100 samples per benchmark. Differences in quality and agreement are noisy.
- bf16 near-ties can make speculative and AR outputs differ at the token level. Check the saved token IDs
  before comparing text between methods.
- The HiViS paper's numbers use a different harness (fp16, 500 new tokens, an image-first prompt, the first
  80 TextVQA test rows). They are context only and are never substituted for these results.
- Re:Verse+ on 7B and KV eviction on 7B are not run.
