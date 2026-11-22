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

These predictions were written before any run. The [Results](#results) section scores them.

Status: **complete** (jobs 412893 to 412899, 2026-11-21). All 31 arms ran 100 of 100 samples. Prediction 1
held, prediction 2 did not hold, and prediction 3 held for acceptance. The divergence diagnosis and fp32 checks
(job 412936, 2026-11-22) passed. The parity check against HiViS's own code (job 412935) passed four of its five
gates. The token-count gate did not pass as written, and every difference it found is a near-tie. A rerun of the
TextVQA prompt-order arms in reverse order (job 412937) showed that the slower `image-first` steps of job 412898
came from when those arms ran, not from the prompt order.

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

### Hardware and run folders

Every job ran on one pinned 48 GB `NVIDIA RTX A6000` on the `preempt` partition, at commit `1a0f282` with a
clean tree. Each job used one GPU (one UUID in every device report of the job).

| Job | Content | Node | Attempt used | Run folders (suffix, in arm order) |
| --- | --- | --- | --- | --- |
| 412893 | `long` TextVQA, primary arms | c4-2 | restart 2 | `pgfapa` AR, `j0x9G4` ViSpec, `GXHZvS` HiViS |
| 412894 | `long` COCO, primary arms | c4-2 | restart 2 | `wlu7kb` AR, `fwQuzN` ViSpec, `k88Sw5` HiViS |
| 412895 | `short`, both benchmarks, primary arms | c4-3 | first | `SZakwq` AR, `G1SxZa` ViSpec, `vIOO0P` HiViS |
| 412896 | `long` TextVQA, tree size | c1-8 | first | `0GBfHd` AR, `K8sx7a` ViSpec, `1jGVHp` HiViS, `i86ItQ` ViSpec 60-5-10, `lJyReB` HiViS 60-5-10 |
| 412897 | `long` COCO, tree size | c4-2 | first | `kvtXKB` AR, `zkjDwi` ViSpec, `fr8ZPu` HiViS, `lhjUo1` ViSpec 60-5-10, `7OaF8J` HiViS 60-5-10 |
| 412898 | `long` TextVQA, prompt order | c4-3 | first | `joa66M` AR, `5m5iUe` ViSpec, `5llhBP` HiViS, `WeVGWv` AR image-first, `zaxpj0` ViSpec image-first, `Yw3DM3` HiViS image-first |
| 412899 | `long` COCO, prompt order | c4-3 | first | `9vX30T` AR, `1UpLyu` ViSpec, `wXVaxz` HiViS, `YfoT55` AR image-first, `iU8WO5` ViSpec image-first, `AIczCS` HiViS image-first |
| 412937 | `long` TextVQA, prompt order rerun with the image-first arms first | c4-2 | first | `zxEcz6` AR image-first, `ipRWfs` ViSpec image-first, `HTwlPl` HiViS image-first, `Y1B21R` AR, `rnC0LL` ViSpec, `tgoWov` HiViS |

Jobs 412893 and 412894 were requeued twice. Their restart 1 attempts also finished every arm, but the tables use
the final attempt (highest `slurm_restart_count` with `exit_code=0`), as the prerequisites above require. The
earlier folders stay in `runs/` and are not used.

Job 412937 ran on 2026-11-22, after the other seven. Its first arm records commit `def2e96` and the other five
record `0f23bf8`, because a documentation commit landed while the job ran. The two commits differ only in two
README files, so every arm ran the same code.

**Completion checks.** All 31 folders exit 0, record 100 of 100 samples per benchmark and have one report.
The GPU checks of job 402297 (final attempt, restart 4, folder `dkKVdV`) passed before these runs: 3B identity
(5 of 5 AR and 5 of 5 ViSpec outputs equal to the 3B references), fp32 exactness of HiViS against AR (5 of 5
long TextVQA questions at `image-last` 30-3-8, and 5 of 5 at `image-first` 60-5-10), a bf16 smoke test, the
large tree and the position ablation.

**Reproducibility between jobs.** Each primary `long` arm (AR, ViSpec and HiViS at 30-3-8, `image-last`) ran in
three jobs per benchmark, on three different GPUs. Its output token IDs and input fingerprints are identical in
all three jobs, for all six arm and benchmark pairs. The timing is not identical. Long TextVQA AR took 12.96 s to
16.62 s in the three jobs (28% longer in the slowest), and long COCO AR took 4.88 s to 5.18 s (6% longer).

### Primary comparison

Speedup over AR is the same-job AR mean latency divided by the method's mean latency (above 1 means faster).
Speedup over ViSpec divides ViSpec's latency by HiViS's. The TTFT share is the method's TTFT divided by its
latency, pooled over samples. Quality is TextVQA VQA accuracy (%) or COCO CIDEr-D x 100. `long` quality is
near zero for every method, because strict scoring rejects paragraph answers, as in [03](../03_qwen_vispec/README.md).

| Style | Benchmark | Job | Method | Latency (s) | Speedup over AR | Speedup over ViSpec | Accepted draft tokens per iteration | TPOT (ms) | TTFT (s) | TTFT share | Exact agreement with AR | Quality |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| long | TextVQA | 412893 | AR | 15.832 | 1.000 | n/a | n/a | 38.2 | 1.049 | 6.6% | 100/100 | 0.0 |
| long | TextVQA | 412893 | ViSpec | 8.028 | 1.972 | n/a | 2.181 | 17.8 | 1.049 | 13.1% | 2/100 | 0.0 |
| long | TextVQA | 412893 | HiViS | 7.960 | 1.989 | 1.009 | 2.282 | 17.2 | 1.050 | 13.2% | 3/100 | 0.0 |
| long | COCO | 412894 | AR | 4.878 | 1.000 | n/a | n/a | 29.5 | 0.251 | 5.1% | 100/100 | 0.0 |
| long | COCO | 412894 | ViSpec | 2.464 | 1.980 | n/a | 2.287 | 14.3 | 0.251 | 10.2% | 18/100 | 0.0 |
| long | COCO | 412894 | HiViS | 2.215 | 2.203 | 1.113 | 2.724 | 12.5 | 0.251 | 11.3% | 9/100 | 0.0 |
| short | TextVQA | 412895 | AR | 1.108 | 1.000 | n/a | n/a | 31.3 | 1.002 | 90.4% | 100/100 | 75.6 |
| short | TextVQA | 412895 | ViSpec | 1.139 | 0.973 | n/a | 0.846 | 53.1 | 1.008 | 88.5% | 100/100 | 75.6 |
| short | TextVQA | 412895 | HiViS | 1.148 | 0.966 | 0.993 | 0.549 | 62.2 | 1.001 | 87.2% | 100/100 | 75.6 |
| short | COCO | 412895 | AR | 0.824 | 1.000 | n/a | n/a | 31.0 | 0.254 | 30.8% | 100/100 | 54.63 |
| short | COCO | 412895 | ViSpec | 0.588 | 1.401 | n/a | 1.794 | 18.6 | 0.250 | 42.6% | 87/100 | 54.50 |
| short | COCO | 412895 | HiViS | 0.529 | 1.559 | 1.113 | 2.345 | 15.2 | 0.249 | 47.1% | 84/100 | 54.43 |

- **Speedup.** In the `long` style, both methods were about twice as fast as AR. HiViS was 1.009 times as fast
  as ViSpec on TextVQA and 1.113 times as fast on COCO.
- **Spread between jobs.** In the three jobs with identical outputs, the `long` speedup over AR ranged from 1.850
  to 1.997 (ViSpec) and from 1.931 to 2.013 (HiViS) on TextVQA, and from 1.980 to 2.075 (ViSpec) and from 2.203
  to 2.339 (HiViS) on COCO. HiViS over ViSpec ranged from 1.008 to 1.044 on TextVQA and from 1.113 to 1.127 on
  COCO. The TextVQA advantage of HiViS is small and varies between jobs, while the COCO advantage is consistent.
- **Acceptance.** HiViS accepted more draft tokens per iteration than ViSpec in the `long` style (TextVQA +0.10,
  COCO +0.44) and on `short` COCO (+0.55). Its time per verification step was about the same as ViSpec's
  (46.1 ms against 46.3 ms on long COCO in job 412894, pooled over samples), so the higher acceptance became a
  lower TPOT.
- **Short TextVQA.** Answers average 4.4 tokens, TTFT is 87% to 90% of the latency, and neither method was
  faster than AR. HiViS accepted fewer draft tokens per iteration than ViSpec here (0.55 against 0.85).
- **Agreement.** In bf16, the `long` outputs of both methods rarely match AR exactly (2 to 18 of 100). The median
  matching prefix before the first differing token is 56.5 (ViSpec) and 60.0 (HiViS) tokens on long TextVQA,
  and 72.5 and 55.0 tokens on long COCO. The `short` outputs match on 84 to 100 of 100, and the `short` quality
  of the three methods is within 0.20 CIDEr-D and 0.0 VQA points. Every differing pair is a bf16 near-tie (see
  [Exactness checks](#exactness-checks-job-412936)).
- **Memory.** The decode peak is 17.20 to 17.28 GiB for HiViS, 17.29 to 17.36 GiB for ViSpec and 17.23 to 17.33
  GiB for AR, in every AR arm and every 30-3-8 arm.

Compared with the 3B results (`experiment/RESULTS.md` on branch `qwen-results`, a different model and other jobs, so context only), ViSpec's
`long` acceptance is almost unchanged on 7B (2.181 against 2.141 on TextVQA, 2.287 against 2.284 on COCO), and its
speedup is somewhat lower (1.85 to 2.00 against 2.15 to 2.20 on TextVQA, 1.98 to 2.07 against 2.13 to 2.32 on COCO).
Exact agreement on long TextVQA is lower (2 of 100 against 22 of 100). The 7B `short` COCO CIDEr-D (54.63 for AR)
is less than half of the 3B value (121.40) on the same 100 images. The 7B AR captions are longer (19.4 output
tokens on average, against 14.8 for 3B AR in job 398923) and more descriptive. They average 15.2 words against
12.1, 13 of 100 are wrapped in quotation marks (none for 3B), and 54% of their words occur in at least one
reference caption, against 76% for 3B. CIDEr-D rewards words and phrases shared with the references, so this
style fits the lower score. The three 7B methods are within 0.20 CIDEr-D of each other, so the method comparison
is unaffected.

### Tree size (jobs 412896 and 412897)

Tokens per iteration and step time are pooled over the 100 samples (total output tokens or total decode time
divided by total verification iterations).

| Benchmark | Method | Accepted draft tokens per iteration, 30-3-8 to 60-5-10 | Output tokens per iteration | Decode time per iteration (ms) | Speedup over own 30-3-8 | Speedup over AR, 30-3-8 to 60-5-10 |
| --- | --- | --- | --- | --- | ---: | --- |
| TextVQA | ViSpec | 2.181 to 2.496 (+0.31) | 3.181 to 3.496 (+9.9%) | 58.80 to 66.41 (+12.9%) | 0.968 | 1.997 to 1.933 |
| TextVQA | HiViS | 2.282 to 2.748 (+0.47) | 3.282 to 3.748 (+14.2%) | 58.65 to 66.75 (+13.8%) | 1.010 | 2.013 to 2.034 |
| COCO | ViSpec | 2.287 to 2.671 (+0.38) | 3.287 to 3.671 (+11.7%) | 46.94 to 52.33 (+11.5%) | 0.995 | 2.075 to 2.065 |
| COCO | HiViS | 2.724 to 3.335 (+0.61) | 3.724 to 4.335 (+16.4%) | 46.08 to 52.40 (+13.7%) | 1.023 | 2.339 to 2.392 |

The larger tree raised acceptance for both methods on both benchmarks, and HiViS gained more. Each verification
step took 11.5% to 13.8% longer, which cancelled most of the gain. ViSpec was 3.2% slower on TextVQA and 0.5% slower
on COCO than at 30-3-8, and HiViS was 1.0% and 2.3% faster. With one run per arm, differences of a few percent
are close to timing noise (the HiViS over ViSpec ratio on TextVQA varied from 1.008 to 1.044 between jobs with
identical outputs). The 60-5-10 tree raised the decode peak by about 0.10 GiB for both methods.

### Prompt order (jobs 412898, 412899 and 412937)

Jobs 412898 and 412899 ran the `image-last` arms first. Job 412937 reran the TextVQA arms with the `image-first`
arms first, to separate the prompt order from the time at which an arm ran.

| Benchmark | Job | Order | AR latency (s) | AR output tokens | AR outputs at the 1,024-token limit | ViSpec speedup over AR | HiViS speedup over AR | Accepted draft tokens per iteration, ViSpec / HiViS | HiViS minus ViSpec |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| TextVQA | 412898 | image-last | 12.960 | 387.8 | 0 | 1.850 | 1.931 | 2.181 / 2.282 | 0.101 |
| TextVQA | 412898 | image-first | 7.591 | 225.2 | 4 | 1.565 | 1.550 | 2.174 / 2.270 | 0.096 |
| TextVQA | 412937 | image-first | 8.073 | 225.2 | 4 | 1.885 | 1.913 | 2.174 / 2.270 | 0.096 |
| TextVQA | 412937 | image-last | 12.935 | 387.8 | 0 | 1.849 | 1.860 | 2.181 / 2.282 | 0.101 |
| COCO | 412899 | image-last | 4.999 | 157.9 | 0 | 2.016 | 2.260 | 2.287 / 2.724 | 0.437 |
| COCO | 412899 | image-first | 5.353 | 172.1 | 0 | 2.019 | 2.209 | 2.272 / 2.721 | 0.449 |

- **Acceptance.** Moving the image first changed the acceptance of each method by at most 0.015. The difference
  between HiViS and ViSpec changed by -0.004 on TextVQA and +0.013 on COCO.
- **Outputs.** The order changed AR's own answers. On TextVQA the mean output fell from 388 to 225 tokens, and
  exact agreement with AR rose to 28 of 100 (ViSpec) and 30 of 100 (HiViS), from 2 and 3. On COCO it was 10 and
  19 of 100, from 18 and 9.
- **TextVQA speed.** Prompt order changes the TextVQA speedup by less than 3%. In job 412898, both `image-first`
  speedups were about 1.56, because the `image-first` ViSpec and HiViS steps took 16% and 21% longer than the
  `image-last` ones, while AR steps did not. In the rerun (412937), all six outputs are token-identical to
  412898, and the steps take the same time in both orders: 48.6 ms against 48.6 ms for ViSpec, 47.0 ms against
  48.6 ms for HiViS, and 31.2 ms against 31.0 ms for AR (median per sample, `image-first` against `image-last`).
  Its `image-first` speedups are 1.885 (ViSpec) and 1.913 (HiViS), against 1.849 and 1.860 with the image last.
- **The slow arms of job 412898.** The slow steps therefore came from when the last two arms of 412898 ran (22:17
  to 22:36 UTC on c4-3), not from the prompt order. The slowdown is uniform across samples and does not depend on
  answer length. The device reports show no GPU clock or temperature change at the start of those arms. Decoding
  is host-bound, so a change in host load on the node is a candidate cause, but host load is not recorded. The
  COCO steps of 412899 did not change between orders (within 1.4%).
- **TTFT share.** The shorter `image-first` answers raise the TTFT share from 14.5% to 23.7% (ViSpec, 412937), yet
  the speedup did not fall.

### Exactness checks (job 412936)

Job 412936 (node c4-3, folder `JaXa78`, commit `def2e96`) ran the plan's diagnosis step after the comparison
runs. It diagnosed the first differing token of every AR and speculative output pair in the three primary jobs,
using the comparisons in the analysis files below, and then ran two fp32 checks.

| Comparison | Job | Differing outputs | Near-ties | Unexplained | Both wrong | fp32 re-runs | fp32 prefers AR / speculative / another token |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| long TextVQA, ViSpec | 412893 | 98 | 98 | 0 | 0 | 50 | 27 / 20 / 3 |
| long TextVQA, HiViS | 412893 | 97 | 97 | 0 | 0 | 50 | 24 / 25 / 1 |
| long COCO, ViSpec | 412894 | 82 | 82 | 0 | 0 | 50 | 27 / 21 / 2 |
| long COCO, HiViS | 412894 | 91 | 91 | 0 | 0 | 50 | 24 / 26 / 0 |
| short COCO, ViSpec | 412895 | 13 | 13 | 0 | 0 | 13 | 4 / 9 / 0 |
| short COCO, HiViS | 412895 | 16 | 16 | 0 | 0 | 16 | 6 / 10 / 0 |

`short` TextVQA has no differing outputs (100 of 100 identical for both methods).

- **Near-ties.** All 397 differences are near-ties. At the first differing token, the stock model's bf16
  margin between its top two logits is at most 0.5 (median 0.0 to 0.125 per comparison).
- **fp32 re-runs.** In the fp32 re-runs (the first 50 cases of each comparison, or all of them), the stock
  model prefers AR's token in 112 of 229 cases, the speculative method's token in 111 and a third token in 6.
  Neither method's token is systematically the better one.
- **fp32 exactness.** In fp32, the 7B ViSpec draft gives the same tokens as AR on 5 of 5 long TextVQA questions
  (`image-last`, 30-3-8), and HiViS does so on 5 of 5 long COCO images with the image first. With job 402297
  (HiViS, 5 of 5 in each prompt order on TextVQA), the speculative methods match AR exactly on all 20 questions
  of the four fp32 checks.

Both gates of the plan passed: no unexplained or both-wrong case at the 0.5-logit threshold, and AR equal to
the speculative method on 5 of 5 in each fp32 check. The low bf16 agreement in the tables above therefore comes
from near-ties, as on 3B.

### Parity with the HiViS reference code (job 412935)

Job 412935 (node c4-2, folder `DRNcWa`, commit `def2e96`) ran HiViS's own code and our port on the same 20
TextVQA questions, the first 20 of the 100 used above. HiViS's code ran from its checkout at `6db36b2`, in its
own environment (torch 2.6.0, Transformers 4.54.0), offline. Both sides used the 60-5-10 tree, fp16, greedy
decoding, at most 500 new tokens and HiViS's own prompt.

| Check | Required | Result |
| --- | --- | --- |
| Inputs | the same samples, digests and settings on both sides | Passed, 20 of 20 |
| Pruned token IDs, kept positions and position offset | exact integers equal | Passed, 20 of 20 |
| First draft tree, with the drafter fed HiViS's recorded hidden states | equal trees | Passed. 14 of 20 equal in fp16. The other 6 are equal once both sides run in fp32, so they are rounding ties in the kernels |
| Output tokens end to end | at least 18 of 20 identical, and every difference a near-tie | **Not passed as written.** 9 of 20 identical. All 11 differences are near-ties, with stock fp16 top-2 margins of 0.0 to 0.078 logits |
| Acceptance on the identical samples | within 3% relative | Passed. 2.8646 on both sides, over 842 iterations each |

- **The token gate fails on the count, not on the kind of difference.** Every difference is a near-tie. HiViS's
  own code agrees with itself less often: its speculative and AR paths give identical tokens on 0 of 20 of these
  outputs, and ours on 6 of 20. We read this as near-tie divergence between two different torch and Transformers
  stacks over 500 fp16 tokens, not as a defect of the port. That reading is an inference.
- **The checks of the port itself pass exactly.** The drafter's pruning and positions are equal, its first tree
  is equal up to fp16 rounding ties, and where the outputs are identical the acceptance is identical.
- **Acceptance over all 20 samples.** 2.896 on HiViS's side and 2.829 on ours (2.3% lower) by the gate's rule,
  and 2.906 and 2.827 by HiViS's own averaging rule. The paper's 3.95 at 60-5-10 used other samples and is
  context only.

The plan treated a failed parity gate as a stop condition, but this check ran after the comparison runs. The
outcome is recorded as not passed, with the evidence above.

### Predictions

| Prediction | Cutoff | Outcome | Evidence |
| --- | --- | --- | --- |
| 1. Both speculative methods beat AR in the `long` style | ViSpec and HiViS mean latency below the same-job AR mean on both benchmarks | **Held** | Speedup over AR 1.972 (ViSpec) and 1.989 (HiViS) on TextVQA (412893), 1.980 and 2.203 on COCO (412894). Every `long` speculative arm in the six `long` jobs was faster than its AR reference (1.550 to 2.392) |
| 2. HiViS's acceptance relative to ViSpec depends on prompt order | The HiViS minus ViSpec acceptance difference changes by at least 0.1 between `image-last` and `image-first` | **Not held** | The change is -0.004 on TextVQA (0.101 to 0.096) and +0.013 on COCO (0.437 to 0.449). HiViS's own acceptance moved by at most 0.013 |
| 3. The 60/5/10 tree raises acceptance for both methods | Acceptance at 60/5/10 above the 30/3/8 value for both methods on both benchmarks | **Held** | +0.31 (ViSpec) and +0.47 (HiViS) on TextVQA, +0.38 and +0.61 on COCO. The latency part was left open. It ranged from 3.2% slower to 2.3% faster |

### Analysis files

`experiment/sweeps/20261121T233327Z_qwen7b_methods_full/` (git-ignored) holds the following files.

- `sweep.md`, `sweep.json` and `sweep.csv` are the `sweep_table --layout methods` output for the 31 folders.
- `run_dirs.txt` lists the folders.
- `token_identity.py` and its `.txt` output give the cross-job token-ID identity check.
- `prefix_vs_ar.py` and its `.txt` output give the matching prefix against AR and the outputs at the token
  limit.
- `pooled_decode.py` and its `.txt` output give the pooled tokens per iteration, step time and TTFT share.

`experiment/sweeps/20261122T020510Z_qwen7b_comparisons/` holds the 22 same-job `compare_metrics --mode method`
comparisons (AR against each speculative arm, within one prompt order), their commands, and the two list files
of the diagnosis job.

`experiment/sweeps/20261122T020838Z_qwen7b_open_questions/` holds `coco_caption_style.py` (the caption length,
quoting and word overlap above) and `order_step_time.py` (per-sample step times by prompt order), with their
outputs for job 412898. For the rerun, `order_rerun_dirs.txt` lists the twelve TextVQA order folders of 412898 and
412937, `order_rerun_token_identity.txt` and `order_rerun_step_time.txt` hold the token identity and step times,
and `order_rerun_sweep/` holds the methods sweep table of 412937.

Each script reads only saved answer files and runs without a GPU.

## Limitations

- One run per arm, 100 samples per benchmark. Differences in quality and agreement are noisy.
- bf16 near-ties can make speculative and AR outputs differ at the token level. Check the saved token IDs
  before comparing text between methods.
- The HiViS paper's numbers use a different harness (fp16, 500 new tokens, an image-first prompt, the first
  80 TextVQA test rows). They are context only and are never substituted for these results.
- Re:Verse+ on 7B and KV eviction on 7B are not run.
- The ViSpec and HiViS drafters are separately trained checkpoints released by their authors. The acceptance
  difference compares the two checkpoints, so it cannot be attributed to the draft's visual input alone.
- The fp32 checks cover 20 questions (5 per check), and the diagnosis covers the primary jobs only, with fp32
  re-runs for the first 50 cases of each comparison. The speedups are observed ratios on bf16 outputs that
  differ from AR at near-ties, with their agreement next to them.
- The parity check compares our port with HiViS's code on 20 TextVQA questions at 60-5-10 in fp16 only. Its
  token-count gate did not pass as written.
- Every job except 412937 runs its arms in the same order: AR first, then ViSpec, then HiViS. Timing inside a
  job is not stable. In job 412898 the last two arms took 16% and 21% longer per step than the `image-last` arms
  of the same job, and the reversed-order rerun, with identical outputs, shows that prompt order does not explain
  it. A speed difference of a few percent between two arms of one run
  can therefore reflect when each arm ran. The device report records the GPU at the start of each arm only, and
  host load is not recorded.
- `compare_metrics` refuses to pair runs from different jobs only when the draft is HiViS. A ViSpec against AR
  comparison still accepts runs from different jobs, as the 3B baselines did, so build 7B comparisons from one
  job's folders, or use `sweep_table`, which refuses to mix jobs.
