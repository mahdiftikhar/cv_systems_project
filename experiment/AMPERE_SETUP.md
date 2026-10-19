# Ampere cluster setup

This guide covers submitting experiments to the Ampere GPU nodes of the local Slurm
cluster. Experiments 00 and 01 were run on Newton; see `NEWTON_SETUP.md` for them. The
Newton scripts (`submit.sh`, `*/run.sbatch`) are unchanged.

## GPU pin

Comparison runs use one 48 GB Ampere GPU model only: `NVIDIA RTX A6000` (49140 MiB).
`ampere_env.sh` exports it as `CLUSTER_EXPECTED_GPU`, and `run_sequence.sbatch` exits with
status 3 before any evaluation if `nvidia-smi` reports a different name.

Defaults in `experiment/ampere_env.sh` (each can be overridden by exporting the variable):

| Variable | Default |
| --- | --- |
| `CLUSTER_PARTITION` | `gpu` |
| `CLUSTER_GPU_GRES` | `gpu:ampere:1` |
| `CLUSTER_EXCLUDE` | `c4-4,c4-5,c5-2` (80 GB nodes and the mixed ampere/ada node) |
| `CLUSTER_EXPECTED_GPU` | `NVIDIA RTX A6000` |
| `PYTHON_BIN` | `/home/aeternum/anaconda3/envs/vipsec/bin/python` |

### Probe results

The 48 GB nodes per `sinfo` are c1-8, c2-1, c2-2, c4-2 and c4-3. This cluster ignores
`srun -w NODE` ("Request for specific node ignored"). The first live probes, each pinned to one
node by excluding the others (jobs 392562 to 392566 on the `gpu` partition, `--time=00:05:00`),
were still pending after about 30 minutes and were cancelled. The pin was therefore first taken
from `nvidia-smi` output in logs of earlier jobs (Dark_Genome results, matched to nodes with
`sacct`).

Two nodes have since been seen live on the `preempt` partition. `sacct` gives the node of each
job, and the job's own output gives the GPU name:

- **c2-2**: probe job 392590 (`nvidia-smi`: `NVIDIA RTX A6000`, 49140 MiB, driver 615.71.09),
  the CIDEr smoke job 392603, and both Week 1 jobs, 392660 and 392661, whose `device.txt` files
  report an `NVIDIA RTX A6000`.
- **c1-8**: the evaluator smoke job 392593, the diagnosis smoke jobs 392616 and 392638, and the
  Week 1 diagnosis jobs 392676 and 392714. Each recorded an `NVIDIA RTX A6000` in its runtime or
  diagnosis file.

The pins for c2-1, c4-2 and c4-3 rest only on the earlier logs.

| Node | Evidence | GPU |
| --- | --- | --- |
| c1-8 | live: jobs 392593, 392616, 392638, 392676, 392714; earlier log: job 339897_2 | NVIDIA RTX A6000, 49140 MiB |
| c2-1 | earlier log only: job 339897_1 | NVIDIA RTX A6000, 49140 MiB |
| c2-2 | live: jobs 392590, 392603, 392660, 392661 | NVIDIA RTX A6000, 49140 MiB, driver 615.71.09 |
| c4-2 | earlier log only: job 339897_3 | NVIDIA RTX A6000, 49140 MiB |
| c4-3 | earlier log only: job 353849 | NVIDIA RTX A6000, 49140 MiB |

The live `srun` test of the GPU-mismatch path (job 392574) also stayed queued and was cancelled.
The exit-3 path was verified with a stub `nvidia-smi` instead.

No node was found with a different 48 GB model, so `CLUSTER_EXCLUDE` was not extended. The
runtime check in `run_sequence.sbatch` still guards against drift: a job that lands on a
different GPU model fails with status 3 and can be resubmitted. The check exists only in
`run_sequence.sbatch`. A direct `run.sh` run inside an allocation, or a command run through
`gpu_check.sh`, skips it, so results of record must be submitted through `submit_ampere.sh`.
Dates in this guide are local time unless marked UTC.

## Submitting

Run from a login or VS Code shell in the repository root. The helpers clear inherited
`SLURM_*` variables first.

```bash
PROMPT_STYLE=long bash experiment/submit_ampere.sh experiment/02_name [experiment/03_name ...]
```

This submits `run_sequence.sbatch` once, by default with 8 CPUs, 64G and 8 hours, and with
offline Hugging Face caches. It runs each `EXP_DIR/run.sh` in order in the same allocation,
continues past a failed experiment, and exits non-zero if any failed. `OMP_NUM_THREADS` is set
from the allocation's `SLURM_CPUS_PER_TASK`. Logs go to `logs/<job>-<id>.log`. Add `sbatch`
options with `SBATCH_EXTRA`, for example `SBATCH_EXTRA=--test-only`. The helper exits with
status 2 before calling `sbatch` if `PROMPT_STYLE` is not `long` or `short`.

Week 1 ran one job per prompt style with both Qwen experiments and leaner resources. The 64G
default did not fit the free host memory on the A6000 nodes at the time (about 24 GB on c2-2).
This is the reproduction command; use `PROMPT_STYLE=short` for the second job:

```bash
PROMPT_STYLE=long CLUSTER_PARTITION=preempt SBATCH_EXTRA="--cpus-per-task=4 --mem=22G --time=03:00:00" \
  bash experiment/submit_ampere.sh experiment/02_qwen_autoregressive experiment/03_qwen_vispec
```

Running an experiment alone, or with the default resources, gives a different allocation and
possibly a different card. `compare_metrics` compares the GPU UUID, the allocated CPU count and
memory, and the CPU affinity, and it refuses to pair runs whose values differ.

The helper exports the exact `%q`-quoted `sbatch` command as `SUBMIT_COMMAND`, together with
`SBATCH_EXTRA`. The job inherits both through `--export=ALL`, and the 02 and 03 runners write
them to `metadata.txt` as `submit_command=` and `sbatch_extra=`. Both read `unset` for a launch
that did not use the helper.

The cluster default is `JobRequeue=0`, so `run_sequence.sbatch` requests `--requeue`, and a job
preempted on the `preempt` partition (`PreemptMode=REQUEUE`) is requeued instead of cancelled.
The requeued job reruns `run_sequence.sbatch` from the start, so each experiment gets a fresh
run folder, and the partial folders stay on disk as evidence. Their reports may be missing,
because the partition has `GraceTime=0` and Slurm sends SIGKILL 30 seconds (`KillWait`) after
SIGTERM. Run folders from the same attempt share `slurm_job_id` and `slurm_restart_count` in
`metadata.txt`. The log is opened in append mode, so it keeps every attempt.

For an ad-hoc check on the pinned GPU:

```bash
SRUN_EXTRA="--time=00:05:00" bash experiment/gpu_check.sh nvidia-smi
```

## Preparing Qwen assets

The Qwen experiments need the Qwen2.5-VL-3B target and ViSpec draft checkpoints, the
TextVQA inputs and scorer, and a 100-image COCO Karpathy-test subset. Run this inside a
Slurm allocation with network access (a CPU node is enough):

```bash
PYTHON_BIN=/home/aeternum/anaconda3/envs/vipsec/bin/python bash prepare.sh --suite qwen
PYTHON_BIN=/home/aeternum/anaconda3/envs/vipsec/bin/python bash prepare.sh --check-only --suite qwen
```

The first command is resumable and records the resolved model and parquet revisions and
the parquet SHA-256 in `.preparation/assets_qwen.json`; it leaves the LLaVA manifest
`.preparation/assets.json` alone. The second validates the shards, processor files,
parquet hash and that all 100 COCO images decode, without downloading. Omitting
`--suite` keeps the original LLaVA preparation.

## Clearing SLURM variables

A shell started inside a Slurm job (such as VS Code job 391999) carries `SLURM_*`
variables that override `srun`/`sbatch` resource requests. After
`source experiment/ampere_env.sh`, run `clear_inherited_slurm`, or equivalently
`for v in $(compgen -v SLURM_); do unset "$v"; done`.

## Environment

CIDEr scoring uses `pycocoevalcap`, whose PTB tokenizer runs a Java program. The vipsec env
needs both the Python package and a `java` binary. These commands were run once in the vipsec
env (Python 3.11.17):

```bash
~/.local/bin/uv pip install --python "$PYTHON_BIN" pycocoevalcap==1.2
conda install -n vipsec -y openjdk=17
```

The conda command uses the default channels, which resolve to `pkgs/main` and install
OpenJDK 17.0.14. It was dry-run first, and its plan only added packages, with no updates or
downgrades to existing ones. The `uv` command installs `pycocoevalcap` 1.2 and
`pycocotools` 2.0.11 (its dependency) into the env.

`java` is not on `PATH` unless the env's `bin` directory is on it, and the jobs do not
activate the env. Prepend the env's bin directory when running CIDEr code:

```bash
PATH="$(dirname "$PYTHON_BIN"):$PATH" PYTHONNOUSERSITE=1 "$PYTHON_BIN" your_script.py
```

Check the install with `"$(dirname "$PYTHON_BIN")/java" -version`, which should report 17.x.
