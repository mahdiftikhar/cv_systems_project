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
`srun -w NODE` ("Request for specific node ignored"), and the gpu partition was busy
(backfill estimates of several days), so live probes did not start within the roughly
30 minute budget. Probes pinned to one node by excluding the others (jobs 392562 to 392566,
`--time=00:05:00`) were still pending when cancelled. The pin therefore rests on earlier
`nvidia-smi` output in logs of previous jobs (Dark_Genome results, matched to nodes with
`sacct`):

| Node | Evidence | GPU |
| --- | --- | --- |
| c1-8 | job 339897_2 | NVIDIA RTX A6000, 49140 MiB |
| c2-1 | job 339897_1 | NVIDIA RTX A6000, 49140 MiB |
| c2-2 | none found (same 48 GB feature set) | unverified |
| c4-2 | job 339897_3 | NVIDIA RTX A6000, 49140 MiB |
| c4-3 | job 353849 | NVIDIA RTX A6000, 49140 MiB |

The live `srun` mismatch test from the Task 1 plan (job 392574) also stayed queued and was cancelled; the exit-3 path was verified with a stub `nvidia-smi` instead.

No node was found with a different 48 GB model, so `CLUSTER_EXCLUDE` was not extended. The
runtime check in `run_sequence.sbatch` guards the unverified c2-2 and any drift: a job that
lands on a different GPU model fails with status 3 and can be resubmitted. Re-run the probe
when the queue is quieter and record the driver version here.

## Submitting

Run from a login or VS Code shell in the repository root. The helpers clear inherited
`SLURM_*` variables first.

```bash
PROMPT_STYLE=long bash experiment/submit_ampere.sh experiment/02_name [experiment/03_name ...]
```

This submits `run_sequence.sbatch` once (8 CPUs, 64G, 8 hours, offline Hugging Face
caches). It runs each `EXP_DIR/run.sh` in order in the same allocation, continues past a
failed experiment, and exits non-zero if any failed. Logs go to `logs/<job>-<id>.log`.
Add `sbatch` options with `SBATCH_EXTRA`, for example `SBATCH_EXTRA=--test-only`.

For an ad-hoc check on the pinned GPU:

```bash
SRUN_EXTRA="--time=00:05:00" bash experiment/gpu_check.sh nvidia-smi
```

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
