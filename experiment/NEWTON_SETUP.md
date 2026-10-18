# Newton: prepare once, run both experiments

- **Project root means the Git repository itself:** `~/cv_systems_project`.
- Use uv to prepare `.venv` in the repository on Newton. Existing `.venv` users can skip creation.
- Jobs invoke `.venv/bin/python` directly; activation, Conda, pip inside `.venv`, and uv on compute nodes at runtime are not required.
- Environment creation/installation and preparation belong on compute nodes.
  [UCF environment policy](https://arcc.ist.ucf.edu/docs/software/anaconda/).
- The [experiment register](README.md) links the current completed runs and their hardware evidence.

## Login shell: request a GPU session

- Replace `your_account_name` with a Slurm account you are authorized to use.

```bash
export NEWTON_ACCOUNT=your_account_name
srun --account="$NEWTON_ACCOUNT" --partition=normal \
  --gres=gpu:nvidia_h100_pcie:1 --exclude=evc26 \
  --nodes=1 --ntasks=1 --cpus-per-task=8 \
  --mem=64G --time=03:00:00 --pty bash
```

- Verify the assigned node with `hostname` and `nvidia-smi`.
- If a previous allocation survives disconnection, reuse its job ID with
  `srun --jobid=JOB_ID --overlap --gres=gpu:nvidia_h100_pcie:1 --pty bash`.
- [UCF submission guidance](https://arcc.ist.ucf.edu/docs/scheduler/scripts/),
  [Slurm job-step options](https://slurm.schedmd.com/srun.html).

## Compute shell: create/install the uv environment

- Go to the checkout containing the current committed preparation/experiment code.
- Ensure `uv` is available (`uv --version`); if needed, follow the [official installation guide](https://docs.astral.sh/uv/getting-started/installation/).
- Create the environment on Newton, not on your laptop; skip creation if `.venv` already exists.

```bash
cd ~/cv_systems_project
# Only if you need a new environment:
uv venv --python 3.11 .venv

export UV_CACHE_DIR="$PWD/.cache/uv"
uv pip install --python .venv/bin/python \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  --index-strategy unsafe-best-match \
  -r requirements.txt
uv pip check --python .venv/bin/python
```

- The index strategy resolves the existing pinned requirements across PyPI and the official PyTorch wheel index.
  [uv environment selection](https://docs.astral.sh/uv/pip/environments/).
- The pinned PyTorch CUDA 12.8 Linux wheel needs glibc >=2.28; check `getconf GNU_LIBC_VERSION`
  if installation cannot find `torch==2.7.0+cu128`.
- [Official PyTorch wheel index](https://download.pytorch.org/whl/cu128/torch/).
- No activation is needed. For a different prepared environment, set `PYTHON_BIN` to its absolute Python path.

## Prepare all downloads with one command

```bash
cd ~/cv_systems_project
bash prepare.sh
```

- Uses `<repo>/.venv/bin/python`, or an explicit `PYTHON_BIN=/absolute/path/to/python`.
- Downloads the fixed LLaVA-1.6 Vicuna 7B and ViSpec checkpoints.
- Downloads TextVQA validation annotations/images and GQA prompts/images/answer annotations.
- Caches the MM-Vet test split and downloads the TextVQA scoring helper.
- `gdown==5.2.0` is included in `requirements.txt`; missing preparation dependencies fail before downloads with setup instructions. Preparation does not install packages.
- Pins model revisions in `.preparation/assets.json` for subsequent preparation reruns.
- Validates model shards, benchmark image paths, reference answers, question IDs, and MM-Vet schema.
- Reuses completed files; interrupted HTTP/Hub downloads and image extraction can resume on rerun.
- Logs combined output and exit status under `.preparation/runs/`.
- Concurrent preparation in the same checkout is rejected to protect partial downloads.
- Existing assets outside the repo are not moved or deleted automatically.
- Google Drive throttling or source outages fail visibly; the next invocation reuses finished work.

```text
cv_systems_project/              # contains .git/
├── prepare.sh                  # preparation entry point
├── models/                     # local base + draft checkpoints
├── bench_data/                 # TextVQA and GQA data/labels
├── downloads/                  # source archives and partial downloads
├── .cache/                     # Hugging Face/MM-Vet and package caches
├── tools/downloaded/           # external TextVQA scoring helper
├── .preparation/               # asset manifest, lock, preparation logs
└── experiment/*/runs/          # evaluation evidence and reports
```

- Generated assets are ignored by default. Only the two run snapshots selected
  in `.gitignore` are versioned for team review. `.rsyncignore` controls transfer exclusions.
- No `PROJECT_ROOT`, `models.env`, or manual source commands are needed.
- Offline check after preparation:

```bash
bash prepare.sh --check-only
```

## Preparation that survives SSH disconnections

- From the repo root on the login node, submit preparation using the existing `.venv` as a batch job.
- It needs CPU/RAM and network access, not a GPU. Six hours is an example time request.

```bash
cd ~/cv_systems_project
mkdir -p .preparation
export PYTHON_BIN="$PWD/.venv/bin/python"
sbatch --account="${NEWTON_ACCOUNT:?Set NEWTON_ACCOUNT to your Slurm account}" --partition=normal \
  --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G --time=06:00:00 \
  --export=ALL --job-name=prepare-vispec \
  --output="$PWD/.preparation/slurm-%j.log" \
  prepare.sh
```

- Monitor with `squeue -u "$USER"` and `tail -f .preparation/slurm-JOB_ID.log`.
- `.preparation/runs/<timestamp>_<suffix>/output.log` contains the preparation output too.
- [UCF recommends batch jobs for work that must survive connection loss](https://arcc.ist.ucf.edu/docs/scheduler/scripts/).

## Run the experiments

- Use a working one-H100 allocation and the same prepared `.venv`.
- Both runners default to the prepared repo-local checkpoints, data, and scoring inputs.
- Explicit `BENCH_DIR`, `BASE_MODEL_PATH`, `SPEC_DIR`, and scoring environment overrides still apply;
  clear old overrides if they point to the previous parent-workspace layout.
- Model path overrides are intended for local copies of the same fixed baseline checkpoints.

```bash
bash experiment/00_baseline_original/run.sh
bash experiment/01_baseline_draft/run.sh
```

- For separate batch jobs, run from the checkout on the login node (no activation needed):

```bash
export NEWTON_ACCOUNT=your_account_name
export NEWTON_PARTITION=normal
export NEWTON_GPU_GRES=gpu:nvidia_h100_pcie:1
bash experiment/submit.sh experiment/00_baseline_original
bash experiment/submit.sh experiment/01_baseline_draft
```

- The submit helper sets the Git repository as `PROJECT_ROOT` and defaults to its `.venv/bin/python`.
- Both batch files invoke that Python directly without modules, Conda activation, installs, or environment synchronization.
- Clear a stale `PYTHON_BIN` override with `unset PYTHON_BIN` to use the default `.venv`.
- Runs record installed distribution names/versions in `packages.txt` using Python's package metadata, without pip or uv. This inventory is not a dependency lockfile.
- Separate jobs may land on different GPUs; the comparison checks hardware identity.
- Keep both runs on the same GPU/allocation for comparisons.
- Runtime evidence includes `prepared-assets.json`, copied from the preparation manifest when present.
- Reports are under each run's `reports/` folder. MM-Vet accuracy needs external judge grades afterward.
- Metric definitions and comparison/re-scoring commands: [METRICS.md](METRICS.md).
- Check truncation before interpreting speedup: the inherited generation limits still differ.
- Preserve raw runs on project storage. The current jobs 950793 and 950803 are
  also versioned for team review through explicit `.gitignore` exceptions.

## Sync code without copying downloads back and forth

```bash
rsync -av --exclude-from=.rsyncignore ./ \
  ma076549@newton.ist.ucf.edu:~/cv_systems_project/
```

- Run rsync from your local repo root; keep the destination's prepared assets in place.
- `.rsyncignore` excludes assets, environments, caches, preparation/evaluation logs, and local secrets.
- Repository metadata is not excluded: runners need a Git checkout to capture the launch commit.
- Data sources: [LLaVA preparation](https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md),
  [GQA downloads](https://cs.stanford.edu/people/dorarad/gqa/download.html),
  [Hugging Face downloads](https://huggingface.co/docs/huggingface_hub/v0.30.2/guides/download).
