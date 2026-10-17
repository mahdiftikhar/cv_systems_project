# Getting started on Newton

Prepare the environment and downloads once, submit an experiment, then check its
logs and reports. The current experiments use LLaVA-1.6 Vicuna 7B on
**TextVQA → GQA → MM-Vet**. No training is required.

Commands below assume the repository is at `~/cv_systems_project` on Newton.
Use storage visible to both login and compute nodes with enough space for large
model checkpoints, image datasets, source archives, caches, and run artifacts.
If using another location, replace the `cd` and transfer paths throughout.

You need a UCF ARCC account, an authorized Slurm account, and SSH access.
Connect to UCF VPN when off campus. See [UCF access instructions](https://arcc.ist.ucf.edu/docs/access/linux/).

## 1. Copy the code — on your laptop

From your local repository root:

```bash
rsync -av --exclude-from=.rsyncignore ./ \
  ma076549@newton.ist.ucf.edu:~/cv_systems_project/
```

Replace the SSH username if needed. This excludes local environments, downloaded
assets, and generated logs; create the Linux environment on Newton. Git metadata
is included because experiment runners record the launch commit and working tree.
Prefer committed code for actual experiment runs.

Connect:

```bash
ssh ma076549@newton.ist.ucf.edu
```

## 2. Select resources — on the login node

Replace `your_account_name` with your Slurm account, not your SSH username:

```bash
export NEWTON_ACCOUNT=your_account_name
export NEWTON_PARTITION=normal
export NEWTON_GPU_GRES=gpu:nvidia_h100_pcie:1

sinfo -N -o '%N %P %G %m %l'
cd ~/cv_systems_project
df -h .
```

Confirm that the partition and GPU type are available to your account; adjust the
variables if needed. These exports must be repeated in a new login session.

## 3. Install dependencies once — on a compute node

Request an interactive allocation from the login node:

```bash
srun --account="$NEWTON_ACCOUNT" --partition="$NEWTON_PARTITION" \
  --gres="$NEWTON_GPU_GRES" \
  --nodes=1 --ntasks=1 --cpus-per-task=8 \
  --mem=64G --time=03:00:00 --pty bash
```

Wait for the compute-node shell. Environment creation and package installation
belong on compute nodes under [UCF's environment policy](https://arcc.ist.ucf.edu/docs/software/anaconda/).
This project uses uv and a repository-local `.venv`; Conda is not needed.

```bash
cd ~/cv_systems_project
hostname
nvidia-smi
uv --version
```

If uv is missing, install it using the [official uv instructions](https://docs.astral.sh/uv/getting-started/installation/)
and ensure its executable is on your `PATH` before continuing.

Create `.venv` only if it does not already exist, then install the requirements:

```bash
# Skip this command if your Newton .venv already exists.
uv venv --python 3.11 .venv

export UV_CACHE_DIR="$PWD/.cache/uv"
uv pip install --python .venv/bin/python \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  --index-strategy unsafe-best-match \
  -r requirements.txt
uv pip check --python .venv/bin/python
```

This includes the `gdown` helper needed for GQA preparation. The index options
allow the pinned CUDA PyTorch wheels and PyPI dependencies to be resolved together.
Do not change the pinned Transformers version for the baseline.

Check imports and a tiny GPU operation without loading model weights:

```bash
.venv/bin/python - <<'PY'
import torch
import transformers
from vispec.model.spec_model import SpecModel
from vispec.model.spec_model_ours import SpecModel as ViSpecModel

print('PyTorch:', torch.__version__)
print('Transformers:', transformers.__version__)
assert torch.cuda.is_available(), 'CUDA is unavailable'
assert torch.cuda.device_count() == 1, 'Expected one allocated GPU'
print('GPU:', torch.cuda.get_device_name(0))
assert torch.ones(16, device='cuda').sum().item() == 16
PY
```

Resolve installation, import, or CUDA errors before submitting evaluation. This
check does not validate actual model execution.

Return to the login node:

```bash
exit
```

## 4. Prepare models and datasets — submit from the login node

Preparation runs as a CPU batch job so downloads survive SSH disconnection.
No environment activation is needed. From the repository root:

```bash
cd ~/cv_systems_project
mkdir -p .preparation
export PYTHON_BIN="$PWD/.venv/bin/python"

sbatch --account="$NEWTON_ACCOUNT" --partition="$NEWTON_PARTITION" \
  --nodes=1 --ntasks=1 --cpus-per-task=4 \
  --mem=16G --time=06:00:00 \
  --export=ALL --job-name=prepare-vispec \
  --output="$PWD/.preparation/slurm-%j.log" \
  prepare.sh
```

Note the returned job ID. You can now disconnect. Reconnect and check progress,
replacing `JOB_ID` with the number Slurm returned:

```bash
cd ~/cv_systems_project
squeue -u "$USER"
tail -f .preparation/slurm-JOB_ID.log
sacct -j JOB_ID --format=JobID,State,Elapsed,ExitCode
```

Press `Ctrl+C` to stop following the log; it does not cancel the job. Preparation
must finish successfully (`COMPLETED`, exit code `0:0`) before evaluation.
Six hours is an initial request, not a verified download-time estimate.

`prepare.sh` downloads and validates:

- The fixed LLaVA-1.6 Vicuna 7B target and ViSpec draft checkpoints, under `models/`.
- TextVQA and GQA images, questions, and reference labels, under `bench_data/`.
- The MM-Vet test split, under `.cache/`, and the TextVQA scoring helper, under `tools/downloaded/`.

Archives stay in `downloads/`; preparation logs and the asset manifest stay in
`.preparation/`. Preparation does not install packages. Rerun the same submission
after an interruption or source failure; completed assets are reused. Do not run
two preparations simultaneously in the same checkout.

After completion, validate prepared files without downloads:

```bash
bash prepare.sh --check-only
```

This offline check is also allowed outside an allocation. It checks assets, not
model inference. Detailed preparation logs and exit status are under
`.preparation/runs/<unique-run>/`.

## 5. Submit the original baseline — on the login node

From the repository root, export the account settings again if this is a new
session. `unset PYTHON_BIN` clears old overrides and selects the repository `.venv`:

```bash
cd ~/cv_systems_project
export NEWTON_ACCOUNT=your_account_name
export NEWTON_PARTITION=normal
export NEWTON_GPU_GRES=gpu:nvidia_h100_pcie:1
unset PYTHON_BIN

bash experiment/submit.sh experiment/00_baseline_original
```

`submit.sh` accepts an experiment folder path and submits its `run.sbatch`.
It creates the log directory and passes the account, partition, GPU request, and
Python path to Slurm. Each experiment owns its resource directives and job name.
The current evaluation requests are one GPU, 8 CPUs, 64 GiB RAM, and six hours;
adjust the experiment's `run.sbatch` to your allocation limits if needed.

Once Slurm returns a job ID, you can disconnect. Keep the checkout, environment,
and prepared assets unchanged while the job is queued or running.
Batch jobs run independently of the SSH connection; see [UCF submission guidance](https://arcc.ist.ucf.edu/docs/scheduler/scripts/).

## 6. Check the job and read results

After reconnecting, replace `JOB_ID` below:

```bash
cd ~/cv_systems_project
squeue -u "$USER"
tail -f logs/llava-original-JOB_ID.log
sacct -j JOB_ID --format=JobID,State,Elapsed,ExitCode
```

The scheduler log includes setup failures that occur before a run directory exists.
Each evaluation launch creates a unique directory under
`experiment/00_baseline_original/runs/`:

| File or directory | What to check |
| --- | --- |
| `exit-code.txt` | `0` indicates successful runner completion; missing means an unclean finish |
| `output.log` | Combined output, exact evaluation commands, and errors |
| `device.txt` | Allocated/visible hardware, driver, and scheduler details |
| `commit.txt`, `git-status.txt`, `tracked-changes.patch` | Launch code evidence |
| `packages.txt`, `metadata.txt`, `prepared-assets.json` | Environment, settings, and prepared asset provenance |
| `results/` | Generated answers and raw measurements |
| `reports/<unique-report>/summary.md` | Per-benchmark performance and available quality results |

Review reports even when Slurm reports success. MM-Vet accuracy needs external
judge grades afterward. The original and draft methods inherit different generation
limits; inspect truncation before interpreting speedup. Keep hardware and settings
matched when comparing runs. Device snapshots alone do not measure runtime utilization
or energy. See [metric definitions and scoring instructions](experiment/METRICS.md).

## 7. Run the draft experiment or repeat a run

After the original baseline succeeds, submit the draft experiment:

```bash
bash experiment/submit.sh experiment/01_baseline_draft
```

Its scheduler log is `logs/llava-draft-JOB_ID.log`; detailed evidence is under
`experiment/01_baseline_draft/runs/`. Separate jobs may receive different physical
GPUs, so inspect both device reports before comparing results.

For later runs, reuse `.venv` and the prepared assets. Reinstall dependencies only
when they change, on a compute node. Repeating a submission creates a fresh run
directory rather than overwriting previous evidence. Preserve raw artifacts on
project storage and commit concise verified results with evidence links in the
[experiment register](experiment/README.md).

## Common problems

| Symptom | Next step |
| --- | --- |
| `sbatch: unrecognized option '--parition=normal'` | Correct the spelling to `--partition=normal` |
| Missing `.venv/bin/python` | Create the environment in step 3 on Newton; do not copy a laptop environment |
| Missing `gdown` or another dependency | Rerun the requirements installation on a compute node |
| `No module named pip` | These scripts do not need pip inside `.venv`; sync the updated code and use `uv pip` for installation |
| Offline model/dataset loading fails | Repair preparation first, then resubmit evaluation |
| Google Drive throttles the GQA download | Retry preparation later; completed downloads are retained |
| Job fails before evaluation | Read the scheduler log, then the run's `output.log` if one exists |
| A requested GPU is unavailable or CUDA fails | Check account/partition/GRES and the assigned node; the existing Newton notes record a prior failure on `evc26` |

The commands match the current scripts and have lightweight local verification.
Downloads and full GPU evaluation still require verification on Newton. Additional
cluster notes are in [NEWTON_SETUP.md](experiment/NEWTON_SETUP.md).
