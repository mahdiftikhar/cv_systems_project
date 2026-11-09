#!/usr/bin/env bash
# Build the separate environment for the HiViS reference run: Python 3.9, torch 2.6.0 (cu124)
# and Transformers 4.54.0, the versions HiViS's own code is written for.
#
# Run it once, in a CPU allocation (the network is used only for the package installs):
#   source experiment/ampere_env.sh && clear_inherited_slurm && \
#     srun -p normal --cpus-per-task=4 --mem=16G --time=01:00:00 \
#     bash experiment/10_qwen7b_methods/hivis_reference/setup_env.sh
#
# It installs only what inference needs. It leaves out the training, demo and API packages that
# HiViS's requirements.txt lists (deepspeed, fschat, gradio, openai, anthropic, protobuf,
# sentencepiece), and ray and flash-attn. It passes no credentials: pip reads no user config
# file, and no Hugging Face token is set.
#
# Outputs, next to each other under .cache/envs/ (git-ignored, on storage that compute nodes
# can see):
#   hivis-ref/              the conda environment
#   hivis-ref.freeze.txt    pip freeze of the environment, copied into every reference run
#
# Before the build it checks the HiViS checkout (pinned commit, clean status, read-only files).
# After the build it imports hivis.model.model_hivis and hivis.evaluation.benchmark_data with
# the checkout on sys.path, on the CPU and offline. The import check loads no weights and runs
# no HiViS function.
set -euo pipefail

if [[ $# -gt 0 ]]; then
    echo "Usage: [HIVIS_ENV=/abs/env] [HIVIS_CHECKOUT=/abs/checkout] $0" >&2
    exit 2
fi

here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$here/../../.." && pwd)"
cd "$repo_dir"
# Resolve .cache to its real location, so that the environment's scripts and records name the
# project storage path, not a worktree link to it.
cache_dir="$(cd -P -- "$repo_dir/.cache" && pwd)"
env_dir="${HIVIS_ENV:-$cache_dir/envs/hivis-ref}"
checkout="${HIVIS_CHECKOUT:-$cache_dir/ext/HiViS}"
freeze_file="${env_dir}.freeze.txt"
hivis_pin="6db36b241eaf733ec313cbc9b86206c0f20762c9"

# No credentials: no pip user config (it could hold an index URL with a token), no Hugging
# Face token, no user site-packages.
export PIP_CONFIG_FILE=/dev/null
export PIP_CACHE_DIR="$cache_dir/pip"
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PYTHONNOUSERSITE=1
unset PYTHONPATH HF_TOKEN HUGGING_FACE_HUB_TOKEN HF_TOKEN_PATH

printf 'setup_started_utc=%s host=%s slurm_job_id=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(hostname)" "${SLURM_JOB_ID:-unset}"
printf 'env_dir=%s\ncheckout=%s\nfreeze_file=%s\n' "$env_dir" "$checkout" "$freeze_file"

# ==========================================
# step 01 -- Check the checkout: pinned commit, clean status, no writable file.
# ==========================================
head_commit="$(git -C "$checkout" rev-parse HEAD)"
if [[ "$head_commit" != "$hivis_pin" ]]; then
    echo "HiViS checkout is at $head_commit, expected $hivis_pin" >&2
    exit 2
fi
checkout_status="$(git -C "$checkout" status --porcelain)"
if [[ -n "$checkout_status" ]]; then
    printf 'HiViS checkout is not clean:\n%s\n' "$checkout_status" >&2
    exit 2
fi
writable="$(find "$checkout" -perm /222 -print -quit)"
if [[ -n "$writable" ]]; then
    echo "HiViS checkout has writable files (first: $writable); run chmod -R a-w on it" >&2
    exit 2
fi
printf 'checkout_head=%s\ncheckout_status=clean\ncheckout_read_only=true\n' "$head_commit"

# ==========================================
# step 02 -- Create the environment. Refuse to reuse or overwrite an existing one.
# ==========================================
if [[ -e "$env_dir" || -e "$freeze_file" ]]; then
    echo "Environment or freeze file already exists: $env_dir; remove both to rebuild." >&2
    exit 2
fi
conda_bin="${CONDA_EXE:-$(command -v conda)}"
mkdir -p "$(dirname "$env_dir")" "$PIP_CACHE_DIR"
"$conda_bin" create -y -p "$env_dir" python=3.9
py="$env_dir/bin/python"
"$py" --version

# ==========================================
# step 03 -- Install torch and torchvision from the cu124 index, then the inference packages.
# ==========================================
# --no-compile skips writing bytecode for every installed file, which is slow on project
# storage; the reference run imports with -B and compiles in memory instead.
"$py" -m pip install --no-input --no-compile \
    --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0 torchvision==0.21.0
"$py" -m pip install --no-input --no-compile \
    transformers==4.54.0 accelerate==1.9.0 datasets==4.0.0 safetensors pillow numpy

# ==========================================
# step 04 -- Record the package list and check the two pinned versions.
# ==========================================
"$py" -m pip freeze > "$freeze_file"
for pin in 'torch==2.6.0+cu124' 'torchvision==0.21.0+cu124' 'transformers==4.54.0' \
    'accelerate==1.9.0' 'datasets==4.0.0'; do
    if ! grep -qxF "$pin" "$freeze_file"; then
        echo "Freeze file lacks $pin; see $freeze_file" >&2
        exit 1
    fi
done
echo "Freeze file: $freeze_file"
cat "$freeze_file"

# ==========================================
# step 05 -- Import check on the CPU, offline, with an empty Hugging Face home.
# ==========================================
# -I keeps the current directory and user site-packages off sys.path; -B writes no bytecode
# (the checkout is read-only in any case). The working directory is a scratch folder.
mkdir -p "$cache_dir/tmp"
work_dir="$(mktemp -d "$cache_dir/tmp/hivis-ref-import-XXXXXX")"
trap 'rm -rf "$work_dir"' EXIT
(
    cd "$work_dir"
    mkdir -p hf_home
    export HF_HOME="$work_dir/hf_home" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
        HF_DATASETS_OFFLINE=1 CUDA_VISIBLE_DEVICES=""
    "$py" -I -B - "$checkout" <<'PY'
import sys

checkout = sys.argv[1]
sys.path.insert(0, checkout)

import torch
import transformers

assert torch.__version__ == "2.6.0+cu124", torch.__version__
assert transformers.__version__ == "4.54.0", transformers.__version__

import hivis.evaluation.benchmark_data as benchmark_data
import hivis.model.model_hivis as model_hivis

for module in (model_hivis, benchmark_data):
    if not module.__file__.startswith(checkout + "/"):
        raise SystemExit(f"{module.__name__} was imported from {module.__file__}, not the checkout")
print(f"python {sys.version.split()[0]}, torch {torch.__version__}, "
      f"transformers {transformers.__version__}, cuda available: {torch.cuda.is_available()}")
print(f"imported {model_hivis.__name__} from {model_hivis.__file__}")
print(f"imported {benchmark_data.__name__} from {benchmark_data.__file__}")
print(f"EaModel: {model_hivis.EaModel.__name__}, draft methods: {sorted(model_hivis.DRAFT_MODELS)}")
print("Import check passed")
PY
)
printf 'setup_finished_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
