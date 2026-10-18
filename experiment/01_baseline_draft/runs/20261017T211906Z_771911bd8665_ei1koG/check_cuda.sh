#!/usr/bin/env bash
# Require usable CUDA before loading models. Usage: bash check_cuda.sh [python]
set -euo pipefail
python_bin="${1:-python}"
printf 'CUDA preflight: host=%s job=%s CUDA_VISIBLE_DEVICES=%s\n' \
    "$(hostname)" "${SLURM_JOB_ID:-unset}" "${CUDA_VISIBLE_DEVICES:-unset}"
"$python_bin" - <<'PY'
import sys

try:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch reports CUDA unavailable")
    # Exercise the same logical device used by the evaluators, including a kernel
    # and synchronization; enumeration alone does not establish usable CUDA.
    value = torch.ones(1, device="cuda:0") + 1
    torch.cuda.synchronize(0)
    if value.item() != 2:
        raise RuntimeError("CUDA computation returned an unexpected result")
    print(f"CUDA preflight passed: cuda:0 = {torch.cuda.get_device_name(0)}", flush=True)
except Exception as error:
    print(f"CUDA preflight FAILED: {type(error).__name__}: {error}", file=sys.stderr)
    print("Stopping before model loading. Inspect device.txt and the assigned node; "
          "keep Slurm's CUDA_VISIBLE_DEVICES unchanged.", file=sys.stderr)
    sys.exit(1)
PY
