#!/usr/bin/env bash
# Shared launch-time hardware snapshot. Usage: bash capture_device.sh [python]
set -uo pipefail
python_bin="${1:-python}"

# Optional tools may be unavailable inside a cluster container. Keep the error in
# the report rather than silently omitting a field or aborting evaluation.
probe() {
    printf '\n###'
    printf ' %q' "$@"
    printf '\n'
    if command -v "$1" >/dev/null 2>&1; then
        "$@" || printf 'Probe failed (exit %s).\n' "$?"
    else
        printf 'Unavailable: %s\n' "$1"
    fi
}

printf 'Hardware snapshot at launch (UTC): %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
probe hostname
probe uname -a
if [[ -r /etc/os-release ]]; then
    probe cat /etc/os-release
fi
probe lscpu
probe free -b
for resource_file in /proc/self/status /proc/self/cgroup \
    /sys/fs/cgroup/cpu.max /sys/fs/cgroup/cpuset.cpus.effective \
    /sys/fs/cgroup/memory.max; do
    if [[ -r "$resource_file" ]]; then
        probe cat "$resource_file"
    fi
done

printf '\n### GPU visibility and scheduler allocation\n'
for variable in CUDA_VISIBLE_DEVICES NVIDIA_VISIBLE_DEVICES CUDA_DEVICE_ORDER \
    SLURM_JOB_ID SLURM_JOB_NODELIST SLURMD_NODENAME SLURM_JOB_PARTITION \
    SLURM_JOB_NUM_NODES SLURM_NTASKS SLURM_CPUS_PER_TASK SLURM_CPUS_ON_NODE \
    SLURM_MEM_PER_NODE SLURM_MEM_PER_CPU SLURM_JOB_GPUS SLURM_STEP_GPUS \
    SLURM_GPUS_ON_NODE SLURM_GPUS_PER_TASK OMP_NUM_THREADS MKL_NUM_THREADS; do
    printf '%s=%s\n' "$variable" "${!variable-unset}"
done

# Includes GPU UUID/model, VRAM, driver, MIG/compute mode, clocks, power limits,
# temperature, utilization, and running processes for devices visible to NVML.
probe nvidia-smi -q
probe nvidia-smi topo -m
probe nvcc --version

printf '\n### Python/PyTorch view of allocated devices\n'
if ! "$python_bin" - <<'PY'
import json
import os
import platform
import sys

report = {
    "python": sys.version,
    "executable": sys.executable,
    "platform": platform.platform(),
    "logical_cpus_on_host": os.cpu_count(),
    "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
}
try:
    import torch

    report.update({
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "torch_cpu_threads": torch.get_num_threads(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_visible_device_count": torch.cuda.device_count(),
        "devices": [],
    })
    for ordinal in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(ordinal)
        report["devices"].append({
            "cuda_ordinal": ordinal,
            "name": props.name,
            "uuid": str(getattr(props, "uuid", "unavailable")),
            "total_memory_bytes": props.total_memory,
            "compute_capability": [props.major, props.minor],
            "multiprocessor_count": props.multi_processor_count,
        })
except Exception as error:
    report["torch_probe_error"] = f"{type(error).__name__}: {error}"
print(json.dumps(report, indent=2))
PY
then
    printf 'Python device probe failed; see error above.\n'
fi
