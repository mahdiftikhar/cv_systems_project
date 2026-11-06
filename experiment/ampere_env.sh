#!/usr/bin/env bash
# Source this file; it only exports defaults and defines helpers.
# Every default can be overridden by exporting the variable first.
export CLUSTER_PARTITION="${CLUSTER_PARTITION:-gpu}"
export CLUSTER_GPU_GRES="${CLUSTER_GPU_GRES:-gpu:ampere:1}"
# 80 GB nodes (c4-4, c4-5) and mixed ampere/ada node (c5-2) are excluded.
export CLUSTER_EXCLUDE="${CLUSTER_EXCLUDE:-c4-4,c4-5,c5-2}"
# Exact nvidia-smi name of the pinned 48 GB Ampere GPU (see AMPERE_SETUP.md).
export CLUSTER_EXPECTED_GPU="${CLUSTER_EXPECTED_GPU:-NVIDIA RTX A6000}"
export PYTHON_BIN="${PYTHON_BIN:-/home/aeternum/anaconda3/envs/vipsec/bin/python}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"

# The interactive shell may itself live inside a Slurm job; its SLURM_* variables
# would leak into srun/sbatch and override the requested resources.
clear_inherited_slurm() {
    local v
    for v in $(compgen -v SLURM_); do unset "$v"; done
}
