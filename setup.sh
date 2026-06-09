#!/bin/bash

# Source this file at the top of every job script.
# Sets up modules, runtime caches, and shared config.

# --- User config: set these for your project ---
export PROJECT_ACCOUNT="${PROJECT_ACCOUNT:-project_462000131}"
export CONTAINER="${CONTAINER:-/appl/local/laifs/containers/lumi-multitorch-latest.sif}"
export SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/${PROJECT_ACCOUNT}/${USER}}"

# --- Modules ---
module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

# --- MIOpen and PyTorch cache ---
# Must be writable per-job directories; MIOpen aborts if it cannot set
# permissions on its runtime files.
MIOPEN_DIR=$(mktemp -d)
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_DIR/cache
export MIOPEN_USER_DB=$MIOPEN_DIR/config
export TORCH_HOME="${TORCH_HOME:-/scratch/${SLURM_JOB_ACCOUNT:-$PROJECT_ACCOUNT}/${USER}/torch_home}"
mkdir -p "$TORCH_HOME"
