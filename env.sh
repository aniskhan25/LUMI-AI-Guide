#!/bin/bash

# Shared configuration for all lessons.
# Set these two values before submitting any jobs:

export PROJECT_ACCOUNT="${PROJECT_ACCOUNT:-project_462000131}"
export CONTAINER="${CONTAINER:-/appl/local/laifs/containers/lumi-multitorch-latest.sif}"

# Derived paths — change only if your project layout differs
export SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/${PROJECT_ACCOUNT}/${USER}}"

# MIOpen cache — must be a writable per-job directory, not the container default.
# Without this the HIP runtime aborts with a filesystem permissions error.
MIOPEN_DIR=$(mktemp -d)
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_DIR/cache
export MIOPEN_USER_DB=$MIOPEN_DIR/config

# PyTorch model cache — keep downloaded weights on scratch, not $HOME.
export TORCH_HOME="${TORCH_HOME:-${SCRATCH_ROOT}/torch_home}"
mkdir -p "$TORCH_HOME"
