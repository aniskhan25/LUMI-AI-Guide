#!/bin/bash

# Source this file at the top of every job script.
# Sets up modules, runtime caches, and shared config.

# --- User config: set these for your project ---
export PROJECT_ACCOUNT="${PROJECT_ACCOUNT:-project_462000131}"
# Pinned to a specific date-stamped image rather than a `latest` symlink, so a new
# container release cannot change behaviour under you. Look in
# /appl/local/laifs/containers/ for newer images and update this line deliberately.
export CONTAINER="${CONTAINER:-/appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260807_115122/lumi-multitorch-full-u24r70f21m50t210-20260807_115122.sif}"
export SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/${PROJECT_ACCOUNT}/${USER}}"

# --- Modules ---
module purge
module load Local-LAIF lumi-aif-singularity-bindings

# --- MIOpen caches ---
# MIOpen defaults to a fixed, non-per-user path under $TMPDIR. On a shared node the
# first user to create it owns it and everyone else gets permission-denied, so pin
# each cache to an explicit per-user path.
export MIOPEN_CUSTOM_CACHE_DIR="/tmp/miopen-cache-${USER}"
export MIOPEN_USER_DB_PATH="/tmp/miopen-config-${USER}"

# /tmp is node-local, so the directories must exist on every node of the allocation.
if [ -n "${SLURM_JOB_ID:-}" ]; then
  srun mkdir -p "$MIOPEN_CUSTOM_CACHE_DIR" "$MIOPEN_USER_DB_PATH"
else
  mkdir -p "$MIOPEN_CUSTOM_CACHE_DIR" "$MIOPEN_USER_DB_PATH"
fi

# --- Framework caches ---
# Keep downloaded models off $HOME, whose file-count quota they will otherwise exhaust.
export TORCH_HOME="${TORCH_HOME:-/scratch/${SLURM_JOB_ACCOUNT:-$PROJECT_ACCOUNT}/${USER}/torch_home}"
export HF_HOME="${HF_HOME:-/scratch/${SLURM_JOB_ACCOUNT:-$PROJECT_ACCOUNT}/${USER}/hf-cache}"
mkdir -p "$TORCH_HOME" "$HF_HOME"
