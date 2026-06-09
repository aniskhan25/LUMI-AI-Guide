#!/bin/bash

# Shared configuration for all lessons.
# Set these two values before submitting any jobs:

export PROJECT_ACCOUNT="${PROJECT_ACCOUNT:-project_462000131}"
export CONTAINER="${CONTAINER:-/appl/local/laifs/containers/lumi-multitorch-latest.sif}"

# Derived paths — change only if your project layout differs
export SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/${PROJECT_ACCOUNT}/${USER}}"

# Bind the per-job local NVMe scratch over /tmp inside the container.
# HIP writes its runtime files (.ufdb.txt) to /tmp and hardcodes that path —
# setting TMPDIR is not enough. Bind-mounting a writable directory over /tmp
# is the correct fix. SINGULARITY_BIND is read automatically by every
# singularity exec call, so no job script changes are needed.
if [ -n "${LOCAL_SCRATCH:-}" ]; then
    export SINGULARITY_BIND="${SINGULARITY_BIND:+${SINGULARITY_BIND},}${LOCAL_SCRATCH}:/tmp"
fi
