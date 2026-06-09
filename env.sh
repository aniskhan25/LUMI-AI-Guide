#!/bin/bash

# Shared configuration for all lessons.
# Set these two values before submitting any jobs:

export PROJECT_ACCOUNT="${PROJECT_ACCOUNT:-project_462000131}"
export CONTAINER="${CONTAINER:-/appl/local/laifs/containers/lumi-multitorch-latest.sif}"

# Derived paths — change only if your project layout differs
export SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/${PROJECT_ACCOUNT}/${USER}}"

# Point the HIP/ROCm runtime's temp files to the per-job local NVMe scratch.
# Without this, the HIP user file database (.ufdb.txt) is written to /tmp
# inside the container, which may not be writable, causing an abort.
export SINGULARITYENV_TMPDIR="${LOCAL_SCRATCH:-/tmp}"
