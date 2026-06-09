#!/bin/bash

# Shared configuration for all lessons.
# Set these two values before submitting any jobs:

export PROJECT_ACCOUNT="${PROJECT_ACCOUNT:-project_462000131}"
export CONTAINER="${CONTAINER:-/appl/local/laifs/containers/lumi-multitorch-latest.sif}"

# Derived paths — change only if your project layout differs
export SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/${PROJECT_ACCOUNT}/${USER}}"
