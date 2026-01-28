#!/bin/bash

# Shared configuration for 3-file-formats scripts.
# This file now sources the repo-level env.sh for cross-repo consistency.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
source "$REPO_ROOT/env.sh"
