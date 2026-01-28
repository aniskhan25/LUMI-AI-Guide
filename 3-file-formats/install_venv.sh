#!/bin/bash
set -euo pipefail
module use /appl/local/containers/ai-modules
module load singularity-AI-bindings

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../env.sh"

if [ -d "venv-extension" ]; then echo 'Removing existing venv-extension'; rm -Rf venv-extension; fi

singularity exec "$CONTAINER" bash -c 'if [ -n "${WITH_CONDA:-}" ]; then $WITH_CONDA; fi && python -m venv venv-extension --system-site-packages && source venv-extension/bin/activate && python -m pip install -r venv-requirements.txt'
