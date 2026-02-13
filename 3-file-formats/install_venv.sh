#!/bin/bash
set -euo pipefail
module use /appl/local/containers/ai-modules
module load singularity-AI-bindings

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../env.sh"
: "${CONTAINER:?Set CONTAINER in env.sh or export it before running this script.}"

cd "$SCRIPT_DIR"
rm -rf venv-extension

singularity exec "$CONTAINER" bash -c '
set -euo pipefail
if [ -n "${WITH_CONDA:-}" ]; then eval "$WITH_CONDA"; fi
python -m venv venv-extension --system-site-packages
source venv-extension/bin/activate
python -m pip install -r venv-requirements.txt
'
