#!/bin/bash
set -euo pipefail

module use /appl/local/containers/ai-modules
module load singularity-AI-bindings

source ../env.sh
: "${CONTAINER:?Set CONTAINER in env.sh or export it before running this script.}"

rm -rf venv-extension

singularity exec "$CONTAINER" bash -c '
set -euo pipefail
python -m venv venv-extension --system-site-packages
venv-extension/bin/python -m pip install -r venv-requirements.txt
'
