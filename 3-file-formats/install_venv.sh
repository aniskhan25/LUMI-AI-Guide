#!/bin/bash
set -euo pipefail
module use /appl/local/containers/ai-modules
module load singularity-AI-bindings

CONTAINER=/appl/local/containers/sif-images/lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.7.1.sif

if [ -d "venv-extension" ]; then echo 'Removing existing venv-extension'; rm -Rf venv-extension; fi

singularity exec "$CONTAINER" bash -c 'if [ -n "${WITH_CONDA:-}" ]; then $WITH_CONDA; fi && python -m venv venv-extension --system-site-packages && source venv-extension/bin/activate && python -m pip install -r venv-requirements.txt'
