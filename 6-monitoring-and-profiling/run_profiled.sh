#!/bin/bash

#SBATCH --job-name=profiled-vit
#SBATCH --account=project_462000131
#SBATCH --partition=small-g

#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-gpu=60G

#SBATCH --time=01:00:00

set -euo pipefail

module use /appl/local/containers/ai-modules
module load singularity-AI-bindings

source ../env.sh
: "${CONTAINER:?Set CONTAINER in ../env.sh}"
: "${TINY_HDF5_PATH:?Set TINY_HDF5_PATH in ../env.sh}"
[[ -f "$TINY_HDF5_PATH" ]] || { echo "ERROR: Missing HDF5 file: $TINY_HDF5_PATH" >&2; exit 1; }
SQSH_PATH="../resources/visiontransformer-env.sqsh"
[[ -f "$SQSH_PATH" ]] || { echo "ERROR: Missing sqsh file: $SQSH_PATH" >&2; exit 1; }

time srun singularity exec -B "$SQSH_PATH":/user-software:image-src=/ "$CONTAINER" \
  /user-software/bin/python visiontransformer_profiled.py
