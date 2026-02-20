#!/bin/bash
#SBATCH --job-name=dataset_compare
#SBATCH --account=project_462000131
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-gpu=60G
#SBATCH --time=00:20:00

set -euo pipefail

# shortcut for getting the binds right
module use /appl/local/containers/ai-modules
module load singularity-AI-bindings

source ../env.sh
: "${CONTAINER:?Set CONTAINER in ../env.sh}"
: "${SQUASH_LARGE:?Set SQUASH_LARGE in ../env.sh}"
[[ -f "$SQUASH_LARGE" ]] || { echo "ERROR: Missing squashfs: $SQUASH_LARGE" >&2; exit 1; }

export MPICH_MPIIO_STATS=1
export MPICH_MEMORY_REPORT=1

time srun singularity exec \
  -B "$SQUASH_LARGE":/train_images:image-src=/Data/CLS-LOC/train/ \
  "$CONTAINER" \
  venv-extension/bin/python compare-dataset.py
