#!/bin/bash
#SBATCH --job-name=noprof-lmdb
#SBATCH --account=project_462000131
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=24:00:00

set -euo pipefail

# shortcut for getting the binds right
module use /appl/local/training/modules/AI-20240529
module load singularity-userfilesystems singularity-CPEbits

source ../env.sh
: "${CONTAINER:?Set CONTAINER in ../env.sh}"
: "${SQUASH_LARGE:?Set SQUASH_LARGE in ../env.sh}"
[[ -f "$SQUASH_LARGE" ]] || { echo "ERROR: Missing squashfs: $SQUASH_LARGE" >&2; exit 1; }

export MPICH_MPIIO_STATS=1
export MPICH_MEMORY_REPORT=1

time srun singularity exec "$CONTAINER" \
  venv-extension/bin/python scripts/lmdb/visualtransformer-lmdb.py

time srun singularity exec \
  -B "$SQUASH_LARGE":/train_images:image-src=/Data/CLS-LOC/train/ \
  "$CONTAINER" \
  venv-extension/bin/python scripts/squashfs/visualtransformer-squashfs.py

### Tiny imagenet scripts
# srun singularity exec "$CONTAINER" venv-extension/bin/python scripts/hdf5/visualtransformer-hdf5.py
# srun singularity exec -B data-formats/squashfs/train.squashfs:/train_images:image-src=/ "$CONTAINER" venv-extension/bin/python scripts/squashfs/visualtransformer-squashfs.py
