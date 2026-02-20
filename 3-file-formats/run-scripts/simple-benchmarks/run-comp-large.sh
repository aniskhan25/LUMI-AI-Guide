#!/bin/bash
#SBATCH --job-name=comp-large
#SBATCH --account=project_462000131
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-gpu=60G
#SBATCH --time=04:00:00

set -euo pipefail

module use /appl/local/training/modules/AI-20240529
module load singularity-userfilesystems singularity-CPEbits

source ../env.sh
: "${CONTAINER:?Set CONTAINER in ../env.sh}"
: "${SQUASH_LARGE:?Set SQUASH_LARGE in ../env.sh}"
SQSH_PATH="../resources/visiontransformer-env.sqsh"
[[ -f "$SQSH_PATH" ]] || { echo "ERROR: Missing sqsh: $SQSH_PATH" >&2; exit 1; }

FORMAT="${1:?Usage: sbatch run-scripts/simple-benchmarks/run-comp-large.sh <squashfs|lmdb|hdf5>}"

echo "Warning: This benchmark requires significant resources"
export MPICH_MPIIO_STATS=1
export MPICH_MEMORY_REPORT=1

case "$FORMAT" in
  squashfs)
    [[ -f "$SQUASH_LARGE" ]] || { echo "ERROR: Missing squashfs: $SQUASH_LARGE" >&2; exit 1; }
    time srun singularity exec \
      -B "$SQSH_PATH":/user-software:image-src=/ \
      -B "$SQUASH_LARGE":/train_images:image-src=/Data/CLS-LOC/train/ \
      "$CONTAINER" \
      /user-software/bin/python run-scripts/simple-benchmarks/compare-dataset-large.py \
      -n 7 -ff squashfs -N 200000
    ;;
  lmdb)
    time srun singularity exec \
      -B "$SQSH_PATH":/user-software:image-src=/ \
      "$CONTAINER" \
      /user-software/bin/python run-scripts/simple-benchmarks/compare-dataset-large.py \
      -n 7 -ff lmdb -N 200000
    ;;
  hdf5)
    echo "ERROR: HDF5 is incompatible with large imagenet dataset." >&2
    exit 1
    ;;
  *)
    echo "ERROR: Unknown format '$FORMAT'. Use one of: squashfs, lmdb, hdf5." >&2
    exit 1
    ;;
esac
