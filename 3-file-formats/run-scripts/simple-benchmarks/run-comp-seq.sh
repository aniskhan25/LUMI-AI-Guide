#!/bin/bash
#SBATCH --job-name=comp-seq
#SBATCH --account=project_462000131
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-gpu=60G
#SBATCH --time=00:20:00

set -euo pipefail

module use /appl/local/training/modules/AI-20240529
module load singularity-userfilesystems singularity-CPEbits

source ../env.sh
: "${CONTAINER:?Set CONTAINER in ../env.sh}"
: "${DATA_PROJECT_DIR:?Set DATA_PROJECT_DIR in ../env.sh}"
SQSH_PATH="../resources/visiontransformer-env.sqsh"
[[ -f "$SQSH_PATH" ]] || { echo "ERROR: Missing sqsh: $SQSH_PATH" >&2; exit 1; }

FORMAT="${1:?Usage: sbatch run-scripts/simple-benchmarks/run-comp-seq.sh <squashfs|lmdb|hdf5>}"

export MPICH_MPIIO_STATS=1
export MPICH_MEMORY_REPORT=1

case "$FORMAT" in
  squashfs)
    TINY_SQSH="$DATA_PROJECT_DIR/data-formats/squashfs/train.squashfs"
    [[ -f "$TINY_SQSH" ]] || { echo "ERROR: Missing squashfs: $TINY_SQSH" >&2; exit 1; }
    time srun singularity exec \
      -B "$SQSH_PATH":/user-software:image-src=/ \
      -B "$TINY_SQSH":/train_images:image-src=/ \
      "$CONTAINER" \
      /user-software/bin/python run-scripts/simple-benchmarks/compare-dataset-tiny.py \
      -n 1 -ff squashfs -N 100000
    ;;
  lmdb)
    time srun singularity exec \
      -B "$SQSH_PATH":/user-software:image-src=/ \
      "$CONTAINER" \
      /user-software/bin/python run-scripts/simple-benchmarks/compare-dataset-tiny.py \
      -n 1 -ff lmdb -N 100000
    ;;
  hdf5)
    time srun singularity exec \
      -B "$SQSH_PATH":/user-software:image-src=/ \
      "$CONTAINER" \
      /user-software/bin/python run-scripts/simple-benchmarks/compare-dataset-tiny.py \
      -n 1 -ff hdf5 -N 100000
    ;;
  *)
    echo "ERROR: Unknown format '$FORMAT'. Use one of: squashfs, lmdb, hdf5." >&2
    exit 1
    ;;
esac
