#!/bin/bash

#SBATCH --account=project_462000131
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=1750
#SBATCH --time=0:10:00

set -euo pipefail

module use /appl/local/training/modules/AI-20240529
module load singularity-userfilesystems singularity-CPEbits

source ../env.sh
: "${CONTAINER:?Set CONTAINER in ../env.sh}"
: "${DATA_PROJECT_DIR:?Set DATA_PROJECT_DIR in ../env.sh}"
SQSH_PATH="../resources/visiontransformer-env.sqsh"

FORMAT="${1:?Usage: sbatch convert.sh <squashfs|lmdb|hdf5>}"

DATA_DIR="$DATA_PROJECT_DIR/data-formats"

case "$FORMAT" in
  squashfs)
    mkdir -p "$DATA_DIR/squashfs"
    for split in val train; do
      in_dir="$DATA_DIR/raw/tiny-imagenet-200/$split"
      out_file="$DATA_DIR/squashfs/$split.squashfs"
      time srun --cpu-bind=none mksquashfs "$in_dir" "$out_file" -processors 16 -no-progress -no-xattrs
    done
    ;;
  lmdb)
    mkdir -p "$DATA_DIR/lmdb"
    [[ -f "$SQSH_PATH" ]] || { echo "ERROR: Missing sqsh: $SQSH_PATH" >&2; exit 1; }
    time srun --cpu-bind=none singularity exec "$CONTAINER" \
      -B "$SQSH_PATH":/user-software:image-src=/ \
      /user-software/bin/python scripts/lmdb/convert_to_lmdb.py
    ;;
  hdf5)
    mkdir -p "$DATA_DIR/hdf5"
    [[ -f "$SQSH_PATH" ]] || { echo "ERROR: Missing sqsh: $SQSH_PATH" >&2; exit 1; }
    time srun --cpu-bind=none singularity exec "$CONTAINER" \
      -B "$SQSH_PATH":/user-software:image-src=/ \
      /user-software/bin/python scripts/hdf5/convert_to_hdf5.py
    ;;
  *)
    echo "ERROR: Unknown format '$FORMAT'. Use one of: squashfs, lmdb, hdf5." >&2
    exit 1
    ;;
esac
