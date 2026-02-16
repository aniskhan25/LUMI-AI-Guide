#!/bin/bash

#SBATCH --account=project_462000131
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=1750
#SBATCH --time=0:10:00
#SBATCH --output=/scratch/project_462000131/anisrahm/slurm/convert-%j.out

set -euo pipefail

module use /appl/local/training/modules/AI-20240529
module load singularity-userfilesystems singularity-CPEbits

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
if [[ "${SUBMIT_DIR##*/}" == "3-file-formats" ]]; then
  REPO_ROOT="$(cd -- "$SUBMIT_DIR/.." && pwd)"
else
  REPO_ROOT="$SUBMIT_DIR"
fi

if [[ ! -f "$REPO_ROOT/env.sh" ]]; then
  echo "ERROR: env.sh not found. Submit from repo root or 3-file-formats." >&2
  exit 1
fi

source "$REPO_ROOT/env.sh"

FORMAT="${1:-}"
: "${FORMAT:?Usage: sbatch convert.sh <squashfs|lmdb|hdf5>}"

cd "$REPO_ROOT/3-file-formats"

DATA_DIR="$DATA_PROJECT_DIR/data-formats"
SRUN="${SRUN:-srun --cpu-bind=none}"

run_in_container () {
  local script="$1"
  time $SRUN singularity exec "$CONTAINER" bash -c \
    '$WITH_CONDA && source venv-extension/bin/activate && python '"$script"
}

case "$FORMAT" in
  squashfs)
    mkdir -p "$DATA_DIR/squashfs"
    time $SRUN mksquashfs "$DATA_DIR/raw/tiny-imagenet-200/val" "$DATA_DIR/squashfs/val.squashfs" -processors 16 -no-progress -no-xattrs
    time $SRUN mksquashfs "$DATA_DIR/raw/tiny-imagenet-200/train" "$DATA_DIR/squashfs/train.squashfs" -processors 16 -no-progress -no-xattrs
    ;;
  lmdb)
    mkdir -p "$DATA_DIR/lmdb"
    run_in_container scripts/lmdb/convert_to_lmdb.py
    ;;
  hdf5)
    mkdir -p "$DATA_DIR/hdf5"
    run_in_container scripts/hdf5/convert_to_hdf5.py
    ;;
  *)
    echo "ERROR: Unknown format '$FORMAT'. Use one of: squashfs, lmdb, hdf5." >&2
    exit 1
    ;;
esac
