#!/bin/bash
#SBATCH --job-name=comp-seq
#SBATCH --output=./run-scripts/simple-benchmarks/comp-seq-%j.out
#SBATCH --account=project_462000131
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-gpu=60G
#SBATCH --time=00:20:00

set -euo pipefail

# shortcut for getting the binds right
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
cd "$REPO_ROOT/3-file-formats"

FORMAT="${1:-}"
: "${FORMAT:?Usage: sbatch run-scripts/simple-benchmarks/run-comp-seq.sh <squashfs|lmdb|hdf5>}"

export MPICH_MPIIO_STATS=1
export MPICH_MEMORY_REPORT=1

run_benchmark() {
  local format="$1"
  local bind_arg="${2:-}"
  srun singularity exec $bind_arg "$CONTAINER" bash -c \
    'if [ -n "${WITH_CONDA:-}" ]; then eval "$WITH_CONDA"; fi; source venv-extension/bin/activate; python run-scripts/simple-benchmarks/compare-dataset-tiny.py -n 1 -ff "'"$format"'" -N 100000'
}

case "$FORMAT" in
  squashfs)
    run_benchmark "squashfs" "-B $DATA_PROJECT_DIR/data-formats/squashfs/train.squashfs:/train_images:image-src=/"
    ;;
  lmdb)
    run_benchmark "lmdb"
    ;;
  hdf5)
    run_benchmark "hdf5"
    ;;
  *)
    echo "ERROR: Unknown format '$FORMAT'. Use one of: squashfs, lmdb, hdf5." >&2
    exit 1
    ;;
esac
