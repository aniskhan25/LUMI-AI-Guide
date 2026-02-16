#!/bin/bash
#SBATCH --account=project_462000131
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-gpu=60G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/project_462000131/anisrahm/slurm/ramfs-%j.out

set -euo pipefail

module use /appl/local/containers/ai-modules
module load singularity-AI-bindings

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
if [[ "${SUBMIT_DIR##*/}" == "4-data-storage" ]]; then
  REPO_ROOT="$(cd -- "$SUBMIT_DIR/.." && pwd)"
else
  REPO_ROOT="$SUBMIT_DIR"
fi

if [[ ! -f "$REPO_ROOT/env.sh" ]]; then
  echo "ERROR: env.sh not found. Submit from repo root or 4-data-storage." >&2
  exit 1
fi

source "$REPO_ROOT/env.sh"

: "${CONTAINER:?Set CONTAINER in env.sh}"
: "${TINY_HDF5_PATH:?Set TINY_HDF5_PATH in env.sh}"
[[ -f "$TINY_HDF5_PATH" ]] || { echo "ERROR: Missing HDF5 file: $TINY_HDF5_PATH" >&2; exit 1; }
SQSH_PATH="$REPO_ROOT/resources/visiontransformer-env.sqsh"
[[ -f "$SQSH_PATH" ]] || { echo "ERROR: Missing sqsh file: $SQSH_PATH" >&2; exit 1; }

OUT_DIR="$DATA_BENCH_DIR/ramfs"
mkdir -p "$OUT_DIR"

export SRC_HDF5="$TINY_HDF5_PATH"
export APP_SCRIPT="$REPO_ROOT/4-data-storage/visiontransformer_ramfs.py"
export DST_MODEL="$OUT_DIR/vit_b_16_imagenet.${SLURM_JOB_ID:-$$}.pth"
export SINGULARITYENV_PREPEND_PATH="/user-software/bin"

srun singularity exec -B "$SQSH_PATH":/user-software:image-src=/ "$CONTAINER" bash -c '
  set -euo pipefail
  if [ -n "${WITH_CONDA:-}" ]; then eval "$WITH_CONDA"; fi
  source /user-software/bin/activate
  time cp -a "$SRC_HDF5" /tmp/train_images.hdf5
  time python "$APP_SCRIPT"
  time /bin/cp -a /tmp/vit_b_16_imagenet.pth "$DST_MODEL"
'
