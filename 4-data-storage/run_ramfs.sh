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

source ../env.sh

: "${CONTAINER:?Set CONTAINER in env.sh}"
: "${TINY_HDF5_PATH:?Set TINY_HDF5_PATH in env.sh}"
[[ -f "$TINY_HDF5_PATH" ]] || { echo "ERROR: Missing HDF5 file: $TINY_HDF5_PATH" >&2; exit 1; }
SQSH_PATH="../resources/visiontransformer-env.sqsh"
[[ -f "$SQSH_PATH" ]] || { echo "ERROR: Missing sqsh file: $SQSH_PATH" >&2; exit 1; }

OUT_DIR="$DATA_BENCH_DIR/ramfs"
mkdir -p "$OUT_DIR"
DST_MODEL="$OUT_DIR/vit_b_16_imagenet.${SLURM_JOB_ID:-$$}.pth"

time srun singularity exec -B "$SQSH_PATH":/user-software:image-src=/ "$CONTAINER" bash -c "
  set -euo pipefail
  cp -a \"$TINY_HDF5_PATH\" /tmp/train_images.hdf5
  /user-software/bin/python visiontransformer_ramfs.py
  cp -a /tmp/vit_b_16_imagenet.pth \"$DST_MODEL\"
"
