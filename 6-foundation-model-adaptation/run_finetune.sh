#!/bin/bash

#SBATCH --job-name=finetune
#SBATCH --account=project_462000131
#SBATCH --partition=dev-g

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-gpu=60G

#SBATCH --time=00:30:00

set -euo pipefail

module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

# MIOpen and PyTorch cache — must be writable per-job directories
MIOPEN_DIR=$(mktemp -d)
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_DIR/cache
export MIOPEN_USER_DB=$MIOPEN_DIR/config
export TORCH_HOME="/scratch/${SLURM_JOB_ACCOUNT}/${USER}/torch_home"
mkdir -p "$TORCH_HOME"

source ../env.sh
: "${CONTAINER:?Set CONTAINER in ../env.sh}"

srun singularity exec "$CONTAINER" \
  python train.py --config configs/baseline.yaml
