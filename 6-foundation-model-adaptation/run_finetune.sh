#!/bin/bash

#SBATCH --job-name=finetune
#SBATCH --account=project_462000131
#SBATCH --partition=small-g

#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-gpu=60G

#SBATCH --time=01:00:00

set -euo pipefail

module use /appl/local/containers/ai-modules
module load singularity-AI-bindings

source ../env.sh
: "${CONTAINER:?Set CONTAINER in ../env.sh}"

# Adaptation mode: head_only (default), lora, or full
MODE=${1:-head_only}

srun singularity exec "$CONTAINER" \
  python finetune.py --mode "$MODE" --output_dir "outputs/finetune-${MODE}"
