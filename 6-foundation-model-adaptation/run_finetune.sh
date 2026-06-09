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

module use /appl/local/containers/ai-modules
module load singularity-AI-bindings

source ../env.sh
: "${CONTAINER:?Set CONTAINER in ../env.sh}"

srun singularity exec "$CONTAINER" \
  python train.py --config configs/baseline.yaml
