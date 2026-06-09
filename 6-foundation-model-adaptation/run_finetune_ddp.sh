#!/bin/bash

#SBATCH --job-name=finetune-ddp
#SBATCH --account=project_462000131
#SBATCH --partition=standard-g

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G

#SBATCH --time=00:45:00

set -euo pipefail

module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

source ../env.sh
: "${CONTAINER:?Set CONTAINER in ../env.sh}"

srun singularity exec "$CONTAINER" \
  python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 \
    train_ddp.py --config configs/baseline.yaml --run-name baseline-run-ddp
