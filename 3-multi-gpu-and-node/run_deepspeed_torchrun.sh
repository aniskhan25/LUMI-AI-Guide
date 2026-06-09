#!/bin/bash

#SBATCH --job-name=deepspeed-torchrun
#SBATCH --account=project_462000131
#SBATCH --partition=standard-g

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --time=01:00:00

set -euo pipefail

source ../setup.sh
cd "$SLURM_SUBMIT_DIR"

time srun singularity run "$CONTAINER" \
  python -m torch.distributed.run --numa-binding=exclusive \
  --standalone --nnodes=1 --nproc_per_node=8 \
  visiontransformer_deepspeed.py --deepspeed --deepspeed_config ds_config.json
