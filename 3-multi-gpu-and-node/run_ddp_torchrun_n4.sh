#!/bin/bash

#SBATCH --job-name=ddp-torchrun-4n
#SBATCH --account=project_462000131
#SBATCH --partition=standard-g

#SBATCH --nodes=4
#SBATCH --tasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --time=01:00:00

set -euo pipefail

source ../setup.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"


export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=PHB

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500

srun singularity exec "$CONTAINER" bash -c '
  python -m torch.distributed.run \
    --nnodes="$SLURM_JOB_NUM_NODES" --nproc_per_node=8 \
    --rdzv_id="$SLURM_JOB_ID" --rdzv_backend=c10d --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    visiontransformer_ddp.py
'
