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

module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

# MIOpen and PyTorch cache — must be writable per-job directories
MIOPEN_DIR=$(mktemp -d)
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_DIR/cache
export MIOPEN_USER_DB=$MIOPEN_DIR/config
export TORCH_HOME="/scratch/${SLURM_JOB_ACCOUNT}/${USER}/torch_home"
mkdir -p "$TORCH_HOME"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source ../env.sh
: "${CONTAINER:?Set CONTAINER in ../env.sh}"

export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=PHB

time srun singularity exec "$CONTAINER" \
  python -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node=8 \
  visiontransformer_deepspeed.py --deepspeed --deepspeed_config ds_config.json
