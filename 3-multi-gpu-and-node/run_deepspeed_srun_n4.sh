#!/bin/bash

#SBATCH --job-name=deepspeed-srun-4n
#SBATCH --account=project_462000131
#SBATCH --partition=standard-g

#SBATCH --nodes=4
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-gpu=60G

#SBATCH --time=01:00:00

set -euo pipefail

module use /appl/local/containers/ai-modules
module load singularity-AI-bindings

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source ../env.sh
: "${CONTAINER:?Set CONTAINER in ../env.sh}"

export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=PHB

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NPROCS

# Pin each rank to the CPU cores closest to its GPU
# See https://docs.lumi-supercomputer.eu/runjobs/scheduled-jobs/distribution-binding/#gpu-binding
CPU_BIND_MASKS="0x00fe000000000000,0xfe00000000000000,0x0000000000fe0000,0x00000000fe000000,0x00000000000000fe,0x000000000000fe00,0x000000fe00000000,0x0000fe0000000000"

srun --cpu-bind="v,mask_cpu=${CPU_BIND_MASKS}" singularity exec "$CONTAINER" bash -c '
  export RANK=$SLURM_PROCID
  export LOCAL_RANK=$SLURM_LOCALID
  python visiontransformer_deepspeed.py --deepspeed --deepspeed_config ds_config.json
'
