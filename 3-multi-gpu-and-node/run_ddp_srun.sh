#!/bin/bash

#SBATCH --job-name=ddp-srun
#SBATCH --account=project_462000131
#SBATCH --partition=standard-g

#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-gpu=60G

#SBATCH --time=01:00:00

set -euo pipefail

source ../setup.sh

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT="1${SLURM_JOB_ID:0-4}"
export WORLD_SIZE=$SLURM_NPROCS
export LOCAL_WORLD_SIZE=$SLURM_GPUS_PER_NODE

# Pin each rank to the CPU cores closest to its GPU
# See https://docs.lumi-supercomputer.eu/runjobs/scheduled-jobs/distribution-binding/#gpu-binding
CPU_BIND_MASKS="0x00fe000000000000,0xfe00000000000000,0x0000000000fe0000,0x00000000fe000000,0x00000000000000fe,0x000000000000fe00,0x000000fe00000000,0x0000fe0000000000"

srun --cpu-bind="v,mask_cpu=${CPU_BIND_MASKS}" singularity run "$CONTAINER" bash -c '
  export RANK=$SLURM_PROCID
  export LOCAL_RANK=$SLURM_LOCALID
  python visiontransformer_ddp.py
'
