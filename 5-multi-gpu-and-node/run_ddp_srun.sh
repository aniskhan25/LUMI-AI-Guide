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
#SBATCH --output=/scratch/project_462000131/anisrahm/slurm/ddp-srun-%j.out

set -euo pipefail

module use /appl/local/containers/ai-modules
module load singularity-AI-bindings

source ../env.sh
: "${CONTAINER:?Set CONTAINER in ../env.sh}"
: "${TINY_HDF5_PATH:?Set TINY_HDF5_PATH in ../env.sh}"
[[ -f "$TINY_HDF5_PATH" ]] || { echo "ERROR: Missing HDF5 file: $TINY_HDF5_PATH" >&2; exit 1; }
SQSH_PATH="../resources/visiontransformer-env.sqsh"
[[ -f "$SQSH_PATH" ]] || { echo "ERROR: Missing sqsh file: $SQSH_PATH" >&2; exit 1; }

export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=PHB

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NPROCS

CPU_BIND_MASKS="0x00fe000000000000,0xfe00000000000000,0x0000000000fe0000,0x00000000fe000000,0x00000000000000fe,0x000000000000fe00,0x000000fe00000000,0x0000fe0000000000"

srun --cpu-bind="v,mask_cpu=${CPU_BIND_MASKS}" singularity exec -B "$SQSH_PATH":/user-software:image-src=/ "$CONTAINER" bash -c '
  set -euo pipefail
  export RANK="$SLURM_PROCID"
  export LOCAL_RANK="$SLURM_LOCALID"
  /user-software/bin/python ddp_visiontransformer.py
'
