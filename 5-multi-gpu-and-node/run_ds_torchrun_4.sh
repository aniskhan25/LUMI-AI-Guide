#!/bin/bash

#SBATCH --job-name=ds-torchrun-4n
#SBATCH --account=project_462000131
#SBATCH --partition=standard-g

#SBATCH --nodes=4
#SBATCH --tasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/project_462000131/anisrahm/slurm/ds-torchrun-4n-%j.out

set -euo pipefail

module use /appl/local/containers/ai-modules
module load singularity-AI-bindings

source ../scripts/slurm_bootstrap.sh
bootstrap_repo --require-sqsh

export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=PHB

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

srun singularity exec -B "$SQSH_PATH":/user-software:image-src=/ "$CONTAINER" bash -c '
  set -euo pipefail
  if [ -n "${WITH_CONDA:-}" ]; then eval "$WITH_CONDA"; fi
  source /user-software/bin/activate
  export CXX=g++-12
  python -m torch.distributed.run --nnodes="$SLURM_JOB_NUM_NODES" --nproc_per_node=8 --node_rank="$SLURM_PROCID" --rdzv_id="$SLURM_JOB_ID" --rdzv_backend=c10d --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" ds_visiontransformer.py --deepspeed --deepspeed_config ds_config.json
'
