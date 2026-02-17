#!/bin/bash

#SBATCH --job-name=ddp-torchrun
#SBATCH --account=project_462000131
#SBATCH --partition=standard-g

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G

#SBATCH --time=01:00:00
#SBATCH --output=/scratch/project_462000131/anisrahm/slurm/ddp-torchrun-%j.out

set -euo pipefail

module use  /appl/local/containers/ai-modules
module load singularity-AI-bindings

source ../scripts/slurm_bootstrap.sh
bootstrap_repo --require-sqsh

export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=PHB

srun singularity exec -B "$SQSH_PATH":/user-software:image-src=/ "$CONTAINER" bash -c '
  set -euo pipefail
  if [ -n "${WITH_CONDA:-}" ]; then eval "$WITH_CONDA"; fi
  source /user-software/bin/activate
  python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 ddp_visiontransformer.py
'
