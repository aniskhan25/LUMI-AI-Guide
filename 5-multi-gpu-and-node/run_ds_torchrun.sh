#!/bin/bash

#SBATCH --job-name=ds-torchrun
#SBATCH --account=project_462000131
#SBATCH --partition=standard-g

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/project_462000131/anisrahm/slurm/ds-torchrun-%j.out

set -euo pipefail

module use /appl/local/containers/ai-modules
module load singularity-AI-bindings

source ../scripts/slurm_bootstrap.sh
bootstrap_repo --require-sqsh

export TORCH_EXTENSIONS_DIR="${SCRATCH_ROOT}/torch_extensions/${SLURM_JOB_ID}"
mkdir -p "$TORCH_EXTENSIONS_DIR"
export SINGULARITYENV_TORCH_EXTENSIONS_DIR="$TORCH_EXTENSIONS_DIR"
export SINGULARITYENV_CXX=g++-12

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

srun --nodes=1 --ntasks=1 singularity exec -B "$SQSH_PATH":/user-software:image-src=/ "$CONTAINER" bash -c '
  set -euo pipefail
  if [ -n "${WITH_CONDA:-}" ]; then eval "$WITH_CONDA"; fi
  source /user-software/bin/activate
  export CXX=g++-12
  python -c "import torch; from deepspeed.ops.adam import FusedAdam; p=torch.nn.Parameter(torch.zeros(1, device=\"cuda\")); FusedAdam([p], lr=1e-3); print(\"fused_adam extension ready\")"
'


srun singularity exec -B "$SQSH_PATH":/user-software:image-src=/ "$CONTAINER" bash -c '
  set -euo pipefail
  if [ -n "${WITH_CONDA:-}" ]; then eval "$WITH_CONDA"; fi
  source /user-software/bin/activate
  export CXX=g++-12
  python -m torch.distributed.run --nproc_per_node=8 --nnodes="$SLURM_NNODES" --node_rank="$SLURM_PROCID" --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" ds_visiontransformer.py --deepspeed --deepspeed_config ds_config.json
'
