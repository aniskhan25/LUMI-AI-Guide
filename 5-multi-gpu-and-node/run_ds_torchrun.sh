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

source ../env.sh
: "${CONTAINER:?Set CONTAINER in ../env.sh}"
: "${TINY_HDF5_PATH:?Set TINY_HDF5_PATH in ../env.sh}"
[[ -f "$TINY_HDF5_PATH" ]] || { echo "ERROR: Missing HDF5 file: $TINY_HDF5_PATH" >&2; exit 1; }
SQSH_PATH="../resources/visiontransformer-env.sqsh"
[[ -f "$SQSH_PATH" ]] || { echo "ERROR: Missing sqsh file: $SQSH_PATH" >&2; exit 1; }

time srun singularity exec -B "$SQSH_PATH":/user-software:image-src=/ "$CONTAINER" \
  /user-software/bin/python -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node=8 \
  visiontransformer_deepspeed.py --deepspeed --deepspeed_config ds_config.json
