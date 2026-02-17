#!/bin/bash

#SBATCH --job-name=profiled-vit
#SBATCH --account=project_462000131
#SBATCH --partition=small-g

#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-gpu=60G

#SBATCH --time=01:00:00
#SBATCH --output=/scratch/project_462000131/anisrahm/slurm/profiled-%j.out

set -euo pipefail

module use /appl/local/containers/ai-modules
module load singularity-AI-bindings

source ../scripts/slurm_bootstrap.sh
bootstrap_repo --require-sqsh

srun singularity exec -B "$SQSH_PATH":/user-software:image-src=/ "$CONTAINER" bash -c '
  set -euo pipefail
  if [ -n "${WITH_CONDA:-}" ]; then eval "$WITH_CONDA"; fi
  source /user-software/bin/activate
  python visiontransformer_profiled.py
'
