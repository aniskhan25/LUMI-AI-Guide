#!/bin/bash

#SBATCH --job-name=profiled-vit
#SBATCH --account=project_462000131
#SBATCH --partition=dev-g

#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-gpu=60G

#SBATCH --time=01:00:00

set -euo pipefail

module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

source ../env.sh
: "${CONTAINER:?Set CONTAINER in ../env.sh}"

time srun singularity exec "$CONTAINER" python visiontransformer_profiled.py
