#!/bin/bash

#SBATCH --job-name=generation
#SBATCH --account=project_462000131
#SBATCH --partition=dev-g

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-gpu=60G

#SBATCH --time=00:30:00

set -euo pipefail

source ../setup.sh


singularity run "$CONTAINER" \
  python run_generation.py --config configs/generation.yaml
