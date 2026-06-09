#!/bin/bash

#SBATCH --job-name=quickstart-vit
#SBATCH --account=project_462000131
#SBATCH --partition=dev-g

#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-gpu=60G

#SBATCH --time=01:00:00

source ../setup.sh
cd "$SLURM_SUBMIT_DIR"

singularity exec "$CONTAINER" python visiontransformer.py
