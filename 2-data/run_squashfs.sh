#!/bin/bash

#SBATCH --job-name=squashfs-demo
#SBATCH --account=project_462000131
#SBATCH --partition=dev-g

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-gpu=60G

#SBATCH --time=00:10:00

source ../setup.sh
cd "$SLURM_SUBMIT_DIR"

srun singularity exec -B demo.squashfs:/data:image-src=/ "$CONTAINER" \
  python read_squashfs.py
