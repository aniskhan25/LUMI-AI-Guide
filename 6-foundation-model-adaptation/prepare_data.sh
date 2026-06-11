#!/bin/bash

#SBATCH --job-name=prepare-ag-news
#SBATCH --account=project_462000131
#SBATCH --partition=small-g

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

#SBATCH --time=00:15:00

set -euo pipefail

source ../setup.sh

singularity run "$CONTAINER" \
  python data/prepare_ag_news.py --output data/ag_news
