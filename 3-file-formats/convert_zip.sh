#!/bin/bash
#SBATCH --job-name=large-convert
#SBATCH --account=project_462000131
#SBATCH --partition=standard
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
#SBATCH --time=0:30:00

set -euo pipefail

module use /appl/local/training/modules/AI-20240529
module load singularity-userfilesystems singularity-CPEbits

source ../env.sh
: "${CONTAINER:?Set CONTAINER in ../env.sh}"
: "${DATA_BENCH_DIR:?Set DATA_BENCH_DIR in ../env.sh}"

mkdir -p "$DATA_BENCH_DIR"

time srun --cpu-bind=none singularity exec "$CONTAINER" \
  venv-extension/bin/python scripts/lmdb/convert_large_to_lmdb.py -o "$DATA_BENCH_DIR"
