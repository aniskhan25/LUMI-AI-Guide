#!/bin/bash
#SBATCH --job-name=large-convert
#SBATCH --account=project_462000131
#SBATCH --partition=standard
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
#SBATCH --time=0:30:00
#SBATCH --output=slurm-convert_zip-%j.out

module use /appl/local/training/modules/AI-20240529
module load singularity-userfilesystems singularity-CPEbits

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

OUT_FOLDER="$DATA_BENCH_DIR"
mkdir -p "$OUT_FOLDER"

srun singularity exec "$CONTAINER" bash -c '$WITH_CONDA && source venv-extension/bin/activate && python scripts/lmdb/convert_large_to_lmdb.py -o "$DATA_BENCH_DIR"'
