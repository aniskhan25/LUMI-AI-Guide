#!/bin/bash
#SBATCH --account=project_462000131
#SBATCH --partition=small
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=1750
#SBATCH --time=0:10:00

module use /appl/local/training/modules/AI-20240529
module load singularity-userfilesystems singularity-CPEbits

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/env.sh"

if [[ $1 == 'squashfs' ]]; then
    mkdir -p data-formats/squashfs/
    time srun bash -c 'mksquashfs data-formats/raw/tiny-imagenet-200/val/ data-formats/squashfs/val.squashfs -processors 16 -no-progress'
    time srun bash -c 'mksquashfs data-formats/raw/tiny-imagenet-200/train/ data-formats/squashfs/train.squashfs -processors 16 -no-progress'
elif [[ $1 == 'lmdb' ]]; then
    mkdir -p data-formats/lmdb/
    time srun singularity exec "$CONTAINER" bash -c '$WITH_CONDA && source venv-extension/bin/activate && python scripts/lmdb/convert_to_lmdb.py'
elif [[ $1 == 'hdf5' ]]; then
    mkdir -p data-formats/hdf5/
    time srun singularity exec "$CONTAINER" bash -c '$WITH_CONDA && source venv-extension/bin/activate && python scripts/hdf5/convert_to_hdf5.py'
fi
