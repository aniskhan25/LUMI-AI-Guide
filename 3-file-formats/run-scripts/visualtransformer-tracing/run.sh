#!/bin/bash
#SBATCH --job-name=noprof-lmdb
#SBATCH --output=noprof-lmdb-%j.out
#SBATCH --account=project_362000131
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=24:00:00


# shortcut for getting the binds right
module use /appl/local/training/modules/AI-20240529
module load singularity-userfilesystems singularity-CPEbits

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
source "$ROOT_DIR/env.sh"

export MPICH_MPIIO_STATS=1
export MPICH_MEMORY_REPORT=1

srun singularity exec "$CONTAINER" bash -c '$WITH_CONDA && source venv-extension/bin/activate && python scripts/lmdb/visualtransformer-lmdb.py'

SQUASH="$SQUASH_LARGE"
IMAGES=/Data/CLS-LOC/train/
srun singularity exec -B "$SQUASH":/train_images:image-src=$IMAGES "$CONTAINER" bash -c '$WITH_CONDA && source venv-extension/bin/activate && python scripts/squashfs/visualtransformer-squashfs.py'

### Tiny imagenet scripts
# srun singularity exec $CONTAINER bash -c '$WITH_CONDA && source venv-extension/bin/activate && python scripts/hdf5/visualtransformer-hdf5.py'
# srun singularity exec -B data-formats/squashfs/train.squashfs:/train_images:image-src=/ $CONTAINER bash -c '$WITH_CONDA && source venv-extension/bin/activate && python scripts/squashfs/visualtransformer-squashfs.py'
