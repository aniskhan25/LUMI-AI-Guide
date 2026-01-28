#!/bin/bash
#SBATCH --job-name=dataset_compare
#SBATCH --output=results/loaderdiff-nw1-cpu7
#SBATCH --account=project_462000131
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-gpu=60G
#SBATCH --time=00:20:00

# shortcut for getting the binds right
module use /appl/local/containers/ai-modules
module load singularity-AI-bindings

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
source "$ROOT_DIR/env.sh"

export MPICH_MPIIO_STATS=1
export MPICH_MEMORY_REPORT=1

SQUASH="$SQUASH_LARGE"
IMAGES=/Data/CLS-LOC/train/
srun singularity exec -B "$SQUASH":/train_images:image-src=$IMAGES "$CONTAINER" bash -c '$WITH_CONDA && source venv-extension/bin/activate && python compare-dataset.py'
