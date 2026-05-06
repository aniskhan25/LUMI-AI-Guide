#!/bin/bash

#SBATCH --job-name=aif-synthdata
#SBATCH --account=project_462000131
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-gpu=60G
#SBATCH --time=00:45:00

set -euo pipefail

module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

source ../../env.sh
: "${CONTAINER:?Set CONTAINER in env.sh}"

singularity exec "$CONTAINER" bash -lc "
set -euo pipefail
cd '${SLURM_SUBMIT_DIR:-$PWD}'
python scripts/identify_weak_cases.py --config configs/generate.yaml
python scripts/generate_candidates.py --config configs/generate.yaml
python scripts/filter_candidates.py --config configs/generate.yaml
python scripts/merge_augmented_dataset.py --config configs/generate.yaml
python scripts/rerun_downstream_task.py --config configs/generate.yaml
python scripts/compare_results.py --config configs/generate.yaml
"
